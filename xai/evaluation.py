
import json
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import torch


class xMILEval:
    def __init__(self, xmodel, classifier, heatmap_type, scores_df=None):
        self.xmodel = xmodel
        self.classifier = classifier
        self.heatmap_type = heatmap_type
        self.scores_df = scores_df

    def _patch_drop_or_add_oneslide(self, batch, attribution_strategy='original', order='morf',
                                    approach='drop', strategy='1%-of-all',
                                    patch_scores=None, verbose=False):
        """
        performs patch dropping for one slide. heatmaps are computed for the target class.
        :param batch:
        :param heatmap_type: (str) can be either 'lrp' or 'attention'
        :param attribution_strategy: (str) 'original' uses the patch scores from the explanation method directly
                                           'absolute' uses the absolute value of the patch scores
                                           'random' shuffles the patch scores (used for building random baseline)
        :param order: (str) 'morf': most relevant first, 'morl': most relevant last
        :param approach: (str) 'drop', 'add'
        :param strategy: (str) can be either 'one-by-one', f'remaining-{P}-perc' where P is the percent of
        remaining patches to be flipped, or '{P}%-of-all' where P is the percentage of all the
        patches in the slide to be dropped in each iteration. 0<=P<=100
        :param patch_scores: (1D numpy array) default None. if not None: these patch scores are used
        :param verbose: (bool)
        :return:
                predicted_probs: numpy array of predicted probabilities (for the target class) over dropping iterations
                false_pred: (bool) True if the model's prediction did not match the target class

        """
        n_patches = batch['bag_size'].item()

        # compute or read patch scores and sort them based on the attribution strategy.
        if attribution_strategy == 'random':
            patch_scores = np.random.randn(n_patches)
        else:
            if patch_scores is None:
                patch_scores = self.xmodel.get_heatmap(batch, self.heatmap_type, verbose)
            patch_scores = patch_scores[-n_patches:]

            if attribution_strategy == 'absolute':
                patch_scores = np.abs(patch_scores)

        # sort them based on the given order
        ind_sorted = np.argsort(patch_scores)  # most relevant last

        if order == 'morf':  # most relevant first
            ind_sorted = ind_sorted[::-1]  # the index of sorted patch scores in descending way

        probs_orig, _, _, _ = self.classifier.validation_step(batch, softmax=True)  # the original probability

        batch_ = deepcopy(batch)
        batch_['features'] = torch.zeros(batch['features'].shape)
        probs_zero, _, _, _ = self.classifier.validation_step(batch_, softmax=True)

        # we keep track of the slides for which the model prediction is false
        false_pred = probs_orig[0, batch['targets'].item()].item() <= 0.5

        # collector for the target class probabilities in each iteration of patch dropping
        if approach == 'drop':
            predicted_probs = [probs_orig[0, self.xmodel.set_explained_class(batch)].item()]
        elif approach == 'add':
            predicted_probs = [probs_zero[0, self.xmodel.set_explained_class(batch)].item()]

        if '%-of-all' in strategy:
            perc = int(strategy[:strategy.index('%')])
            perc = np.arange(perc, 101, perc)
            if perc[-1] != 100:
                perc = np.append(perc, 100)
            percentiles = np.percentile(patch_scores, perc)
            bins = np.append(patch_scores.min(), percentiles)
            n_drop_array, _ = np.histogram(patch_scores, bins=bins)

        elif strategy == 'one-by-one':
            n_drop_array = np.array([1 for _ in range(n_patches)])

        ind_add = []
        flag_empty_bag = False
        flag_full_bag = False
        for n_drop in n_drop_array:
            ind_add += list(ind_sorted[:n_drop])  # keep the n_drop top-ranked
            ind_sorted = np.delete(ind_sorted, [i for i in range(n_drop)])  # drop the n_drop top-ranked patches

            batch_ = deepcopy(batch)  # the batch dictionary for the kept patches after dropping/adding

            if approach == 'drop':
                if ind_sorted.size > 0:
                    # the remaining patches - morf:=most relevant first
                    batch_['features'] = batch['features'][..., sorted(ind_sorted), :]
                    bag_size = len(ind_sorted)
                else:
                    flag_empty_bag = True
                    bag_size = n_patches
            elif approach == 'add':
                if len(ind_add) == n_patches:
                    flag_full_bag = True
                    bag_size = n_patches
                else:
                    batch_['features'] = batch['features'][..., sorted(ind_add), :]
                    bag_size = len(ind_add)

            batch_['bag_size'] = torch.tensor([bag_size])

            if flag_empty_bag:
                probs = probs_zero
            elif flag_full_bag:
                probs = probs_orig
            else:
                probs, _, _, _ = self.classifier.validation_step(batch_, softmax=True)

            predicted_probs.append(probs[0, self.xmodel.set_explained_class(batch)].item())

        return predicted_probs, false_pred

    def patch_drop_or_add(self, data_loader, attribution_strategy='original',
                          order='morf', approach='drop',
                          strategy='remaining-10-perc', max_bag_size=None, min_bag_size=0, verbose=False):

        """

        :param data_loader:
        :param heatmap_type: (str) can be either 'lrp' or 'attention'
        :param attribution_strategy: (str) 'original' uses the patch scores from the explanation method directly
                                           'absolute' uses the absolute value of the patch scores
                                           'random' shuffles the patch scores (used for building random baseline)
        :param order: (str) 'morf': most relevant first, 'morl': most relevant last
        :param approach: (str) 'drop', 'add'
        :param strategy: (str) can be either 'one-by-one' or f'remaining-{P}-perc' where P is the percent of
        remaining patches to be flipped
        :param flip_threshold: (str) f'{N}-percentile' or f'{N}%-of-max-val' or 'random-relevance'.
        the latter sets a threshold to total relevance divided by number of patches
        :param max_bag_size: (int) if not None, the slides with more than max_bag_size patches will be skipped
        :param min_bag_size: (int)
        :param verbose: (bool)
        :return:
            df_results: with columns slide_id (str), false_pred (bool), and predicted_probs (list)

        """

        # containers for results of each batch

        max_bag_size_ = torch.inf if (max_bag_size is None or max_bag_size < 0) else max_bag_size

        df_results = pd.DataFrame()

        for i_batch, batch in enumerate(tqdm(data_loader)):
            torch.cuda.empty_cache()
            slide_id = batch['sample_ids']['slide_id'][0]

            if self.scores_df is not None:
                if slide_id not in self.scores_df['slide_id'].values:
                    res_this_slide = pd.DataFrame(data={
                        'slide_id': [slide_id],
                        'predicted_probs': [None],
                        'false_pred': [None],
                    })
                    df_results = pd.concat([df_results, res_this_slide], ignore_index=True)
                    continue

                df_this_slide = self.scores_df[self.scores_df['slide_id'] == slide_id]
                patch_scores = np.array(json.loads(df_this_slide[f'patch_scores_{self.heatmap_type}'].item()))

            else:
                patch_scores = None

            if min_bag_size <= batch['bag_size'].item() <= max_bag_size_:
                predicted_probs_, false_pred_ = \
                    self._patch_drop_or_add_oneslide(batch, attribution_strategy=attribution_strategy,
                                                     order=order, approach=approach, strategy=strategy,
                                                     patch_scores=patch_scores, verbose=verbose)
                res_this_slide = pd.DataFrame(data={
                    'slide_id': [slide_id],
                    'predicted_probs': [predicted_probs_],
                    'false_pred': [false_pred_],
                })

            else:
                res_this_slide = pd.DataFrame(data={
                    'slide_id': [slide_id],
                    'predicted_probs': [None],
                    'false_pred': [None],
                })
            df_results = pd.concat([df_results, res_this_slide], ignore_index=True)

        return df_results

