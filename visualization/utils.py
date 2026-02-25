import numpy as np
from copy import deepcopy
from scipy.stats import ttest_rel
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.cbook import boxplot_stats
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def clean_outliers_fliers(data, return_idxs=False):
    data_clean = deepcopy(data)
    stat = boxplot_stats(data)[0]
    data_clean[data < stat["whislo"]] = stat["whislo"]
    data_clean[data > stat["whishi"]] = stat["whishi"]
    filter_idxs = np.logical_and(data >= stat["whislo"], data <= stat["whishi"])
    data_no_outlier = data[filter_idxs]
    if return_idxs:
        return data_clean, data_no_outlier, filter_idxs
    return data_clean, data_no_outlier


def convert2rgb(mat, cmap_name="coolwarm", zero_centered=True):
    """
    converts the given matrix to RGB values
    """
    cmap = plt.get_cmap(cmap_name)
    if zero_centered:
        max_scalar = np.max(np.abs(mat))
        rgb_values = (max_scalar + mat) / (2 * max_scalar)
    else:
        min_scalar = np.min(mat)
        max_scalar = np.max(mat)
        rgb_values = (mat - min_scalar) / (max_scalar - min_scalar)
    return cmap(rgb_values).squeeze()[:, :-1]


def plot_colorbar(ax, data, cmap="coolwarm", ori="vertical", zero_centered=True):
    """
    lazy plotting of a colormap corresponding to the given data
    """
    if zero_centered:
        max_scalar = np.max(np.abs(data))
        norm = Normalize(vmin=-max_scalar, vmax=max_scalar)
    else:
        norm = Normalize(vmin=min(data), vmax=max(data))
    _ = ColorbarBase(ax, cmap=cmap, norm=norm, orientation=ori)


def plot_line_mean_se(
    x, mean_values, std_errors, n_se, color="blue", alpha=0.3, label=""
):
    plt.plot(x, mean_values, color=color, label=label)
    plt.fill_between(
        x,
        mean_values - n_se * std_errors,
        mean_values + n_se * std_errors,
        color=color,
        alpha=alpha,
    )


def plot_boxplot_paired(
    data,
    xticks,
    ylabel,
    violinplot=True,
    datapoints=None,
    paired=None,
    pair_linewidth=0.1,
    datapoint_size=3,
    alpha=0.5,
    datapoints_color="lightskyblue",
    jitter_std=0.05,
    notch=True,
    palette="colorblind",
    showfliers=False,
):

    if violinplot:
        ax = sns.violinplot(
            data, palette=palette, showfliers=showfliers, width=1.2, linewidth=1.7
        )
    else:
        ax = sns.boxplot(data, notch=notch, palette=palette, showfliers=False)

    ax.set_xticklabels(xticks)
    plt.ylabel(ylabel)

    datapoints = [] if datapoints is None else datapoints

    for i_data in datapoints:

        if not showfliers:
            outliers = [
                y for stat in boxplot_stats(data[i_data]) for y in stat["fliers"]
            ]
            data_i = [d for d in data[i_data] if d not in outliers]
        else:
            data_i = data[i_data]

        n_points = len(data_i)
        plt.plot(
            np.ones(n_points) * i_data + np.random.randn(n_points) * jitter_std,
            data_i,
            ".",
            color=datapoints_color,
            markersize=datapoint_size,
        )

        mean_i = np.mean(data[i_data])
        plt.plot(i_data, mean_i, ".", color="red", markersize=datapoint_size * 10)

    if paired is not None:
        if not showfliers:
            outliers_0 = [
                y for stat in boxplot_stats(data[paired[0]]) for y in stat["fliers"]
            ]
            outliers_1 = [
                y for stat in boxplot_stats(data[paired[1]]) for y in stat["fliers"]
            ]
        for d1, d2 in zip(data[paired[0]], data[paired[1]]):
            if showfliers or (
                not showfliers and d1 not in outliers_0 and d2 not in outliers_1
            ):
                x = np.array(list(paired))
                y = np.array([d1, d2])
                plt.plot(x, y, "-", linewidth=pair_linewidth, alpha=alpha)

    plt.grid(True, zorder=0)
    ax.set_axisbelow(True)


def significance_bar_helper(x1, x2, pvalue, ax, level, tag="star"):
    if pvalue > 0.01:
        return
    bottom, top = ax.get_ylim()
    y_range = top - bottom

    bar_height = (y_range * 0.01 * level) + top
    bar_tips = bar_height - (y_range * 0.01)
    text_height = bar_height + (y_range * 0.0001)

    if pvalue < 1e-4:
        sig_symbol = "***" if tag == "star" else "p < 0.0001"
    elif pvalue < 1e-3:
        sig_symbol = "**" if tag == "star" else "p < 0.001"
    elif pvalue < 1e-2:
        sig_symbol = "*" if tag == "star" else "p < 0.01"

    plt.plot(
        [x1, x1, x2, x2], [bar_tips, bar_height, bar_height, bar_tips], lw=1.3, c="k"
    )
    plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha="center", va="bottom", c="k")


def compute_SemiBluecher_faithfulness(probs_add, probs_drop, skip_ids):
    ave_auc = dict()
    for heatmap_type in probs_add.keys():
        ave_auc[heatmap_type] = [
            np.mean(p_add) - np.mean(p_drop)
            for i, (p_add, p_drop) in enumerate(
                zip(probs_add[heatmap_type], probs_drop[heatmap_type])
            )
            if i not in skip_ids
        ]
    return ave_auc


def plot_boxplot_with_significance(
    ax,
    data_list,
    box_labels,
    ylabel,
    palette,
    showfliers=False,
    significance_bars=True,
    verbose=False,
):
    plot_boxplot_paired(
        data_list,
        box_labels,
        ylabel,
        violinplot=False,
        datapoints=list(range(len(box_labels))),
        paired=None,
        jitter_std=0.05,
        datapoints_color="black",
        pair_linewidth=0.001,
        datapoint_size=1,
        alpha=0.2,
        notch=False,
        palette=palette,
        showfliers=showfliers,
    )

    bonferroni_factor = 2 * len(box_labels) - 1
    # 1st data type vs others
    ind1 = 0
    for ind2 in range(1, len(box_labels)):
        tval, pval = ttest_rel(data_list[ind1], data_list[ind2])
        pval *= bonferroni_factor
        if verbose:
            print(f"{box_labels[ind1]} vs {box_labels[ind2]}: t={tval}, p={pval}")
        if significance_bars and pval < 0.01:
            significance_bar_helper(ind1, ind2, pval, ax, 1)

    # all vs the last data type
    ind2 = len(data_list) - 1
    for ind1 in range(1, len(box_labels) - 1):
        tval, pval = ttest_rel(data_list[ind1], data_list[ind2])
        pval *= bonferroni_factor
        if verbose:
            print(f"{box_labels[ind1]} vs {box_labels[ind2]}: t={tval}, p={pval}")
        if significance_bars and pval < 0.01:
            significance_bar_helper(ind1, ind2, pval, ax, 1)


def compute_auc(data, skip_ids):
    ave_prob = []
    for i_heatmap, heatmap_type in enumerate(data.keys()):
        ave_prob.append(
            [
                np.mean(prob)
                for i, prob in enumerate(data[heatmap_type])
                if i not in skip_ids
            ]
        )

    return ave_prob


def plot_tensorboard(path_list, legend=False, root_path=None):
    """
    plots the tensorboard events in the given paths
    :param path_list: [list] a list of paths with tensorboard events
    :param legend: [bool] defines whether the legends are shown
    :param root_path: [str] a path to the root path of all the events. effective only if legend=True.
    if legend=True and root_path=None, root_path=path_list[0]
    """
    plt.figure(figsize=(15, 10))

    root_path = path_list[0] if root_path is None else root_path

    for path1 in path_list:

        label = path1[len(root_path) :]  # the label of this path on the legend

        event_file_path = path1

        event_acc = EventAccumulator(event_file_path)
        event_acc.Reload()

        scalar_tags_ = event_acc.Tags()["scalars"]
        scalar_tags = ["auc/train", "auc/val", "auc/test"]

        # backward compatibility --------------
        # scalar_tags = ['auc/train', 'auc/val', 'auc/test',
        #                'loss/epoch/train', 'loss/epoch/val', 'loss/test']

        if "loss/epoch/train" in scalar_tags_:
            scalar_tags.append("loss/epoch/train")
        else:
            scalar_tags.append("loss/train")

        if "loss/epoch/val" in scalar_tags_:
            scalar_tags.append("loss/epoch/val")
        else:
            scalar_tags.append("loss/val")

        scalar_tags.append("loss/test")

        for i, tag in enumerate(scalar_tags):
            plt.subplot(2, 3, i + 1)
            events = event_acc.Scalars(tag)

            steps = [event.step for event in events]
            values = [event.value for event in events]

            if "test" in tag:
                plt.plot(steps, values, "o", label=label)
            else:
                plt.plot(steps, values, label=label)
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title(tag)
            if legend:
                plt.legend()

            plt.grid(True)
    plt.show()
