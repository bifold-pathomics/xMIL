import random

from torch.utils.data import Sampler


def sampling_args_factory(sampler, dataset):
    if sampler is not None:
        if sampler == "bts":
            sampling_args = {"sampler": BTSSampler(dataset)}
        else:
            raise ValueError(f"Unknwon sampler: {sampler}")
    else:
        sampling_args = {"shuffle": True}
    return sampling_args


class BTSSampler(Sampler):

    def __init__(self, dataset, tss_column="tss", label_column="label"):
        super(BTSSampler, self).__init__(dataset)
        if len(dataset.label_cols) > 1:
            raise NotImplementedError("BTSSampler does not support multi-task learning")
        # Load and check metadata
        self.metadata = dataset.get_metadata()
        if tss_column not in self.metadata.columns:
            ValueError(
                f"BTSSampler requires metadata to contain a column: {tss_column}"
            )
        if label_column not in self.metadata.columns:
            ValueError(
                f"BTSSampler requires metadata to contain a column: {label_column}"
            )
        self.tss_column = tss_column
        self.label_column = label_column
        # Compute sampling parameters
        self.labels = self.metadata[label_column].unique().tolist()
        self.min_sample_count = (
            self.metadata.groupby([tss_column])[label_column]
            .value_counts()
            .unstack()
            .fillna(0)
            .min(axis=1)
        )
        self.tss_weights = self.min_sample_count / self.min_sample_count.sum()

    def __len__(self):
        return len(self.metadata)

    def __iter__(self):
        counter = 0
        while counter < self.__len__():
            tss_draw = self.tss_weights.sample(
                n=1, weights=self.tss_weights, replace=True
            ).index.values[0]
            tss_samples = []
            for label in self.labels:
                tss_samples.append(
                    self.metadata[
                        (self.metadata[self.tss_column] == tss_draw)
                        & (self.metadata[self.label_column] == label)
                    ]
                    .sample(1)
                    .index.values[0]
                )
            random.shuffle(tss_samples)
            for sample in tss_samples:
                yield sample
                counter += 1
                if counter == self.__len__():
                    break
