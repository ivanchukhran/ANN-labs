from datasets.dataset import Dataset
import random


class DataLoader:
    """
        Base class for all data loaders.

        Args:
            dataset (Dataset): Dataset to load.
            shuffle (bool): If True, shuffles the data.
    """
    dataset: Dataset
    shuffle: bool

    def __init__(self, dataset: Dataset, shuffle: bool = False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.len = len(self.dataset)

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(indices)

        for index in indices:
            yield self.dataset[index]
