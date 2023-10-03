import pandas as pd


class Dataset:
    """Base class for all datasets.
    The dataset creates a Pandas DataFrame and serves as a base class for all datasets.

    Args:
        root_dir (str): Root directory of the dataset.
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        """
    root_dir: str
    train: bool
    batch_size: int
    data: pd.DataFrame

    def __init__(self, root_dir: str, train: bool = True, batch_size: int = 1):
        self.root_dir = root_dir
        self.train = train
        self.batch_size = batch_size
        # TODO: Implement data loading from CSV or other format.
        # TODO: Implement batch loading.

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        """Returns sample from dataset in format (x, y) where x is numpy.ndarray, and y is scalar or ndarray too."""
        raise NotImplementedError("You should implement this method in subclass.")
