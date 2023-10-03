from typing import List

import pandas as pd
import os
from pathlib import Path
import numpy as np


class Dataset:
    """Base class for all datasets.
    The dataset creates a Pandas DataFrame and serves as a base class for all datasets.

    Args:
        root_dir (str): Root directory of the dataset.
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        batch_size (int): Size of batch.
        x_labels (List[str]): List of labels for x values.
        y_labels (List[str]): List of labels for y values.
    """
    root_dir: str
    train: bool
    batch_size: int
    data: pd.DataFrame
    x_labels: List[str]
    y_labels: List[str]

    def __init__(self,
                 root_dir: str,
                 x_labels: List[str],
                 y_labels: List[str],
                 train: bool = True,
                 batch_size: int = 1):

        self.root_dir = root_dir
        self.train = train
        self.batch_size = batch_size
        self.y_labels = y_labels
        self.x_labels = x_labels

        if not os.path.isfile(self.root_dir):
            raise FileNotFoundError("Dataset file not found.")

        match Path(self.root_dir).suffix:
            case ".csv":
                self.data = pd.read_csv(self.root_dir)
            case ".json":
                self.data = pd.read_json(self.root_dir)
            case ".xlsx":
                self.data = pd.read_excel(self.root_dir)
            case _:
                raise NotImplementedError("Dataset file format not supported.")

    def __len__(self):
        """Returns length of dataset."""
        return len(self.data) - self.batch_size

    def __getitem__(self, idx: int) -> (np.ndarray, np.ndarray):
        """Returns sample from dataset in format (x, y) where x is numpy.ndarray, and y is scalar or ndarray too."""
        x = self.data[self.x_labels].iloc[idx:idx + self.batch_size]
        y = self.data[self.y_labels].iloc[idx:idx + self.batch_size]
        return x, y
