from typing import List

import pandas as pd
import os
from pathlib import Path
import numpy as np
from sklearn.base import TransformerMixin


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

    __x_data: pd.DataFrame
    __y_data: pd.DataFrame

    def __init__(self,
                 root_dir: str,
                 x_labels: List[str],
                 y_labels: List[str],
                 train: bool = True,
                 batch_size: int = 1,
                 scaler: str = "StandardScaler"):

        self.root_dir = root_dir
        self.train = train
        self.batch_size = batch_size
        self.y_labels = y_labels
        self.x_labels = x_labels

        self.__x_data = self.data[self.x_labels]
        self.__y_data = self.data[self.y_labels]

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

        match scaler:
            case "StandardScaler":
                from sklearn.preprocessing import StandardScaler
                self.__x_transform = StandardScaler()
                self.__y_transform = StandardScaler()
            case "MinMaxScaler":
                from sklearn.preprocessing import MinMaxScaler
                self.__x_transform = MinMaxScaler()
                self.__y_transform = MinMaxScaler()
            case "RobustScaler":
                from sklearn.preprocessing import RobustScaler
                self.__x_transform = RobustScaler()
                self.__y_transform = RobustScaler()
            case _:
                raise NotImplementedError("Scaler not supported.")

        self.__transform_x_y()

    def __len__(self):
        """Returns length of dataset."""
        return len(self.data) - self.batch_size

    def __getitem__(self, idx: int) -> (np.ndarray, np.ndarray):
        """Returns sample from dataset in format (x, y) where x is numpy.ndarray, and y is scalar or ndarray too."""
        x = self.__x_data.iloc[idx:idx + self.batch_size]
        y = self.__y_data.iloc[idx:idx + self.batch_size]
        return x, y

    def __transform_x_y(self) -> None:
        """Transforms x and y data using any of Scalers."""
        self.__x_data = self.__x_transform.fit_transform(self.__x_data)
        self.__y_data = self.__y_transform.fit_transform(self.__y_data)

    def x_untransform(self, x: np.ndarray) -> np.ndarray:
        """Untransforms x data using x scaler."""
        return self.__x_transform.inverse_transform(x)

    def y_untransform(self, y: np.ndarray) -> np.ndarray:
        """Untransforms y data using y scaler."""
        return self.__y_transform.inverse_transform(y)
