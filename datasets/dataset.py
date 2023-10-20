import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from torchvision import datasets, transforms

from sklearn.preprocessing import LabelEncoder

from tensor import Tensor


class Dataset:
    """Base Interface for all datasets."""

    def __len__(self):
        """Returns length of dataset."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> (np.ndarray, np.ndarray):
        """Returns sample from dataset in format (x, y) where x is numpy.ndarray, and y is scalar or ndarray too."""
        raise NotImplementedError


class Dataset_imgs(Dataset):
    """Class for images of numbers dataset."""

    def __init__(self, name: str = "MNIST"):
        # Load dataset
        match name:
            case "MNIST":
                self.data = datasets.MNIST(root="datasets", train=True, download=True)
            case "CIFAR10":
                self.data = datasets.CIFAR10(root="datasets", train=True, download=True)
            case "CIFAR100":
                self.data = datasets.CIFAR100(root="datasets", train=True, download=True)
            case _:
                raise NotImplementedError("Dataset not supported.")

        # Split dataset into x and y
        self.x = np.array(self.data.data)
        self.y = np.array(self.data.targets)

        # Transform x
        self.x = self.x / 255.0

    def __len__(self):
        """Returns length of dataset."""
        return len(self.x)

    def __getitem__(self, idx: int) -> (np.ndarray, np.ndarray):
        """Returns sample from dataset in format (x, y) where x is numpy.ndarray, and y is scalar or ndarray too."""
        x = self.x[idx].reshape(1, -1)
        y = np.zeros((10, 1))
        y[self.y[idx]] = 1
        return Tensor(x), Tensor(y)


class Dataset_nums(Dataset):
    """Base class for all numeric datasets.
    The dataset creates a Pandas DataFrame and serves as a base class for all datasets.

    Args:
        root_dir (str): Root directory of the dataset.
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        batch_size (int): Size of batch.
        x_labels (List[str]): List of labels for x values.
        y_labels (List[str]): List of labels for y values.
        leave_last (bool): If True, leaves last batch if it is smaller than batch_size.
    """
    root_dir: str
    train: bool
    batch_size: int
    data: pd.DataFrame
    x_labels: List[str]
    y_labels: List[str]
    leave_last: bool

    __x_data: pd.DataFrame
    __y_data: pd.DataFrame
    __labeled: bool = False

    def __init__(self,
                 root_dir: str,
                 x_labels: List[str],
                 y_labels: List[str],
                 train: bool = True,
                 batch_size: int = 1,
                 leave_last: bool = False,
                 scaler: str = "StandardScaler",
                 *args, **kwargs):

        self.root_dir = root_dir
        self.train = train
        self.batch_size = batch_size
        self.y_labels = y_labels
        self.x_labels = x_labels
        self.leave_last = leave_last

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

        self.__x_data = pd.DataFrame(self.data[self.x_labels])
        self.__y_data = pd.DataFrame(self.data[self.y_labels])

        self.__label_encoder = LabelEncoder()

        # Check if __y_data is categorical string data
        if self.__y_data.dtypes[0] == "object" or self.__y_data.dtypes[0] == "string":
            self.__labeled = True
            self.__y_data = pd.DataFrame(self.__label_encoder.fit_transform(self.__y_data))

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
        if self.leave_last:
            return len(self.__x_data) // self.batch_size + 1
        else:
            return len(self.__x_data) // self.batch_size

    def __getitem__(self, idx: int) -> (np.ndarray, np.ndarray):
        """Returns sample from dataset in format (x, y) where x is numpy.ndarray, and y is scalar or ndarray too."""
        if idx >= len(self) or idx < 0:
            raise IndexError("Index out of range.")

        if idx == len(self) - 1 and not self.leave_last:
            x = self.__x_data.iloc[idx * self.batch_size:].values
            y = self.__y_data.iloc[idx * self.batch_size:].values
        else:
            x = self.__x_data.iloc[idx * self.batch_size:(idx + 1) * self.batch_size].values
            y = self.__y_data.iloc[idx * self.batch_size:(idx + 1) * self.batch_size].values

        return x, y

    def __transform_x_y(self) -> None:
        """Transforms x and y data using any of Scalers."""
        self.__x_data = pd.DataFrame(self.__x_transform.fit_transform(self.__x_data))
        self.__y_data = pd.DataFrame(self.__y_transform.fit_transform(self.__y_data))

    def x_untransform(self, x: np.ndarray) -> np.ndarray:
        """Untransforms x data using x scaler."""
        return self.__x_transform.inverse_transform(x)

    def y_untransform(self, y: np.ndarray) -> np.ndarray:
        """Untransforms y data using y scaler."""
        y = self.__y_transform.inverse_transform(y)
        if self.__labeled:
            y = y.astype(int)
            y = self.__label_encoder.inverse_transform(y)
        return y


if __name__ == '__main__':
    dataset = Dataset_imgs()
    x, y = dataset[0]
    print(x.shape)
    print(y.shape)
