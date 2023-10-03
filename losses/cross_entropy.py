import numpy as np

from .loss import Loss


class BinaryCrossEntropy(Loss):
    """Computes binary cross entropy loss."""
    @staticmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class CrossEntropy(Loss):
    """Computes cross entropy loss."""
    @staticmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
