import numpy as np

from .loss import Loss


class MSE(Loss):
    """Computes mean squared error loss."""

    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray):
        return 2 * (y_pred - y_true) / y_true.shape[0]

    @staticmethod
    def __call__(y_true: np.ndarray, y_pred: np.ndarray):
        return MSE.forward(y_true, y_pred)
