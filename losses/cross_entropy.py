import numpy as np

from .loss import Loss
from tensor import binary_cross_entropy_, cross_entropy_


class BinaryCrossEntropy(Loss):
    """Computes binary cross entropy loss."""

    def forward(self, y_pred, y_true):
        return binary_cross_entropy_(y_pred, y_true)


class CrossEntropy(Loss):
    """Computes cross entropy loss."""

    def forward(self, y_pred, y_true):
        return cross_entropy_(y_pred, y_true)
