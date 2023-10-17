import numpy as np

from .loss import Loss
from tensor import binary_cross_entropy_, cross_entropy_


class BinaryCrossEntropy(Loss):
    """Computes binary cross entropy loss."""

    def forward(self, y_true, y_pred):
        return binary_cross_entropy_(y_true, y_pred)


class CrossEntropy(Loss):
    """Computes cross entropy loss."""

    def forward(self, y_true, y_pred):
        return cross_entropy_(y_true, y_pred)
