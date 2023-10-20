from tensor import mse_
from .loss import Loss


class MSE(Loss):
    """Computes mean squared error loss."""

    def forward(self, y_pred, y_true):
        return mse_(y_pred, y_true)
