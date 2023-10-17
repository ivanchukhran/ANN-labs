from tensor import mse_
from .loss import Loss


class MSE(Loss):
    """Computes mean squared error loss."""

    def forward(self, y_true, y_pred):
        return mse_(y_true, y_pred)
