from tensor import mse
from .loss import Loss


class MSE(Loss):
    """Computes mean squared error loss."""

    def forward(self, y_true, y_pred):
        return mse(y_true, y_pred)
