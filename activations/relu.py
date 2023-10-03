import numpy as np
from .activation import Activation


class ReLU(Activation):
    """
    ReLU activation function.
    Examples:
    >>> relu = ReLU()
    >>> relu(np.array([-1, 0, 1]))
    array([0, 0, 1])
        """

    def forward(self, x: np.ndarray):
        return np.maximum(0, x)

    def backward(self, x: np.ndarray, grad_output: np.ndarray):
        return grad_output * (x > 0)
