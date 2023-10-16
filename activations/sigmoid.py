import numpy as np
from .activation import Activation
from tensor import sigmoid


class Sigmoid(Activation):
    """
    Sigmoid activation function.
    Examples:
    >>> sigmoid = Sigmoid()
    >>> sigmoid(np.array([0, 1]))
    array([0.5       , 0.73105858])
    """
    def forward(self, x: np.ndarray):
        return sigmoid(x)

