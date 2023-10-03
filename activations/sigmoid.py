import numpy as np
from .activation import Activation


class Sigmoid(Activation):
    """
    Sigmoid activation function.
    Examples:
    >>> sigmoid = Sigmoid()
    >>> sigmoid(np.array([0, 1]))
    array([0.5       , 0.73105858])
    """
    def forward(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.ndarray, grad_output: np.ndarray):
        return grad_output * self.forward(x) * (1 - self.forward(x))
