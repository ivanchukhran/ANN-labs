import numpy as np
from .activation import Activation
from tensor import relu


class ReLU(Activation):
    """
    ReLU activation function.
    Examples:
    >>> relu = ReLU()
    >>> relu(np.array([-1, 0, 1]))
    array([0, 0, 1])
        """

    def forward(self, x):
        return relu(x)
