from activations import Activation
from tensor import softmax


class Softmax(Activation):
    """
    Softmax activation function.
    Examples:
    >>> softmax = Softmax()
    >>> softmax(np.array([0, 1]))
    array([0.26894142, 0.73105858])
    """

    def forward(self, x):
        return softmax(x)
