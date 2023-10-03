import numpy as np
from .activation import Activation


class Tanh(Activation):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, x, grad_output):
        return grad_output * (1 - np.tanh(x) ** 2)