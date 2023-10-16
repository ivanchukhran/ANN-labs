import numpy as np
from .activation import Activation
from tensor import tanh


class Tanh(Activation):
    def forward(self, x):
        return tanh(x)
