import numpy as np
from .activation import Activation
from tensor import tanh_


class Tanh(Activation):
    def forward(self, x):
        return tanh_(x)
