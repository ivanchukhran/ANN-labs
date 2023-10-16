import numpy as np

from modules import module
from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, model, lr=0.001):
        super().__init__(model, lr)

    def step(self):
        for param in self.model.parameters():
            param.data = param.data - self.lr * param.grad
