import numpy as np
from .optimizer import Optimizer


class Adagrad(Optimizer):
    def __init__(self, model, lr=0.001, eps=1e-8):
        super().__init__(model, lr)
        self.eps = eps
        self.t = 0

        self.v = []
        for param in self.model.parameters():
            self.v.append(np.zeros_like(param.data))

    def step(self):
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.model.parameters(), self.model.grads())):
            self.v[i] += grad ** 2
            param.data = param.data - self.lr * grad / (np.sqrt(self.v[i]) + self.eps)
