import numpy as np
from .optimizer import Optimizer


class Adagrad(Optimizer):
    def __init__(self, model, lr=0.001, eps=1e-8):
        super().__init__(model, lr)
        self.eps = eps
        self.t = 0

        self.v = []
        for i, module in enumerate(self.model.modules):
            if module.has_parameters():
                self.v.append([])
                for param in module.parameters():
                    self.v[i].append(np.zeros_like(param))

    def step(self):
        self.t += 1
        for i, module in enumerate(self.model.modules):
            if module.has_parameters():
                for j, (param, grad) in enumerate(zip(module.parameters(), module.grads())):
                    self.v[i][j] += grad ** 2
                    param -= self.lr * grad / (np.sqrt(self.v[i][j]) + self.eps)
