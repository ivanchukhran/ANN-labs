import numpy as np
from .optimizer import Optimizer


class RMSProp(Optimizer):
    def __init__(self, model, lr=0.001, beta=0.9, eps=1e-8):
        super().__init__(model, lr)
        self.beta = beta
        self.eps = eps
        self.s = []
        for i, module in enumerate(self.model.modules):
            if module.has_parameters():
                self.s.append([])
                for param in module.parameters():
                    self.s[i].append(np.zeros_like(param))

    def step(self):
        for i, module in enumerate(self.model.modules):
            if module.has_parameters():
                for j, (param, grad) in enumerate(zip(module.parameters(), module.grads())):
                    self.s[i][j] = self.beta * self.s[i][j] + (1 - self.beta) * grad ** 2
                    param -= self.lr * grad / (np.sqrt(self.s[i][j]) + self.eps)
