import numpy as np
from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, model, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m = []
        self.v = []
        for i, module in enumerate(self.model.modules):
            if module.has_parameters():
                self.m.append([])
                self.v.append([])
                for param in module.parameters():
                    self.m[i].append(np.zeros_like(param))
                    self.v[i].append(np.zeros_like(param))

    def step(self):
        self.t += 1
        for i, module in enumerate(self.model.modules):
            if module.has_parameters():
                for j, (param, grad) in enumerate(zip(module.parameters(), module.grad_parameters())):
                    self.m[i][j] = self.beta1 * self.m[i][j] + (1 - self.beta1) * grad
                    self.v[i][j] = self.beta2 * self.v[i][j] + (1 - self.beta2) * grad ** 2
                    m_hat = self.m[i][j] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[i][j] / (1 - self.beta2 ** self.t)
                    param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
