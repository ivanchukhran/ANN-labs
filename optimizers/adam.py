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
        for param in self.model.parameters():
            self.m.append(np.zeros_like(param.data))
            self.v.append(np.zeros_like(param.data))

    def step(self):
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.model.parameters(), self.model.grads())):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
