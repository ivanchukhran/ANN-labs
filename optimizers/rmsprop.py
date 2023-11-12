import numpy as np
from .optimizer import Optimizer


class RMSProp(Optimizer):
    def __init__(self, model, lr=0.001, beta=0.9, eps=1e-8):
        super().__init__(model, lr)
        self.beta = beta
        self.eps = eps
        self.s = []
        for param in self.model.parameters():
            self.s.append(np.zeros_like(param.data))

    def step(self):
        for i, (param, grad) in enumerate(zip(self.model.parameters(), self.model.grads())):
            self.s[i] = self.beta * self.s[i] + (1 - self.beta) * grad ** 2
            param.data = param.data - self.lr * grad / (np.sqrt(self.s[i]) + self.eps)


    def get_state(self):
        state = {
            self.__class__.__name__: {
                'lr': self.lr,
                'beta': self.beta,
                'eps': self.eps,
            }
        }
        return state

    def set_state(self, config):
        self.lr = config[self.__class__.__name__]['lr']
        self.beta = config[self.__class__.__name__]['beta']
        self.eps = config[self.__class__.__name__]['eps']

    def save_state(self, file: str):
        np.save(file=file, arr=self.get_state(), allow_pickle=True)

    def load_state(self, file: str):
        config = np.load(file, allow_pickle=True).item()
        self.set_state(config)
