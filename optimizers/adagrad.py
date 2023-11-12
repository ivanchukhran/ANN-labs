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

    def get_state(self):
        state = {
            self.__class__.__name__: {
                'lr': self.lr,
                'eps': self.eps,
                't': self.t,
            }
        }
        return state

    def set_state(self, config):
        self.lr = config[self.__class__.__name__]['lr']
        self.eps = config[self.__class__.__name__]['eps']
        self.t = config[self.__class__.__name__]['t']

    def save_state(self, file: str):
        np.save(file=file, arr=self.get_state(), allow_pickle=True)

    def load_state(self, file: str):
        config = np.load(file, allow_pickle=True).item()
        self.set_state(config)