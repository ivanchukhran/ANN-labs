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

    def get_state(self):
        state = {
            self.__class__.__name__: {
                'lr': self.lr,
                'beta1': self.beta1,
                'beta2': self.beta2,
                'eps': self.eps,
                't': self.t,
            }
        }
        return state

    def set_state(self, config):
        self.lr = config[self.__class__.__name__]['lr']
        self.beta1 = config[self.__class__.__name__]['beta1']
        self.beta2 = config[self.__class__.__name__]['beta2']
        self.eps = config[self.__class__.__name__]['eps']
        self.t = config[self.__class__.__name__]['t']

    def save_state(self, file: str):
        np.save(file=file, arr=self.get_state(), allow_pickle=True)

    def load_state(self, file: str):
        config = np.load(file, allow_pickle=True).item()
        self.set_state(config)

