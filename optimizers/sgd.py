import numpy as np

from modules import module, Module
from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, model: Module, lr=0.001):
        super().__init__(model, lr)

    def step(self):
        for param in self.model.parameters():
            param.data = param.data - self.lr * param.grad

    def get_state(self):
        state = {
            self.__class__.__name__: {
                'lr': self.lr,
            }
        }
        return state

    def set_state(self, config):
        self.lr = config[self.__class__.__name__]['lr']

    def save_state(self, file: str):
        np.save(file=file, arr=self.get_state(), allow_pickle=True)

    def load_state(self, file: str):
        config = np.load(file, allow_pickle=True).item()
        self.set_state(config)
