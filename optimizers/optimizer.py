import torch

from modules import Module


class Optimizer:
    lr: float
    model: Module

    def __init__(self, model, lr: float = 0.001, *args, **kwargs):
        self.model = model
        self.lr = lr

    def step(self):
        raise NotImplementedError("You should implement this method in subclass.")

    def zero_grad(self):
        for param in self.model.parameters():
            param.grad = 0.0

    def get_state(self):
        raise NotImplementedError("You should implement this method in subclass.")

    def save_state(self, path: str, *args, **kwargs):
        raise NotImplementedError("You should implement this method in subclass.")

    def load_state(self):
        raise NotImplementedError("You should implement this method in subclass.")
