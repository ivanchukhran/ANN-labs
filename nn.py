import numpy as np

import modules
from activations import Activation
from modules import *


class NeuralNetwork(Module):
    modules: list[Module]
    train: bool

    def from_config(self, config: dict):
        layers = config['neural_network']['layers']
        self.modules = list()
        for layer in layers:
            name, params = layer['name'], layer['params']
            maybe_layer = None
            try:
                maybe_layer = globals().get(name)
            except Exception as e:
                print(f'Layer {name} not found. Error: {e}')
            if maybe_layer is not None:
                self.modules.append(maybe_layer(**params))

    def __init__(self, config: dict):
        self.from_config(config)
        self.train = True

    def forward(self, x: np.ndarray, *args, **kwargs):
        for module in self.modules:
            x = module(x)
        return x

    def backward(self, x: np.ndarray, grad_output: np.ndarray = None):
        if grad_output is None:
            grad_output = np.ones_like(x)
        for module in reversed(self.modules):
            grad_output = module.backward(x, grad_output)
        return grad_output

    def __repr__(self):
        return f'NeuralNetwork(modules={self.modules})'

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
