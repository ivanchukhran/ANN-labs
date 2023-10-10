import numpy as np

import modules
from activations import Activation
from modules import *


class NeuralNetwork(Module):
    modules: list[Module]
    train: bool

    def from_config(self, config: dict):
        layers = config['layers']
        for layer in layers:
            name, params = layer['type'], layer['params']
            maybe_layer = None
            try:
                maybe_layer = globals().get(name)
            except Exception as e:
                print(f'Layer {name} not found. Error: {e}')
            if maybe_layer is not None:
                self.modules.append(maybe_layer(**params))

    def __init__(self, config: dict):
        self.modules = list()
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

    def to_dict(self):
        return {self.__class__.__name__: list(map(lambda x: x.to_dict(), self.modules))}

    def from_dict(self, config: dict):
        if self.__class__.__name__ not in config.keys():
            raise KeyError(f"Module {self.__class__.__name__} not found in config. "
                           f"The expected module is {self.__class__.__name__} but got {config.keys()} instead.")

        for module, config_value in zip(self.modules, config[self.__class__.__name__]):
            module.from_dict(config_value)

    def save_state(self, path: str):
        np.save(path, self.to_dict(), allow_pickle=True)

    def load_state(self, path: str):
        file = np.load(path, allow_pickle=True).item()
        self.from_dict(file)
