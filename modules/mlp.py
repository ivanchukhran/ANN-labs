from typing import Optional

from . import Linear
from .module import Module
import activations
from activations import *


class MLP(Module):
    def __init__(self, features: list, activation_: Optional[str] = None):
        super().__init__()
        for i, (in_features, out_features) in enumerate(features):
            self.__setattr__(f'linear_{i}', Linear(in_features, out_features))
        try:
            self.__setattr__('activation', getattr(activations, activation_)())
        except AttributeError:
            self.__setattr__('activation', None)

    def forward(self, x):
        for module in self.modules():
            x = module(x)
        return x
