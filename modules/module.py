from collections import OrderedDict
from typing import Iterator, Optional

from tensor import Parameter


class Module:
    """
    Abstract class for all neural network modules.
    Your models should also inherit from this class.
    """
    _training: bool
    _parameters: OrderedDict
    _modules: OrderedDict

    def __init__(self, *args, **kwargs):
        super.__setattr__(self, '_parameters', OrderedDict())
        super.__setattr__(self, '_modules', OrderedDict())
        super.__setattr__(self, '_training', True)

    def __setattr__(self, key, value):
        if isinstance(value, Parameter):
            self.register_parameter(key, value)
        elif isinstance(value, Module):
            self.register_module(key, value)
        super.__setattr__(self, key, value)

    def register_parameter(self, name: str, param: Optional[Parameter]):
        if name in self._parameters:
            raise KeyError(f"Parameter {name} already exists.")
        self._parameters[name] = param

    def named_parameters(self) -> Iterator[Parameter]:
        for k, v in self._parameters.items():
            yield k, v

    def register_module(self, name: str, module: Optional['Module']):
        if name in self._modules:
            raise KeyError(f"Module {name} already exists.")
        self._modules[name] = module
        for k, v in module.named_parameters():
            self.register_parameter(f'{name}.{k}', v)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward pass is not implemented. You should implement it in subclass.")

    def parameters(self) -> Iterator[Parameter]:
        for k, v in self._parameters.items():
            yield v

    def get_parameters(self) -> OrderedDict:
        return self._parameters

    def has_parameters(self):
        return bool(self._parameters)

    def grads(self) -> Iterator[Parameter]:
        for p in self._parameters.values():
            yield p.grad

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def to_dict(self) -> dict:
        return {self.__class__.__name__: self.parameters()}

    def from_dict(self, config: dict):
        try:
            self._parameters = config[self.__class__.__name__]
        except KeyError:
            raise KeyError(f"Module {self.__class__.__name__} not found in config. "
                           f"The expected module is {self.__class__.__name__} but got {config.keys()} instead.")

