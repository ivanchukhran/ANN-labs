from collections import OrderedDict
from typing import Iterator, Optional

import numpy as np

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

    def modules(self) -> Iterator['Module']:
        for k, v in self._modules.items():
            yield v

    def get_parameters(self) -> OrderedDict:
        return self._parameters

    def has_parameters(self):
        return bool(self._parameters)

    def grads(self) -> Iterator[np.ndarray]:
        for p in self._parameters.values():
            yield p.grad

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def get_state(self) -> dict:
        return {
            self.__class__.__name__: {
                '_parameters': {pk: pv.get_state() for pk, pv in self._parameters.items()},
            }
        }

    def set_state(self, config: dict):
        if self.__class__.__name__ not in config:
            raise KeyError(f"Module {self.__class__.__name__} not found in config. "
                           f"The expected module is {self.__class__.__name__} but got {config.keys()} instead.")
        params = config[self.__class__.__name__]['_parameters']

        def set_param_recursive(split_name: list, param_, params_, modules_):
            """split_name is a list of split strings of the name of the parameter.
            param is the parameter to be assigned.
            params is the OrderedDict of parameters of the current module.
            modules is the OrderedDict of modules of the current module.
            """
            if len(split_name) < 1:
                raise ValueError(f"What the fuck do you want to assign?")
            full_name_ = '.'.join(split_name)
            if full_name_ in params_:
                params_[full_name_].set_state(param_)
            head, tail = split_name[0], split_name[1:]
            if head in modules_:
                module_ = modules_[head]
                set_param_recursive(tail, param_, module_.parameters(), module_.modules())

        for full_name, param in params.items():
            set_param_recursive(full_name.split('.'), param, self._parameters, self._modules)

    def save_state(self, file: str):
        np.save(file=file, arr=self.get_state(), allow_pickle=True)

    def load_state(self, file: str):
        config = np.load(file, allow_pickle=True).item()
        self.set_state(config)
