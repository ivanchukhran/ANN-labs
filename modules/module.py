import abc


class Module(abc.ABC):
    __parameters: list = None
    __grads: list = None
    """
    Abstract class for all neural network modules.
    Your models should also inherit from this class.
    """

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward pass is not implemented. You should implement it in subclass.")

    def backward(self, *args, **kwargs):
        raise NotImplementedError("Backward pass is not implemented. You should implement it in subclass.")

    def parameters(self):
        return self.__parameters

    def has_parameters(self):
        return self.__parameters is not None

    def grads(self):
        return self.__grads

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self):
        return f'{self.__class__.__name__}({list(map(len, self.parameters()))})'

    def update_parameters(self):
        raise NotImplementedError("Parameters update is not implemented. You should implement it in subclass.")

    def to_dict(self) -> dict:
        return {self.__class__.__name__: self.parameters()}

    def from_dict(self, config: dict):
        try:
            self.__parameters = config[self.__class__.__name__]
        except KeyError:
            raise KeyError(f"Module {self.__class__.__name__} not found in config. "
                           f"The expected module is {self.__class__.__name__} but got {config.keys()[0]} instead.")

