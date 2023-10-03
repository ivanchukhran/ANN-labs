import abc


class Module(abc.ABC):
    """
    Abstract class for all neural network modules.
    Your models should also inherit from this class.
    """

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward pass is not implemented. You should implement it in subclass.")

    def backward(self, *args, **kwargs):
        raise NotImplementedError("Backward pass is not implemented. You should implement it in subclass.")

    def parameters(self):
        raise NotImplementedError("Parameters getter is not implemented. You should implement it in subclass.")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self):
        return f'{self.__class__.__name__}({list(map(len, self.parameters()))})'

    def update_parameters(self):
        raise NotImplementedError("Parameters update is not implemented. You should implement it in subclass.")
