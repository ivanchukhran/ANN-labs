from modules import Module


class Activation(Module):
    """Abstract class for all activation functions."""

    def forward(self, x):
        raise NotImplementedError("Forward pass is not implemented. You should implement it in subclass.")

    def backward(self, x, grad_output):
        raise NotImplementedError("Backward pass is not implemented. You should implement it in subclass.")

    def parameters(self):
        raise AttributeError("Activation function has no parameters.")

    def update_parameters(self):
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __call__(self, x):
        return self.forward(x)