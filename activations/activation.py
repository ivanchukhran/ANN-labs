from modules import Module


class Activation(Module):
    """Abstract class for all activation functions."""

    def forward(self, x):
        raise NotImplementedError("Forward pass is not implemented. You should implement it in subclass.")

    def parameters(self):
        return None

    def has_parameters(self):
        return False

    def update_parameters(self):
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __call__(self, x):
        return self.forward(x)