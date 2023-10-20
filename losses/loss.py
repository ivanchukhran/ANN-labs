class Loss:
    """Base class for loss functions."""
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Loss function is not implemented. You should implement it in subclass.")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}()"
