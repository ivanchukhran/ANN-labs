class Loss:
    """Base class for loss functions."""
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Loss function is not implemented. You should implement it in subclass.")

    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)
