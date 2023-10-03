class Loss:

    @staticmethod
    def __call__(*args, **kwargs):
        raise NotImplementedError("Loss function is not implemented. You should implement it in subclass.")
