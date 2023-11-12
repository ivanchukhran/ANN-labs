import numpy as np

from ._tensor import Tensor


class Parameter(Tensor):
    def __init__(self, *features):
        self.data = np.random.randn(*features)
        super().__init__(self.data, requires_grad=True)

    def __repr__(self):
        return f"Parameter({self.data})"
