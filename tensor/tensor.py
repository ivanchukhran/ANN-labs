import numpy as np
from functions import Function

_convertable = (int, float, list)


class Tensor:
    data: np.ndarray
    requires_grad: bool
    grad: np.ndarray
    grad_fn: 'Function'

    def __init__(self, data: np.ndarray | int | float | list, requires_grad: bool = False, grad_fn: 'Function' = None):
        if isinstance(data, _convertable):
            data = np.array(data)
        if not isinstance(data, np.ndarray):
            raise TypeError(f"unsupported operand type(s) for +: 'Tensor' and '{type(data)}'")
        self.data = data
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(data) if requires_grad else None
        self.grad_fn = grad_fn

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        if isinstance(other, (_convertable, np.ndarray)):
            other = Tensor(other)
        if not isinstance(other, Tensor):
            raise TypeError(f"unsupported operand type(s) for +: 'Tensor' and '{type(other)}'")
        return Tensor(self.data + other.data, requires_grad=(self.requires_grad or other.requires_grad))

    def __radd__(self, other):
        if isinstance(other, (_convertable, np.ndarray)):
            other = Tensor(other)
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (_convertable, np.ndarray)):
            other = Tensor(other)
        if not isinstance(other, Tensor):
            raise TypeError(f"unsupported operand type(s) for -: 'Tensor' and '{type(other)}'")
        return Tensor(self.data - other.data, requires_grad=(self.requires_grad or other.requires_grad))

    def __rsub__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            other = Tensor(other)
        return other.__sub__(self)

    def __mul__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            other = Tensor(other)
        if not isinstance(other, Tensor):
            raise TypeError(f"unsupported operand type(s) for *: 'Tensor' and '{type(other)}'")
        return Tensor(self.data * other.data, requires_grad=(self.requires_grad or other.requires_grad))

    def __rmul__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            other = Tensor(other)
        return self.__mul__(other)

    def __neg__(self):
        return self.__mul__(-1)

    def __matmul__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            other = Tensor(other)
        if not isinstance(other, Tensor):
            raise TypeError(f"unsupported operand type(s) for @: 'Tensor' and '{type(other)}'")
        return Tensor(self.data @ other.data, requires_grad=(self.requires_grad or other.requires_grad))

    def __rmatmul__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            other = Tensor(other)
        return other.__matmul__(self)

    T = property(lambda self: self.transpose())

    def transpose(self):
        return Tensor(self.data.T, self.requires_grad)

