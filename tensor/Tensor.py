from functools import reduce
from typing import Any

import numpy as np

_convertable = (int, float, list)


def to_tensor(other):
    if isinstance(other, (_convertable, np.ndarray)):
        other = Tensor(other)
    if not isinstance(other, Tensor):
        raise TypeError(f"unsupported type(s): expected 'Tensor' or {_convertable} but got '{type(other)}' instead")
    return other


class Tensor:
    data: np.ndarray
    requires_grad: bool
    grad: np.ndarray
    grad_fn: Any

    def __init__(self, data: np.ndarray | int | float | list, requires_grad: bool = False, grad_fn: 'Function' = None):
        if isinstance(data, _convertable):
            data = np.array(data)
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Can't convert {type(data)} to numpy.ndarray")
        self.data = data
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(data) if requires_grad else None
        self.grad_fn = grad_fn

    def backward(self, grad_output: Any = None):
        if self.grad_fn is None or not self.requires_grad:
            raise RuntimeError("Can't call backward on Tensor that does not require grad")
        if grad_output is None:
            grad_output = 1
        self.grad_fn.backward(grad_output)

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        other = to_tensor(other)
        f = Add((self, other))
        return f()

    def __radd__(self, other):
        other = to_tensor(other)
        return self.__add__(other)

    def __sub__(self, other):
        other = to_tensor(other)
        return Tensor(self.data - other.data, requires_grad=(self.requires_grad or other.requires_grad))

    def __rsub__(self, other):
        other = to_tensor(other)
        return other.__sub__(self)

    def __mul__(self, other):
        other = to_tensor(other)
        return Tensor(self.data * other.data, requires_grad=(self.requires_grad or other.requires_grad))

    def __rmul__(self, other):
        other = to_tensor(other)
        return self.__mul__(other)

    def __neg__(self):
        return self.__mul__(-1)

    def __matmul__(self, other):
        other = to_tensor(other)
        return Tensor(self.data @ other.data, requires_grad=(self.requires_grad or other.requires_grad))

    def __rmatmul__(self, other):
        other = to_tensor(other)
        return other.__matmul__(self)

    T = property(lambda self: self.transpose())

    def transpose(self):
        return Tensor(self.data.T, self.requires_grad)


class Function:
    """
    Represent each tensor operation as a Function object.

    Each Function object has a forward method and a backward method.
    The forward method computes the output tensor given the input tensor.
    The backward method computes the gradient of the loss with respect to the input tensor/s.

    saved_tensors is a list of input tensors that are saved for the backward pass (just for context).
    next_functions is a list of Function objects that represent the next operation/s.

    """

    def __init__(self, saved_tensors: Tensor | tuple = None):
        self.saved_tensors = saved_tensors
        self.next_functions = [t.grad_fn for t in saved_tensors if saved_tensors is not None]

    def forward(self, *args):
        raise NotImplementedError("Forward not implemented. "
                                  "You should implement forward method in your Function subclass")

    def backward(self, *grad_output):
        raise NotImplementedError("Backward not implemented. "
                                  "You should implement backward method in your Function subclass")


class Add(Function):
    def __init__(self, saved_tensors: tuple):
        super().__init__(saved_tensors)

    def forward(self):
        return reduce(
            lambda x, y: Tensor(x.data + y.data, requires_grad=(x.requires_grad or y.requires_grad), grad_fn=self),
            self.saved_tensors
        )

    def backward(self, grad_output: Any = None):
        # print(f"grad_fn: {self}")
        # print(f"grad_output: {grad_output}")
        for t in self.saved_tensors:
            if t.requires_grad:
                t.grad += grad_output
                t.grad_fn.backward(t.grad) if t.grad_fn is not None else None

    def __call__(self, *args, **kwargs):
        return self.forward()

    def __repr__(self):
        tensor_string = ', '.join([str(t) for t in self.saved_tensors])
        return f"<Add ({tensor_string})>"


class MatMul(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        return reduce(lambda x, y: x @ y, self.saved_tensors)

    def backward(self, grad_output: Any = None):
        first, last = self.saved_tensors
        first.grad += grad_output @ last.T
        last.grad += grad_output @ first.T
        for t in self.saved_tensors:
            if t.requires_grad:
                t.grad_fn.backward(grad_output) if t.grad_fn is not None else None


    def __call__(self, *args, **kwargs):
        return self.forward()


class Mul(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        return reduce(lambda x, y: x * y, self.saved_tensors)

    def backward(self, grad_output: Any = None):
        # TODO: fix this
        for t in self.saved_tensors:
            if t.requires_grad:
                t.grad += grad_output.grad

    def __call__(self, *args, **kwargs):
        return self.forward()


class Sub(Function):
    def __init__(self, saved_tensors):
        super().__init__(saved_tensors)

    def forward(self):
        return reduce(lambda x, y: x - y, self.saved_tensors)

    def backward(self, grad_output: Any = None):
        # TODO: fix this
        for t in self.saved_tensors:
            if t.requires_grad:
                t.grad += grad_output.grad

    def __call__(self, *args, **kwargs):
        return self.forward()
