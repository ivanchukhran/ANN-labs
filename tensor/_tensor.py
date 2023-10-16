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

    def __init__(self, data: np.ndarray | int | float | list, requires_grad: bool = False, grad_fn: Any = None):
        if isinstance(data, _convertable):
            data = np.array(data)
        if 'numpy' not in str(type(data)):
            raise TypeError(f"Can't convert {type(data)} to numpy.ndarray")
        self.data = data
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(data) if requires_grad else None
        self.grad_fn = grad_fn

    def backward(self, grad_output: Any = None):
        if self.grad_fn is None or not self.requires_grad:
            raise RuntimeError("Can't call backward on Tensor that does not require grad")
        if grad_output is None:
            grad_output = np.ones_like(self.data)
        self.grad_fn.backward(grad_output)

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        other = to_tensor(other)
        return Add((self, other))()

    def __radd__(self, other):
        other = to_tensor(other)
        return self.__add__(other)

    def __sub__(self, other):
        other = to_tensor(other)
        return self.__add__(other.__neg__())

    def __rsub__(self, other):
        other = to_tensor(other)
        return other.__sub__(self)

    def __mul__(self, other):
        other = to_tensor(other)
        return Mul((self, other))()

    def __rmul__(self, other):
        other = to_tensor(other)
        return self.__mul__(other)

    def __neg__(self):
        return self.__mul__(-1)

    def __matmul__(self, other):
        other = to_tensor(other)
        return MatMul((self, other))()

    def __rmatmul__(self, other):
        other = to_tensor(other)
        return other.__matmul__(self)

    T = property(lambda self: self.transpose())

    def transpose(self):
        return Tensor(self.data.T, self.requires_grad, self.grad_fn)


class Function:
    """
    Represent each tensor operation as a Function object.

    Each Function object has a forward method and a backward method.
    The forward method computes the output tensor given the input tensor.
    The backward method computes the gradient of the loss with respect to the input tensor/s.

    saved_tensors is a list of input tensors that are saved for the backward pass (just for context).
    next_functions is a list of Function objects that represent the next operation/s.

    """

    def __init__(self, saved_tensors: Any):
        self.saved_tensors = saved_tensors
        self.next_functions = [t.grad_fn for t in saved_tensors if saved_tensors is not None]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward not implemented. "
                                  "You should implement forward method in your Function subclass")

    def backward(self, *grad_output):
        raise NotImplementedError("Backward not implemented. "
                                  "You should implement backward method in your Function subclass")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self):
        return f"<{self.__class__.__name__}>"


class Add(Function):

    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        x, y = self.saved_tensors
        t = Tensor(x.data + y.data)
        t.requires_grad = x.requires_grad or y.requires_grad
        t.grad_fn = self if t.requires_grad else None
        return t

    def backward(self, grad_output: Any = None):
        # print(f"grad_fn: {self}")
        # print(f"grad_output: {grad_output}")
        for t in self.saved_tensors:
            if t.requires_grad:
                t.grad += grad_output
                t.grad_fn.backward(t.grad) if t.grad_fn is not None else None

    def __call__(self, *args, **kwargs):
        return self.forward()


class MatMul(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        x, y = self.saved_tensors
        t = Tensor(x.data @ y.data)
        t.requires_grad = x.requires_grad or y.requires_grad
        t.grad_fn = self if t.requires_grad else None
        return t

    def backward(self, grad_output: Any = None):
        for i in range(len(self.saved_tensors)):
            a, b = self.saved_tensors[i], self.saved_tensors[i - 1]
            if a.requires_grad:
                a.grad += np.dot(grad_output, b.data.T)
                a.grad_fn.backward(grad_output) if a.grad_fn is not None else None


class Mul(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        x, y = self.saved_tensors
        t = Tensor(x.data * y.data)
        t.requires_grad = x.requires_grad or y.requires_grad
        t.grad_fn = self if t.requires_grad else None
        return t

    def backward(self, grad_output: Any = None):
        print(f"backward init grad: {grad_output}")
        for i in range(len(self.saved_tensors)):
            a, b = self.saved_tensors[i], self.saved_tensors[i - 1]
            if a.requires_grad:
                a.grad += grad_output * b.data
                a.grad_fn.backward(grad_output) if a.grad_fn is not None else None


class ReLU(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        x = self.saved_tensors
        t = Tensor(np.maximum(0, x.data))
        t.requires_grad = x.requires_grad
        t.grad_fn = self if t.requires_grad else None
        return t

    def backward(self, grad_output: Any = None):
        x = self.saved_tensors
        if x.requires_grad:
            x.grad += grad_output * (x.data > 0)
            x.grad_fn.backward(grad_output) if x.grad_fn is not None else None


def relu(x: Tensor):
    return ReLU((x,))()


class Sigmoid(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        x = self.saved_tensors
        t = Tensor(1 / (1 + np.exp(-x.data)))
        t.requires_grad = x.requires_grad
        t.grad_fn = self if t.requires_grad else None
        return t

    def backward(self, grad_output: Any = None):
        x = self.saved_tensors
        if x.requires_grad:
            x.grad += grad_output * (x.data * (1 - x.data))
            x.grad_fn.backward(grad_output) if x.grad_fn is not None else None


def sigmoid(x: Tensor):
    return Sigmoid((x,))()


class Tanh(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        x = self.saved_tensors
        t = Tensor(np.tanh(x.data))
        t.requires_grad = x.requires_grad
        t.grad_fn = self if t.requires_grad else None
        return t

    def backward(self, grad_output: Any = None):
        x = self.saved_tensors
        if x.requires_grad:
            x.grad += grad_output * (1 - np.tanh(x.data) ** 2)
            x.grad_fn.backward(grad_output) if x.grad_fn is not None else None


def tanh(x: Tensor):
    return Tanh((x,))()


class Softmax(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        x = self.saved_tensors
        t = Tensor(np.exp(x.data) / np.sum(np.exp(x.data)))
        t.requires_grad = x.requires_grad
        t.grad_fn = self if t.requires_grad else None
        return t

    def backward(self, grad_output: Any = None):
        x = self.saved_tensors
        if x.requires_grad:
            x.grad += grad_output * (x.data * (1 - x.data))
            x.grad_fn.backward(grad_output) if x.grad_fn is not None else None


def softmax(x: Tensor):
    return Softmax((x,))()


class CrossEntropy(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        x, y = self.saved_tensors
        t = Tensor(-np.log(x.data[y.data]))
        t.requires_grad = x.requires_grad
        t.grad_fn = self if t.requires_grad else None
        return t

    def backward(self, grad_output: Any = None):
        x, y = self.saved_tensors
        if x.requires_grad:
            x.grad += grad_output * (x.data - y.data)
            x.grad_fn.backward(grad_output) if x.grad_fn is not None else None


def cross_entropy(x: Tensor, y: Tensor):
    return CrossEntropy((x, y))()


class BinaryCrossEntropy(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        x, y = self.saved_tensors
        t = Tensor(-np.log(x.data[y.data]) - np.log(1 - x.data[1 - y.data]))
        t.requires_grad = x.requires_grad
        t.grad_fn = self if t.requires_grad else None
        return t

    def backward(self, grad_output: Any = None):
        x, y = self.saved_tensors
        if x.requires_grad:
            x.grad += grad_output * (x.data - y.data)
            x.grad_fn.backward(grad_output) if x.grad_fn is not None else None


def binary_cross_entropy(x: Tensor, y: Tensor):
    return BinaryCrossEntropy((x, y))()


class MSE(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        x, y = self.saved_tensors
        t = Tensor(np.mean((x.data - y.data) ** 2))
        t.requires_grad = x.requires_grad
        t.grad_fn = self if t.requires_grad else None
        return t

    def backward(self, grad_output: Any = None):
        x, y = self.saved_tensors
        if x.requires_grad:
            x.grad += grad_output * (2 * (x.data - y.data) / x.data.shape[0])
            x.grad_fn.backward(grad_output) if x.grad_fn is not None else None


def mse(x: Tensor, y: Tensor):
    return MSE((x, y))()
