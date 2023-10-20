from functools import reduce
from typing import Any, Callable

import numpy as np

_convertable = (int, float, list)


def to_tensor(other):
    if isinstance(other, (_convertable, np.ndarray)):
        other = Tensor(other)
    if not isinstance(other, Tensor):
        raise TypeError(f"unsupported type(s): expected 'Tensor' or {_convertable} but got '{type(other)}' instead")
    return other


def make_tensor_from_ops(*tensors, ops: Callable, backward_fn: Callable = None):
    tensors = list([*tensors])
    t = Tensor(ops(*[t.data for t in tensors]))
    t.requires_grad = reduce(lambda x, y: x or y, [t.requires_grad for t in tensors])
    t.grad_fn = backward_fn if t.requires_grad else None
    t.grad = np.zeros_like(t.data) if t.requires_grad else None
    return t


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
        if data.ndim == 1:
            data = data.reshape(1, -1)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(data) if requires_grad else None
        self.grad_fn = grad_fn

    def backward(self, grad_output: Any = None):
        if not self.requires_grad:
            raise RuntimeError("Can't call backward on Tensor that does not require grad")
        if grad_output is None:
            grad_output = np.ones_like(self.data)
        else:
            grad_output = grad_output * np.ones_like(self.data)
        if self.grad_fn:
            self.grad_fn.backward(grad_output)

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        other = to_tensor(other)
        return add_(self, other)

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
        return mul_(self, other)

    def __rmul__(self, other):
        other = to_tensor(other)
        return self.__mul__(other)

    def __neg__(self):
        return self.__mul__(-1)

    def __matmul__(self, other):
        other = to_tensor(other)
        return matmul_(self, other)

    def __rmatmul__(self, other):
        return other.__matmul__(self)

    def __truediv__(self, other):
        other = to_tensor(other)
        return div_(self, other)

    def __rtruediv__(self, other):
        other = to_tensor(other)
        return other.__truediv__(self)

    def __pow__(self, power):
        power = to_tensor(power)
        return power_(self, power)

    def __eq__(self, other):
        other = to_tensor(other)
        return self.data == other.data

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self.data)

    T = property(lambda self: self.transpose())

    def transpose(self):
        return permute_(self, lambda a: a.T)

    def sum(self):
        return sum_(self)

    def argmax(self, axis=None):
        return Tensor(np.argmax(self.data, axis=axis), self.requires_grad, self.grad_fn)

    @property
    def shape(self):
        return self.data.shape

    def get_state(self) -> dict:
        return {
            self.__class__.__name__: {
                'data': self.data,
                'requires_grad': self.requires_grad,
                'grad': self.grad,
                'grad_fn': self.grad_fn
            }
        }

    def set_state(self, state: dict):
        if self.__class__.__name__ not in state.keys():
            raise KeyError(f"Tensor {self.__class__.__name__} not found in config. "
                           f"The expected tensor is {self.__class__.__name__} but got {state.keys()} instead.")
        state_values = state[self.__class__.__name__]
        self.data = state_values['data']
        self.requires_grad = state_values['requires_grad']
        self.grad = state_values['grad']
        self.grad_fn = state_values['grad_fn']


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
        # self.next_functions = [t.grad_fn for t in saved_tensors if saved_tensors is not None]

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
        x, y = self.saved_tensors[0], self.saved_tensors[1]
        t = make_tensor_from_ops(x, y, ops=lambda a, b: a + b, backward_fn=self)
        return t

    def backward(self, grad_output: Any = None):
        x, y = self.saved_tensors[0], self.saved_tensors[1]
        if x.requires_grad:
            x.grad = x.grad + grad_output * np.ones_like(x.data)
            x.backward(x.grad) 
        if y.requires_grad:
            y.grad = y.grad + grad_output * np.ones_like(y.data)
            y.backward(y.grad)

    def __call__(self, *args, **kwargs):
        return self.forward()


def add_(x: Tensor, y: Tensor):
    return Add((x, y))()


class Sum(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        x = self.saved_tensors
        t = make_tensor_from_ops(x, ops=lambda a: np.sum(a), backward_fn=self)
        return t

    def backward(self, grad_output: Any = None):
        x = self.saved_tensors[0]
        if x.requires_grad:
            x.grad = x.grad + grad_output * np.ones_like(x.data)
            x.backward(x.grad) 


def sum_(x: Tensor):
    return Sum(x)()


class Div(Function):
    def __init__(self, saved_tensors):
        super().__init__(saved_tensors)

    def forward(self):
        x, y = self.saved_tensors[0], self.saved_tensors[1]
        t = make_tensor_from_ops(x, y, ops=lambda a, b: a / b, backward_fn=self)
        return t

    def backward(self, grad_output: Any = None):
        x, y = self.saved_tensors[0], self.saved_tensors[1]
        if x.requires_grad:
            x.grad = x.grad + grad_output * (1 / y.data)
            x.backward(x.grad) 
        if y.requires_grad:
            y.grad = y.grad + grad_output * (-x.data / (y.data ** 2))
            y.backward(y.grad) 


def div_(x: Tensor, y: Tensor):
    return Div((x, y))()


class MatMul(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        x, y = self.saved_tensors[0], self.saved_tensors[1]
        t = make_tensor_from_ops(x, y, ops=lambda a, b: a @ b, backward_fn=self)
        return t

    def backward(self, grad_output: Any = None):
        """
        c = input @ weight
        suppose dc = grad_output then d_input = dc @ weight.T, d_weight = input.T @ dc
        """
        input, weight = self.saved_tensors
        d_input = grad_output @ weight.data.T
        d_weight = input.data.T @ grad_output
        if input.requires_grad:
            input.grad = input.grad + d_input
            input.backward(input.grad)
        if weight.requires_grad:
            weight.grad = weight.grad + d_weight
            weight.backward(weight.grad)


def matmul_(x: Tensor, y: Tensor):
    return MatMul((x, y))()


class Mul(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        x, y = self.saved_tensors[0], self.saved_tensors[1]
        t = make_tensor_from_ops(x, y, ops=lambda a, b: a * b, backward_fn=self)
        return t

    def backward(self, grad_output: Any = None):
        for i in range(len(self.saved_tensors)):
            a, b = self.saved_tensors[i], self.saved_tensors[i - 1]
            if a.requires_grad:
                a.grad = a.grad + grad_output * b.data
                a.backward(grad_output) 


def mul_(x: Tensor, y: Tensor):
    return Mul((x, y))()


class Power(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        x, y = self.saved_tensors[0], self.saved_tensors[1]
        t = make_tensor_from_ops(x, y, ops=lambda a, b: a ** b, backward_fn=self)
        return t

    def backward(self, grad_output: Any = None):
        x, y = self.saved_tensors[0], self.saved_tensors[1]
        if x.requires_grad:
            x.grad = x.grad + grad_output * y.data * x.data ** (y.data - 1)
            x.backward(grad_output) 
        if y.requires_grad:
            y.grad = y.grad + grad_output * np.log(x.data) * x.data ** y.data
            y.backward(y.grad) 


def power_(x: Tensor, y: Tensor):
    return Power((x, y))()


class Permute(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self, operation: Callable):
        x = self.saved_tensors
        self.operation = operation
        t = make_tensor_from_ops(x, ops=operation, backward_fn=self)
        return t

    def backward(self, grad_output: Any = None):
        x = self.saved_tensors
        if x.requires_grad:
            x.grad = x.grad + self.operation(grad_output)
            x.backward(x.grad)


def permute_(x: Tensor, operation: Callable):
    return Permute(x)(operation)


class ReLU(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        x = self.saved_tensors
        t = make_tensor_from_ops(x, ops=lambda a: np.maximum(a, 0), backward_fn=self)
        return t

    def backward(self, grad_output: Any = None):
        x = self.saved_tensors
        if x.requires_grad:
            x.grad = x.grad + grad_output * (x.data > 0)
            x.backward(x.grad) 


def relu_(x: Tensor):
    return ReLU(x)()


class Sigmoid(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        x = self.saved_tensors
        t = make_tensor_from_ops(x, ops=lambda a: 1 / (1 + np.exp(-a)), backward_fn=self)
        return t

    def backward(self, grad_output: Any = None):
        x = self.saved_tensors
        if x.requires_grad:
            x.grad = x.grad + grad_output * (x.data * (1 - x.data))
            x.backward(x.grad) 


def sigmoid_(x: Tensor):
    return Sigmoid(x)()


class Tanh(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        x = self.saved_tensors
        t = make_tensor_from_ops(x, ops=lambda a: np.tanh(a), backward_fn=self)
        return t

    def backward(self, grad_output: Any = None):
        x = self.saved_tensors
        if x.requires_grad:
            x.grad = x.grad + grad_output * (1 - np.tanh(x.data) ** 2)
            x.backward(x.grad) 


def tanh_(x: Tensor):
    return Tanh(x)()


class Softmax(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        x = self.saved_tensors
        t = make_tensor_from_ops(x, ops=lambda a: np.exp(a) / np.sum(np.exp(a)), backward_fn=self)
        return t

    def backward(self, grad_output: Any = None):
        x = self.saved_tensors
        if x.requires_grad:
            x.grad = x.grad + grad_output * (x.data * (1 - x.data))
            x.backward(x.grad) 


def softmax_(x: Tensor):
    return Softmax(x)()


class CrossEntropy(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        y_hat, y = self.saved_tensors[0], self.saved_tensors[1]
        # t = make_tensor_from_ops(x, y, ops=lambda a, b: -np.log(a[b]), backward_fn=self)
        t = make_tensor_from_ops(y_hat, y, ops=lambda a, b: -np.sum(b * np.log(a)) / len(a), backward_fn=self)
        return t

    def backward(self, grad_output: Any = None):
        x, y = self.saved_tensors[0], self.saved_tensors[1]
        if x.requires_grad:
            x.grad = x.grad + grad_output * (x.data - y.data)
            x.backward(x.grad) 


def cross_entropy_(x: Tensor, y: Tensor):
    return CrossEntropy((x, y))()


class BinaryCrossEntropy(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        x, y = self.saved_tensors[0], self.saved_tensors[1]
        t = make_tensor_from_ops(x, y, ops=lambda a, b: -a * np.log(b) - (1 - b) * np.log(1 - a) / len(a),
                                 backward_fn=self)
        return t

    def backward(self, grad_output: Any = None):
        x, y = self.saved_tensors[0], self.saved_tensors[1]
        if x.requires_grad:
            x.grad = x.grad + grad_output * (x.data - y.data)
            x.backward(x.grad)


def binary_cross_entropy_(x: Tensor, y: Tensor):
    return BinaryCrossEntropy((x, y))()


class MSE(Function):
    def __init__(self, saved_tensors: Any):
        super().__init__(saved_tensors)

    def forward(self):
        x, y = self.saved_tensors[0], self.saved_tensors[1]
        t = make_tensor_from_ops(x, y, ops=lambda a, b: np.sum((a - b) ** 2) / a.shape[0], backward_fn=self)
        return t

    def backward(self, grad_output: Any = None):
        x, y = self.saved_tensors[0], self.saved_tensors[1]
        if x.requires_grad:
            x.grad = x.grad + grad_output * (2 * (x.data - y.data) / x.data.shape[0])
            x.backward(x.grad) 


def mse_(x: Tensor, y: Tensor):
    return MSE((x, y))()
