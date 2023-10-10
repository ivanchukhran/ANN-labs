from functools import reduce

from tensor import Tensor
from .function import Function


class MatMul(Function):
    def __init__(self, saved_tensors: list):
        super().__init__(saved_tensors)

    def forward(self):
        return reduce(lambda x, y: x @ y, self.saved_tensors)

    def backward(self, grad_output: Tensor = None):
        for t in self.saved_tensors:
            if t.requires_grad:
                t.grad += grad_output.grad

    def __call__(self, *args, **kwargs):
        return self.forward()
