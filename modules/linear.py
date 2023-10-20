import numpy as np
from .module import Module
from tensor import Parameter, Tensor


class Linear(Module):
    """Linear layer of neural network."""

    def __init__(self, in_features: int, out_features: int, use_bias: bool = True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(out_features, in_features)
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = Parameter(out_features)

    def forward(self, x: Tensor):
        output = x @ self.weight.T
        if self.use_bias:
            output = output + self.bias
        return output

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, use_bias={self.use_bias})"

