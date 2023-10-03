import numpy as np
from .module import Module


class Linear(Module):
    """Linear layer of neural network."""

    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.randn(in_features, out_features)
        self.bias = np.random.randn(out_features)
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, x):
        return np.dot(x, self.weight) + self.bias

    def backward(self, x: np.ndarray, grad_output: np.ndarray):
        self.grad_weight = np.dot(x.T, grad_output)
        self.grad_bias = np.sum(grad_output, axis=0)
        return np.dot(grad_output, self.weight.T)

    def parameters(self):
        return [self.weight, self.bias]

    def grad_parameters(self):
        return [self.grad_weight, self.grad_bias]


