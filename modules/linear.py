import numpy as np
from .module import Module


class Linear(Module):
    """Linear layer of neural network."""

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        weight = np.random.randn(in_features, out_features)
        bias = np.random.randn(out_features)
        self._parameters = [weight, bias]
        grad_weight = np.zeros_like(weight)
        grad_bias = np.zeros_like(bias)
        self._grads = [grad_weight, grad_bias]
        super(Linear, self).__init__()

    def forward(self, x):
        return np.dot(x, self._parameters[0]) + self._parameters[1]

    def backward(self, x: np.ndarray, grad_output: np.ndarray):
        self._grads[0] = np.dot(x.T, grad_output)
        self._grads[1] = np.sum(grad_output, axis=0)
        return np.dot(grad_output, self._parameters[0].T)
