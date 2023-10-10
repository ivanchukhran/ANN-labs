import numpy as np
from .module import Module


class Linear(Module):
    """Linear layer of neural network."""

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.randn(out_features, in_features)
        self.bias = np.random.randn(out_features)
        self._parameters = [self.weight, self.bias]
        grad_weight = np.zeros_like(self.weight)
        grad_bias = np.zeros_like(self.bias)
        self._grads = [grad_weight, grad_bias]
        super(Linear, self).__init__()

    def forward(self, x):
        return np.dot(x, self.weight.T) + self.bias

    def backward(self, x: np.ndarray, grad_output: np.ndarray = None):
        # if grad_output is None:
        #     grad_output = 1.0
        # self._grads[0] = np.dot(x, grad_output)
        # self._grads[1] = np.sum(grad_output, axis=0)
        # print(self._grads[0].shape, self._grads[1].shape)
        # return np.dot(grad_output, self._grads[0])

        if grad_output is None:
            grad_output = np.ones_like(x)
        self._grads[0] += np.dot(x.T, grad_output)
        self._grads[1] += np.sum(grad_output, axis=0)
        return np.dot(grad_output, self._parameters[0])
