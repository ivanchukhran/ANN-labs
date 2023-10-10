from tensor.tensor import Tensor


class Function:
    """
    Represent each tensor operation as a Function object.

    Each Function object has a forward method and a backward method.
    The forward method computes the output tensor given the input tensor.
    The backward method computes the gradient of the loss with respect to the input tensor/s.

    """

    def __init__(self, saved_tensors: Tensor | list[Tensor] = None):
        self.saved_tensors = saved_tensors
        self.next_functions = [t.grad_fn for t in saved_tensors if saved_tensors is not None]

    def forward(self, *args):
        raise NotImplementedError("Forward not implemented. "
                                  "You should implement forward method in your Function subclass")

    def backward(self, *grad_output):
        raise NotImplementedError("Backward not implemented. "
                                  "You should implement backward method in your Function subclass")
