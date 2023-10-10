from nn import NeuralNetwork


class Optimizer:
    lr: float
    model: NeuralNetwork

    def __init__(self, model: NeuralNetwork, lr: float = 0.001, *args, **kwargs):
        self.model = model
        self.lr = lr

    def step(self):
        raise NotImplementedError("You should implement this method in subclass.")

    def zero_grad(self):
        for module in self.model.modules:
            if module.has_parameters():
                for grad in module.grads():
                    grad *= 0

    def save_state(self):
        raise NotImplementedError("You should implement this method in subclass.")

    def load_state(self):
        raise NotImplementedError("You should implement this method in subclass.")