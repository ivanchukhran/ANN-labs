from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, model, lr=0.001):
        super().__init__(model, lr)

    def step(self):
        for module in self.model.modules:
            if module.has_parameters():
                for param, grad in zip(module.parameters(), module.grad_parameters()):
                    param -= self.lr * grad
