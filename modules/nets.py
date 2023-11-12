from activations import *
from modules import *

class WiNET(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(11, 1)
        # self.linear1 = Linear(6, 1)
        # self.relu = ReLU()

    def forward(self, x):
        x = self.linear(x)
        # x = self.relu(x)
        # x = self.linear1(x)
        # x = self.sigmoid(x)
        return x