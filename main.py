import numpy as np

from utils import fetch_json
from modules import *
from nn import NeuralNetwork


def main():
    config = fetch_json('fully_connected.json')
    nn = NeuralNetwork(config)
    print(nn)
    x = np.random.randn(1, 784)
    output = nn(x)
    print(output)
    back = nn.backward(output)
    print(back)


if __name__ == '__main__':
    main()