import numpy as np

from optimizers import SGD, Adam, RMSProp, Adagrad
from utils import fetch_json
from nn import NeuralNetwork


def main():
    config = fetch_json('fc.json')
    nn = NeuralNetwork(config)
    print("saving state")
    nn.save_state("fc.npy")
    print("done")
    print("loading state")
    nn.load_state("fc.npy")
    print("done")

    # print(nn.to_dict())
    # optim = RMSProp(nn, lr=0.001)
    # print(nn)
    # print(f"parameters: {nn.modules[0].parameters()[0][:10]}")
    # x = np.random.randn(1, 784)
    # output = nn(x)
    # print(output)
    # back = nn.backward(output)
    # optim.step()
    # print(f"parameters: {nn.modules[0].parameters()[0][:10]}")
    # print(back)


if __name__ == '__main__':
    main()