import os
from os.path import join
import matplotlib.pyplot as plt
from tqdm import tqdm

import losses
import optimizers
from core import train_epoch, validation_epoch
from datasets import *
from datasets.dataset import Dataset_imgs
from modules.mlp import MLP
from nn import NeuralNetwork
from utils import dump_json

from utils.logging import setup_logger


def accuracy(output, y):
    return (output.argmax(axis=1) == y).sum() / len(output)


def precision(output, y):
    tp = ((output.argmax(axis=1) == y) & (output.argmax(axis=1) == 1)).sum()
    fp = ((output.argmax(axis=1) != y) & (output.argmax(axis=1) == 1)).sum()
    return tp / (tp + fp)


def recall(output, y):
    tp = ((output.argmax(axis=1) == y) & (output.argmax(axis=1) == 1)).sum()
    fn = ((output.argmax(axis=1) != y) & (output.argmax(axis=1) == 0)).sum()
    return tp / (tp + fn)


def f1_score(output, y):
    prec = precision(output, y)
    rec = recall(output, y)
    return 2 * (prec * rec) / (prec + rec)


def main(optimizer_: str, activation_: str, loss_: str, ) -> None:
    """Main function to run the training loop."""

    logger = setup_logger("train_logger", "train.log")
    stats = {
        "losses": {
            "train": [],
            "validation": []
        },
        "accuracies": {
            "train": [],
            "validation": []
        }
    }
    epochs = 20
    save_state_frequency = 1
    min_save_epoch = 0
    weights_dir = "weights/mnist"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    nn = MLP([(1 * 28 * 28, 64), (64, 10)], activation_=activation_)
    for param in nn.parameters():
        print(param.shape)
    optimizer = getattr(optimizers, optimizer_)(nn, lr=0.001)
    loss_fn = getattr(losses, loss_)()
    print(loss_fn)
    dataset = Dataset_imgs()
    dataloader = DataLoader(dataset, shuffle=True)

    for epoch in tqdm(range(epochs)):
        train_avg_loss, train_avg_accuracy = train_epoch(nn, optimizer, loss_fn, dataloader)
        validation_avg_loss, validation_avg_accuracy = validation_epoch(nn, loss_fn, dataloader)

        stats["losses"]["train"].append(train_avg_loss)
        stats["losses"]["validation"].append(validation_avg_loss)
        stats["accuracies"]["train"].append(train_avg_accuracy)
        stats["accuracies"]["validation"].append(validation_avg_accuracy)

        if epoch % save_state_frequency == 0 and epoch >= min_save_epoch:
            nn.save_state(join(weights_dir, f"mnist_{epoch}.npy"))
            # TODO: implement save optimizer state
            # optimizer.save_state(join(weights_dir, f"iris_{epoch}_optimizer.npy"))
            dump_json(join(weights_dir, f"stats.json"), stats)

        logger.info(f"Epoch {epoch + 1}/{epochs} | loss= t:{train_avg_loss:.5f} v:{validation_avg_loss:.5f} "
                    f"| accuracy= t:{train_avg_accuracy:.5f} v:{validation_avg_accuracy:.5f}")

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        for i, type_ in enumerate(stats.keys()):
            ax[i].plot(stats[type_]["train"], label="train")
            ax[i].plot(stats[type_]["validation"], label="validation")
            ax[i].set_title(type_.capitalize())
            ax[i].legend()
        plt.show()


if __name__ == '__main__':
    main(optimizer_="SGD", activation_="ReLU", loss_="MSE")
