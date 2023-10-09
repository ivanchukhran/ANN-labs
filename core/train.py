from os.path import join
import matplotlib.pyplot as plt

import losses
import optimizers
from core import train_epoch, validation_epoch
from datasets import *
from nn import NeuralNetwork
from utils import dump_json

from utils.logging import setup_logger


def main(config: dict) -> None:
    """Main function to run the training loop.

    Args:
        config (dict): Configuration dictionary.
    """

    logger = setup_logger("train_logger", "train.log")
    weights_dir = config["training"]["weights_dir"]

    x_labels = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    y_labels = ['Species']

    ds_config = config["dataset"]

    dataset = Dataset(root_dir=ds_config["root_dir"], x_labels=x_labels, y_labels=y_labels,
                      scaler=ds_config["scaler"], batch_size=ds_config["batch_size"],
                      leave_last=ds_config["leave_last"])
    dataloader = DataLoader(dataset, shuffle=ds_config["shuffle"])

    nn_config = config["neural_network"]
    nn = NeuralNetwork(nn_config)

    loss_fn = getattr(losses, nn_config["loss_fn"])()
    optimizer_config = config["optimizer"]
    optimizer = getattr(optimizers, config["optimizer"]["type"])(nn, **optimizer_config["params"])

    epochs = config["training"]["epochs"]
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

    for epoch in range(epochs):
        train_avg_loss, train_avg_accuracy = train_epoch(nn, optimizer, loss_fn, dataloader)
        validation_avg_loss, validation_avg_accuracy = validation_epoch(nn, loss_fn, dataloader)

        stats["losses"]["train"].append(train_avg_loss)
        stats["losses"]["validation"].append(validation_avg_loss)
        stats["accuracies"]["train"].append(train_avg_accuracy)
        stats["accuracies"]["validation"].append(validation_avg_accuracy)

        if epoch % config["training"]["save_state_frequency"] == 0 and epoch >= config["training"]["min_save_epoch"]:
            # nn.save_state(f"weights/iris/iris_{epoch}.npy")
            # optimizer.save_state(f"weights/iris/iris_{epoch}_optimizer.npy")
            nn.save_state(join(weights_dir, f"iris_{epoch}.npy"))
            optimizer.save_state(join(weights_dir, f"iris_{epoch}_optimizer.npy"))
            dump_json(join(weights_dir, f"iris_{epoch}_config.json"), config)

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
    config = {
        "neural_network": {
            "layers": [
                {
                    "type": "Linear",
                    "params": {
                        "in_features": 4,
                        "out_features": 10
                    },
                },
                {
                    "type": "Linear",
                    "params": {
                        "in_features": 10,
                        "out_features": 10
                    }
                },
                {
                    "type": "Linear",
                    "params": {
                        "in_features": 10,
                        "out_features": 3
                    },
                }
            ]

        },
        "optimizer": {
            "type": "SGD",
            "params": {
                "lr": 0.001
            }
        },
        "dataset": {
            "root_dir": "data/iris/Iris.csv",
            "scaler": "MinMaxScaler",
            "batch_size": 32,
            "leave_last": True,
            "shuffle": True
        },
        "training": {
            "epochs": 10,
            "save_state_frequency": 1,
            "min_save_epoch": 0,
            "weights_dir": "weights/iris"
        }

    }
    main(config)
