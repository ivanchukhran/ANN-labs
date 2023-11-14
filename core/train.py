import os
from logging import Logger
from os.path import join
from typing import Callable

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import losses
import modules
import optimizers
from core import train_epoch, validation_epoch
from datasets import *
from datasets.dataset import *
from losses import Loss
from modules import *
from optimizers import Optimizer
from .evaluations import *

from utils.logging import setup_logger


def train(model: Module,
          optimizer: Optimizer,
          loss_fn: Loss,
          dataloaders: tuple[DataLoader, DataLoader],
          metric_fn: Callable,
          epochs: int,
          stats: dict,
          save_state_frequency: int,
          min_save_epoch: int, weights_dir: str,
          logger: Logger,
          validate: bool = True,
          verbose: bool = False):
    """Train the model for the given number of epochs.
    
    Args:
        model (Module): The model to train.
        optimizer (Optimizer): The optimizer to use for training.
        loss_fn (Loss): The loss function to use for training.
        dataloaders (tuple[DataLoader, DataLoader]): The train and validation dataloaders.
        metric_fn (Callable): The metric function to use for evaluation.
        epochs (int): The number of epochs to train the model for.
        stats (dict): The dictionary to store the training and validation losses and accuracies.
        save_state_frequency (int): The frequency at which to save the model and optimizer states.
        min_save_epoch (int): The minimum epoch from which to start saving the model and optimizer states.
        weights_dir (str): The directory to save the model and optimizer states.
        logger (Logger): The logger to use for logging.
        validate (bool, optional): Whether to validate the model after each epoch. Defaults to True.
        verbose (bool, optional): The display mode to use for displaying the training and validation losses and accuracies. Defaults to None.
    """
    train_dl, validation_dl = dataloaders
    for epoch in tqdm(range(epochs)):
        train_avg_loss, train_avg_accuracy = train_epoch(model, optimizer, loss_fn, train_dl, metric_fn)
        if validate:
            validation_avg_loss, validation_avg_accuracy = validation_epoch(model, loss_fn, validation_dl, metric_fn)
            stats["losses"]["validation"].append(validation_avg_loss)
            stats["accuracies"]["validation"].append(validation_avg_accuracy)
            val_avg_loss_str = f"v:{validation_avg_loss:.5f}"
            val_avg_accuracy_str = f"v:{validation_avg_accuracy:.5f}"
        else:
            val_avg_loss_str = ""
            val_avg_accuracy_str = ""

        stats["losses"]["train"].append(train_avg_loss)
        stats["accuracies"]["train"].append(train_avg_accuracy)

        if save_state_frequency and epoch % save_state_frequency == 0 and epoch >= min_save_epoch:
            model.save_state(join(weights_dir, f"{model.__class__.__name__}_{epoch}.npy"))
            optimizer.save_state(join(weights_dir, f"{optimizer.__class__.__name__}_{epoch}.npy"))
            np.save(file=join(weights_dir, f"stats.json"), arr=stats, allow_pickle=True)

        logger.info(f"Epoch {epoch + 1}/{epochs} | loss= t:{train_avg_loss:.5f} {val_avg_loss_str} "
                    f"| accuracy= t:{train_avg_accuracy:.5f} {val_avg_accuracy_str}")
        if verbose:
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            for i, type_ in enumerate(stats.keys()):
                ax[i].plot(stats[type_]["train"], label="train")
                ax[i].plot(stats[type_]["validation"], label="validation")
                ax[i].set_title(type_.capitalize())
                ax[i].legend()
            plt.show()


def main(dataset: str, model: str, optimizer_: str, loss_: str, ) -> None:
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
    weights_dir = "../weights/wine"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    model = getattr(modules, model)()
    optimizer = getattr(optimizers, optimizer_)(model, lr=1e-4)
    loss_fn = getattr(losses, loss_)()
    wine_df = pd.read_csv("../data/wine/winequality-red.csv")

    Y = wine_df['quality']
    X = wine_df.drop('quality', axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X.values)
    Y = scaler.fit_transform(Y.values.reshape(-1, 1))

    batch_size = 1

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)
    train_ds = WineDataset((X_train, Y_train), batch_size=batch_size)
    test_ds = WineDataset((X_test, Y_test), batch_size=batch_size)
    train_dl = DataLoader(train_ds, shuffle=True)
    test_dl = DataLoader(test_ds, shuffle=True)

    train(model=model, optimizer=optimizer, loss_fn=loss_fn, dataloaders=(train_dl, test_dl), metric_fn=rmse, epochs=epochs,
          stats=stats, save_state_frequency=save_state_frequency, min_save_epoch=min_save_epoch, weights_dir=weights_dir, logger=logger)


if __name__ == '__main__':
    main(dataset='wine', model='WiNET', optimizer_="Adam", loss_="MSE")
