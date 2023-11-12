from typing import Tuple, Callable

from datasets import DataLoader
from losses import Loss
from modules import Module
from optimizers import Optimizer


def train_epoch(model: Module, optimizer: Optimizer, loss_fn: Loss, dataloader: DataLoader, metric_fn: Callable) -> Tuple:
    avg_accuracy = .0
    avg_loss = .0

    for i, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        accuracy = metric_fn(output.data, y.data)
        loss.backward()
        optimizer.step()
        avg_loss += loss.data / len(y)
        avg_accuracy += accuracy / len(y)

    return avg_loss, avg_accuracy


def validation_epoch(model: Module, loss_fn: Loss, dataloader: DataLoader, metric_fn: Callable) -> Tuple:
    avg_accuracy = .0
    avg_loss = .0

    for i, (x, y) in enumerate(dataloader):
        output = model(x)
        loss = loss_fn(output, y)
        accuracy = metric_fn(output.data, y.data)

        avg_loss += loss.data / len(y)
        avg_accuracy += accuracy / len(y)

    return avg_loss, avg_accuracy
