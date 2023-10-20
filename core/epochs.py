from typing import Tuple

from datasets import DataLoader
from losses import Loss
from modules import Module
from optimizers import Optimizer


def train_epoch(model: Module, optimizer: Optimizer, loss_fn: Loss, dataloader: DataLoader) -> Tuple:
    avg_accuracy = .0
    avg_loss = .0

    for i, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        accuracy = (output.argmax(axis=1) == y).sum()
        loss.backward()
        optimizer.step()

        avg_loss += loss / len(y)
        avg_accuracy += accuracy / len(y)

    return avg_loss, avg_accuracy


def validation_epoch(model: Module, loss_fn: Loss, dataloader: DataLoader) -> Tuple:
    avg_accuracy = .0
    avg_loss = .0

    for i, (x, y) in enumerate(dataloader):
        output = model(x)
        loss = loss_fn(output, y)
        accuracy = (output.argmax(axis=1) == y).sum()

        avg_loss += loss / len(y)
        avg_accuracy += accuracy / len(y)

    return avg_loss, avg_accuracy
