import numpy as np


def accuracy(output, y):
    return (output == y).sum() / len(output)


def precision(output, y):
    tp = ((output == y) & (output == 1)).sum()
    fp = ((output != y) & (output == 1)).sum()
    return tp / (tp + fp)


def recall(output, y):
    tp = ((output == y) & (output == 1)).sum()
    fn = ((output != y) & (output == 0)).sum()
    return tp / (tp + fn)


def f1_score(output, y):
    prec = precision(output, y)
    rec = recall(output, y)
    return 2 * (prec * rec) / (prec + rec)

def mae(output, y):
    return np.abs(output - y).mean()

def mse(output, y):
    return ((output - y) ** 2).mean()

def rmse(output, y):
    return np.sqrt(mse(output, y))