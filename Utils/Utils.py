import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidDerivative(x):
    return (x * (1 - x))

def error(y,p,type):
    if (type == "MSE"):
        return MSE(y,p)
    if (type == "MEE"):
        return MEE(y, p)
    if (type == "accuracy"):
        return accuracy(y, p)

def MSE(y,p):
    return np.mean(np.sum(np.square(y - p), axis=1))

def MEE(y,p):
    return np.mean(np.sqrt(np.sum(np.square(y - p), axis=1)))

def accuracy(y,p):
    err = 0
    for l in range(0, len(y)):

        if (predictClass(y[l]) != p[l]):
            err += 1
    return 100 - ((err / len(y)) * 100)

def predictClass(x):
    if x<0.5:
        return 0.0
    else:
        return 1.0
