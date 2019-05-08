import numpy as np

from src.Utils import ValidationUtils as vu
from multiprocessing import Process


def proc(k, X, y, hidden, l, alfa, la,type="MSE"):
    for a in alfa:
        for h in hidden:
            vu.kFold(k, X, y, h, l, a, la, type=type)

class Grid():
    def __init__(self, X, y,type="MSE"):
        hidden = np.array([5,6,7,8,9,10,14,16,18,20,25,30,50,70,80,100])
        learning = np.array([0.05,0.01,0.003,0.005,0.007,0.001,0.0003,0.0005,0.0007])
        alfa = np.array([0.5, 0.6,0.65, 0.7,0.75, 0.8,0.85, 0.9])
        lamb = np.array([0.1,0.01,0.001,0.05,0.005])

        for la in lamb:
            for l in learning:
                 p = Process(target=proc, args=(5, X, y, hidden, l, alfa, la,type))
                 p.start()
