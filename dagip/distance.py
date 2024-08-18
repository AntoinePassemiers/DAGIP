import numpy as np
from scipy.spatial.distance import cdist


def kernel_matrix(X, Y, gc_content):

    weights = 1. / (1. + np.abs(gc_content[:-1] - gc_content[1:]))

    X = X[:, 1:] - X[:, :-1]
    Y = Y[:, 1:] - Y[:, :-1]
    X = X * np.sqrt(weights)[np.newaxis, :]
    Y = Y * np.sqrt(weights)[np.newaxis, :]
    D = cdist(X, Y) ** 2
    d1 = np.sum(X ** 2, axis=1)[:, np.newaxis]
    d2 = np.sum(Y ** 2, axis=1)[np.newaxis, :]
    return d1 + d2 - D
