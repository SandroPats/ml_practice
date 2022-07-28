import numpy as np
import sklearn.neighbors
import sklearn.metrics


vbrute = sklearn.neighbors.VALID_METRICS['brute']
vmetrics = sklearn.metrics.pairwise._VALID_METRICS
pairwise = sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS


def install_metric(f):
    name = f.__name__
    if name not in vbrute:
        vbrute.append(name)
    if name not in vmetrics:
        vmetrics.append(name)
    pairwise[name] = f
    return f


@install_metric
def euclidean_distance(X, Y):
    YN = Y
    XN = X
    if len(Y.shape) == 1:
        YN = Y[np.newaxis, :]
    if len(X.shape) == 1:
        XN = X[np.newaxis, :]
    Mul = -2 * (XN @ YN.T)
    Nsum = Mul + (XN ** 2).sum(axis=1)[:, np.newaxis] + (YN ** 2).sum(axis=1)
    if (len(Y.shape) == 1) and (len(X.shape) == 1):
        return ((Nsum) ** 0.5)[0, 0]
    else:
        return (Nsum) ** 0.5


@install_metric
def cosine_distance(X, Y):
    YN = Y
    XN = X
    if len(Y.shape) == 1:
        YN = Y[np.newaxis, :]
    if len(X.shape) == 1:
        XN = X[np.newaxis, :]
    Mul = XN @ YN.T
    Xnorm = ((XN**2).sum(axis=1)[:, np.newaxis]) ** 0.5
    Ynorm = (YN**2).sum(axis=1) ** 0.5
    if (len(Y.shape) == 1) and (len(X.shape) == 1):
        return (1 - (Mul/Xnorm)/Ynorm)[0, 0]
    else:
        return 1 - (Mul/Xnorm)/Ynorm
