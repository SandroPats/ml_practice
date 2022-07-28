import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp
from scipy.special import expit


class BaseSmoothOracle:
    def func(self, w):
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):

    def __init__(self, l2_coef):
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        Mrgn = X.dot(w) * y
        Loss = np.logaddexp(0, -Mrgn)
        return Loss.mean() + 0.5 * self.l2_coef * np.dot(w, w)

    def grad(self, X, y, w):
        if (X.shape[0] == 1) and sp.issparse(X):
            X = X.toarray()
        Mrgn = X.dot(w) * y
        grad = ((X.T).dot(-y * expit(-Mrgn))/X.shape[0] +
                self.l2_coef * w)
        return grad
