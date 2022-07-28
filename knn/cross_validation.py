import numpy as np
import nearest_neighbors as nn
from sklearn.neighbors import NearestNeighbors


def kfold(n, n_folds):
    inds = np.arange(n)
    val_folds = np.array_split(inds, n_folds)
    fold_list = []
    if n_folds == 1:
        return inds
    for i in range(n_folds):
        train_fold = np.hstack((val_folds[0:i] + val_folds[i+1:]))
        fold_list.append((train_fold, val_folds[i]))
    return fold_list


def knn_cross_val_score(X, y, k_list, score, cv=[], **kwargs):
    def count_votes(inds, y_train):
        a = np.arange(0)
        for v in np.unique(y_train):
            if isinstance(inds, tuple):
                wght = 1 / (inds[0]+10**(-5))
                y_sum = (wght*(y_train[inds[1]] == v))
            else:
                y_sum = (y_train[inds] == v)
            if len(y_sum.shape) < 2:
                y_sum = np.array([y_sum.sum()])
            else:
                y_sum = y_sum.sum(axis=1)
            if a.size > 1:
                a = np.concatenate([a, y_sum[:, np.newaxis]], axis=1)
            else:
                a = y_sum[:, np.newaxis]
        return a.argmax(axis=1)

    k_score = {k: np.arange(0) for k in k_list}
    if not cv:
        cv = kfold(X.shape[0], 3)
    knn = nn.KNNClassifier(max(k_list), **kwargs)
    for val in cv:
        knn.fit(X[val[0]], y[val[0]])
        if 'weights' in kwargs:
            wght = kwargs['weights']
        else:
            wght = True
        nb = knn.find_kneighbors(X[val[1]], wght)
        for k in k_list:
            if wght:
                a = count_votes((nb[0][:, :k], nb[1][:, :k]), y[val[0]])
            else:
                a = count_votes(nb[:, :k], y[val[0]])
            accur = np.mean(a == y[val[1]])
            k_score[k] = np.append(k_score[k], accur)
    return k_score
