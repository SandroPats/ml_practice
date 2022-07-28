import numpy as np
import distances as dst
from sklearn.neighbors import NearestNeighbors


class KNNClassifier:
    strat_set = {'my_own', 'brute', 'kd_tree', 'ball_tree'}
    metric_dict = {'cosine': dst.cosine_distance,
                   'euclidean': dst.euclidean_distance}

    def __init__(self, k, strategy='brute', metric='euclidean',
                 weights=True, test_block_size=0):
        self.k = k
        if strategy not in self.strat_set:
            raise AttributeError('invalid attribute provided: Second')
        self.strategy = strategy
        if metric not in self.metric_dict:
            raise AttributeError('invalid attribute provided: Third')
        self.metric = metric
        self.weights = weights
        self.tbs = test_block_size

    def fit(self, X, y):
        self.y_train = y
        self.y_unique = np.unique(y)
        self.X_train = X
        if self.strategy == 'brute':
            metr = self.metric_dict[self.metric]
            self.brute = NearestNeighbors(n_neighbors=self.k,
                                          algorithm='brute',
                                          metric=metr.__name__).fit(X)
        elif self.strategy == 'kd_tree':
            self.kd_tree = NearestNeighbors(n_neighbors=self.k,
                                            algorithm='kd_tree',
                                            metric='euclidean').fit(X)
        elif self.strategy == 'ball_tree':
            self.ball_tree = NearestNeighbors(n_neighbors=self.k,
                                              algorithm='ball_tree',
                                              metric='euclidean').fit(X)

    def find_kneighbors(self, X, return_distance=True):
        dist = self.metric_dict[self.metric](X, self.X_train)
        if return_distance:
            inds = np.argsort(dist, axis=1)
            dist_sorted = np.take_along_axis(dist, inds, axis=1)[:, :self.k]
            return dist_sorted, inds[:, :self.k]
        else:
            return dist.argsort(axis=1)[:, :self.k]

    def find_kneighbors_blockwise(self, X, return_distance):
        if not self.tbs or X.shape[0] % self.tbs:
            if X.shape[0] < 1000:
                self.tbs = X.shape[0]
            else:
                for i in range(3, 11):
                    if X.shape[0] % i == 0:
                        self.tbs = X.shape[0] // i
                        break
        iters = range(X.shape[0] // self.tbs)
        inds = np.arange(0)
        dist = np.arange(0)
        if return_distance:
            for i in iters:
                bdist, block = self.find_kneighbors(
                                                X[i*self.tbs:(i+1)*self.tbs],
                                                return_distance)
                if inds.size >= 2:
                    inds = np.concatenate([inds, block])
                    dist = np.concatenate([dist, bdist])
                else:
                    inds = block
                    dist = bdist
            if (inds.shape[0] <= 1) and (X.shape[0] != 2):
                print(self.tbs)
                print(X.shape[0])
            return dist, inds
        else:
            for i in iters:
                block = self.find_kneighbors(X[i*self.tbs:(i+1)*self.tbs],
                                             return_distance)
                if inds.size >= 2:
                    inds = np.concatenate([inds, block])
                else:
                    inds = block
            if (inds.shape[0] <= 1) and (X.shape[0] != 2):
                print(self.tbs)
                print(X.shape[0])
            return inds

    def count_votes(self, inds):
        a = np.arange(0)
        for v in self.y_unique:
            if self.weights:
                wght = 1 / (inds[0]+10**(-5))
                y_sum = (wght*(self.y_train[inds[1]] == v))
            else:
                y_sum = (self.y_train[inds] == v)
            if len(y_sum.shape) < 2:
                y_sum = np.array([y_sum.sum()])
            else:
                y_sum = y_sum.sum(axis=1)
            if a.size > 1:
                a = np.concatenate([a, y_sum[:, np.newaxis]], axis=1)
            else:
                a = y_sum[:, np.newaxis]
        a = a.argmax(axis=1)
        return a

    def predict(self, X):
        a_res = np.arange(0)
        if self.strategy == 'my_own':
            inds = self.find_kneighbors_blockwise(X, self.weights)
        else:
            inds = self.__dict__[self.strategy].kneighbors(
                                                X,
                                                return_distance=self.weights)
        return self.count_votes(inds)
