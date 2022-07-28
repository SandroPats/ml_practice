import oracles
import numpy as np
from scipy.special import logsumexp
from scipy.special import expit
from sklearn.model_selection import train_test_split
import time


class GDClassifier:

    def __init__(
        self, loss_function, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, **kwargs
    ):
        self.loss_func = loss_function
        self.alpha = step_alpha
        self.beta = step_beta
        self.tol = tolerance
        self.max_iter = max_iter
        self.oracle = oracles.BinaryLogistic(**kwargs)

    def fit(self, X, y, w_0=np.arange(0), trace=False, valid=False):
        self.y_unique = np.unique(y)
        history = {'time': [], 'func': []} if trace else None
        if valid:
            X, X_val, y, y_val = train_test_split(X, y, test_size=0.3,
                                                  random_state=42)
            history['accuracy'] = []

        def history_push(w_k, tstart, f_val):
            history['time'].append(time.time() - tstart)
            history['func'].append(f_val)
            if valid:
                pred = np.sign(X_val.dot(w_k))
                pred[pred == 0] = -1
                pred = (pred == y_val).mean()
                history['accuracy'].append(pred)

        if w_0.size > 0:
            w_k = w_0
        else:
            w_k = np.zeros(X.shape[1])
        start = time.time()
        func_val = self.oracle.func(X, y, w_k)
        if trace:
            history_push(w_k, start, func_val)
        start = time.time()
        for k in range(1, self.max_iter+1):
            grad_k = self.oracle.grad(X, y, w_k)
            lambd = self.alpha / k**self.beta
            w_pr = w_k.copy()
            w_k -= grad_k * lambd
            func_val = self.oracle.func(X, y, w_k)
            if trace:
                history_push(w_k, start, func_val)
            dif = func_val - self.oracle.func(X, y, w_pr)
            if abs(dif) < self.tol:
                self.w = w_k
                return history
            start = time.time()
        self.w = w_k
        return history

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)*2 - 1

    def predict_proba(self, X):
        Mrgn = X.dot(self.w)[:, np.newaxis] * self.y_unique
        return expit(Mrgn)

    def get_objective(self, X, y):
        return self.oracle.func(X, y, self.w)

    def get_gradient(self, X, y):
        return self.oracle.grad(X, y, self.w)

    def get_weights(self):
        return self.w


class SGDClassifier(GDClassifier):

    def __init__(
        self, loss_function, batch_size, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs
    ):
        self.loss_func = loss_function
        self.batch_size = batch_size
        self.alpha = step_alpha
        self.beta = step_beta
        self.tol = tolerance
        self.max_iter = max_iter
        self.oracle = oracles.BinaryLogistic(**kwargs)
        self.y_unique = np.array([-1, 1])
        np.random.seed(random_seed)

    def fit_from_iter(self, iterator):
        for k in range(1, self.max_iter):
            X_b, y_b = next(iterator)
            if k == 1:
                w_k = np.zeros(X_b.shape[1])
            grad_k = self.oracle.grad(X_b, y_b, w_k)
            lambd = self.alpha / k**self.beta
            w_k -= grad_k * lambd
        self.w = w_k
        
    def fit(self, X, y, w_0=np.arange(0), trace=False, log_freq=1, valid=False):
        if trace:
            history = {'epoch_num': [], 'time': [],
                       'func': [],
                       'weights_diff': []}
        else:
            history = None
        if valid and trace:
            X, X_val, y, y_val = train_test_split(X, y, test_size=0.3,
                                                  random_state=42)
            history['accuracy'] = []

        def history_push(X, y, w_k, tstart, w_diff, processed_elems):
            history['time'].append(time.time() - tstart)
            history['epoch_num'].append(processed_elems / X.shape[0])
            f_val = self.oracle.func(X, y, w_k)
            history['func'].append(f_val)
            history['weights_diff'].append(np.dot(w_diff, w_diff))
            if valid:
                pred = np.sign(X_val.dot(w_k))
                pred[pred == 0] = -1
                pred = (pred == y_val).mean()
                history['accuracy'].append(pred)

        def batch_generator(X, y):
            inds = np.arange(X.shape[0])
            while True:
                np.random.shuffle(inds)
                iters = range(0, X.shape[0], self.batch_size)
                X_shd = X[inds]
                y_shd = y[inds]
                for i in iters:
                    yield (X_shd[i:i+self.batch_size],
                           y_shd[i:i+self.batch_size])

        if w_0.size > 0:
            w_k = w_0.copy()
        else:
            w_k = np.zeros(X.shape[1])
        func_prev = 0
        batch_gen = batch_generator(X, y)
        processed = 0
        w_pr = 0
        prev_ep = 0
        worktime = 0
        start = time.time()
        if trace:
            history_push(X, y, w_k, start, 0, 0)
        start = time.time()
        for k in range(1, self.max_iter*(X.shape[0]//self.batch_size) + 1):
            X_b, y_b = next(batch_gen)
            grad_k = self.oracle.grad(X_b, y_b, w_k)
            processed += self.batch_size
            lambd = self.alpha / k**self.beta
            w_k -= grad_k * lambd
            if trace and (abs(history['epoch_num'][-1] -
                              processed/X.shape[0]) > log_freq):
                history_push(X, y, w_k, start,
                             w_k - w_pr, processed)
                w_pr = w_k.copy()
                start = time.time()
            epoch = abs(prev_ep - processed/X.shape[0])
            if epoch >= 1:
                func_val = self.oracle.func(X, y, w_k)
                dif = abs(func_val - func_prev)
                if dif < self.tol:
                    if trace:
                        history_push(X, y, w_k, start,
                                     w_k - w_pr, processed)
                    self.w = w_k
                    return history
                func_prev = func_val
                prev_ep = processed/X.shape[0]
        self.w = w_k
        if trace:
            history_push(X, y, w_k, start,
                         w_k - w_pr, processed)
        return history

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)*2 - 1

    def predict_proba(self, X):
        Mrgn = X.dot(self.w)[:, np.newaxis] * self.y_unique
        return expit(Mrgn)

    def get_objective(self, X, y):
        return self.oracle.func(X, y, self.w)

    def get_gradient(self, X, y):
        return self.oracle.grad(X, y, self.w)

    def get_weights(self):
        return self.w
