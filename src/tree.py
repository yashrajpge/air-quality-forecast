"""A regression tree written from scratch with NumPy.

Each split is chosen by scanning every feature threshold for the largest
reduction in squared error. The scan is vectorized with cumulative sums,
so a fit over a few hundred rows takes milliseconds.
"""

import numpy as np


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "n")

    def __init__(self, value, n):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = value
        self.n = n


def _sse(y):
    if len(y) == 0:
        return 0.0
    return float(((y - y.mean()) ** 2).sum())


class RegressionTree:
    def __init__(self, max_depth=4, min_samples_split=10, min_gain=1e-9):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.root_ = self._grow(X, y, depth=0)
        return self

    def _best_split(self, X, y):
        """Return (gain, feature, threshold) for the best squared-error split."""
        n, n_features = X.shape
        base = _sse(y)
        best = None
        for f in range(n_features):
            order = np.argsort(X[:, f], kind="mergesort")
            xs, ys = X[order, f], y[order]
            c1 = np.cumsum(ys)
            c2 = np.cumsum(ys * ys)
            left_n = np.arange(1, n)
            right_n = n - left_n
            left_sse = c2[:-1] - c1[:-1] ** 2 / left_n
            right_sse = (c2[-1] - c2[:-1]) - (c1[-1] - c1[:-1]) ** 2 / right_n
            total = left_sse + right_sse
            total[xs[:-1] == xs[1:]] = np.inf  # no threshold between equal values
            i = int(np.argmin(total))
            gain = base - float(total[i])
            if best is None or gain > best[0]:
                t = float(xs[i] + xs[i + 1]) / 2.0
                # adjacent floats: the midpoint can round to an endpoint,
                # which would put no sample on one side - nudge it inside
                if t <= xs[i]:
                    t = float(np.nextafter(xs[i], xs[i + 1]))
                elif t >= xs[i + 1]:
                    t = float(np.nextafter(xs[i + 1], xs[i]))
                best = (gain, f, t)
        return best

    def _grow(self, X, y, depth):
        if len(y) == 0:
            return _Node(0.0, 0)  # can only arise from float rounding at a parent split
        node = _Node(float(y.mean()), len(y))
        if depth >= self.max_depth or len(y) < self.min_samples_split or _sse(y) < 1e-12:
            return node
        gain, f, t = self._best_split(X, y)
        if gain < self.min_gain:
            return node
        left = X[:, f] <= t
        if left.all() or not left.any():
            return node
        node.feature, node.threshold = f, t
        node.left = self._grow(X[left], y[left], depth + 1)
        node.right = self._grow(X[~left], y[~left], depth + 1)
        return node

    def _predict_one(self, x, node):
        while node.feature is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.root_) for x in X])
