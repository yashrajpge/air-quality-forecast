"""Gradient boosting over the hand-written regression trees.

Squared-error loss, so each stage fits a new tree to the residuals of
the current ensemble and adds it scaled by the learning rate.
"""

import numpy as np

from tree import RegressionTree


class GradientBoostingRegressor:
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=3,
                 min_samples_split=10, subsample=1.0, seed=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.base_ = float(y.mean())
        pred = np.full(len(y), self.base_)
        rng = np.random.default_rng(self.seed)
        self.trees_ = []
        self.train_rmse_ = []
        for _ in range(self.n_estimators):
            residuals = y - pred
            if self.subsample < 1.0:
                idx = rng.choice(len(y), size=int(len(y) * self.subsample), replace=False)
            else:
                idx = np.arange(len(y))
            tree = RegressionTree(max_depth=self.max_depth,
                                  min_samples_split=self.min_samples_split)
            tree.fit(X[idx], residuals[idx])
            pred = pred + self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
            self.train_rmse_.append(float(np.sqrt(((y - pred) ** 2).mean())))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        pred = np.full(len(X), self.base_)
        for tree in self.trees_:
            pred = pred + self.learning_rate * tree.predict(X)
        return pred
