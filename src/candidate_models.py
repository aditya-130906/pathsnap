"""Small dependency-light candidate classifiers for Phase 1 experiments.

These are deliberately implemented here rather than relying on unavailable
third-party ML packages. They are experiment baselines, not a final model.
"""
from __future__ import annotations
import time
import numpy as np


EPS = 1e-9


class Preprocessor:
    def fit(self, x: np.ndarray) -> "Preprocessor":
        self.median = np.nanmedian(x, axis=0)
        filled = np.where(np.isfinite(x), x, self.median)
        self.mean = filled.mean(axis=0)
        self.scale = np.maximum(filled.std(axis=0), 1e-6)
        return self
    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.where(np.isfinite(x), x, self.median) - self.mean) / self.scale


class DistanceBaseline:
    name = "distance_baseline"
    def fit(self, x, y): return self
    def predict_proba(self, x): return 1 / (1 + np.exp(np.clip(x[:, 0], -40, 40)))


class GaussianNB:
    name = "gaussian_naive_bayes"
    def fit(self, x, y):
        self.prior = np.clip(y.mean(), EPS, 1 - EPS)
        self.mean0, self.mean1 = x[y == 0].mean(axis=0), x[y == 1].mean(axis=0)
        self.var0 = x[y == 0].var(axis=0) + 1e-4
        self.var1 = x[y == 1].var(axis=0) + 1e-4
        return self
    def predict_proba(self, x):
        log0 = np.log(1 - self.prior) - .5 * np.sum(np.log(self.var0) + (x-self.mean0)**2/self.var0, axis=1)
        log1 = np.log(self.prior) - .5 * np.sum(np.log(self.var1) + (x-self.mean1)**2/self.var1, axis=1)
        return 1 / (1 + np.exp(-np.clip(log1-log0, -40, 40)))


class LogisticGD:
    name = "logistic_regression_gd"
    def __init__(self, learning_rate=.08, iterations=500, l2=.01): self.learning_rate, self.iterations, self.l2 = learning_rate, iterations, l2
    def fit(self, x, y):
        self.weights = np.zeros(x.shape[1]); self.bias = 0.0
        pos_weight = len(y) / max(2 * y.sum(), 1); neg_weight = len(y) / max(2 * (len(y)-y.sum()), 1)
        weights = np.where(y == 1, pos_weight, neg_weight)
        for _ in range(self.iterations):
            p = 1 / (1 + np.exp(-np.clip(x @ self.weights + self.bias, -40, 40)))
            error = (p - y) * weights
            self.weights -= self.learning_rate * ((x.T @ error) / len(y) + self.l2 * self.weights)
            self.bias -= self.learning_rate * error.mean()
        return self
    def predict_proba(self, x): return 1 / (1 + np.exp(-np.clip(x @ self.weights + self.bias, -40, 40)))


class KNN:
    name = "knn"
    def __init__(self, k=15, max_negative=2500, seed=42): self.k, self.max_negative, self.seed = k, max_negative, seed
    def fit(self, x, y):
        rng = np.random.default_rng(self.seed); pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
        neg = rng.choice(neg, size=min(len(neg), self.max_negative), replace=False)
        keep = np.concatenate([pos, neg]); self.x, self.y = x[keep], y[keep]
        return self
    def predict_proba(self, x):
        out = []
        for batch in np.array_split(x, max(1, len(x)//300 + 1)):
            d2 = ((batch[:, None, :] - self.x[None, :, :])**2).sum(axis=2)
            nearest = np.argpartition(d2, min(self.k, d2.shape[1]-1), axis=1)[:, :self.k]
            out.append(self.y[nearest].mean(axis=1))
        return np.concatenate(out)


class RandomStumpForest:
    name = "random_stump_forest"
    def __init__(self, trees=80, seed=42): self.trees, self.seed = trees, seed
    def fit(self, x, y):
        rng = np.random.default_rng(self.seed); self.stumps = []
        for _ in range(self.trees):
            feature = int(rng.integers(x.shape[1])); threshold = float(rng.choice(x[:, feature]))
            left, right = y[x[:, feature] <= threshold], y[x[:, feature] > threshold]
            if len(left) < 5 or len(right) < 5: continue
            # Keep only stumps with useful class separation.
            if abs(left.mean() - right.mean()) < .01: continue
            self.stumps.append((feature, threshold, max(EPS, min(1-EPS, left.mean())), max(EPS, min(1-EPS, right.mean()))))
        return self
    def predict_proba(self, x):
        if not self.stumps: return np.full(len(x), .5)
        scores = [np.where(x[:, f] <= t, p_left, p_right) for f, t, p_left, p_right in self.stumps]
        return np.mean(scores, axis=0)


def evaluate(y: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> dict:
    p = np.clip(scores, EPS, 1-EPS); predicted = p >= .5
    tp = int(((predicted == 1) & (y == 1)).sum()); fp = int(((predicted == 1) & (y == 0)).sum()); fn = int(((predicted == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    top1 = []
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        top1.append(int(y[rows[np.argmax(p[rows])]] == 1))
    return {"candidate_precision": precision, "candidate_recall": recall, "candidate_f1": 2*precision*recall/max(precision+recall, EPS),
            "candidate_log_loss": float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean()), "observation_top1_accuracy": float(np.mean(top1)), "observations": len(top1)}


def fit_and_score(model, x_train, y_train, x_test, y_test, groups_test):
    started = time.perf_counter(); model.fit(x_train, y_train); train_s = time.perf_counter() - started
    started = time.perf_counter(); scores = model.predict_proba(x_test); infer_s = time.perf_counter() - started
    result = evaluate(y_test, scores, groups_test); result.update(model=model.name, train_seconds=train_s, inference_seconds=infer_s)
    return result
