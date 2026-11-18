"""Logistic Regression in numpy: the same objective as scikit-learn's LogisticRegression(penalty="l2").

Why this exists
---------------
The training laptop's Windows Application Control policy blocks the compiled extension files of
scikit-learn, scipy and pandas, while numpy (including numpy.linalg) loads. This module implements the
identical model so training can run locally. It is a full pipeline with the same steps as the sklearn
pipeline in train_logreg.make_model:

    signed log1p (fixed columns) -> median imputer -> standard scaler -> L2 logistic regression

Objective (identical to sklearn, intercept not penalised):

    minimise  C * sum_i s_i * logloss(y_i, sigmoid(x_i . w + b))  +  0.5 * ||w||^2

s_i are sample weights (balanced class weights here). The problem is strictly convex, so it has one global
optimum; sklearn's lbfgs and the Newton (IRLS) iterations used here converge to the same solution. Newton's
method uses the exact Hessian  C * X^T diag(s p (1-p)) X + diag(1,...,1,0)  and a backtracking line search,
and stops when the gradient's max-norm falls below `tol` (scaled by the number of weighted samples).
Convergence history (objective, gradient norm per iteration) is recorded in `history_`.
"""

from __future__ import annotations

import json

import numpy as np


def _sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _log1p_exp(z):
    """log(1 + exp(z)) without overflow."""
    return np.logaddexp(0.0, z)


class NumpyLogisticRegression:
    def __init__(self, C=1.0, max_iter=100, tol=1e-6, log1p_columns=(), **_ignored):
        self.C, self.max_iter, self.tol = float(C), int(max_iter), float(tol)
        self.log1p_columns = list(log1p_columns)

    # ---- preprocessing (fitted on the training rows passed to fit only) ----
    def _transform(self, X):
        X = np.array(X, dtype=np.float64, copy=True)
        for c in self.log1p_columns:
            X[:, c] = np.sign(X[:, c]) * np.log1p(np.abs(X[:, c]))
        nan = np.isnan(X)
        if nan.any():
            X[nan] = np.take(self.median_, np.nonzero(nan)[1])
        return (X - self.mean_) / self.scale_

    def _fit_preprocessing(self, X):
        X = np.array(X, dtype=np.float64, copy=True)
        for c in self.log1p_columns:
            X[:, c] = np.sign(X[:, c]) * np.log1p(np.abs(X[:, c]))
        self.median_ = np.nanmedian(X, axis=0)
        nan = np.isnan(X)
        if nan.any():
            X[nan] = np.take(self.median_, np.nonzero(nan)[1])
        self.mean_ = X.mean(axis=0)
        sd = X.std(axis=0)
        self.scale_ = np.where(sd > 0, sd, 1.0)          # constant columns stay at 0 after centring
        return (X - self.mean_) / self.scale_

    # ---- model ----
    def _objective(self, beta, Z, y, s):
        z = Z @ beta
        loss = np.sum(s * (_log1p_exp(z) - y * z))
        return self.C * loss + 0.5 * np.dot(beta[:-1], beta[:-1])

    def fit(self, X, y, logreg__sample_weight=None, sample_weight=None):
        s = logreg__sample_weight if logreg__sample_weight is not None else sample_weight
        y = np.asarray(y, dtype=np.float64)
        s = np.ones(len(y)) if s is None else np.asarray(s, dtype=np.float64)
        Z = np.c_[self._fit_preprocessing(X), np.ones(len(y))]
        d = Z.shape[1]
        reg = np.ones(d); reg[-1] = 0.0                    # intercept not penalised
        beta = np.zeros(d)
        f = self._objective(beta, Z, y, s)
        self.history_ = []
        self.converged_ = False
        for it in range(self.max_iter):
            p = _sigmoid(Z @ beta)
            grad = self.C * Z.T @ (s * (p - y)) + reg * beta
            gnorm = float(np.max(np.abs(grad)))
            self.history_.append({"iteration": it, "objective": float(f), "grad_max": gnorm})
            if gnorm <= self.tol * max(s.sum(), 1.0) * self.C:
                self.converged_ = True
                break
            W = s * p * (1 - p)
            H = self.C * (Z.T * W) @ Z + np.diag(reg)
            step = np.linalg.solve(H + 1e-12 * np.eye(d), grad)
            t = 1.0
            while True:                                     # backtracking (Armijo) line search
                new = beta - t * step
                fn = self._objective(new, Z, y, s)
                if fn <= f - 1e-4 * t * np.dot(grad, step) or t < 1e-10:
                    break
                t *= 0.5
            beta, f = new, fn
        self.n_iter_ = len(self.history_)
        self.coef_ = beta[:-1][None, :]                    # standardised-feature coefficients
        self.intercept_ = np.array([beta[-1]])
        self.classes_ = np.array([0, 1])
        self.final_gradient_max_ = self.history_[-1]["grad_max"]
        return self

    def decision_function(self, X):
        return self._transform(X) @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X):
        p = _sigmoid(self.decision_function(X))
        return np.c_[1 - p, p]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    # ---- portable export (used later by the API; no pickle needed) ----
    def to_dict(self, feature_names):
        return {"model": "logistic_regression", "implementation": "numpy Newton/IRLS, sklearn-equivalent objective",
                "C": self.C, "feature_names": list(feature_names),
                "log1p_features": [feature_names[c] for c in self.log1p_columns],
                "median": self.median_.tolist(), "mean": self.mean_.tolist(), "scale": self.scale_.tolist(),
                "coef_standardised": self.coef_[0].tolist(), "intercept": float(self.intercept_[0]),
                "n_iter": self.n_iter_, "converged": self.converged_, "final_gradient_max": self.final_gradient_max_}

    @classmethod
    def from_dict(cls, d):
        names = d["feature_names"]
        m = cls(C=d["C"], log1p_columns=[names.index(c) for c in d["log1p_features"]])
        m.median_, m.mean_, m.scale_ = (np.array(d[k]) for k in ("median", "mean", "scale"))
        m.coef_ = np.array(d["coef_standardised"])[None, :]
        m.intercept_ = np.array([d["intercept"]]); m.classes_ = np.array([0, 1])
        return m

    def save_json(self, path, feature_names):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(feature_names), f, indent=2)
