"""Exact numpy inference for a fitted scikit-learn HistGradientBoostingClassifier (binary, no categorical features).

Why: the deployment must not depend on every laptop being able to load scikit-learn's compiled extensions
(the development laptop's Windows Application Control blocks some of them). This module reads the fitted trees out
of the saved joblib bundle and evaluates them with the same rules as sklearn's TreePredictor on raw features:

    at an internal node: if x is NaN -> go left iff missing_go_to_left, else go left iff x <= num_threshold
    raw score = baseline_prediction + sum of leaf values over all trees
    P(class 1) = sigmoid(raw score)                       (HalfBinomialLoss)

When scikit-learn loads normally, `load_hgb` returns the real estimator instead; `verify_against` compares both.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class NumpyHGB:
    def __init__(self, baseline: float, trees: list[dict], n_features: int, params: dict):
        self.baseline = float(baseline)
        self.trees = trees
        self.n_features_in_ = n_features
        self.params = params
        self.classes_ = np.array([0, 1])

    @classmethod
    def from_sklearn_like(cls, est) -> "NumpyHGB":
        if getattr(est, "n_trees_per_iteration_", 1) != 1:
            raise ValueError("only binary HistGradientBoostingClassifier is supported")
        if est.is_categorical_ is not None and np.any(est.is_categorical_):
            raise ValueError("categorical features are not supported")
        trees = []
        for it in est._predictors:
            nodes = it[0].nodes
            trees.append({k: np.asarray(nodes[k]) for k in
                          ("value", "feature_idx", "num_threshold", "missing_go_to_left", "left", "right", "is_leaf")})
        params = {k: getattr(est, k) for k in ("learning_rate", "max_iter", "max_leaf_nodes", "max_depth",
                                                "min_samples_leaf", "l2_regularization", "max_bins")
                  if hasattr(est, k)}
        return cls(float(np.ravel(est._baseline_prediction)[0]), trees, int(est.n_features_in_), params)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected (n, {self.n_features_in_}) features, got {X.shape}")
        raw = np.full(len(X), self.baseline)
        rows = np.arange(len(X))
        for t in self.trees:
            node = np.zeros(len(X), dtype=np.int64)
            active = ~t["is_leaf"][node].astype(bool)
            while active.any():
                i = rows[active]; n = node[i]
                x = X[i, t["feature_idx"][n]]
                nan = np.isnan(x)
                left = np.where(nan, t["missing_go_to_left"][n].astype(bool), x <= t["num_threshold"][n])
                node[i] = np.where(left, t["left"][n], t["right"][n])
                active[i] = ~t["is_leaf"][node[i]].astype(bool)
            raw += t["value"][node]
        return raw

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self.decision_function(X)
        p = np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))
        return np.column_stack([1 - p, p])


def _load_bundle_without_sklearn(path: Path):
    """Unpickle the joblib bundle, replacing sklearn classes by attribute holders (numpy arrays load as usual)."""
    import joblib
    from joblib.numpy_pickle import NumpyUnpickler

    class _Holder:
        def __setstate__(self, state):
            self.__dict__.update(state if isinstance(state, dict) else {"_state": state})

    original = NumpyUnpickler.find_class

    def find_class(self, module, name):
        if module.startswith("sklearn"):
            return type(name, (_Holder,), {})
        return original(self, module, name)

    NumpyUnpickler.find_class = find_class
    try:
        return joblib.load(path)
    finally:
        NumpyUnpickler.find_class = original


def load_hgb(path: str | Path, prefer_sklearn: bool = True):
    """Return (model, feature_names, backend). Uses sklearn if it imports cleanly, else the numpy evaluator."""
    path = Path(path)
    if prefer_sklearn:
        try:
            import joblib
            from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: F401  (fails if DLLs are blocked)
            bundle = joblib.load(path)
            model = bundle["model"] if isinstance(bundle, dict) else bundle
            names = bundle.get("feature_names") if isinstance(bundle, dict) else None
            return model, list(names) if names is not None else None, "sklearn"
        except Exception:
            pass
    bundle = _load_bundle_without_sklearn(path)
    est = bundle["model"] if isinstance(bundle, dict) else bundle
    names = bundle.get("feature_names") if isinstance(bundle, dict) else None
    return NumpyHGB.from_sklearn_like(est), list(names) if names is not None else None, "numpy"
