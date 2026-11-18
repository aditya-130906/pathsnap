"""Candidate- and observation-level metrics (numpy only, so they can be unit-tested anywhere)."""

from __future__ import annotations

import numpy as np


def candidate_metrics(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Binary candidate classification metrics. Positive class = "this candidate is the reference arc"."""
    y = np.asarray(y).astype(int)
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-7, 1 - 1e-7)
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())

    def f1(p, r):
        return 2 * p * r / (p + r) if p + r else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    neg_precision = tn / (tn + fn) if tn + fn else 0.0
    neg_recall = tn / (tn + fp) if tn + fp else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1(precision, recall),
        "macro_f1": (f1(precision, recall) + f1(neg_precision, neg_recall)) / 2,
        "candidate_accuracy": (tp + tn) / max(len(y), 1),
        "log_loss": float(-np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob))),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def observation_top1(obs_key: np.ndarray, y: np.ndarray, prob: np.ndarray, n_valid_observations: int) -> dict:
    """Top-1 accuracy: pick the highest-probability candidate per observation and check it.

    `obs_key` identifies the observation of each candidate row (e.g. trip/observation encoded as one int).
    `n_valid_observations` counts every validly labelled observation, including ones whose reference arc is
    missing from their candidate set, which therefore count as wrong (candidate misses are not hidden).
    """
    obs_key = np.asarray(obs_key); y = np.asarray(y).astype(int); prob = np.asarray(prob, dtype=np.float64)
    order = np.lexsort((-prob, obs_key))
    first = np.r_[True, obs_key[order][1:] != obs_key[order][:-1]]
    best = order[first]
    uniq, inv = np.unique(obs_key, return_inverse=True)
    ref_present = np.zeros(len(uniq), bool)
    np.logical_or.at(ref_present, inv, y == 1)
    correct = int(y[best].sum())
    present = int(ref_present.sum())
    return {
        "top1_accuracy": correct / max(n_valid_observations, 1),
        "top1_given_reference_in_candidates": correct / max(present, 1),
        "candidate_miss_rate": 1 - present / max(n_valid_observations, 1),
        "observations_evaluated": int(n_valid_observations),
    }


def route_continuity(obs_index: np.ndarray, arc: np.ndarray, from_node: np.ndarray, to_node: np.ndarray) -> float:
    """Share of consecutive top-1 predictions (one trip, sorted by observation) that stay on the same arc or
    move to an arc that starts where the previous one ends."""
    order = np.argsort(obs_index)
    a, f, t = np.asarray(arc)[order], np.asarray(from_node)[order], np.asarray(to_node)[order]
    if len(a) < 2:
        return 1.0
    ok = (a[1:] == a[:-1]) | (f[1:] == t[:-1])
    return float(ok.mean())
