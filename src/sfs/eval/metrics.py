"""Retrieval metrics over a ranked list of relevance flags (spec-crops-queries-baseline §5).

Input everywhere: `rel`, a boolean array in ranked order (rel[0] = top result). It must cover the
whole searched corpus, so the total number of relevant items is `rel.sum()`.
"""
from __future__ import annotations

import numpy as np


def rank(ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """ids ordered by descending score; ties broken by ascending id so results are deterministic."""
    order = np.lexsort((ids, -scores))
    return ids[order]


def average_precision(rel: np.ndarray) -> float:
    rel = np.asarray(rel, bool)
    n_rel = rel.sum()
    if n_rel == 0:
        return float("nan")
    hits = np.cumsum(rel)
    return float((hits[rel] / (np.flatnonzero(rel) + 1)).sum() / n_rel)


def precision_at_k(rel: np.ndarray, k: int) -> float:
    rel = np.asarray(rel, bool)
    return float(rel[:k].sum() / k)


def recall_at_k(rel: np.ndarray, k: int) -> float:
    rel = np.asarray(rel, bool)
    n_rel = rel.sum()
    return float("nan") if n_rel == 0 else float(rel[:k].sum() / n_rel)


def r_precision(rel: np.ndarray) -> float:
    """Precision at R, where R = number of relevant items (equals recall at R)."""
    rel = np.asarray(rel, bool)
    n_rel = int(rel.sum())
    return float("nan") if n_rel == 0 else float(rel[:n_rel].sum() / n_rel)


def reciprocal_rank(rel: np.ndarray) -> float:
    hits = np.flatnonzero(np.asarray(rel, bool))
    return 0.0 if len(hits) == 0 else float(1.0 / (hits[0] + 1))
