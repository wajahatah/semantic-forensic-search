"""Cascade evaluation over cached scores (spec-rerank-cascade D8).

A system = a first-stage score (LoRA, or LoRA fused with the student) optionally followed by the
VLM re-ordering that query's top k candidates. Cost = VLM calls per query = k (capped at the
corpus size). Inside the top k the order is z(VLM) + gamma * z(first stage), z over those k
items; candidates below k keep their first-stage order. k = None means every candidate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sfs.eval.metrics import rank


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-12 else np.zeros_like(x)


def topk_rerank(ids: np.ndarray, base: np.ndarray, vlm_top: np.ndarray, k: int, gamma: float) -> np.ndarray:
    """New scores (aligned with ids): the base top-k re-ordered by z(vlm) + gamma * z(base), kept above
    the rest; the rest unchanged. `vlm_top` holds the VLM margins of the base top-k, in ranked order."""
    ids, base = np.asarray(ids), np.asarray(base, np.float64)
    if k <= 0:
        return base.copy()
    top = rank(ids, base)[:k]
    if len(vlm_top) != len(top):
        raise ValueError("vlm_top must hold one margin per top-k candidate")
    pos = {int(i): n for n, i in enumerate(ids)}
    rows = np.array([pos[int(i)] for i in top])
    fused = _z(vlm_top) + gamma * _z(base[rows])
    out = base.copy()
    out[rows] = base.max() + 1.0 + (fused - fused.min())
    return out


@dataclass(frozen=True)
class System:
    name: str
    first: str                 # key into the first-stage score matrices
    k: int | None = 0          # VLM calls per query; None = every candidate
    gamma: float = 0.0         # weight of the first-stage score inside the re-ranked top k


def top_needed(queries, inst_ids: np.ndarray, first: np.ndarray, text_col: dict, k: int) -> set[tuple[int, str]]:
    """(instance, text) pairs the VLM must score so that `first` + VLM@k can be evaluated."""
    pos = {int(i): n for n, i in enumerate(inst_ids)}
    need = set()
    for q in queries:
        ids = np.array(sorted(q.corpus), np.int64)
        col = text_col[(q.activity, q.phrasing_idx)]
        top = ids if k is None else rank(ids, first[[pos[i] for i in ids], col])[:k]
        need |= {(int(i), q.text) for i in top}
    return need


def evaluate_systems(queries, inst_ids: np.ndarray, firsts: dict[str, np.ndarray], text_col: dict,
                     vlm: dict, systems: list[System]) -> pd.DataFrame:
    """One row per (query, system): the metrics of sfs.eval.baseline.score_query plus `cost`."""
    from sfs.eval.baseline import score_query
    pos = {int(i): n for n, i in enumerate(inst_ids)}
    rows = []
    for q in queries:
        ids = np.array(sorted(q.corpus), np.int64)
        col = text_col[(q.activity, q.phrasing_idx)]
        relevant = np.fromiter(q.relevant, np.int64)
        for s in systems:
            base = firsts[s.first][[pos[i] for i in ids], col]
            k = len(ids) if s.k is None else min(s.k, len(ids))
            if k:
                top = rank(ids, base)[:k]
                try:
                    v = np.array([vlm[(int(i), q.text)] for i in top])
                except KeyError as e:
                    raise KeyError(f"no VLM score for {e.args[0]} (system {s.name}); score it first") from None
                scores = topk_rerank(ids, base, v, k, s.gamma)
            else:
                scores = base
            rel = np.isin(rank(ids, scores), relevant)
            rows.append({"qid": q.qid, "kind": q.kind, "activity": q.activity, "phrasing_idx": q.phrasing_idx,
                         "system": s.name, "cost": k, "n_corpus": len(ids), **score_query(rel)})
    return pd.DataFrame(rows)
