"""Instance-level scoring from frame embeddings (spec-retrieval-system §5.2).

An instance's score for a query = mean of its top-m frame similarities. Mean-of-top-m rather than
max limits the bias towards instances that simply have more frames.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sfs.data.regions import VIEWS


@dataclass
class InstanceIndex:
    instance_ids: np.ndarray   # (n_inst,) sorted unique
    emb: np.ndarray            # (n_rows, dim) float32, rows grouped by instance
    starts: np.ndarray         # (n_inst + 1,) row offsets per instance

    @classmethod
    def from_shard(cls, shard: dict, view: str, instance_subset=None) -> "InstanceIndex":
        sel = shard["view"] == VIEWS.index(view)
        if instance_subset is not None:
            sel &= np.isin(shard["instance_id"], np.fromiter(instance_subset, np.int64))
        inst = shard["instance_id"][sel].astype(np.int64)
        order = np.argsort(inst, kind="stable")
        inst = inst[order]
        emb = shard["emb"][sel][order].astype(np.float32)
        ids, first = np.unique(inst, return_index=True)
        return cls(ids, emb, np.append(first, len(inst)))

    def score(self, queries: np.ndarray, m: int = 3) -> np.ndarray:
        """(n_inst, n_queries) instance scores for L2-normalised query vectors."""
        sims = self.emb @ np.asarray(queries, np.float32).T
        out = np.empty((len(self.instance_ids), sims.shape[1]), np.float32)
        for i in range(len(self.instance_ids)):
            block = sims[self.starts[i]:self.starts[i + 1]]
            k = min(m, len(block))
            out[i] = np.sort(block, axis=0)[-k:].mean(axis=0)
        return out
