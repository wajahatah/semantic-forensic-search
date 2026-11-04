"""Frames for the VLM: an instance's tight or context crops from the crop store, in time order
(spec-rerank-cascade D2)."""
from __future__ import annotations

import glob
import os
from collections import OrderedDict

import numpy as np

from sfs.data.regions import VIEWS
from sfs.store.crops import CropShard, read_crop_rows

N_FRAMES = 8


def pick_even(n: int, k: int = N_FRAMES) -> list[int]:
    """k indices spread evenly over range(n), first and last included; all of them if n <= k."""
    if n <= 0:
        raise ValueError("no frames")
    if n <= k:
        return list(range(n))
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


class CropIndex:
    """instance_id -> (shard, row numbers per view, sorted by frame). Reads row arrays only;
    image bytes are loaded one shard at a time (shards are per video, so jobs should visit
    instances grouped by clip)."""

    def __init__(self, crops_dir: str, cache_shards: int = 2):
        self.paths = sorted(glob.glob(os.path.join(crops_dir, "*.npz")))
        if not self.paths:
            raise FileNotFoundError(f"no crop shards in {crops_dir}")
        self.where: dict[int, tuple[int, dict[str, np.ndarray]]] = {}
        self.fps: list[float] = []
        store_ids = set()
        for s, path in enumerate(self.paths):
            r = read_crop_rows(path)
            store_ids.add(r["meta"]["store_id"])
            self.fps.append(float(r["meta"]["fps"]))
            for inst in np.unique(r["instance_id"]):
                sel = r["instance_id"] == inst
                per_view = {}
                for vi, view in enumerate(VIEWS):
                    rows = np.flatnonzero(sel & (r["view"] == vi))
                    if len(rows):
                        per_view[view] = rows[np.argsort(r["frame_idx"][rows], kind="stable")]
                self.where[int(inst)] = (s, per_view)
        if len(store_ids) != 1:
            raise ValueError(f"crop shards come from different stores: {sorted(store_ids)}")
        self.store_id = store_ids.pop()
        self._cache: OrderedDict[int, CropShard] = OrderedDict()
        self._cache_n = cache_shards

    def _shard(self, s: int) -> CropShard:
        if s in self._cache:
            self._cache.move_to_end(s)
        else:
            self._cache[s] = CropShard(self.paths[s])
            if len(self._cache) > self._cache_n:
                self._cache.popitem(last=False)
        return self._cache[s]

    def shard_of(self, instance_id: int) -> int:
        return self.where[int(instance_id)][0]

    def fps_of(self, instance_id: int) -> float:
        return self.fps[self.shard_of(instance_id)]

    def n_frames(self, instance_id: int, view: str) -> int:
        return len(self.where[int(instance_id)][1].get(view, ()))

    def frames(self, instance_id: int, view: str, k: int = N_FRAMES) -> tuple[list[np.ndarray], list[int]]:
        """(RGB uint8 frames, their video frame numbers), evenly picked, in time order."""
        s, per_view = self.where[int(instance_id)]
        rows = per_view.get(view)
        if rows is None or not len(rows):
            raise KeyError(f"instance {instance_id} has no {view} crops")
        shard = self._shard(s)
        picked = rows[pick_even(len(rows), k)]
        return [shard.image(int(r)) for r in picked], [int(shard.frame_idx[r]) for r in picked]
