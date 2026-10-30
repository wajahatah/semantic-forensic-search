"""Embedding shards: one ``.npz`` per (encoder, video). Atomic writes; validated reads.

Arrays per row: emb (float16, L2-normalised), instance_id, frame_idx, view (index into VIEWS),
fallback (crop fell back to full frame), box_side (actor box size in px). Plus a JSON ``meta``.
"""
from __future__ import annotations

import json
import os

import numpy as np

ARRAYS = ("emb", "instance_id", "frame_idx", "view", "fallback", "box_side")
_DTYPES = {"emb": np.float16, "instance_id": np.int32, "frame_idx": np.int32,
           "view": np.uint8, "fallback": np.bool_, "box_side": np.int32}


def write_shard(path: str, meta: dict, **arrays) -> None:
    missing = set(ARRAYS) - set(arrays)
    if missing:
        raise ValueError(f"missing arrays: {sorted(missing)}")
    n = len(arrays["emb"])
    if any(len(arrays[k]) != n for k in ARRAYS):
        raise ValueError("array lengths differ")
    if "encoder_id" not in meta:
        raise ValueError("meta must contain encoder_id")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, meta=np.array(json.dumps(meta, sort_keys=True)),
                 **{k: np.asarray(arrays[k], dtype=_DTYPES[k]) for k in ARRAYS})
    os.replace(tmp, path)


def read_shard(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out["meta"] = json.loads(str(out["meta"]))
    return out


def shard_is_valid(path: str, encoder_id: str | None = None) -> bool:
    if not os.path.exists(path):
        return False
    try:
        d = read_shard(path)
    except Exception:
        return False
    n = len(d.get("emb", []))
    if any(k not in d or len(d[k]) != n for k in ARRAYS):
        return False
    if n and not np.isfinite(d["emb"].astype(np.float32)).all():
        return False
    return encoder_id is None or d["meta"].get("encoder_id") == encoder_id


def consolidate(paths: list[str], out_path: str) -> dict:
    """Concatenate shards from ONE encoder. Refuses to mix encoder ids."""
    if not paths:
        raise ValueError("no shards to consolidate")
    shards = [read_shard(p) for p in paths]
    ids = {s["meta"]["encoder_id"] for s in shards}
    if len(ids) != 1:
        raise ValueError(f"shards come from different encoders: {sorted(ids)}")
    meta = {"encoder_id": ids.pop(), "encoder": shards[0]["meta"].get("encoder"),
            "n_shards": len(shards), "clip_stems": [s["meta"].get("clip_stem") for s in shards]}
    write_shard(out_path, meta, **{k: np.concatenate([s[k] for s in shards]) for k in ARRAYS})
    return meta
