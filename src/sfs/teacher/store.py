"""VLM score files: (instance_id, text) -> margin, with the scorer's identity (spec-rerank-cascade D5).

One .npz per scorer configuration and request set. Texts are stored once in the metadata and
referenced by index, so no pickle is needed. EVAL-side files are kept apart from training-side
files: `load_scores(..., allow_eval=False)` refuses a file whose meta says it holds eval scores.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os

import numpy as np


def save_scores(path: str, instance_ids, texts: list[str], margins, meta: dict) -> None:
    vocab = sorted(set(texts))
    tid = {t: i for i, t in enumerate(vocab)}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, instance_id=np.asarray(instance_ids, np.int64),
                 text_id=np.array([tid[t] for t in texts], np.int32),
                 margin=np.asarray(margins, np.float32),
                 meta=np.array(json.dumps({**meta, "texts": vocab}, sort_keys=True)))
    os.replace(tmp, path)


def read_scores(path: str) -> tuple[np.ndarray, list[str], np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        texts = [meta["texts"][i] for i in z["text_id"]]
        return z["instance_id"], texts, z["margin"], meta


def scorer_id(meta: dict) -> str:
    keys = ("model_id", "kind", "view", "n_frames", "frame_side", "prompt", "store_id")
    return hashlib.sha1(json.dumps({k: meta.get(k) for k in keys}, sort_keys=True).encode()).hexdigest()[:12]


def load_scores(path: str, allow_eval: bool = False) -> tuple[dict, str]:
    """({(instance_id, text): margin}, scorer id)."""
    ids, texts, margins, meta = read_scores(path)
    if meta.get("subset") == "eval" and not allow_eval:
        raise PermissionError(f"{path} holds EVAL-split scores; training code must not read them")
    return {(int(i), t): float(m) for i, t, m in zip(ids, texts, margins)}, scorer_id(meta)


def write_requests(path: str, pairs: list[tuple[int, str]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "text"])
        w.writerows(sorted(set((int(i), t) for i, t in pairs)))


def read_requests(path: str) -> list[tuple[int, str]]:
    with open(path, encoding="utf-8") as f:
        return [(int(r["instance_id"]), r["text"]) for r in csv.DictReader(f)]
