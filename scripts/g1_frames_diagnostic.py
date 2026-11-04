"""E3 diagnostic (spec-vlm-followups, Result 05): inside the G1 top-50 validation pools, does scoring
with 16 frames separate correct from wrong clips better than 8 frames?

Clips are split by how many frames the crop store holds for them (tight view). A clip with <= 8 stored
frames gets the same frames at 8 and 16, so only the "long" clips can change. For each query that has
both a correct and a wrong clip in a group, the AUC of the VLM score is computed; the mean over queries
is reported per protocol, model, group and frame count. Validation only.

Usage (repo root):
  python scripts/g1_frames_diagnostic.py --out runs/cascade/g1_16f_diagnostic.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def auc(pos, neg) -> float | None:
    """P(score of a correct clip > score of a wrong clip), ties count half; None if a side is empty."""
    if not len(pos) or not len(neg):
        return None
    p, n = np.asarray(pos, dtype=float)[:, None], np.asarray(neg, dtype=float)[None, :]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def long_short_auc(queries, pool: dict, n_frames: dict, stores: dict, limit: int = 8) -> dict:
    """Mean per-query AUC for clips with > limit stored frames ("long") and the rest ("short").
    queries: objects with .text and .relevant; pool: text -> clip ids; n_frames: clip id -> stored frames;
    stores: name -> {(clip id, text): score}."""
    res = {g: {name: [] for name in stores} for g in ("long", "short")}
    for q in queries:
        for grp in res:
            sel = [i for i in pool[q.text] if (n_frames[i] > limit) == (grp == "long")]
            for name, s in stores.items():
                a = auc([s[(i, q.text)] for i in sel if i in q.relevant],
                        [s[(i, q.text)] for i in sel if i not in q.relevant])
                if a is not None:
                    res[grp][name].append(a)
    return {g: {name: {"mean_auc": round(float(np.mean(x)), 4) if x else None, "n_queries": len(x)}
                for name, x in d.items()} for g, d in res.items()}


def main() -> int:
    from sfs.data.manifest import read_manifest
    from sfs.eval.queries import load_phrasings
    from sfs.store.shards import read_shard
    from sfs.teacher.frames import CropIndex
    from sfs.teacher.gate import LORA_EMB, ValQ1, _phrasing_embeddings
    from sfs.teacher.store import load_scores

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="manifest_v2_drive/manifest_v2.csv")
    ap.add_argument("--phrasings", default="queries/phrasings.csv")
    ap.add_argument("--crops", default="data/crops_v2")
    ap.add_argument("--base", default="data/teacher/g1", help="8-frame G1 stores, <base>/<model>/scores.npz")
    ap.add_argument("--more", default="data/teacher/g1_16f", help="16-frame G1 stores")
    ap.add_argument("--models", nargs="*", default=["instruct-tight", "reranker-tight"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = read_manifest(a.manifest)
    phr = load_phrasings(a.phrasings)
    pe = _phrasing_embeddings(phr)
    crops = CropIndex(a.crops)
    vals = {p: ValQ1(m, p, phr, pe, read_shard(LORA_EMB[p])) for p in "AB"}
    out = {}
    for model in a.models:
        stores = {"8": load_scores(f"{a.base}/{model}/scores.npz")[0],
                  "16": load_scores(f"{a.more}/{model}/scores.npz")[0]}
        for p, v in vals.items():
            pool: dict[str, list[int]] = {}
            for iid, text in v.top():
                pool.setdefault(text, []).append(iid)
            nf = {i: crops.n_frames(i, "tight") for ids in pool.values() for i in ids}
            out[f"{p}/{model}"] = long_short_auc(v.queries, pool, nf, stores)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
