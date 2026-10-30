"""Validation-only checks after LoRA re-embedding (docs/results/03-lora.md).

Usage (repo root): python scripts/lora_val_check.py A-middle A-uniform B-uniform B-middle


1. Consistency: Q1 val mAP recomputed from the re-embedded vectors must match the val mAP the
   trainer recorded for the best epoch (same model, same queries) -> proves the re-embedded store
   is the selected model.
2. Frozen per-activity val AP (baseline vectors), for H4 and the val curve context.
Writes runs/lora/val_check.json.
"""
import json
import sys

import numpy as np
import pandas as pd
import torch

from sfs.data.manifest import read_manifest
from sfs.embed.encoder import Encoder
from sfs.eval.baseline import evaluate
from sfs.eval.queries import build_q1, load_phrasings
from sfs.eval.retrieval import InstanceIndex
from sfs.store.shards import read_shard

COLS = {"A": "split", "B": "split_b"}
m = read_manifest("manifest_v2_drive/manifest_v2.csv")
phr = load_phrasings("queries/phrasings.csv")
enc = Encoder("siglip2")
texts = [(a, j, t) for a in sorted(phr) for j, t in enumerate(phr[a])]
T_all = enc.embed_texts([t for _, _, t in texts])


def val_scores(shard, proto):
    col = COLS[proto]
    val = m[m[col] == "val"]
    counts = val.activity.value_counts()
    acts = sorted(a for a in phr if counts.get(a, 0) >= 5)
    qs = build_q1(m, col, acts, phr, subset="val")
    sel = [i for i, (a, _, _) in enumerate(texts) if a in acts]
    text_col = {(texts[i][0], texts[i][1]): c for c, i in enumerate(sel)}
    idx = InstanceIndex.from_shard(shard, "tight", set(val.instance_id.astype(int)))
    df = evaluate(qs, idx.instance_ids, idx.score(T_all[sel], 3), text_col)
    per = df.groupby("activity").ap.mean()
    return float(per.mean()), per.round(4).to_dict()


out = {"frozen": {}, "runs": {}}
base = read_shard("manifest_v2_drive/embeddings_v2_siglip2.npz")
for proto in "AB":
    out["frozen"][proto] = dict(zip(("val_map", "per_activity"), val_scores(base, proto)))
del base
for run in sys.argv[1:]:
    proto = run[0]
    best = torch.load(f"runs/lora/{run}/best/adapter.pt", map_location="cpu", weights_only=False)
    vm, per = val_scores(read_shard(f"data/embeddings_lora/{run}/embeddings.npz"), proto)
    out["runs"][run] = {"recorded_val_map": best["val_map"], "recomputed_val_map": vm,
                        "abs_diff": abs(vm - best["val_map"]), "best_epoch": best["epoch"],
                        "recorded_per_activity": best["per_activity"], "recomputed_per_activity": per}
    print(f"{run}: recorded {best['val_map']:.4f} recomputed {vm:.4f}")
for p, v in out["frozen"].items():
    print(f"frozen {p}: val mAP {v['val_map']:.4f}")
json.dump(out, open("runs/lora/val_check.json", "w"), indent=1)
