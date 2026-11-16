"""Demo bundle (spec-search-demo §4.2): everything the search engine needs for one protocol's eval
corpus, built once from the stores the evaluation used, so the demo searches exactly what was scored.

    <out>/meta.sqlite    table clips: instance id, annotated activity, site, camera, date, hour,
                         frame range, video stem, crop shard file
    <out>/vectors.npz    instance_ids (sorted) + LoRA and frozen tight frame embeddings with row
                         offsets (InstanceIndex layout) + the student features of those instances
    <out>/student.pt     copy of the selected student checkpoint
    <out>/bundle.json    sources with sha256, fusion weights, model settings

    python -m sfs.demo.index --protocol A --out data/demo/A
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import time

import numpy as np

PROTOCOLS = {"A": {"split_col": "split", "lora_run": "A-middle"},
             "B": {"split_col": "split_b", "lora_run": "B-uniform"}}
CLIP_COLUMNS = ("instance_id", "activity", "site", "camera", "date", "hour", "frame_start", "frame_end",
                "clip_stem", "crop_shard")
VERSION = 1


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _index_arrays(shard: dict, corpus: set, prefix: str) -> dict:
    from sfs.eval.retrieval import InstanceIndex
    idx = InstanceIndex.from_shard(shard, "tight", corpus)
    if len(idx.instance_ids) != len(corpus):
        raise ValueError(f"{prefix}: {len(corpus) - len(idx.instance_ids)} corpus instances lack embeddings")
    # stored as float16 exactly as in the shard; InstanceIndex converts to float32 the same way on load
    return {"instance_ids": idx.instance_ids, f"{prefix}_emb": idx.emb.astype(np.float16),
            f"{prefix}_starts": idx.starts}


def crop_shard_names(crops_dir: str, corpus: set) -> dict[int, str]:
    """instance id -> crop shard file name (reads row arrays only)."""
    import glob

    from sfs.store.crops import read_crop_rows
    out = {}
    for path in sorted(glob.glob(os.path.join(crops_dir, "*.npz"))):
        r = read_crop_rows(path)
        for i in np.unique(r["instance_id"]):
            if int(i) in corpus:
                out[int(i)] = os.path.basename(path)
    missing = corpus - set(out)
    if missing:
        raise ValueError(f"{len(missing)} corpus instances have no crops, e.g. {sorted(missing)[:3]}")
    return out


def build(protocol: str, manifest, frozen_path: str, lora_path: str, features_path: str, student_dir: str,
          crops_dir: str, choices_path: str, vlm_kind: str, out_dir: str, log=print) -> dict:
    import torch

    from sfs.rerank.features import load_features
    from sfs.store.shards import read_shard
    cfg = PROTOCOLS[protocol]
    sub = manifest[manifest[cfg["split_col"]] == "eval"].copy()
    corpus = set(sub.instance_id.astype(int))
    log(f"[{protocol}] corpus: {len(corpus):,} instances, {sub.clip_stem.nunique()} videos")

    lora = _index_arrays(read_shard(lora_path), corpus, "lora")
    frozen = _index_arrays(read_shard(frozen_path), corpus, "frozen")
    if not np.array_equal(lora["instance_ids"], frozen["instance_ids"]):
        raise ValueError("LoRA and frozen stores disagree on the corpus")
    ids = lora["instance_ids"]
    feats = load_features(features_path)
    pos = {int(i): r for r, i in enumerate(feats["instance_id"])}
    if any(int(i) not in pos for i in ids):
        raise ValueError("some corpus instances lack student features")
    rows = np.array([pos[int(i)] for i in ids])

    selection = json.load(open(os.path.join(student_dir, "selection.json")))[protocol]
    student_run = selection["selected"]
    best_path = os.path.join(student_dir, student_run, "best.pt")
    beta = float(torch.load(best_path, map_location="cpu", weights_only=False)["beta"])
    choices = json.load(open(choices_path))["choices"][f"{protocol}/{vlm_kind}"]

    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "vectors.npz.tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **lora, **{k: v for k, v in frozen.items() if k != "instance_ids"},
                 feats=feats["feats"][rows], mask=feats["mask"][rows])
    os.replace(tmp, os.path.join(out_dir, "vectors.npz"))
    shutil.copyfile(best_path, os.path.join(out_dir, "student.pt"))

    shards = crop_shard_names(crops_dir, corpus)
    sub["instance_id"] = sub.instance_id.astype(int)
    sub["crop_shard"] = sub.instance_id.map(shards)
    db = os.path.join(out_dir, "meta.sqlite")
    if os.path.exists(db):
        os.remove(db)
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE clips (instance_id INTEGER PRIMARY KEY, activity TEXT, site TEXT, camera TEXT,"
                    " date TEXT, hour INTEGER, frame_start INTEGER, frame_end INTEGER, clip_stem TEXT,"
                    " crop_shard TEXT)")
        con.executemany(f"INSERT INTO clips VALUES ({','.join('?' * len(CLIP_COLUMNS))})",
                        [tuple(_plain(r[c]) for c in CLIP_COLUMNS)
                         for r in sub.sort_values("instance_id").to_dict("records")])
        for c in ("site", "camera", "date", "hour"):
            con.execute(f"CREATE INDEX idx_{c} ON clips({c})")

    info = {"version": VERSION, "protocol": protocol, "split_col": cfg["split_col"], "subset": "eval",
            "n_clips": int(len(ids)), "lora_run": cfg["lora_run"], "student_run": student_run, "beta": beta,
            "vlm": {"kind": vlm_kind, "config": choices["config"], "gamma": float(choices["gamma_student"]),
                    "view": choices["config"].split("-", 1)[1]},
            "encoder": "siglip2", "lora_encoder_id": read_shard(lora_path)["meta"]["encoder_id"],
            "sources": {k: {"path": p, "sha256": file_sha(p)} for k, p in
                        (("frozen", frozen_path), ("lora", lora_path), ("features", features_path),
                         ("student", best_path), ("choices", choices_path))},
            "files": {n: file_sha(os.path.join(out_dir, n)) for n in ("vectors.npz", "student.pt")},
            "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out_dir, "bundle.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    log(f"[{protocol}] bundle written to {out_dir}: {len(ids):,} clips, student {student_run} (beta {beta}), "
        f"VLM {choices['config']} gamma {info['vlm']['gamma']}")
    return info


def _plain(v):
    return v.item() if isinstance(v, np.generic) else v


def main(argv=None) -> int:
    from sfs.data.manifest import read_manifest
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--protocol", default="A", choices=sorted(PROTOCOLS))
    ap.add_argument("--manifest", default="manifest_v2_drive/manifest_v2.csv")
    ap.add_argument("--frozen", default="manifest_v2_drive/embeddings_v2_siglip2.npz")
    ap.add_argument("--lora", default=None, help="default data/embeddings_lora/<run>/embeddings.npz")
    ap.add_argument("--features", default=None, help="default data/rerank/features_<run>.npz")
    ap.add_argument("--students", default="runs/student")
    ap.add_argument("--crops", default="data/crops_v2")
    ap.add_argument("--choices", default="runs/cascade/04_cascade.json", help="Result 04 report (gamma)")
    ap.add_argument("--vlm-kind", default="instruct")
    ap.add_argument("--out", default=None, help="default data/demo/<protocol>")
    a = ap.parse_args(argv)
    run = PROTOCOLS[a.protocol]["lora_run"]
    build(a.protocol, read_manifest(a.manifest), a.frozen,
          a.lora or f"data/embeddings_lora/{run}/embeddings.npz", a.features or f"data/rerank/features_{run}.npz",
          a.students, a.crops, a.choices, a.vlm_kind, a.out or f"data/demo/{a.protocol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
