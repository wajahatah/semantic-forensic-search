"""Per-instance frame sequences for the student re-ranker (spec-rerank-cascade D6).

Each sampled frame (<= 16, time order) contributes:
- the LoRA-adapted tight embedding (768),
- the frozen SigLIP 2 context embedding (768),
- BOX_FEATURES box-motion features from the MEVA geometry, scale-invariant (normalised by the
  instance's own median box height; the crop store does not record frame size, and relative
  motion transfers across cameras better than absolute pixel positions).

Built once per LoRA run and cached as one .npz: feats (N, T_MAX, F) float16, mask (N, T_MAX).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

from sfs.data.kpf import Tracks, parse_geom
from sfs.data.regions import VIEWS, lookup_union

T_MAX = 16
EMB = 768
BOX_FEATURES = ("dx_cum", "dy_cum", "dx_step", "dy_step", "log_h", "log_w", "log_aspect",
                "n_actors", "pair_dist", "log_side", "t_sec", "has_box")
DRIVE_ANNO = "/content/drive/MyDrive/DL-Project/meva/annotations/meva-data-repo/"


def local_geom_path(anno_file: str, anno_root: str) -> str:
    rel = anno_file[len(DRIVE_ANNO):] if anno_file.startswith(DRIVE_ANNO) else anno_file
    return os.path.join(anno_root, *rel.split("/")).replace(".activities.yml", ".geom.yml")


def parse_actors(actor_ids: str) -> tuple[int, ...]:
    return tuple(int(a) for a in str(actor_ids).split(";") if a.strip())


def _centre(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def box_features(tracks: Tracks, actors: tuple[int, ...], frames: list[int], sides: list[int],
                 fps: float, window: int = 5) -> np.ndarray:
    """(len(frames), len(BOX_FEATURES)) float32. Missing boxes -> zeros with has_box = 0."""
    boxes = [lookup_union(tracks, actors, f, window) for f in frames]
    out = np.zeros((len(frames), len(BOX_FEATURES)), np.float32)
    heights = [b[3] - b[1] for b in boxes if b is not None and b[3] > b[1]]
    if not heights:
        out[:, BOX_FEATURES.index("t_sec")] = [(f - frames[0]) / fps / 8.0 for f in frames]
        return out
    h_ref = float(np.median(heights))
    first = next(b for b in boxes if b is not None)
    c0 = _centre(first)
    prev = c0
    for t, (f, b) in enumerate(zip(frames, boxes)):
        row = out[t]
        row[BOX_FEATURES.index("t_sec")] = (f - frames[0]) / fps / 8.0
        row[BOX_FEATURES.index("log_side")] = math.log(max(sides[t], 1) / 224.0)
        per_actor = [tracks[a].get(f) for a in actors if a in tracks]
        per_actor = [p for p in per_actor if p is not None]
        row[BOX_FEATURES.index("n_actors")] = min(len(per_actor), 4) / 4.0
        if len(per_actor) >= 2:
            (ax, ay), (bx, by) = _centre(per_actor[0]), _centre(per_actor[1])
            row[BOX_FEATURES.index("pair_dist")] = math.hypot(ax - bx, ay - by) / h_ref
        if b is None or b[3] <= b[1] or b[2] <= b[0]:
            continue
        c = _centre(b)
        w, h = b[2] - b[0], b[3] - b[1]
        row[BOX_FEATURES.index("dx_cum")] = (c[0] - c0[0]) / h_ref
        row[BOX_FEATURES.index("dy_cum")] = (c[1] - c0[1]) / h_ref
        row[BOX_FEATURES.index("dx_step")] = (c[0] - prev[0]) / h_ref
        row[BOX_FEATURES.index("dy_step")] = (c[1] - prev[1]) / h_ref
        row[BOX_FEATURES.index("log_h")] = math.log(h / h_ref)
        row[BOX_FEATURES.index("log_w")] = math.log(w / h_ref)
        row[BOX_FEATURES.index("log_aspect")] = math.log(w / h)
        row[BOX_FEATURES.index("has_box")] = 1.0
        prev = c
    return np.clip(out, -5.0, 5.0)


def _rows_by_instance(shard: dict, view: str) -> dict[int, np.ndarray]:
    sel = np.flatnonzero(shard["view"] == VIEWS.index(view))
    inst = shard["instance_id"][sel]
    order = np.lexsort((shard["frame_idx"][sel], inst))
    sel, inst = sel[order], inst[order]
    starts = np.flatnonzero(np.r_[True, inst[1:] != inst[:-1]])
    ends = np.r_[starts[1:], len(inst)]
    return {int(inst[s]): sel[s:e] for s, e in zip(starts, ends)}


def build_features(manifest, lora: dict, frozen: dict, anno_root: str, fps: float = 30.0) -> dict:
    """lora: re-embedded tight shard of one LoRA run; frozen: baseline SigLIP 2 shard (context view)."""
    tight = _rows_by_instance(lora, "tight")
    ctx = _rows_by_instance(frozen, "context")
    ids = np.array(sorted(int(i) for i in manifest.instance_id), np.int64)
    feats = np.zeros((len(ids), T_MAX, 2 * EMB + len(BOX_FEATURES)), np.float16)
    mask = np.zeros((len(ids), T_MAX), bool)
    m = manifest.set_index(manifest.instance_id.astype(np.int64))
    geoms: dict[str, Tracks] = {}
    for n, iid in enumerate(ids):
        rt, rc = tight.get(int(iid)), ctx.get(int(iid))
        if rt is None or rc is None:
            raise ValueError(f"instance {iid} lacks tight or context embeddings")
        ft, fc = lora["frame_idx"][rt], frozen["frame_idx"][rc]
        if not np.array_equal(ft, fc):
            raise ValueError(f"instance {iid}: tight and context frames differ")
        rt, rc, frames = rt[:T_MAX], rc[:T_MAX], [int(f) for f in ft[:T_MAX]]
        row = m.loc[iid]
        gp = local_geom_path(row.anno_file, anno_root)
        if gp not in geoms:
            geoms[gp] = parse_geom(gp)
        boxes = box_features(geoms[gp], parse_actors(row.actor_ids), frames,
                             [int(s) for s in lora["box_side"][rt]], fps)
        k = len(frames)
        feats[n, :k, :EMB] = lora["emb"][rt]
        feats[n, :k, EMB:2 * EMB] = frozen["emb"][rc]
        feats[n, :k, 2 * EMB:] = boxes
        mask[n, :k] = True
    return {"instance_id": ids, "feats": feats, "mask": mask}


def save_features(path: str, data: dict, meta: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, meta=np.array(json.dumps(meta, sort_keys=True)), **data)
    os.replace(tmp, path)


def load_features(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "meta"}
        out["meta"] = json.loads(str(z["meta"]))
    return out


def main(argv=None) -> int:
    from sfs.data.manifest import read_manifest
    from sfs.store.shards import read_shard

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lora", required=True, help="re-embedded tight embeddings of one LoRA run")
    ap.add_argument("--frozen", required=True, help="baseline SigLIP 2 embeddings (all views)")
    ap.add_argument("--anno-root", default="data/meva/annotations_local")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    lora, frozen = read_shard(a.lora), read_shard(a.frozen)
    data = build_features(read_manifest(a.manifest), lora, frozen, a.anno_root)
    meta = {"lora_encoder_id": lora["meta"]["encoder_id"], "frozen_encoder_id": frozen["meta"]["encoder_id"],
            "box_features": list(BOX_FEATURES), "t_max": T_MAX}
    save_features(a.out, data, meta)
    print(f"wrote {a.out}: {len(data['instance_id'])} instances, "
          f"{data['mask'].sum()} frames, box coverage {data['feats'][..., -1][data['mask']].mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
