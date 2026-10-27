"""Manifest v2 (spec-offline-embedding D1, D2, §8 row 4).

One row per annotated activity instance that has matching video. Adds:
- `split`   protocol A: official MEVA split (kitware-meva-training -> train/val, kitware -> eval)
- `split_b` protocol B: unseen-camera split (eval cameras never appear in train)
Validation is held out BY CLIP, never by instance, so no footage appears on both sides.

CLI:
    python -m sfs.data.manifest --anno <meva-data-repo> --videos <video root> --out manifest.csv
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

from sfs.data.kpf import clip_stem, parse_activities, parse_stem

DEDUPE_KEY = ["clip_stem", "activity", "frame_start", "frame_end"]
VIDEO_EXTS = (".avi", ".mp4", ".mkv")
GATE_MIN_EVAL_INSTANCES = 1500
GATE_MIN_PER_ACTIVITY = 20


def read_manifest(path: str) -> pd.DataFrame:
    """Actor ids reach ~6e18; read them as text so they can never round-trip through float."""
    return pd.read_csv(path, dtype={"actor_ids": str, "clip_stem": str, "camera": str})


def split_from_path(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    if "kitware-meva-training" in parts:
        return "train"
    if "kitware" in parts:
        return "eval"
    return "other"


def index_videos(video_root: str) -> dict[str, str]:
    index: dict[str, str] = {}
    for root, _, files in os.walk(video_root):
        for name in files:
            if name.lower().endswith(VIDEO_EXTS):
                index.setdefault(clip_stem(name), os.path.join(root, name))
    return index


def iter_activity_files(anno_root: str):
    for root, _, files in os.walk(anno_root):
        for name in sorted(files):
            if name.endswith(".activities.yml"):
                yield os.path.join(root, name)


def build_manifest(anno_root: str, videos: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    rows, total, files, unparsable = [], 0, 0, 0
    for path in iter_activity_files(anno_root):
        files += 1
        stem = clip_stem(path)
        try:
            key = parse_stem(stem)
            acts = parse_activities(path)
        except Exception:
            unparsable += 1
            continue
        total += len(acts)
        video = videos.get(stem)
        if not video:
            continue
        split = split_from_path(path)
        for a in acts:
            rows.append({
                "clip_stem": stem, "video_path": video, "anno_file": path, "split": split,
                "activity": a.activity, "frame_start": a.frame_start, "frame_end": a.frame_end,
                "actor_ids": ";".join(map(str, a.actor_ids)),
                "date": key.date, "hour": key.hour, "site": key.site, "camera": key.camera,
            })
    cols = ["clip_stem", "video_path", "anno_file", "split", "activity", "frame_start",
            "frame_end", "actor_ids", "date", "hour", "site", "camera"]
    df = pd.DataFrame(rows, columns=cols)
    before = len(df)
    df = df.drop_duplicates(DEDUPE_KEY).sort_values(["clip_stem", "frame_start", "activity"])
    df = df.reset_index(drop=True)
    df.insert(0, "instance_id", np.arange(len(df), dtype=np.int64))
    stats = {"activity_files": files, "unparsable_files": unparsable,
             "instances_in_annotations": total, "instances_with_video": before,
             "duplicates_dropped": before - len(df)}
    return df, stats


def check_leakage(df: pd.DataFrame) -> None:
    train = set(df.loc[df.split.isin(["train", "val"]), "clip_stem"])
    evals = set(df.loc[df.split == "eval", "clip_stem"])
    both = sorted(train & evals)
    if both:
        raise ValueError(f"{len(both)} clips appear in both train and eval, e.g. {both[:3]}")


def _holdout_clips(stems: list[str], frac: float, seed: int) -> set[str]:
    stems = sorted(set(stems))
    if not stems or frac <= 0:
        return set()
    order = np.random.default_rng(seed).permutation(len(stems))
    k = max(1, math.ceil(frac * len(stems)))
    return {stems[i] for i in order[:k]}


def assign_validation(df: pd.DataFrame, frac: float = 0.1, seed: int = 0) -> pd.DataFrame:
    df = df.copy()
    val = _holdout_clips(df.loc[df.split == "train", "clip_stem"].tolist(), frac, seed)
    df.loc[df.clip_stem.isin(val), "split"] = "val"
    return df


def assign_protocol_b(df: pd.DataFrame, eval_frac: float = 0.2, val_frac: float = 0.1,
                      seed: int = 0) -> pd.DataFrame:
    """Hold out whole cameras until >= eval_frac of instances are covered."""
    df = df.copy()
    counts = df.groupby("camera").size()
    cams = sorted(counts.index)
    target = eval_frac * len(df)
    eval_cams, covered = set(), 0
    for i in np.random.default_rng(seed).permutation(len(cams)):
        if covered >= target:
            break
        eval_cams.add(cams[i])
        covered += int(counts[cams[i]])
    df["split_b"] = np.where(df.camera.isin(eval_cams), "eval", "train")
    val = _holdout_clips(df.loc[df.split_b == "train", "clip_stem"].tolist(), val_frac, seed)
    df.loc[df.clip_stem.isin(val), "split_b"] = "val"
    return df


def camera_overlap(df: pd.DataFrame, col: str = "split") -> set[str]:
    train = set(df.loc[df[col].isin(["train", "val"]), "camera"])
    return train & set(df.loc[df[col] == "eval", "camera"])


def gate_warnings(df: pd.DataFrame) -> list[str]:
    ev = df[df.split == "eval"]
    warns = []
    if len(ev) < GATE_MIN_EVAL_INSTANCES:
        warns.append(f"eval has {len(ev)} instances (< {GATE_MIN_EVAL_INSTANCES}): "
                     "consider a targeted download of eval clips")
    thin = ev.activity.value_counts()
    thin = thin[thin < GATE_MIN_PER_ACTIVITY]
    if len(thin):
        warns.append(f"{len(thin)} activity types have < {GATE_MIN_PER_ACTIVITY} eval instances: "
                     + ", ".join(f"{a}={n}" for a, n in thin.items()))
    return warns


def summarize(df: pd.DataFrame, stats: dict) -> str:
    lines = [f"{k}: {v:,}" for k, v in stats.items()]
    for col in ("split", "split_b"):
        if col in df:
            g = df.groupby(col).agg(instances=("instance_id", "size"),
                                    clips=("clip_stem", "nunique"),
                                    cameras=("camera", "nunique"))
            lines += [f"\n[{col}]", g.to_string()]
            lines.append(f"cameras shared train/eval: {sorted(camera_overlap(df, col))}")
    lines.append("\n[activity x split]")
    ct = pd.crosstab(df.activity, df.split)
    lines.append(ct.loc[ct.sum(axis=1).sort_values(ascending=False).index].to_string())
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--anno", required=True, help="meva-data-repo root (or any subfolder)")
    ap.add_argument("--videos", required=True, help="root folder searched recursively for videos")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--b-eval-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    df, stats = build_manifest(a.anno, index_videos(a.videos))
    if df.empty:
        print("NO ROWS: no annotation file matched a video. Check --anno/--videos paths.")
        return 1
    check_leakage(df)
    df = assign_validation(df, a.val_frac, a.seed)
    df = assign_protocol_b(df, a.b_eval_frac, a.val_frac, a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(summarize(df, stats))
    for w in gate_warnings(df):
        print("WARNING:", w)
    print(f"\nmanifest written: {a.out} ({len(df):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
