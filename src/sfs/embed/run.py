"""Embedding job (spec-offline-embedding D6, D7): one pass per video, resumable, one shard per encoder.

CLI:
    python -m sfs.embed.run --manifest manifest.csv --out shards/ --encoders siglip,siglip2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from sfs.data.kpf import Tracks, clip_stem, geom_path_for, parse_geom
from sfs.data.regions import (VIEWS, box_side, context_region, crop, full_region,
                              lookup_union, tight_region)
from sfs.data.sampling import sample_frames
from sfs.store.shards import shard_is_valid, write_shard


# Bump whenever a code change alters WHAT gets embedded (not just how fast). It is part of the
# encoder_id, so shards produced by older code are treated as stale and recomputed.
#   1 -> 2: geom parser fixed (v1 found boxes in only 2 of 252 clips; crops fell back to full frame)
PIPELINE_VERSION = 2


@dataclass(frozen=True)
class EmbedConfig:
    pipeline_version: int = PIPELINE_VERSION
    rate: float = 2.0
    n_max: int = 16
    tight_pad: float = 0.25
    tight_min: int = 224
    ctx_scale: float = 3.0
    ctx_min: int = 448
    box_window: int = 5
    views: tuple[str, ...] = VIEWS

    def as_dict(self) -> dict:
        d = asdict(self)
        d["views"] = list(self.views)
        return d


def plan(instances, fps: float, cfg: EmbedConfig) -> dict[int, list[tuple[int, tuple[int, ...]]]]:
    """frame index -> [(instance_id, actor_ids), ...] for every sampled frame of every instance."""
    by_frame: dict[int, list] = defaultdict(list)
    for r in instances:
        actors = tuple(int(a) for a in str(r["actor_ids"]).split(";") if a not in ("", "nan"))
        for f in sample_frames(int(r["frame_start"]), int(r["frame_end"]), fps, cfg.rate, cfg.n_max):
            by_frame[f].append((int(r["instance_id"]), actors))
    return dict(by_frame)


def check_track_coverage(instances, tracks: Tracks, stem: str) -> None:
    """Fail loudly when a clip's actors have no box tracks at all.

    Otherwise every crop silently falls back to the full frame and the tight/context views
    become copies of the full view (what happened on the first Colab run: 99.5% fallback).
    """
    actors = {int(a) for r in instances for a in str(r["actor_ids"]).split(";")
              if a not in ("", "nan")}
    if actors and not actors & tracks.keys():
        raise ValueError(f"{stem}: none of {len(actors)} actors have box tracks "
                         f"(geom parsed {len(tracks)} tracks) - check the geom file format")


def regions_for(actors, frame_idx: int, tracks: Tracks, img_w: int, img_h: int, cfg: EmbedConfig):
    """[(view_index, region_box, fallback, actor_box_side)] for one instance at one frame."""
    box = lookup_union(tracks, actors, frame_idx, cfg.box_window) if actors else None
    out = []
    for vi, view in enumerate(VIEWS):
        if view not in cfg.views:
            continue
        if view == "full" or box is None:
            region, fallback = full_region(img_w, img_h), view != "full"
        elif view == "tight":
            region, fallback = tight_region(box, img_w, img_h, cfg.tight_pad, cfg.tight_min), False
        else:
            region, fallback = context_region(box, img_w, img_h, cfg.ctx_scale, cfg.ctx_min), False
        out.append((vi, region, fallback, box_side(box)))
    return out


def iter_frames(cap, needed: list[int]):
    """Sequential decode (no seeking: AVI seeks can land on the wrong frame). Yields (idx, BGR|None)."""
    pos = 0
    for target in needed:
        while pos < target:
            if not cap.grab():
                return  # stream ended early: the caller reports unseen targets as missing
            pos += 1
        ok, frame = cap.read()
        pos += 1
        yield target, frame if ok else None


class _Batch:
    def __init__(self, encoders, batch_size: int):
        self.encoders, self.batch_size = encoders, batch_size
        self.crops, self.rows = [], []
        self.out = {e.name: [] for e in encoders}
        self.rows_done = []

    def add(self, img, row):
        self.crops.append(img)
        self.rows.append(row)
        if len(self.crops) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.crops:
            return
        for e in self.encoders:
            self.out[e.name].append(e.embed_images(self.crops))
        self.rows_done.extend(self.rows)
        self.crops, self.rows = [], []


def process_video(video_path: str, instances, tracks: Tracks, encoders, cfg: EmbedConfig,
                  out_dir: str, encoder_ids: dict[str, str], batch_size: int = 64,
                  debug_dir: str | None = None, debug_n: int = 0) -> dict:
    import cv2

    stem = clip_stem(video_path)
    paths = {e.name: os.path.join(out_dir, e.name, f"{stem}.npz") for e in encoders}
    if all(shard_is_valid(paths[e.name], encoder_ids[e.name]) for e in encoders):
        return {"clip_stem": stem, "skipped": True}
    check_track_coverage(instances, tracks, stem)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    img_w, img_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        raise IOError(f"invalid fps {fps} for {video_path}")

    by_frame = plan(instances, fps, cfg)
    batch = _Batch(encoders, batch_size)
    missing, saved = [], 0
    try:
        seen = set()
        for idx, frame in iter_frames(cap, sorted(by_frame)):
            seen.add(idx)
            if frame is None:
                missing.append(idx)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for inst_id, actors in by_frame[idx]:
                for vi, region, fallback, side in regions_for(actors, idx, tracks, img_w, img_h, cfg):
                    img = crop(rgb, region)
                    batch.add(img, (inst_id, idx, vi, fallback, side))
                    if debug_dir and saved < debug_n:
                        os.makedirs(debug_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(debug_dir, f"{stem}_i{inst_id}_f{idx}_{VIEWS[vi]}.jpg"),
                                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        saved += 1
        missing += [f for f in by_frame if f not in seen]
        batch.flush()
    finally:
        cap.release()

    rows = np.array(batch.rows_done, dtype=np.int64).reshape(-1, 5)
    for e in encoders:
        emb = (np.concatenate(batch.out[e.name]) if batch.out[e.name]
               else np.zeros((0, e.dim), np.float32))
        meta = {"encoder_id": encoder_ids[e.name], "encoder": e.name, "clip_stem": stem,
                "fps": fps, "width": img_w, "height": img_h,
                "frames_planned": len(by_frame), "frames_missing": len(missing)}
        write_shard(paths[e.name], meta, emb=emb, instance_id=rows[:, 0], frame_idx=rows[:, 1],
                    view=rows[:, 2], fallback=rows[:, 3].astype(bool), box_side=rows[:, 4])
    fallback_rate = float(rows[:, 3].mean()) if len(rows) else 0.0
    return {"clip_stem": stem, "skipped": False, "vectors": len(rows),
            "frames_missing": len(missing), "fallback_rate": round(fallback_rate, 4)}


def main(argv=None) -> int:
    from sfs.data.manifest import read_manifest
    from sfs.embed.encoder import Encoder, make_encoder_id

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="shard root; one subfolder per encoder")
    ap.add_argument("--encoders", default="siglip,siglip2")
    ap.add_argument("--splits", default="train,val,eval")
    ap.add_argument("--limit-videos", type=int, default=0, help="0 = all")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--debug-crops", type=int, default=0, help="save N crops per video as JPG")
    a = ap.parse_args(argv)

    cfg = EmbedConfig()
    df = read_manifest(a.manifest)
    df = df[df.split.isin(a.splits.split(","))]
    encoders = [Encoder(n) for n in a.encoders.split(",")]
    ids = {e.name: make_encoder_id(e.identity(cfg.as_dict())) for e in encoders}
    print("encoder ids:", ids)

    videos = list(df.groupby("video_path"))
    if a.limit_videos:
        videos = videos[: a.limit_videos]
    t0, done, failed = time.time(), 0, []
    for i, (video_path, g) in enumerate(videos, 1):
        try:
            tracks = parse_geom(geom_path_for(g.anno_file.iloc[0]))
            stats = process_video(video_path, g.to_dict("records"), tracks, encoders, cfg, a.out, ids,
                                  a.batch_size, os.path.join(a.out, "_debug") if a.debug_crops else None,
                                  a.debug_crops)
            done += 1
            print(f"[{i}/{len(videos)}] {stats}  ({time.time() - t0:.0f}s)", flush=True)
        except Exception as exc:  # one bad video must not kill a multi-hour run
            failed.append(video_path)
            print(f"[{i}/{len(videos)}] FAILED {video_path}: {type(exc).__name__}: {exc}", flush=True)
    print(f"\ndone={done} failed={len(failed)}")
    for f in failed:
        print("  failed:", f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
