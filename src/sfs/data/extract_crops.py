"""Crop extraction job (spec-crops-queries-baseline §3).

Reuses the embedding job's tested planning, region and decoding functions, so stored crops are
exactly the regions that were embedded. One packed shard per video; resumable.

CLI:
    python -m sfs.data.extract_crops --manifest manifest_v2.csv --out crops/
    python -m sfs.data.extract_crops verify --crops crops/ --embeddings embeddings_v2_siglip2.npz
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

from sfs.data.kpf import Tracks, clip_stem, geom_path_for, parse_geom
from sfs.data.regions import VIEWS, crop
from sfs.embed.run import EmbedConfig, check_track_coverage, iter_frames, plan, regions_for
from sfs.store.crops import CropShard, crop_shard_is_valid, write_crop_shard

DEFAULT_VIEWS = ("tight", "context")
INTEGRITY_MIN_COSINE = 0.98


def make_store_id(cfg: EmbedConfig, views, size: int, quality: int) -> str:
    ident = {"embed_config": cfg.as_dict(), "views": list(views), "size": size, "quality": quality}
    return hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]


def encode_crop(rgb: np.ndarray, size: int, quality: int) -> bytes:
    import cv2
    h, w = rgb.shape[:2]
    interp = cv2.INTER_AREA if max(h, w) > size else cv2.INTER_CUBIC
    img = cv2.resize(rgb, (size, size), interpolation=interp)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def extract_video(video_path: str, instances, tracks: Tracks, cfg: EmbedConfig, out_dir: str,
                  store_id: str, views=DEFAULT_VIEWS, size: int = 336, quality: int = 95) -> dict:
    import cv2

    stem = clip_stem(video_path)
    path = os.path.join(out_dir, f"{stem}.npz")
    if crop_shard_is_valid(path, store_id):
        return {"clip_stem": stem, "skipped": True}
    check_track_coverage(instances, tracks, stem)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    img_w, img_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        raise IOError(f"invalid fps {fps} for {video_path}")

    keep = {VIEWS.index(v) for v in views}
    by_frame = plan(instances, fps, cfg)
    jpegs, rows, seen, missing = [], [], set(), 0
    try:
        for idx, frame in iter_frames(cap, sorted(by_frame)):
            seen.add(idx)
            if frame is None:
                missing += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for inst_id, actors in by_frame[idx]:
                for vi, region, _fallback, _side in regions_for(actors, idx, tracks, img_w, img_h, cfg):
                    if vi in keep:
                        jpegs.append(encode_crop(crop(rgb, region), size, quality))
                        rows.append((inst_id, idx, vi))
    finally:
        cap.release()

    missing += len(by_frame) - len(seen)  # targets never reached: stream ended early
    r = np.array(rows, np.int64).reshape(-1, 3)
    meta = {"store_id": store_id, "clip_stem": stem, "fps": fps, "size": size, "quality": quality,
            "views": list(views), "frames_planned": len(by_frame), "frames_missing": missing}
    write_crop_shard(path, jpegs, meta, instance_id=r[:, 0], frame_idx=r[:, 1], view=r[:, 2])
    return {"clip_stem": stem, "skipped": False, "crops": len(jpegs),
            "mb": round(sum(map(len, jpegs)) / 1e6, 1), "frames_missing": missing}


def build_index(out_dir: str) -> "pd.DataFrame":
    import pandas as pd
    parts = []
    for p in sorted(glob.glob(os.path.join(out_dir, "*.npz"))):
        s = CropShard(p)
        parts.append(pd.DataFrame({"instance_id": s.instance_id, "frame_idx": s.frame_idx,
                                   "view": [VIEWS[v] for v in s.view],
                                   "file": os.path.basename(p), "row": np.arange(len(s))}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def verify(crops_dir: str, embeddings_path: str, encoder_name: str = "siglip2",
           n: int = 2000, seed: int = 0) -> dict:
    """Integrity gate: re-embed stored tight crops; compare with the video-derived vectors."""
    from sfs.embed.encoder import Encoder
    from sfs.store.shards import read_shard

    idx = build_index(crops_dir)
    idx = idx[idx.view == "tight"]
    sample = idx.sample(min(n, len(idx)), random_state=seed)
    emb = read_shard(embeddings_path)
    key = (emb["instance_id"].astype(np.int64) << 32) | emb["frame_idx"].astype(np.int64)
    lookup = {k: i for i, k in enumerate(key[emb["view"] == VIEWS.index("tight")])}
    tight_rows = np.flatnonzero(emb["view"] == VIEWS.index("tight"))

    enc, shards, cos = Encoder(encoder_name), {}, []
    for file, g in sample.groupby("file"):
        shard = shards.setdefault(file, CropShard(os.path.join(crops_dir, file)))
        imgs = [shard.image(int(r)) for r in g.row]
        new = enc.embed_images(imgs)
        for (_, r), v in zip(g.iterrows(), new):
            j = lookup.get((int(r.instance_id) << 32) | int(r.frame_idx))
            if j is not None:
                cos.append(float(v @ emb["emb"][tight_rows[j]].astype(np.float32)))
    cos = np.array(cos)
    result = {"compared": len(cos), "mean_cosine": round(float(cos.mean()), 4),
              "p01_cosine": round(float(np.percentile(cos, 1)), 4),
              "passed": bool(len(cos) and cos.mean() >= INTEGRITY_MIN_COSINE)}
    return result


def main(argv=None) -> int:
    from sfs.data.manifest import read_manifest

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "verify":
        ap = argparse.ArgumentParser()
        ap.add_argument("--crops", required=True)
        ap.add_argument("--embeddings", required=True)
        ap.add_argument("--encoder", default="siglip2")
        ap.add_argument("--n", type=int, default=2000)
        a = ap.parse_args(argv[1:])
        r = verify(a.crops, a.embeddings, a.encoder, a.n)
        print(r)
        print("INTEGRITY", "PASSED" if r["passed"] else f"FAILED (need mean >= {INTEGRITY_MIN_COSINE})")
        return 0 if r["passed"] else 1

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", default=",".join(DEFAULT_VIEWS))
    ap.add_argument("--size", type=int, default=336)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--limit-videos", type=int, default=0)
    a = ap.parse_args(argv)

    cfg, views = EmbedConfig(), tuple(a.views.split(","))
    store_id = make_store_id(cfg, views, a.size, a.quality)
    print("store id:", store_id)
    videos = list(read_manifest(a.manifest).groupby("video_path"))
    if a.limit_videos:
        videos = videos[: a.limit_videos]
    t0, failed = time.time(), []
    for i, (video_path, g) in enumerate(videos, 1):
        try:
            tracks = parse_geom(geom_path_for(g.anno_file.iloc[0]))
            stats = extract_video(video_path, g.to_dict("records"), tracks, cfg, a.out, store_id,
                                  views, a.size, a.quality)
            print(f"[{i}/{len(videos)}] {stats}  ({time.time() - t0:.0f}s)", flush=True)
        except Exception as exc:
            failed.append(video_path)
            print(f"[{i}/{len(videos)}] FAILED {video_path}: {type(exc).__name__}: {exc}", flush=True)
    idx = build_index(a.out)
    idx.to_csv(os.path.join(a.out, "crops_index.csv"), index=False)
    print(f"\nindex: {len(idx):,} crops in {idx.file.nunique() if len(idx) else 0} shards | failed={len(failed)}")
    for f in failed:
        print("  failed:", f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
