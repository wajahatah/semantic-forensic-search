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
from sfs.store.crops import CropShard, crop_shard_is_valid, read_crop_rows, write_crop_shard

INTEGRITY_MIN_COSINE = 0.98

# Per-view storage format (D-014). `tight` is embedded and trained on, so it is stored
# pixel-exact: native resolution, lossless PNG, the encoder's own resize is the only resize.
# Storing it resized to 336 px JPEG gave mean cosine 0.964 vs the original embeddings, mostly
# because typical tight crops are 224 px and were upscaled. `context` is read only by the VLM
# teacher, so it is downscaled to <=448 px and stored as JPEG to keep the store small.
VIEW_FORMATS = {
    "tight": {"format": "png", "max_side": None},
    "context": {"format": "jpg", "max_side": 448, "quality": 95},
}
DEFAULT_VIEWS = tuple(VIEW_FORMATS)


def make_store_id(cfg: EmbedConfig, formats: dict) -> str:
    ident = {"embed_config": cfg.as_dict(), "formats": formats}
    return hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]


def encode_crop(rgb: np.ndarray, format: str = "png", max_side: int | None = None,
                quality: int = 95) -> bytes:
    """Encode an RGB crop. Downscales (never upscales) only if `max_side` is exceeded."""
    import cv2
    img = rgb
    h, w = img.shape[:2]
    if max_side and max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if format == "png":
        ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    elif format == "jpg":
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        raise ValueError(f"unknown crop format {format!r}")
    if not ok:
        raise ValueError(f"{format} encoding failed")
    return buf.tobytes()


def extract_video(video_path: str, instances, tracks: Tracks, cfg: EmbedConfig, out_dir: str,
                  store_id: str, formats: dict = VIEW_FORMATS) -> dict:
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

    fmt_by_view = {VIEWS.index(v): f for v, f in formats.items()}
    by_frame = plan(instances, fps, cfg)
    blobs, rows, seen, missing = [], [], set(), 0
    try:
        for idx, frame in iter_frames(cap, sorted(by_frame)):
            seen.add(idx)
            if frame is None:
                missing += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for inst_id, actors in by_frame[idx]:
                for vi, region, _fallback, _side in regions_for(actors, idx, tracks, img_w, img_h, cfg):
                    if vi in fmt_by_view:
                        blobs.append(encode_crop(crop(rgb, region), **fmt_by_view[vi]))
                        rows.append((inst_id, idx, vi))
    finally:
        cap.release()

    missing += len(by_frame) - len(seen)  # targets never reached: stream ended early
    r = np.array(rows, np.int64).reshape(-1, 3)
    meta = {"store_id": store_id, "clip_stem": stem, "fps": fps, "formats": formats,
            "frames_planned": len(by_frame), "frames_missing": missing}
    write_crop_shard(path, blobs, meta, instance_id=r[:, 0], frame_idx=r[:, 1], view=r[:, 2])
    return {"clip_stem": stem, "skipped": False, "crops": len(blobs),
            "mb": round(sum(map(len, blobs)) / 1e6, 1), "frames_missing": missing}


def build_index(out_dir: str) -> "pd.DataFrame":
    import pandas as pd
    parts = []
    for p in sorted(glob.glob(os.path.join(out_dir, "*.npz"))):
        r = read_crop_rows(p)
        parts.append(pd.DataFrame({"instance_id": r["instance_id"], "frame_idx": r["frame_idx"],
                                   "view": [VIEWS[v] for v in r["view"]],
                                   "file": os.path.basename(p), "row": np.arange(len(r["view"]))}))
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

    enc, cos = Encoder(encoder_name), []
    for file, g in sample.groupby("file"):
        # One shard in memory at a time. Caching them all held the whole store in RAM
        # (~10+ GB as PNG) and crashed the Colab runtime.
        shard = CropShard(os.path.join(crops_dir, file))
        imgs = [shard.image(int(r)) for r in g.row]
        del shard
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
    ap.add_argument("--limit-videos", type=int, default=0)
    a = ap.parse_args(argv)

    cfg = EmbedConfig()
    formats = {v: VIEW_FORMATS[v] for v in a.views.split(",")}
    store_id = make_store_id(cfg, formats)
    print("store id:", store_id)
    videos = list(read_manifest(a.manifest).groupby("video_path"))
    if a.limit_videos:
        videos = videos[: a.limit_videos]
    t0, failed = time.time(), []
    for i, (video_path, g) in enumerate(videos, 1):
        try:
            tracks = parse_geom(geom_path_for(g.anno_file.iloc[0]))
            stats = extract_video(video_path, g.to_dict("records"), tracks, cfg, a.out, store_id,
                                  formats)
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
