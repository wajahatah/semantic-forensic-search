"""Re-embed every tight crop with an adapted (LoRA) encoder (spec-lora-finetune D10).

Reads the packed tight-crop file, embeds in fixed blocks written as resumable part files, then
consolidates into one embeddings file in the standard shard format, so `sfs.eval.baseline` runs on
it unchanged. The encoder id includes the adapter hash and training config: adapted vectors can
never be mixed with baseline vectors.

CLI:
    python -m sfs.embed.reembed --packed data/packed_tight --baseline embeddings_v2_siglip2.npz \
        --adapter runs/lora/A-uniform/best/adapter.pt --out data/embeddings_lora/A-uniform
    python -m sfs.embed.reembed ... --check-baseline 2000     # adapter OFF: must reproduce baseline
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from sfs.data.regions import VIEWS

BLOCK = 4096


def baseline_lookup(baseline: dict) -> dict:
    """(instance_id, frame_idx) -> row of the baseline's tight vectors."""
    sel = np.flatnonzero(baseline["view"] == VIEWS.index("tight"))
    return {(int(baseline["instance_id"][r]), int(baseline["frame_idx"][r])): int(r) for r in sel}


def embed_rows(vision, packed, rows, device: str, batch: int = 128) -> np.ndarray:
    import torch

    from sfs.train.data import preprocess_eval, to_pixels
    from sfs.train.lora import image_features

    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
        for s in range(0, len(rows), batch):
            x = np.stack([preprocess_eval(packed.image(int(r))) for r in rows[s:s + batch]])
            out.append(image_features(vision, to_pixels(x).to(device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def load_encoder(adapter_path: str | None, device: str):
    """Vision tower with the adapter applied (or the untouched base model when adapter_path is None)."""
    import torch

    from sfs.embed.encoder import ENCODERS
    from sfs.train.lora import adapter_hash, load_adapter_state, load_vision_tower, wrap_lora

    info = {"adapter": None, "train_config": None}
    if adapter_path is None:
        vision, _, _, rev = load_vision_tower(ENCODERS["siglip2"])
    else:
        best = torch.load(adapter_path, map_location="cpu", weights_only=False)
        cfg = best["config"]
        vision, _, _, rev = load_vision_tower(ENCODERS[cfg["encoder"]], cfg["model_revision"])
        vision = wrap_lora(vision, cfg["lora_rank"], cfg["lora_alpha"], cfg["lora_dropout"])
        load_adapter_state(vision, best["adapter"])
        info = {"adapter": adapter_hash(best["adapter"]), "train_config": cfg,
                "best_epoch": best["epoch"], "val_map": best["val_map"]}
    info["revision"] = rev
    return vision.to(device).eval(), info


def reembed(packed, baseline: dict, vision, info: dict, out_dir: str, device: str, log=print) -> str:
    from sfs.embed.encoder import make_encoder_id
    from sfs.store.shards import consolidate, shard_is_valid, write_shard

    lookup = baseline_lookup(baseline)
    keys = list(zip(packed.instance_id.tolist(), packed.frame_idx.tolist()))
    missing = [k for k in keys if k not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} packed crops have no baseline row, e.g. {missing[:3]}")
    base_meta = baseline["meta"]
    # the baseline id already hashes model, revision and sampling/crop config (pipeline v2)
    identity = {"base_encoder_id": base_meta["encoder_id"], "revision": info["revision"],
                "store_id": packed.meta["store_id"], "adapter": info["adapter"],
                "train_config": info["train_config"]}
    encoder_id = make_encoder_id(identity)
    meta = {"encoder": "siglip2", "encoder_id": encoder_id, "identity": identity,
            "adapter": info["adapter"]}
    parts_dir = os.path.join(out_dir, "parts")
    paths = []
    for b, s in enumerate(range(0, len(keys), BLOCK)):
        path = os.path.join(parts_dir, f"part_{b:05d}.npz")
        paths.append(path)
        if shard_is_valid(path, encoder_id):
            continue
        rows = np.arange(s, min(s + BLOCK, len(keys)))
        emb = embed_rows(vision, packed, rows, device)
        brow = np.array([lookup[keys[r]] for r in rows])
        write_shard(path, meta, emb=emb, instance_id=packed.instance_id[rows],
                    frame_idx=packed.frame_idx[rows], view=np.full(len(rows), VIEWS.index("tight")),
                    fallback=baseline["fallback"][brow], box_side=baseline["box_side"][brow])
        log(f"part {b + 1}/{-(-len(keys) // BLOCK)} written")
    out = os.path.join(out_dir, "embeddings.npz")
    consolidate(paths, out)
    return out


def check_baseline(packed, baseline: dict, vision, device: str, n: int, seed: int = 0) -> dict:
    """Adapter OFF must reproduce the baseline vectors (D10 sanity check)."""
    lookup = baseline_lookup(baseline)
    rows = np.sort(np.random.default_rng(seed).choice(len(packed), size=min(n, len(packed)), replace=False))
    emb = embed_rows(vision, packed, rows, device)
    ref = np.stack([baseline["emb"][lookup[(int(packed.instance_id[r]), int(packed.frame_idx[r]))]]
                    for r in rows]).astype(np.float32)
    cos = (emb * ref).sum(1) / np.linalg.norm(ref, axis=1)
    return {"n": len(rows), "cos_min": float(cos.min()), "cos_mean": float(cos.mean())}


def main(argv=None) -> int:
    import torch

    from sfs.store.shards import read_shard
    from sfs.train.data import PackedCrops

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--packed", required=True)
    ap.add_argument("--baseline", required=True, help="baseline SigLIP 2 embeddings (.npz)")
    ap.add_argument("--adapter", help="best/adapter.pt of a training run")
    ap.add_argument("--out", help="output directory")
    ap.add_argument("--check-baseline", type=int, default=0,
                    help="embed N random crops with NO adapter and compare with the baseline")
    a = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    packed = PackedCrops(a.packed)
    baseline = read_shard(a.baseline)
    if a.check_baseline:
        vision, _ = load_encoder(None, device)
        r = check_baseline(packed, baseline, vision, device, a.check_baseline)
        print(f"adapter off vs baseline over {r['n']} crops: cos min {r['cos_min']:.5f}, "
              f"mean {r['cos_mean']:.5f}")
        return 0 if r["cos_min"] >= 0.98 else 1
    if not (a.adapter and a.out):
        ap.error("--adapter and --out are required unless --check-baseline is given")
    vision, info = load_encoder(a.adapter, device)
    print(f"adapter {info['adapter']} (epoch {info['best_epoch']}, val mAP {info['val_map']:.4f})")
    print("wrote", reembed(packed, baseline, vision, info, a.out, device))
    return 0


if __name__ == "__main__":
    sys.exit(main())
