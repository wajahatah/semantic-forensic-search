"""Zero-shot CLIP ViT-B/16 baseline (spec-submission §2, D-066).

Measures openai/clip-vit-base-patch16 exactly as the frozen SigLIP baselines were measured, on the
tight view (the selected view, D-011), from the local packed tight crops:

  embed  every tight crop with CLIP -> one embeddings file in the standard shard format
         (resumable part files; same rows, fallback and box_side as the SigLIP 2 baseline file).
  val    Result 01 protocol on the validation split (Protocol A ``split == "val"``): activity types
         with >= 5 validation instances, prompt "a surveillance camera photo of a <activity words>",
         instance score = mean of the top-3 frame similarities, per activity AUC and P@10,
         macro-averaged. Run for every encoder given, so the same code must reproduce Result 01's
         SigLIP / SigLIP 2 numbers before the CLIP row is read.
  eval   Result 02 protocol (Q1 / Q2 over the eval split, 66 frozen phrasings, Protocols A and B)
         through ``sfs.eval.baseline.evaluate``/``summarise`` — the functions that made Result 02.

A reported comparison only: no decision depends on it (encoder chosen 2026-10-23); whatever CLIP
scores, the system does not change (rule fixed in the spec before any CLIP number).

CLI:
    python scripts/clip_baseline.py embed --packed data/packed_tight \
        --baseline manifest_v2_drive/embeddings_v2_siglip2.npz --out data/embeddings_clip
    python scripts/clip_baseline.py val --manifest manifest_v2_drive/manifest_v2.csv \
        --emb siglip=manifest_v2_drive/embeddings_v2_siglip.npz \
        --emb siglip2=manifest_v2_drive/embeddings_v2_siglip2.npz \
        --emb clip=data/embeddings_clip/embeddings.npz --out runs/clip/val.json
    python scripts/clip_baseline.py eval --manifest manifest_v2_drive/manifest_v2.csv \
        --phrasings queries/phrasings.csv --emb siglip2=... --emb clip=... --out runs/clip/eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

CLIP_ID = "openai/clip-vit-base-patch16"
CLIP_REVISION = "57c216476eefef5ab752ec549e440a49ae4ae5f3"  # downloaded and hash-checked 2026-11-17
PROMPT = "a surveillance camera photo of a {}"
MIN_VAL_INSTANCES = 5
TOP_M = 3
BLOCK = 4096


# ---------------------------------------------------------------- encoders
def _normalise(t) -> np.ndarray:
    t = t.float()
    return (t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().numpy()


def _features(out):
    """transformers 5 returns a model output from get_*_features on some models; 4.x a tensor."""
    import torch
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("pooler_output", "image_embeds", "text_embeds"):
        val = getattr(out, attr, None)
        if isinstance(val, torch.Tensor):
            return val
    raise TypeError(f"cannot extract features from {type(out).__name__}")


class ClipEncoder:
    """CLIP with its own preprocessing (shortest side 224 bicubic, centre crop, CLIP mean/std) and
    its own tokenizer (77 tokens). Each encoder is measured with the preprocessing it was trained with."""

    def __init__(self, model, image_processor, tokenizer=None, device: str = "cpu"):
        import torch
        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = model.to(device, self.dtype).eval()
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    @classmethod
    def load(cls, device: str | None = None, revision: str = CLIP_REVISION) -> "ClipEncoder":
        import torch
        from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = CLIPModel.from_pretrained(CLIP_ID, revision=revision)
        enc = cls(model, CLIPImageProcessor.from_pretrained(CLIP_ID, revision=revision),
                  CLIPTokenizer.from_pretrained(CLIP_ID, revision=revision), device)
        enc.revision = revision
        return enc

    @property
    def dim(self) -> int:
        return int(self.model.config.projection_dim)

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        import torch
        if not images:
            return np.zeros((0, self.dim), np.float32)
        px = self.image_processor(images=images, return_tensors="pt")["pixel_values"]
        with torch.inference_mode():
            return _normalise(_features(self.model.get_image_features(
                pixel_values=px.to(self.device, self.dtype))))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        import torch
        tok = self.tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
        with torch.inference_mode():
            return _normalise(_features(self.model.get_text_features(
                input_ids=tok["input_ids"].to(self.device),
                attention_mask=tok["attention_mask"].to(self.device))))


# ---------------------------------------------------------------- embed
def embed_packed(packed, baseline: dict, encoder, out_dir: str, identity: dict,
                 batch: int = 256, log=print) -> str:
    """Embed every packed tight crop; rows carry the baseline's fallback / box_side for the same
    (instance, frame). Resumable: finished part files with the same encoder id are kept."""
    from sfs.data.regions import VIEWS
    from sfs.embed.encoder import make_encoder_id
    from sfs.embed.reembed import baseline_lookup
    from sfs.store.shards import consolidate, shard_is_valid, write_shard

    lookup = baseline_lookup(baseline)
    keys = list(zip(packed.instance_id.tolist(), packed.frame_idx.tolist()))
    missing = [k for k in keys if k not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} packed crops have no baseline row, e.g. {missing[:3]}")
    identity = {**identity, "store_id": packed.meta["store_id"], "view": "tight"}
    encoder_id = make_encoder_id(identity)
    meta = {"encoder": identity["encoder"], "encoder_id": encoder_id, "identity": identity}
    parts_dir = os.path.join(out_dir, "parts")
    n_blocks = -(-len(keys) // BLOCK)
    paths = []
    for b, s in enumerate(range(0, len(keys), BLOCK)):
        path = os.path.join(parts_dir, f"part_{b:05d}.npz")
        paths.append(path)
        if shard_is_valid(path, encoder_id):
            continue
        rows = np.arange(s, min(s + BLOCK, len(keys)))
        emb = np.concatenate([encoder.embed_images([packed.image(int(r)) for r in rows[i:i + batch]])
                              for i in range(0, len(rows), batch)])
        brow = np.array([lookup[keys[r]] for r in rows])
        write_shard(path, meta, emb=emb.astype(np.float16), instance_id=packed.instance_id[rows],
                    frame_idx=packed.frame_idx[rows], view=np.full(len(rows), VIEWS.index("tight")),
                    fallback=baseline["fallback"][brow], box_side=baseline["box_side"][brow])
        log(f"part {b + 1}/{n_blocks} written")
    out = os.path.join(out_dir, "embeddings.npz")
    consolidate(paths, out)
    return out


# ---------------------------------------------------------------- validation (Result 01 protocol)
def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC with ties counted as half (average ranks, as sfs.eval.cascade_run._auc)."""
    import pandas as pd
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def val_activities(manifest, split_col: str = "split", min_n: int = MIN_VAL_INSTANCES) -> list[str]:
    counts = manifest.loc[manifest[split_col] == "val", "activity"].value_counts()
    return sorted(counts[counts >= min_n].index)


def prompt_for(activity: str) -> str:
    return PROMPT.format(activity.replace("_", " "))


def zero_shot_val(instance_ids: np.ndarray, labels: np.ndarray, scores: np.ndarray,
                  activities: list[str]) -> dict:
    """scores (n_inst, n_act): per-activity AUC (matching vs non-matching) and P@10, macro."""
    per = {}
    for j, a in enumerate(activities):
        pos = labels == a
        order = np.argsort(-scores[:, j], kind="stable")
        per[a] = {"n_pos": int(pos.sum()), "auc": auc(scores[pos, j], scores[~pos, j]),
                  "p10": float(pos[order[:10]].mean())}
    return {"n_instances": int(len(instance_ids)), "n_activities": len(activities),
            "auc": float(np.mean([v["auc"] for v in per.values()])),
            "p10": float(np.mean([v["p10"] for v in per.values()])),
            "chance_p10": float(np.mean([v["n_pos"] for v in per.values()]) / len(instance_ids)),
            "per_activity": per}


def text_encoder(name: str, device: str | None = None):
    if name == "clip":
        return ClipEncoder.load(device)
    from sfs.embed.encoder import Encoder
    return Encoder(name, device)


def run_val(manifest_path: str, embs: dict[str, str], out: str) -> dict:
    from sfs.data.manifest import read_manifest
    from sfs.eval.retrieval import InstanceIndex
    from sfs.store.shards import read_shard

    m = read_manifest(manifest_path)
    val = m[m["split"] == "val"]
    acts = val_activities(m)
    # Result 01 scores all 739 validation instances; those of the rarer types count as negatives
    result = {"protocol": "Result 01: validation split, tight view, top-3 mean", "prompt": PROMPT,
              "activities": acts, "encoders": {}}
    for name, path in embs.items():
        shard = read_shard(path)
        if shard["meta"].get("encoder") != name:
            raise ValueError(f"{path} holds {shard['meta'].get('encoder')!r}, not {name!r}")
        corpus = set(val.instance_id.astype(int))
        idx = InstanceIndex.from_shard(shard, "tight", corpus)
        if len(idx.instance_ids) != len(corpus):
            raise ValueError(f"{name}: {len(corpus) - len(idx.instance_ids)} val instances lack embeddings")
        T = text_encoder(name).embed_texts([prompt_for(a) for a in acts])
        lab = val.set_index(val.instance_id.astype(int)).loc[idx.instance_ids, "activity"].to_numpy()
        r = zero_shot_val(idx.instance_ids, lab, idx.score(T, TOP_M), acts)
        r["encoder_id"] = shard["meta"]["encoder_id"]
        result["encoders"][name] = r
        print(f"{name:8s} AUC {r['auc']:.3f}  P@10 {r['p10']:.3f}  "
              f"({r['n_instances']} instances, {r['n_activities']} activities)")
    _write_json(out, result)
    return result


# ---------------------------------------------------------------- eval (Result 02 protocol)
def run_eval(manifest_path: str, phrasings_path: str, embs: dict[str, str], out: str) -> dict:
    import pandas as pd

    from sfs.data.manifest import read_manifest
    from sfs.eval.baseline import PROTOCOLS, evaluate, summarise
    from sfs.eval.queries import (build_q1, build_q2, check_phrasings, eligible_activities,
                                  load_phrasings)
    from sfs.eval.retrieval import InstanceIndex
    from sfs.store.shards import read_shard

    m = read_manifest(manifest_path)
    acts = eligible_activities(m, "split")
    phr = load_phrasings(phrasings_path)
    check_phrasings(phr, acts)
    texts = [(a, j, t) for a in acts for j, t in enumerate(phr[a])]
    text_col = {(a, j): c for c, (a, j, _) in enumerate(texts)}
    result = {"protocol": "Result 02: eval split, tight view, top-3 mean, 66 frozen phrasings",
              "n_phrasings": len(texts), "protocols": {}}
    rows = []
    shards = {name: read_shard(p) for name, p in embs.items()}
    T = {name: text_encoder(name).embed_texts([t for _, _, t in texts]) for name in embs}
    for proto, col in PROTOCOLS.items():
        corpus = set(m.loc[m[col] == "eval", "instance_id"].astype(int))
        queries = build_q1(m, col, acts, phr) + build_q2(m, col, acts, phr)
        result["protocols"][proto] = {"corpus_instances": len(corpus), "systems": {}}
        for name, shard in shards.items():
            idx = InstanceIndex.from_shard(shard, "tight", corpus)
            if len(idx.instance_ids) != len(corpus):
                raise ValueError(f"{name}: {len(corpus) - len(idx.instance_ids)} eval instances lack embeddings")
            df = evaluate(queries, idx.instance_ids, idx.score(T[name], TOP_M), text_col)
            df.insert(0, "system", f"{name}-tight")
            df.insert(0, "protocol", proto)
            rows.append(df)
            s = summarise(df)
            s["encoder_id"] = shard["meta"]["encoder_id"]
            result["protocols"][proto]["systems"][f"{name}-tight"] = s
            print(f"{proto} {name:8s} Q1 mAP {s['q1']['macro']['ap']:.3f}  P@10 {s['q1']['macro']['p10']:.3f}  "
                  f"Q2 S@1 {s['q2']['macro']['s1']:.3f}  S@10 {s['q2']['macro']['s10']:.3f}  "
                  f"MRR {s['q2']['macro']['rr']:.3f}")
    _write_json(out + ".json", result)
    pd.concat(rows).to_csv(out + "_queries.csv", index=False)
    return result


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _embs(pairs: list[str]) -> dict[str, str]:
    out = {}
    for p in pairs:
        name, _, path = p.partition("=")
        if not path:
            raise SystemExit(f"--emb takes name=path, got {p!r}")
        out[name] = path
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--packed", required=True)
    e.add_argument("--baseline", required=True, help="SigLIP 2 baseline embeddings (row metadata)")
    e.add_argument("--out", required=True)
    e.add_argument("--batch", type=int, default=256)
    v = sub.add_parser("val")
    v.add_argument("--manifest", required=True)
    v.add_argument("--emb", action="append", required=True, help="name=path (siglip, siglip2, clip)")
    v.add_argument("--out", required=True)
    q = sub.add_parser("eval")
    q.add_argument("--manifest", required=True)
    q.add_argument("--phrasings", required=True)
    q.add_argument("--emb", action="append", required=True)
    q.add_argument("--out", required=True, help="prefix: writes <out>.json and <out>_queries.csv")
    a = ap.parse_args(argv)

    if a.cmd == "embed":
        from sfs.store.shards import read_shard
        from sfs.train.data import PackedCrops
        enc = ClipEncoder.load()
        identity = {"encoder": "clip", "model_id": CLIP_ID, "revision": enc.revision,
                    "preprocess": "CLIPImageProcessor (shortest side 224 bicubic, centre crop 224)",
                    "dtype": str(enc.dtype)}
        print("wrote", embed_packed(PackedCrops(a.packed), read_shard(a.baseline), enc, a.out,
                                    identity, a.batch))
    elif a.cmd == "val":
        run_val(a.manifest, _embs(a.emb), a.out)
    else:
        run_eval(a.manifest, a.phrasings, _embs(a.emb), a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
