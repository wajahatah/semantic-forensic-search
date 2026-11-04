"""Gates G0 (laptop feasibility) and G1 (does the VLM add information?) — spec-rerank-cascade D3/D4.

G0, per model: 20 validation clips; peak VRAM, time per clip / per pair, determinism, and for the
instruct model that the cached path equals the full forward pass. Since 2026-11-03 (owner, Q4;
spec-vlm-followups §4) "equals" is judged against the bf16 noise floor, not a fixed 0.1: on 3 clips x
16 captions the cached-vs-full largest difference must not exceed the difference between two slow
passes that differ only in the attention kernel (eager vs sdpa), and the rank correlation of cached
and full scores must be >= SHORTCUT_MIN_SPEARMAN. `--frames` sets the frames per clip (E3: 16).
G1, per (model, view), VALIDATION only: re-rank the LoRA top-K of every validation Q1 query with
z(VLM) + w * z(LoRA) (w from WEIGHTS, chosen on validation); the gain is the best fused validation
mAP minus LoRA alone. Pass = gain >= G1_MIN_GAIN on at least one protocol. A failed gate does not
stop the work (owner, 2026-10-31): it is recorded and flagged "needs owner feedback".

Office server (spec-vlm-followups E4, D-029): `g0 --remote <url>` runs G0 against the scoring service
(memory read on the server; times are end-to-end from the laptop, transfer included); `--max-vram-gb`
sets the memory limit for that GPU (laptop default 7.5). `remote-check` proves the service adds no
error: the same 2B model served remotely must match the laptop's own scores within the bf16 noise floor
(the same rule as the shortcut check, on the G0 clips' first 3 x 16 captions).

CLI:
  python -m sfs.teacher.gate g0 --model ... --kind instruct --out runs/cascade/g0_instruct.json
  python -m sfs.teacher.gate remote-check --model data/models/Qwen3-VL-2B-Instruct --remote http://127.0.0.1:8765 \
      --out runs/cascade/e4_remote_check.json
  python -m sfs.teacher.gate g1-requests --out data/teacher/requests_g1.csv
  python -m sfs.teacher.gate g1 --scores data/teacher/g1/<model>-<view>/scores.npz ... --out runs/cascade/g1.json
Whole validation (spec-vlm-followups E2): `--k 0` re-ranks every validation clip. `g1-requests --k 0
--exclude-scored <scores.npz>...` lists only pairs not scored yet; `g1 --scores a.npz+b.npz` merges stores
into one configuration (named after the first store's folder).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from sfs.eval.cascade import topk_rerank

G1_K = 50
G1_MIN_GAIN = 0.02
SHORTCUT_MIN_SPEARMAN = 0.99
WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0)
SPLIT_COL = {"A": "split", "B": "split_b"}
LORA_EMB = {"A": "data/embeddings_lora/A-middle/embeddings.npz",
            "B": "data/embeddings_lora/B-uniform/embeddings.npz"}


class ValQ1:
    """Validation Q1 for one protocol: the LoRA score matrix and the queries (as in LoRA selection)."""

    def __init__(self, manifest, protocol: str, phrasings: dict, phr_emb: dict, lora_shard: dict,
                 min_instances: int = 5, top_m: int = 3):
        from sfs.eval.queries import build_q1
        from sfs.eval.retrieval import InstanceIndex
        col = SPLIT_COL[protocol]
        val = manifest[manifest[col] == "val"]
        counts = val.activity.value_counts()
        self.activities = sorted(a for a in phrasings if counts.get(a, 0) >= min_instances)
        self.queries = build_q1(manifest, col, self.activities, phrasings, subset="val")
        self.texts = [(a, j, phrasings[a][j]) for a in self.activities for j in range(len(phrasings[a]))]
        self.text_col = {(a, j): c for c, (a, j, _) in enumerate(self.texts)}
        idx = InstanceIndex.from_shard(lora_shard, "tight", set(val.instance_id.astype(int)))
        self.ids = idx.instance_ids
        self.lora = idx.score(np.stack([phr_emb[a][j] for a, j, _ in self.texts]), top_m)

    def top(self, k: int | None = G1_K) -> list[tuple[int, str]]:
        """Pairs the VLM must score for G1 at depth k; k None = every validation clip."""
        from sfs.eval.metrics import rank
        out = []
        for c, (_, _, text) in enumerate(self.texts):
            out += [(int(i), text) for i in rank(self.ids, self.lora[:, c])[:k]]
        return out

    def mean_ap(self, scores: np.ndarray) -> tuple[float, dict]:
        from sfs.eval.baseline import evaluate
        per = evaluate(self.queries, self.ids, scores, self.text_col).groupby("activity").ap.mean()
        return float(per.mean()), per.round(4).to_dict()

    def reranked(self, vlm: dict, k: int | None, w: float) -> np.ndarray:
        from sfs.eval.metrics import rank
        k = len(self.ids) if k is None else k
        out = np.empty_like(self.lora, dtype=np.float64)
        for c, (_, _, text) in enumerate(self.texts):
            top = rank(self.ids, self.lora[:, c])[:k]
            out[:, c] = topk_rerank(self.ids, self.lora[:, c], np.array([vlm[(int(i), text)] for i in top]), k, w)
        return out


def g1_evaluate(val: ValQ1, vlm: dict, k: int | None = G1_K) -> dict:
    lora_map, lora_per = val.mean_ap(val.lora)
    by_w = {}
    for w in WEIGHTS:
        m, per = val.mean_ap(val.reranked(vlm, k, w))
        by_w[w] = {"val_map": m, "per_activity": per}
    best_w = max(WEIGHTS, key=lambda w: (by_w[w]["val_map"], -w))
    return {"lora_val_map": lora_map, "lora_per_activity": lora_per, "by_weight": by_w, "best_weight": best_w,
            "best_val_map": by_w[best_w]["val_map"], "gain": by_w[best_w]["val_map"] - lora_map,
            "vlm_only_val_map": by_w[0.0]["val_map"]}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    import pandas as pd
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def shortcut_check(cached: np.ndarray, full: np.ndarray, full_other_kernel: np.ndarray) -> dict:
    """Cached path vs full pass, judged against the bf16 noise between two full passes (Q4, 2026-11-03)."""
    diff, noise = np.abs(cached - full), np.abs(full_other_kernel - full)
    out = {"cached_vs_full_max_abs_diff": float(diff.max()), "cached_vs_full_mean_abs_diff": float(diff.mean()),
           "cached_vs_full_spearman": _spearman(cached, full),
           "noise_floor_max_abs_diff": float(noise.max()), "noise_floor_mean_abs_diff": float(noise.mean()),
           "rule": f"cached-vs-full max <= eager-vs-sdpa max and spearman >= {SHORTCUT_MIN_SPEARMAN}"}
    out["shortcut_pass"] = bool(out["cached_vs_full_max_abs_diff"] <= out["noise_floor_max_abs_diff"]
                                and out["cached_vs_full_spearman"] >= SHORTCUT_MIN_SPEARMAN)
    return out


def g0_clips_and_texts(manifest, captions: dict, n_clips: int = 20, seed: int = 0):
    val = sorted(int(i) for i in manifest.instance_id[manifest.split == "val"])
    clips = sorted(np.random.default_rng(seed).choice(val, n_clips, replace=False).tolist())
    texts = [captions[c][j] for c in sorted(captions) for j in (0, 1)]
    return clips, texts


def g0(scorer, crops, manifest, captions: dict, n_clips: int = 20, view: str = "context", seed: int = 0,
       n_frames: int | None = None, max_vram_gb: float = 7.5) -> dict:
    """Feasibility on validation clips; the text load is the teacher-labelling load (2 captions x 32)."""
    from sfs.teacher.frames import N_FRAMES
    n_frames = n_frames or N_FRAMES
    clips, texts = g0_clips_and_texts(manifest, captions, n_clips, seed)
    scorer.reset_peak_memory()
    per_clip, per_pair = [], []
    for n, iid in enumerate(clips):
        frames, nums = crops.frames(iid, view, n_frames)
        fps = crops.fps_of(iid)
        scorer.sync()
        t0 = time.perf_counter()
        if scorer.kind == "instruct":
            scorer.score_clip(frames, nums, fps, texts)
            scorer.sync()
            per_clip.append(time.perf_counter() - t0)
            t1 = time.perf_counter()
            scorer.score_full(frames, nums, fps, texts[:2])
            scorer.sync()
            per_pair.append((time.perf_counter() - t1) / 2)
        else:
            scorer.score_full(frames, nums, fps, texts[:8])
            scorer.sync()
            per_pair.append((time.perf_counter() - t0) / 8)
    report = {"model_id": scorer.model_id, "kind": scorer.kind, "view": view, "n_frames": n_frames, "clips": clips,
              "peak_vram_gb": scorer.peak_memory_gb(), "max_vram_gb": max_vram_gb,
              "seconds_per_pair_uncached": float(np.median(per_pair)), "texts_per_clip": len(texts)}
    if per_clip:
        report["seconds_per_clip_cached_64"] = float(np.median(per_clip))
    frames, nums = crops.frames(clips[0], view, n_frames)
    fps = crops.fps_of(clips[0])
    a = scorer.score_full(frames, nums, fps, texts[:4])
    b = scorer.score_full(frames, nums, fps, texts[:4])
    report["determinism_max_abs_diff"] = float(np.abs(a - b).max())
    report["shortcut"] = bool(getattr(scorer, "shortcut", True))
    if getattr(scorer, "placement", None):                       # Gemma 4 (E6): what sits where
        report["placement"] = scorer.placement
    if len(getattr(scorer, "last_mass", ())):                    # does the model answer yes / no?
        report["mean_yes_no_mass"] = float(np.mean(scorer.last_mass))
    if scorer.kind == "instruct":
        inputs = [(crops.frames(iid, view, n_frames), crops.fps_of(iid)) for iid in clips[:3]]
        full = np.concatenate([scorer.score_full(f, nums, fps, texts[:16]) for (f, nums), fps in inputs])
        cached = np.concatenate([scorer.score_clip(f, nums, fps, texts[:16]) for (f, nums), fps in inputs])
        scorer.set_attn("eager")
        try:
            other = np.concatenate([scorer.score_full(f, nums, fps, texts[:16]) for (f, nums), fps in inputs])
        finally:
            scorer.set_attn("sdpa")
        report.update(shortcut_check(cached, full, other))
    report["pass"] = bool(report["peak_vram_gb"] <= max_vram_gb and report["determinism_max_abs_diff"] < 1e-3
                          and (scorer.kind != "instruct" or (report["seconds_per_clip_cached_64"] <= 3.0
                                                             and report["shortcut_pass"])))
    return report


def _phrasing_embeddings(phr: dict) -> dict:
    from sfs.embed.encoder import Encoder
    acts = sorted(phr)
    emb = np.asarray(Encoder("siglip2").embed_texts([t for a in acts for t in phr[a]]), np.float32)
    out, k = {}, 0
    for a in acts:
        out[a] = emb[k:k + len(phr[a])]
        k += len(phr[a])
    return out


def remote_check(local, remote, crops, manifest, captions: dict, view: str = "context",
                 n_frames: int | None = None) -> dict:
    """The served model must reproduce the laptop's scores: remote-vs-local (full and cached) judged by
    the shortcut rule against the laptop's own eager-vs-sdpa noise floor, on 3 G0 clips x 16 captions."""
    from sfs.teacher.frames import N_FRAMES
    n_frames = n_frames or N_FRAMES
    clips, texts = g0_clips_and_texts(manifest, captions)
    inputs = [(crops.frames(iid, view, n_frames), crops.fps_of(iid)) for iid in clips[:3]]

    def full(s):
        return np.concatenate([s.score_full(f, nums, fps, texts[:16]) for (f, nums), fps in inputs])

    local_full, remote_full = full(local), full(remote)
    local.set_attn("eager")
    try:
        local_other = full(local)
    finally:
        local.set_attn("sdpa")
    report = {"local_model_id": local.model_id, "remote_model_id": remote.model_id, "remote_info": remote.info,
              "view": view, "n_frames": n_frames, "clips": clips[:3], "texts": 16,
              "full": shortcut_check(remote_full, local_full, local_other)}
    if remote.kind == "instruct":
        remote_cached = np.concatenate([remote.score_clip(f, nums, fps, texts[:16]) for (f, nums), fps in inputs])
        report["cached"] = shortcut_check(remote_cached, local_full, local_other)
    report["pass"] = all(report[k]["shortcut_pass"] for k in ("full", "cached") if k in report)
    return report


def main(argv=None) -> int:
    from sfs.data.manifest import read_manifest
    from sfs.eval.queries import load_phrasings
    from sfs.store.shards import read_shard
    from sfs.teacher.store import load_scores, write_requests

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["g0", "g1-requests", "g1", "remote-check"])
    ap.add_argument("--manifest", default="manifest_v2_drive/manifest_v2.csv")
    ap.add_argument("--phrasings", default="queries/phrasings.csv")
    ap.add_argument("--captions", default="queries/train_captions.csv")
    ap.add_argument("--crops", default="data/crops_v2")
    ap.add_argument("--model")
    ap.add_argument("--remote", help="g0 / remote-check: scoring service URL (SSH tunnel)")
    ap.add_argument("--max-vram-gb", type=float, default=7.5, help="g0: GPU memory limit (laptop 7.5)")
    ap.add_argument("--gpu-budget-gb", type=float, help="g0, Gemma 4: GPU memory for weights")
    ap.add_argument("--fp8-at-load", action="store_true", help="g0, Qwen-VL: language model in FP8 (E7)")
    ap.add_argument("--kind", choices=["instruct", "reranker"])
    ap.add_argument("--view", default="context")
    ap.add_argument("--scores", nargs="*", default=[],
                    help="g1: scores.npz of each (model, view) run; a.npz+b.npz merges stores into one run")
    ap.add_argument("--k", type=int, default=G1_K, help="G1 depth; 0 = every validation clip (E2)")
    ap.add_argument("--frames", type=int, default=0, help="g0: frames per clip (0 = default 8)")
    ap.add_argument("--exclude-scored", nargs="*", default=[], help="g1-requests: leave out pairs in these stores")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    m = read_manifest(a.manifest)

    if a.cmd in ("g0", "remote-check"):
        from sfs.teacher.frames import CropIndex
        from sfs.teacher.run import make_scorer
        from sfs.train.captions import load_captions
        if a.cmd == "g0":
            rep = g0(make_scorer(a.model, a.remote, a.kind, a.gpu_budget_gb, a.fp8_at_load), CropIndex(a.crops), m,
                     load_captions(a.captions),
                     view=a.view, n_frames=a.frames or None, max_vram_gb=a.max_vram_gb)
        else:
            from sfs.teacher.remote import RemoteScorer
            from sfs.teacher.vlm import load_scorer
            remote = RemoteScorer(a.remote, log=lambda s: print(s, flush=True))
            rep = remote_check(load_scorer(a.model, remote.kind), remote, CropIndex(a.crops), m,
                               load_captions(a.captions), view=a.view, n_frames=a.frames or None)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(rep, open(a.out, "w"), indent=1)
        print(json.dumps({k: v for k, v in rep.items() if k != "clips"}, indent=1))
        return 0

    phr = load_phrasings(a.phrasings)
    pe = _phrasing_embeddings(phr)
    vals = {p: ValQ1(m, p, phr, pe, read_shard(LORA_EMB[p])) for p in "AB"}
    k = a.k or None
    if a.cmd == "g1-requests":
        pairs = set(vals["A"].top(k)) | set(vals["B"].top(k))
        for path in a.exclude_scored:
            pairs -= set(load_scores(path)[0])
        pairs = sorted(pairs)
        write_requests(a.out, pairs)
        print(f"wrote {a.out}: {len(pairs):,} pairs over {len({i for i, _ in pairs}):,} validation instances")
        return 0

    report = {"k": k if k else "all", "min_gain": G1_MIN_GAIN, "configs": {}}
    for spec in a.scores:
        paths = spec.split("+")
        vlm, sid = {}, []
        for path in paths:
            more, one = load_scores(path)
            vlm, sid = {**more, **vlm}, sid + [one]
        sid = sid[0] if len(sid) == 1 else sid
        name = os.path.basename(os.path.dirname(paths[0]))
        res = {p: g1_evaluate(vals[p], vlm, k) for p in "AB"}
        report["configs"][name] = {"scorer_id": sid, **res,
                                   "pass": any(res[p]["gain"] >= G1_MIN_GAIN for p in "AB"),
                                   "mean_gain": float(np.mean([res[p]["gain"] for p in "AB"]))}
        print(f"{name}: " + "  ".join(f"{p}: LoRA {res[p]['lora_val_map']:.4f} -> {res[p]['best_val_map']:.4f} "
                                      f"(w={res[p]['best_weight']}, VLM-only {res[p]['vlm_only_val_map']:.4f})"
                                      for p in "AB"))
    best = max(report["configs"], key=lambda n: report["configs"][n]["mean_gain"])
    report["teacher"] = best
    report["pass"] = report["configs"][best]["pass"]
    report["needs_owner_feedback"] = not report["pass"]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"teacher: {best}; G1 {'PASS' if report['pass'] else 'FAIL (needs owner feedback; continuing)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
