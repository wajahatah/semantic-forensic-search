"""Resumable VLM scoring job (spec-rerank-cascade D5): score every requested (instance, text) pair.

Requests are grouped by instance and visited clip by clip (one crop shard in memory at a time).
- instruct model: one video pass per instance, all its texts from the cached state;
- reranker: one full pass per pair (its format puts the query first).
Results are written every `part_size` instances to <out>/parts/part_XXXXX.npz; re-running the
same command skips finished parts and consolidates to <out>/scores.npz. The identity (model,
revision, view, frames, prompt, crop store, request-file hash, subset) is checked on resume.

CLI:
  python -m sfs.teacher.run --model data/models/Qwen3-VL-2B-Instruct --kind instruct --view context \
      --requests data/teacher/requests_g1.csv --subset val --out data/teacher/g1/instruct-context
Office server (spec-vlm-followups E4): `--remote http://127.0.0.1:8765` instead of `--model` scores
through the SSH tunnel with `sfs.teacher.remote.RemoteScorer`; the model id comes from the server.
Shared GPU (D-037): `--reserve-gb N` holds GPU memory up to N GB for the whole job (PyTorch keeps the
freed block cached), so a model another user loads mid-run cannot take memory the job still needs.
It does not touch other processes; they use whatever is left.
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

from sfs.teacher.frames import N_FRAMES, CropIndex
from sfs.teacher.store import read_requests, read_scores, save_scores


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def group_requests(pairs: list[tuple[int, str]], crops: CropIndex) -> list[tuple[int, list[str]]]:
    by_inst: dict[int, list[str]] = {}
    for i, t in pairs:
        by_inst.setdefault(int(i), []).append(t)
    order = sorted(by_inst, key=lambda i: (crops.shard_of(i), i))
    return [(i, sorted(set(by_inst[i]))) for i in order]


def score_instance(scorer, crops: CropIndex, iid: int, texts: list[str], view: str, n_frames: int):
    frames, nums = crops.frames(iid, view, n_frames)
    fps = crops.fps_of(iid)
    if scorer.kind == "instruct":
        return scorer.score_clip(frames, nums, fps, texts)
    return scorer.score_full(frames, nums, fps, texts)


def run(scorer, crops: CropIndex, requests_path: str, view: str, out_dir: str, subset: str,
        n_frames: int = N_FRAMES, part_size: int = 100, log=print) -> str:
    from sfs.teacher.vlm import FRAME_SIDE, INSTRUCT_PROMPT, RERANK_INSTRUCTION

    pairs = read_requests(requests_path)
    groups = group_requests(pairs, crops)
    meta = {"model_id": scorer.model_id, "kind": scorer.kind, "view": view, "n_frames": n_frames,
            "frame_side": FRAME_SIDE, "prompt": INSTRUCT_PROMPT if scorer.kind == "instruct" else RERANK_INSTRUCTION,
            "store_id": crops.store_id, "requests_sha": _sha(requests_path), "subset": subset}
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)
    ident = os.path.join(out_dir, "identity.json")
    if os.path.exists(ident):
        if json.load(open(ident)) != meta:
            raise ValueError(f"{out_dir} was started with a different configuration; use a new directory")
    else:
        json.dump(meta, open(ident, "w"), indent=1)
    final = os.path.join(out_dir, "scores.npz")
    if os.path.exists(final):
        log(f"[{out_dir}] complete; skipped")
        return final

    n_parts = (len(groups) + part_size - 1) // part_size
    t_start, done_inst = time.time(), 0
    for p in range(n_parts):
        path = os.path.join(out_dir, "parts", f"part_{p:05d}.npz")
        if os.path.exists(path):
            continue
        ids, texts, margins, secs = [], [], [], []
        for iid, tx in groups[p * part_size:(p + 1) * part_size]:
            scorer.sync()
            t0 = time.perf_counter()
            m = score_instance(scorer, crops, iid, tx, view, n_frames)
            scorer.sync()
            secs.append(time.perf_counter() - t0)
            ids += [iid] * len(tx)
            texts += tx
            margins += [float(x) for x in m]
        save_scores(path, ids, texts, margins, {**meta, "part": p, "seconds_per_instance": secs,
                                                "texts_per_instance": [len(t) for _, t in
                                                                       groups[p * part_size:(p + 1) * part_size]]})
        done_inst += len(secs)
        rate = done_inst / (time.time() - t_start)
        left = sum(1 for q in range(p + 1, n_parts)
                   if not os.path.exists(os.path.join(out_dir, "parts", f"part_{q:05d}.npz")))
        log(f"[{out_dir}] part {p + 1}/{n_parts}: {np.mean(secs):.2f} s/instance, "
            f"~{left * part_size / max(rate, 1e-9) / 60:.0f} min left", )

    ids, texts, margins, secs, ntext = [], [], [], [], []
    for path in sorted(glob.glob(os.path.join(out_dir, "parts", "part_*.npz"))):
        i, t, m, pm = read_scores(path)
        if any(pm.get(k) != v for k, v in meta.items()):
            raise ValueError(f"{path} has a different identity")
        ids.append(i)
        texts += t
        margins.append(m)
        secs += pm["seconds_per_instance"]
        ntext += pm["texts_per_instance"]
    if len(secs) != len(groups):
        raise ValueError(f"{len(secs)} of {len(groups)} instances scored; re-run to finish")
    save_scores(final, np.concatenate(ids), texts, np.concatenate(margins),
                {**meta, "instances": len(secs), "pairs": len(texts),
                 "mean_seconds_per_instance": float(np.mean(secs)),
                 "mean_texts_per_instance": float(np.mean(ntext))})
    log(f"[{out_dir}] wrote {final}: {len(texts):,} pairs over {len(secs):,} instances, "
        f"{np.mean(secs):.2f} s/instance")
    return final


def reserve_gpu(total_gb: float) -> float:
    """Grow this process's cached GPU memory to `total_gb` (never past what is free, minus 256 MB) and
    keep it: the block is freed into PyTorch's cache, not returned to the driver. Returns reserved GB."""
    import torch
    if total_gb <= 0 or not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    want = min(int(total_gb * 1e9) - torch.cuda.memory_reserved(), free - 256 * 2 ** 20)
    if want > 0:
        block = torch.empty(want, dtype=torch.uint8, device="cuda")
        del block
    return torch.cuda.memory_reserved() / 1e9


def make_scorer(model: str | None, remote: str | None, kind: str, gpu_budget_gb: float | None = None,
                fp8_at_load: bool = False, clip_batch: int | None = None, pin: bool | None = None):
    """Local scorer (Qwen3-VL or Gemma 4, by the model's config), or RemoteScorer whose server must
    serve a model of `kind`."""
    if remote:
        from sfs.teacher.remote import RemoteScorer
        scorer = RemoteScorer(remote, log=lambda m: print(m, flush=True))
        if scorer.kind != kind:
            raise ValueError(f"--kind {kind} but the server serves a {scorer.kind} model ({scorer.model_id})")
        return scorer
    from sfs.teacher.vlm import load_scorer
    scorer = load_scorer(model, kind, gpu_budget_gb=gpu_budget_gb, fp8_at_load=fp8_at_load, pin=pin)
    if clip_batch:
        scorer.clip_batch = clip_batch
    return scorer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="local weights directory")
    src.add_argument("--remote", help="scoring service URL, e.g. http://127.0.0.1:8765 (SSH tunnel)")
    ap.add_argument("--kind", required=True, choices=["instruct", "reranker"])
    ap.add_argument("--view", required=True, choices=["tight", "context"])
    ap.add_argument("--requests", required=True)
    ap.add_argument("--subset", required=True, choices=["trainval", "val", "eval"],
                    help="what the requests cover; eval scores can only be read by the cascade report")
    ap.add_argument("--crops", default="data/crops_v2")
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--gpu-budget-gb", type=float, help="Gemma 4: GPU memory for weights (rest kept in CPU memory)")
    ap.add_argument("--reserve-gb", type=float, default=0.0, help="hold this much GPU memory for the whole job")
    ap.add_argument("--fp8-at-load", action="store_true", help="Qwen-VL: language model in FP8 (E7, Qwen2.5-VL-7B)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    scorer = make_scorer(a.model, a.remote, a.kind, a.gpu_budget_gb, a.fp8_at_load)
    if a.reserve_gb and not a.remote:
        print(f"holding {reserve_gpu(a.reserve_gb):.1f} GB of GPU memory for this job", flush=True)
    run(scorer, CropIndex(a.crops), a.requests, a.view, a.out, a.subset, a.frames,
        log=lambda m: print(m, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
