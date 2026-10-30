"""LoRA training loop with validation-only selection and crash-safe resume (spec-lora-finetune D4–D8, D13).

CLI (all four runs; re-running the same command resumes after any interruption):
    python -m sfs.train.run --runs A-uniform,A-middle,B-uniform,B-middle --out runs/lora \
        --packed data/packed_tight --manifest manifest_v2_drive/manifest_v2.csv \
        --captions queries/train_captions.csv --phrasings queries/phrasings.csv

Per run directory:
    last.pt / prev.pt   full training state after the latest / previous epoch (atomic writes)
    best/adapter.pt     LoRA weights of the best validation epoch (the model that is used)
    train_log.csv       one row per epoch, flushed every epoch
    done.json           written when the run has stopped; a finished run is skipped
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

from sfs.train.data import (FRAME_MODES, EpochDraws, InstanceSampler, PackedCrops, TrainDataset,
                            preprocess_eval, to_pixels)

PROTOCOLS = {"A": "split", "B": "split_b"}
LOG_FIELDS = ["epoch", "steps", "train_loss", "val_loss", "val_map", "lr", "lr_factor",
              "since_best", "best_epoch", "seconds"]


@dataclass(frozen=True)
class TrainConfig:
    protocol: str
    frame_mode: str
    encoder: str = "siglip2"
    model_revision: str = "unpinned"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lr: float = 1e-4
    weight_decay: float = 0.01
    batch: int = 64
    max_epochs: int = 100
    warmup_epochs: int = 2
    plateau_patience: int = 2
    plateau_factor: float = 0.5
    stop_patience: int = 5
    min_delta: float = 1e-4
    val_min_instances: int = 5
    top_m: int = 3
    seed: int = 0
    max_steps_per_epoch: int = 0       # 0 = full epoch; >0 only for smoke runs
    captions_sha: str = ""
    phrasings_sha: str = ""
    store_id: str = ""

    def __post_init__(self):
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"protocol must be one of {sorted(PROTOCOLS)}")
        if self.frame_mode not in FRAME_MODES:
            raise ValueError(f"frame_mode must be one of {FRAME_MODES}")

    def hash(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def file_sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


# ---------------------------------------------------------------- learning-rate rule (D6, option 1)
def lr_at(cfg: TrainConfig, step: int, steps_per_epoch: int, factor: float) -> float:
    """Linear warm-up over `warmup_epochs`, then constant × plateau factor."""
    warm = max(1, cfg.warmup_epochs * steps_per_epoch)
    return cfg.lr * factor * min(1.0, (step + 1) / warm)


def after_epoch(cfg: TrainConfig, state: dict, val_map: float, epoch: int) -> bool:
    """Update best / patience / plateau factor. Returns True when this epoch is a new best."""
    improved = val_map > state["best_map"] + cfg.min_delta
    if improved:
        state.update(best_map=val_map, best_epoch=epoch, since_best=0)
    else:
        state["since_best"] += 1
        if state["since_best"] % cfg.plateau_patience == 0:
            state["factor"] *= cfg.plateau_factor
    return improved


def should_stop(cfg: TrainConfig, state: dict, epoch: int) -> str | None:
    if state["since_best"] >= cfg.stop_patience:
        return f"no validation improvement for {cfg.stop_patience} epochs"
    if epoch >= cfg.max_epochs:
        return "max epochs"
    return None


# ---------------------------------------------------------------- checkpoints (D13)
def atomic_save(obj, path: str) -> None:
    import torch
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    if os.path.exists(path):
        os.replace(path, path.replace("last.pt", "prev.pt"))
    os.replace(tmp, path)


def rng_states(sampler, cap_rng) -> dict:
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "sampler": sampler.rng.bit_generator.state, "captions": cap_rng.bit_generator.state}


def set_rng_states(s: dict, sampler, cap_rng) -> None:
    import torch
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    sampler.rng.bit_generator.state = s["sampler"]
    cap_rng.bit_generator.state = s["captions"]


def load_checkpoint(out_dir: str, config_hash: str):
    """The latest readable checkpoint (last.pt, else prev.pt), or None. Refuses a different config."""
    import torch
    for name in ("last.pt", "prev.pt"):
        p = os.path.join(out_dir, name)
        if not os.path.exists(p):
            continue
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue                       # unreadable (e.g. disk failure): fall back to prev.pt
        if ck["config_hash"] != config_hash:
            raise ValueError(f"{p} was written with a different training config "
                             f"({ck['config_hash']} != {config_hash}); use a new output directory")
        return ck
    return None


# ---------------------------------------------------------------- validation (D7)
class Validator:
    """Q1 category queries over the protocol's VALIDATION subset, frozen eval phrasings as queries."""

    def __init__(self, manifest, split_col, packed: PackedCrops, phrasings, query_emb, cap_proto,
                 classes, cfg: TrainConfig):
        from sfs.eval.queries import build_q1
        val = manifest[manifest[split_col] == "val"]
        counts = val.activity.value_counts()
        self.activities = sorted(a for a in phrasings if counts.get(a, 0) >= cfg.val_min_instances)
        self.queries = build_q1(manifest, split_col, self.activities, phrasings, subset="val")
        texts = [(a, j) for a in self.activities for j in range(len(phrasings[a]))]
        self.text_col = {t: c for c, t in enumerate(texts)}
        self.Q = np.stack([query_emb[a][j] for a, j in texts]).astype(np.float32)
        ids = sorted(int(i) for i in val.instance_id if len(packed.rows_of(i)))
        if len(ids) != len(val):
            raise ValueError(f"{len(val) - len(ids)} validation instances have no crops")
        rows = np.concatenate([packed.rows_of(i) for i in ids])
        self.inst = packed.instance_id[rows]
        self.pixels = np.stack([preprocess_eval(packed.image(int(r))) for r in rows])   # uint8 cache
        self.labels = np.array([classes.index(a) for a in
                                val.set_index("instance_id").loc[self.inst, "activity"]])
        self.cap_proto = cap_proto
        self.cfg = cfg

    def __call__(self, vision, scale, bias, device) -> dict:
        import torch
        from sfs.eval.baseline import evaluate
        from sfs.eval.retrieval import InstanceIndex
        from sfs.train.loss import label_aware_sigmoid_loss
        from sfs.train.lora import image_features

        vision.eval()
        embs = []
        with torch.no_grad(), _autocast(device):
            for s in range(0, len(self.pixels), 128):
                x = to_pixels(self.pixels[s:s + 128]).to(device)
                embs.append(image_features(vision, x).float().cpu())
        emb = torch.cat(embs)
        with torch.no_grad():
            loss = float(label_aware_sigmoid_loss(emb, torch.as_tensor(self.cap_proto), torch.as_tensor(
                self.labels), scale.detach().cpu().float(), bias.detach().cpu().float()))
        ids, first = np.unique(self.inst, return_index=True)
        idx = InstanceIndex(ids, emb.numpy(), np.append(first, len(self.inst)))
        df = evaluate(self.queries, idx.instance_ids, idx.score(self.Q, self.cfg.top_m), self.text_col)
        per_act = df.groupby("activity").ap.mean()
        return {"val_map": float(per_act.mean()), "val_loss": loss, "per_activity": per_act.round(4).to_dict()}


def _autocast(device: str):
    import torch
    return torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda"))


# ---------------------------------------------------------------- one run
def train(cfg: TrainConfig, vision, scale, bias, packed: PackedCrops, manifest, cap_emb: dict,
          phrasings: dict, query_emb: dict, out_dir: str, device: str = "cpu", workers: int = 0,
          log=print) -> dict:
    """Train one protocol / frame-mode run. `vision` must already carry LoRA adapters.

    cap_emb: activity -> (n_captions, D) L2-normalised training-caption embeddings (frozen text tower)
    query_emb: activity -> (n_phrasings, D) evaluation-phrasing embeddings (validation queries only)
    """
    import torch
    from torch.utils.data import DataLoader

    from sfs.train.loss import label_aware_sigmoid_loss
    from sfs.train.lora import (adapter_hash, adapter_state, image_features, load_adapter_state,
                                trainable_report)

    os.makedirs(out_dir, exist_ok=True)
    done_path = os.path.join(out_dir, "done.json")
    if os.path.exists(done_path):
        with open(done_path, encoding="utf-8") as f:
            done = json.load(f)
        if done["config_hash"] != cfg.hash():
            raise ValueError(f"{out_dir} holds a finished run with a different config")
        log(f"[{out_dir}] already finished — skipped")
        return done

    split_col = PROTOCOLS[cfg.protocol]
    classes = sorted(cap_emb)
    sampler = InstanceSampler(manifest, split_col, packed, cfg.frame_mode, cfg.seed)
    missing = sorted(set(sampler.activities) - set(classes))
    if missing:
        raise ValueError(f"training activities without captions: {missing}")
    labels = np.array([classes.index(a) for a in sampler.activities])
    cap_bank = [torch.as_tensor(np.asarray(cap_emb[a], np.float32)) for a in classes]
    proto = np.stack([_unit(np.asarray(cap_emb[a], np.float32).mean(0)) for a in classes])
    validator = Validator(manifest, split_col, packed, phrasings, query_emb, proto, classes, cfg)

    vision.to(device)
    scale = torch.nn.Parameter(scale.detach().clone().float().to(device))
    bias = torch.nn.Parameter(bias.detach().clone().float().to(device))
    lora_params = [p for n, p in vision.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora_params, "weight_decay": cfg.weight_decay},
                             {"params": [scale, bias], "weight_decay": 0.0}], lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    cap_rng = np.random.default_rng(cfg.seed + 1)
    steps_per_epoch = -(-len(sampler) // cfg.batch)
    if cfg.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, cfg.max_steps_per_epoch)

    state = {"epoch": 0, "step": 0, "best_map": -1.0, "best_epoch": 0, "since_best": 0, "factor": 1.0}
    log_path = os.path.join(out_dir, "train_log.csv")
    best_dir = os.path.join(out_dir, "best")
    os.makedirs(best_dir, exist_ok=True)

    ck = load_checkpoint(out_dir, cfg.hash())
    if ck is not None:
        load_adapter_state(vision, ck["adapter"])
        with torch.no_grad():
            scale.copy_(ck["scale"]), bias.copy_(ck["bias"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        best = ck["best"]
        set_rng_states(ck["rng"], sampler, cap_rng)
        _write_best(best_dir, best)          # the best of the checkpointed epochs, never a later orphan
        _truncate_log(log_path, state["epoch"])
        log(f"[{out_dir}] resumed after epoch {state['epoch']} (best {state['best_map']:.4f} "
            f"at epoch {state['best_epoch']})")
    else:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LOG_FIELDS)
        # epoch 0 = the untouched model (LoRA B = 0): the frozen baseline on validation.
        t0 = time.time()
        v = validator(vision, scale, bias, device)
        after_epoch(cfg, state, v["val_map"], 0)
        best = _best_payload(vision, scale, bias, 0, v, cfg)
        _write_best(best_dir, best)
        _append_log(log_path, {"epoch": 0, "steps": 0, "train_loss": "", **_vlog(v), "lr": 0.0,
                               "lr_factor": 1.0, "since_best": 0, "best_epoch": 0,
                               "seconds": round(time.time() - t0, 1)})
        atomic_save(_checkpoint(cfg, vision, scale, bias, opt, scaler, state, sampler, cap_rng, best),
                    os.path.join(out_dir, "last.pt"))
        log(f"[{out_dir}] epoch 0 (frozen) val mAP {v['val_map']:.4f}")

    draws_holder = EpochDraws()
    loader = DataLoader(TrainDataset(packed, labels), batch_size=cfg.batch, sampler=draws_holder,
                        num_workers=workers, persistent_workers=workers > 0,
                        pin_memory=device.startswith("cuda"))
    reason = should_stop(cfg, state, state["epoch"])
    while reason is None:
        epoch = state["epoch"] + 1
        t0 = time.time()
        draws = sampler.epoch()
        if cfg.max_steps_per_epoch:
            draws = draws[:cfg.max_steps_per_epoch * cfg.batch]
        draws_holder.set(draws)
        vision.train()
        losses = []
        for x, y in loader:
            lr = lr_at(cfg, state["step"], steps_per_epoch, state["factor"])
            for g in opt.param_groups:
                g["lr"] = lr
            pick = cap_rng.integers(0, [len(b) for b in cap_bank])
            caps = torch.stack([b[k] for b, k in zip(cap_bank, pick)]).to(device)
            with _autocast(device):
                feats = image_features(vision, to_pixels(x).to(device, non_blocking=True))
            loss = label_aware_sigmoid_loss(feats, caps, y.to(device), scale, bias)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, step {state['step']}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
            state["step"] += 1
        v = validator(vision, scale, bias, device)
        state["epoch"] = epoch
        if after_epoch(cfg, state, v["val_map"], epoch):
            best = _best_payload(vision, scale, bias, epoch, v, cfg)
            _write_best(best_dir, best)
        _append_log(log_path, {"epoch": epoch, "steps": state["step"],
                               "train_loss": round(float(np.mean(losses)), 5), **_vlog(v),
                               "lr": lr, "lr_factor": state["factor"], "since_best": state["since_best"],
                               "best_epoch": state["best_epoch"], "seconds": round(time.time() - t0, 1)})
        atomic_save(_checkpoint(cfg, vision, scale, bias, opt, scaler, state, sampler, cap_rng, best),
                    os.path.join(out_dir, "last.pt"))
        log(f"[{out_dir}] epoch {epoch}: loss {np.mean(losses):.4f} val mAP {v['val_map']:.4f} "
            f"(best {state['best_map']:.4f} @ {state['best_epoch']}) lr {lr:.2e} {time.time() - t0:.0f}s"
            + (f" peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if device.startswith("cuda") else ""))
        reason = should_stop(cfg, state, epoch)

    done = {"config": asdict(cfg), "config_hash": cfg.hash(), "stopped": reason,
            "epochs_run": state["epoch"], "best_epoch": state["best_epoch"],
            "best_val_map": state["best_map"], "frozen_val_map": _frozen_map(log_path),
            "adapter_hash": adapter_hash(best["adapter"]), "trainable": trainable_report(vision)}
    with open(done_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(done, f, indent=1)
    os.replace(done_path + ".tmp", done_path)
    log(f"[{out_dir}] finished: {reason}; best epoch {state['best_epoch']} val mAP {state['best_map']:.4f}")
    return done


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def _vlog(v):
    return {"val_loss": round(v["val_loss"], 5), "val_map": round(v["val_map"], 5)}


def _checkpoint(cfg, vision, scale, bias, opt, scaler, state, sampler, cap_rng, best) -> dict:
    from sfs.train.lora import adapter_state
    return {"config": asdict(cfg), "config_hash": cfg.hash(), "adapter": adapter_state(vision),
            "scale": scale.detach().cpu(), "bias": bias.detach().cpu(), "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(), "state": dict(state), "rng": rng_states(sampler, cap_rng),
            "best": best}


def _best_payload(vision, scale, bias, epoch, v, cfg) -> dict:
    from sfs.train.lora import adapter_state
    return {"adapter": adapter_state(vision), "scale": scale.detach().cpu().clone(),
            "bias": bias.detach().cpu().clone(), "epoch": epoch, "val_map": v["val_map"],
            "per_activity": v["per_activity"], "config": asdict(cfg)}


def _write_best(best_dir, best) -> None:
    import torch
    path = os.path.join(best_dir, "adapter.pt")
    torch.save(best, path + ".tmp")
    os.replace(path + ".tmp", path)


def _append_log(path, row) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, LOG_FIELDS).writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _truncate_log(path, last_epoch) -> None:
    """Drop log rows after the resumed checkpoint (an epoch that ran but was not checkpointed)."""
    if not os.path.exists(path):
        return
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, LOG_FIELDS)
        w.writeheader()
        w.writerows(r for r in rows if int(r["epoch"]) <= last_epoch)


def _frozen_map(log_path):
    with open(log_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["epoch"] == "0":
                return float(r["val_map"])
    return None


# ---------------------------------------------------------------- selection between frame modes (D4)
def select(out_root: str, runs: list[str]) -> dict:
    """Per protocol, the frame mode with the higher VALIDATION mAP. Eval is never read here."""
    sel = {}
    for proto in sorted({r.split("-")[0] for r in runs}):
        cands = {}
        for r in runs:
            if r.startswith(proto + "-"):
                with open(os.path.join(out_root, r, "done.json"), encoding="utf-8") as f:
                    cands[r] = json.load(f)["best_val_map"]
        best = max(sorted(cands), key=lambda r: cands[r])
        sel[proto] = {"selected": best, "val_map": cands}
    with open(os.path.join(out_root, "selection.json"), "w", encoding="utf-8") as f:
        json.dump(sel, f, indent=1)
    return sel


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    import torch

    from sfs.data.manifest import read_manifest
    from sfs.embed.encoder import ENCODERS, Encoder
    from sfs.eval.queries import load_phrasings
    from sfs.train.captions import disjointness_violations, load_captions
    from sfs.train.lora import load_vision_tower, trainable_report, wrap_lora

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default="A-uniform,A-middle,B-uniform,B-middle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--packed", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--captions", required=True)
    ap.add_argument("--phrasings", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--max-steps-per-epoch", type=int, default=0, help="smoke runs only")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = read_manifest(a.manifest)
    packed = PackedCrops(a.packed)
    phr = load_phrasings(a.phrasings)
    caps = load_captions(a.captions)
    bad = disjointness_violations(caps, phr)
    if bad:
        raise ValueError(f"training captions overlap the evaluation phrasings: {bad[:5]}")

    enc = Encoder("siglip2", device=device)            # frozen text tower, same revision as baseline
    flat = [(act, c) for act in sorted(caps) for c in caps[act]]
    E = enc.embed_texts([c for _, c in flat])
    cap_emb = {act: E[[i for i, (b, _) in enumerate(flat) if b == act]] for act in sorted(caps)}
    query_emb = {act: enc.embed_texts(phr[act]) for act in phr}
    revision = enc.revision
    del enc
    torch.cuda.empty_cache()

    runs = a.runs.split(",")
    for run in runs:
        proto, mode = run.split("-")
        cfg = TrainConfig(protocol=proto, frame_mode=mode, model_revision=revision,
                          max_epochs=a.max_epochs, max_steps_per_epoch=a.max_steps_per_epoch,
                          batch=a.batch, captions_sha=file_sha(a.captions),
                          phrasings_sha=file_sha(a.phrasings), store_id=packed.meta["store_id"])
        out_dir = os.path.join(a.out, run)
        if os.path.exists(os.path.join(out_dir, "done.json")):
            train(cfg, None, None, None, packed, m, cap_emb, phr, query_emb, out_dir)
            continue
        vision, scale, bias, rev = load_vision_tower(ENCODERS[cfg.encoder], revision)
        vision = wrap_lora(vision, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout)
        rep = trainable_report(vision)
        print(f"[{run}] trainable {rep['trainable']:,} of {rep['total']:,} ({rep['percent']} %), "
              f"{rep['adapted_modules']} adapted projections; config {cfg.hash()}")
        train(cfg, vision, scale, bias, packed, m, cap_emb, phr, query_emb, out_dir, device, a.workers,
              log=lambda msg: print(msg, flush=True))     # visible at once when redirected to a file
        del vision
        torch.cuda.empty_cache()
    if all(os.path.exists(os.path.join(a.out, r, "done.json")) for r in runs):
        print(json.dumps(select(a.out, runs), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
