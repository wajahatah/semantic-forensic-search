"""Student re-ranker training from human labels (GT), teacher scores (KD) or both
(spec-rerank-cascade D6/D7).

- Pair pool per protocol: for every training class, all its training instances (positives)
  plus negatives — half hard (highest LoRA score among other classes), half random. The
  teacher scores exactly these pairs (D5), so all three variants train on the same pool.
- Target = alpha * label + (1 - alpha) * sigmoid(teacher margin);  GT alpha=1, GTKD 0.5, KD 0.
- Validation (only): Q1 mAP over the protocol's validation subset with the frozen eval
  phrasings; the student score is fused with the LoRA score, z(student) + beta * z(LoRA), and
  beta is chosen from BETAS on validation. The LoRA score is never a student input.
- Crash-safe like LoRA training (D13): last.pt / prev.pt after every epoch, exact resume,
  resume refused if the config changed; best.pt; done.json.

CLI:
  python -m sfs.rerank.train pool  --protocol A --lora-emb ... --manifest ... --captions ... --out data/rerank/pool_A.csv
  python -m sfs.rerank.train train --protocol A --lora-run A-middle --variants GT,GTKD,KD ...
  --text-emb FILE: caption / phrasing embeddings exported by scripts/export_text_embeddings.py, for a
  machine without the SigLIP 2 weights (office server, D-045); same numbers as computing them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

LABEL_CAPTIONS = (0, 1)              # caption indices per class scored by the teacher (D5)
ALPHAS = {"GT": 1.0, "GTKD": 0.5, "KD": 0.0}
BETAS = (0.0, 0.25, 0.5, 1.0)
SPLIT_COL = {"A": "split", "B": "split_b"}
LOG_COLS = ["epoch", "train_loss", "val_map", "beta", "seconds"]


@dataclass(frozen=True)
class StudentConfig:
    protocol: str
    variant: str
    lora_run: str
    features_id: str
    teacher_id: str
    captions_sha: str
    phrasings_sha: str
    pool_sha: str
    d: int = 256
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    frame_drop: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch: int = 256
    max_epochs: int = 50
    patience: int = 5
    min_delta: float = 1e-4
    val_min_instances: int = 5
    top_m: int = 3
    seed: int = 0

    def __post_init__(self):
        if self.variant not in ALPHAS:
            raise ValueError(f"variant must be one of {sorted(ALPHAS)}")
        if self.variant != "GT" and not self.teacher_id:
            raise ValueError("KD variants need teacher scores")

    @property
    def alpha(self) -> float:
        return ALPHAS[self.variant]

    def hash(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def file_sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def zcols(s: np.ndarray) -> np.ndarray:
    """Per-query (column) z-score over the corpus, so scores from different models can be fused."""
    s = np.asarray(s, np.float64)
    return (s - s.mean(0)) / np.maximum(s.std(0), 1e-9)


# ---------------------------------------------------------------- pair pool
def build_pool(manifest, split_col: str, classes: list[str], cap_proto: np.ndarray, lora_shard: dict,
               n_hard: int = 100, n_rand: int = 100, seed: int = 0) -> pd.DataFrame:
    from sfs.eval.retrieval import InstanceIndex
    train = manifest[manifest[split_col] == "train"]
    idx = InstanceIndex.from_shard(lora_shard, "tight", set(train.instance_id.astype(int)))
    if len(idx.instance_ids) != len(train):
        raise ValueError("some training instances lack LoRA embeddings")
    act = train.set_index(train.instance_id.astype(np.int64)).loc[idx.instance_ids, "activity"].to_numpy()
    scores = idx.score(cap_proto, 3)
    rng = np.random.default_rng(seed)
    rows = []
    for c, cls in enumerate(classes):
        pos = np.flatnonzero(act == cls)
        if not len(pos):
            continue
        neg = np.flatnonzero(act != cls)
        hard = neg[np.argsort(-scores[neg, c], kind="stable")[:n_hard]]
        rest = np.setdiff1d(neg, hard)
        rand = rng.choice(rest, size=min(n_rand, len(rest)), replace=False)
        rows += [(int(idx.instance_ids[i]), c, 1) for i in pos]
        rows += [(int(idx.instance_ids[i]), c, 0) for i in np.concatenate([hard, np.sort(rand)])]
    return pd.DataFrame(rows, columns=["instance_id", "cls", "y"]).drop_duplicates(["instance_id", "cls"])


def teacher_requests(pool: pd.DataFrame, classes: list[str], captions: dict[str, list[str]]) -> list[tuple[int, str]]:
    """(instance_id, caption text) pairs the teacher must score for this pool."""
    out = {(int(r.instance_id), captions[classes[r.cls]][j]) for r in pool.itertuples() for j in LABEL_CAPTIONS}
    return sorted(out)


# ---------------------------------------------------------------- scoring
def score_matrix(model, feats: np.ndarray, mask: np.ndarray, rows: np.ndarray, text_emb: np.ndarray,
                 device: str, batch: int = 2048) -> np.ndarray:
    """(len(rows), len(text_emb)) student logits: every listed instance against every text."""
    import torch
    model.eval()
    out = np.empty((len(rows), len(text_emb)), np.float32)
    x_all = torch.from_numpy(feats[rows]).float()
    m_all = torch.from_numpy(mask[rows])
    t_all = torch.from_numpy(np.asarray(text_emb, np.float32))
    with torch.no_grad():
        for j in range(len(text_emb)):
            for s in range(0, len(rows), batch):
                x, m = x_all[s:s + batch].to(device), m_all[s:s + batch].to(device)
                t = t_all[j].to(device).expand(len(x), -1)
                out[s:s + batch, j] = model(x, m, t).float().cpu().numpy()
    return out


class StudentValidator:
    """Q1 over the protocol's validation subset (frozen eval phrasings), student fused with LoRA."""

    def __init__(self, manifest, split_col: str, phrasings: dict, phr_emb: dict, lora_shard: dict,
                 feat_index: dict, cfg: StudentConfig):
        from sfs.eval.queries import build_q1
        from sfs.eval.retrieval import InstanceIndex
        val = manifest[manifest[split_col] == "val"]
        counts = val.activity.value_counts()
        self.activities = sorted(a for a in phrasings if counts.get(a, 0) >= cfg.val_min_instances)
        self.queries = build_q1(manifest, split_col, self.activities, phrasings, subset="val")
        texts = [(a, j) for a in self.activities for j in range(len(phrasings[a]))]
        self.text_col = {t: c for c, t in enumerate(texts)}
        self.Q = np.stack([phr_emb[a][j] for a, j in texts]).astype(np.float32)
        idx = InstanceIndex.from_shard(lora_shard, "tight", set(val.instance_id.astype(int)))
        if len(idx.instance_ids) != len(val):
            raise ValueError("some validation instances lack LoRA embeddings")
        self.ids = idx.instance_ids
        self.lora = idx.score(self.Q, cfg.top_m)
        self.rows = np.array([feat_index[int(i)] for i in self.ids])
        self.lora_map = self._map(self.lora)

    def _map(self, scores: np.ndarray) -> tuple[float, dict]:
        from sfs.eval.baseline import evaluate
        per = evaluate(self.queries, self.ids, scores, self.text_col).groupby("activity").ap.mean()
        return float(per.mean()), per.round(4).to_dict()

    def __call__(self, model, feats, mask, device) -> dict:
        s = score_matrix(model, feats, mask, self.rows, self.Q, device)
        best = None
        for beta in BETAS:
            m, per = self._map(zcols(s) + beta * zcols(self.lora))
            if best is None or m > best["val_map"] + 1e-12:
                best = {"val_map": m, "beta": beta, "per_activity": per}
        return best


# ---------------------------------------------------------------- training
def _rng_states(rng) -> dict:
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "gen": rng.bit_generator.state}


def _set_rng_states(s: dict, rng) -> None:
    import torch
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    rng.bit_generator.state = s["gen"]


def _append_log(path: str, row: dict) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, LOG_COLS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _truncate_log(path: str, last_epoch: int) -> None:
    if not os.path.exists(path):
        return
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if int(r["epoch"]) <= last_epoch]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, LOG_COLS)
        w.writeheader()
        w.writerows(rows)


def train_student(cfg: StudentConfig, feats: dict, pool: pd.DataFrame, cap_emb: np.ndarray,
                  teacher: dict | None, captions: dict, classes: list[str], validator: StudentValidator,
                  out_dir: str, device: str, log=print) -> dict:
    """cap_emb: (n_classes, n_captions, 768). teacher: {(instance_id, caption text): margin}."""
    import torch
    from sfs.rerank.model import TemporalReranker, n_params
    from sfs.train.run import atomic_save, load_checkpoint

    os.makedirs(out_dir, exist_ok=True)
    done_path = os.path.join(out_dir, "done.json")
    if os.path.exists(done_path):
        done = json.load(open(done_path))
        if done["config_hash"] != cfg.hash():
            raise ValueError(f"{out_dir} holds a finished run with a different config")
        log(f"[{out_dir}] already finished (best epoch {done['best_epoch']}); skipped")
        return done

    feat_index = {int(i): r for r, i in enumerate(feats["instance_id"])}
    X, M = feats["feats"], feats["mask"]
    rows = np.array([feat_index[int(i)] for i in pool.instance_id])
    cls = pool.cls.to_numpy()
    y = pool.y.to_numpy().astype(np.float32)
    margins = np.full((len(pool), len(LABEL_CAPTIONS)), np.nan, np.float32)
    if cfg.alpha < 1.0:
        for k, r in enumerate(pool.itertuples()):
            for j, ci in enumerate(LABEL_CAPTIONS):
                margins[k, j] = teacher[(int(r.instance_id), captions[classes[r.cls]][ci])]
    soft = 1.0 / (1.0 + np.exp(-margins))

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = TemporalReranker(X.shape[-1], cap_emb.shape[-1], cfg.d, cfg.layers, cfg.heads, X.shape[1],
                             cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    state = {"epoch": 0, "best_map": -1.0, "best_epoch": 0, "since_best": 0}
    log_path = os.path.join(out_dir, "train_log.csv")
    ck = load_checkpoint(out_dir, cfg.hash())
    if ck:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        _set_rng_states(ck["rng"], rng)
        _truncate_log(log_path, state["epoch"])
        log(f"[{out_dir}] resumed after epoch {state['epoch']}")
    else:
        _truncate_log(log_path, -1)
        log(f"[{out_dir}] {cfg.variant} student, {n_params(model):,} parameters, {len(pool):,} pairs")

    X_t, M_t = torch.from_numpy(X), torch.from_numpy(M)
    caps = torch.from_numpy(np.asarray(cap_emb, np.float32))
    lab_idx = torch.tensor(LABEL_CAPTIONS)
    bce = torch.nn.BCEWithLogitsLoss()
    stopped = "max_epochs"
    for epoch in range(state["epoch"] + 1, cfg.max_epochs + 1):
        t0 = time.time()
        model.train()
        perm = rng.permutation(len(pool))
        pick = rng.integers(0, len(LABEL_CAPTIONS), len(pool))
        total, n = 0.0, 0
        for s in range(0, len(perm), cfg.batch):
            b = perm[s:s + cfg.batch]
            x = X_t[rows[b]].to(device).float()
            m = M_t[rows[b]].to(device)
            keep = (torch.rand(m.shape, device=device) >= cfg.frame_drop) & m
            empty = keep.sum(1) == 0
            keep[empty] = m[empty]                      # never drop every frame
            t = caps[torch.from_numpy(cls[b]), lab_idx[torch.from_numpy(pick[b])]].to(device)
            target = cfg.alpha * y[b]
            if cfg.alpha < 1.0:
                target = target + (1 - cfg.alpha) * soft[b, pick[b]]
            loss = bce(model(x, keep, t), torch.from_numpy(target.astype(np.float32)).to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(b)
            n += len(b)
        v = validator(model, X, M, device)
        improved = v["val_map"] > state["best_map"] + cfg.min_delta
        state.update(epoch=epoch)
        if improved:
            state.update(best_map=v["val_map"], best_epoch=epoch, since_best=0)
            atomic_save({"model": model.state_dict(), "beta": v["beta"], "epoch": epoch, "val_map": v["val_map"],
                         "per_activity": v["per_activity"], "config": asdict(cfg)},
                        os.path.join(out_dir, "best.pt"))
        else:
            state["since_best"] += 1
        _append_log(log_path, {"epoch": epoch, "train_loss": round(total / n, 5), "val_map": round(v["val_map"], 5),
                               "beta": v["beta"], "seconds": round(time.time() - t0, 1)})
        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                     "rng": _rng_states(rng), "config_hash": cfg.hash()}, os.path.join(out_dir, "last.pt"))
        log(f"[{out_dir}] epoch {epoch}: loss {total / n:.4f} val mAP {v['val_map']:.4f} (beta {v['beta']}) "
            f"best {state['best_map']:.4f} @ {state['best_epoch']}")
        if state["since_best"] >= cfg.patience:
            stopped = f"no validation improvement for {cfg.patience} epochs"
            break
    best = torch.load(os.path.join(out_dir, "best.pt"), map_location="cpu", weights_only=False)
    done = {"config": asdict(cfg), "config_hash": cfg.hash(), "stopped": stopped, "epochs_run": state["epoch"],
            "best_epoch": best["epoch"], "best_val_map": best["val_map"], "beta": best["beta"],
            "lora_only_val_map": validator.lora_map[0], "n_pairs": int(len(pool)), "params": n_params(model)}
    json.dump(done, open(done_path, "w"), indent=1)
    log(f"[{out_dir}] finished: {stopped}; best epoch {best['epoch']} val mAP {best['val_map']:.4f} "
        f"(LoRA alone {validator.lora_map[0]:.4f})")
    return done


def select(out_root: str, protocol: str, variants: list[str]) -> dict:
    """Per protocol, the variant with the highest validation mAP (written before eval is read)."""
    path = os.path.join(out_root, "selection.json")
    sel = json.load(open(path)) if os.path.exists(path) else {}
    vals = {v: json.load(open(os.path.join(out_root, f"{protocol}-{v}", "done.json")))["best_val_map"]
            for v in variants}
    sel[protocol] = {"selected": f"{protocol}-{max(vals, key=vals.get)}", "val_map": vals}
    json.dump(sel, open(path, "w"), indent=1)
    return sel


# ---------------------------------------------------------------- CLI
def _text_embeddings(texts: list[str], cache: str | None = None) -> np.ndarray:
    """SigLIP 2 text embeddings, in order. `cache`: an .npz from scripts/export_text_embeddings.py holding
    groups `texts_<i>` / `emb_<i>`, each embedded in one call as training does (GPU results depend
    slightly on the batch, so a group is only used for exactly the same list of texts)."""
    if cache:
        with np.load(cache, allow_pickle=False) as z:
            for i in range(sum(k.startswith("texts_") for k in z.files)):
                if z[f"texts_{i}"].tolist() == list(texts):
                    return z[f"emb_{i}"].astype(np.float32)
        raise KeyError(f"{cache} holds no group with exactly these {len(texts)} texts")
    from sfs.embed.encoder import Encoder
    return np.asarray(Encoder("siglip2").embed_texts(texts), np.float32)


def main(argv=None) -> int:
    import torch
    from sfs.data.manifest import read_manifest
    from sfs.eval.queries import load_phrasings
    from sfs.rerank.features import load_features
    from sfs.store.shards import read_shard
    from sfs.train.captions import load_captions

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["pool", "train"])
    ap.add_argument("--protocol", required=True, choices=["A", "B"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--captions", default="queries/train_captions.csv")
    ap.add_argument("--phrasings", default="queries/phrasings.csv")
    ap.add_argument("--lora-emb", required=True, help="re-embedded tight embeddings of the protocol's LoRA run")
    ap.add_argument("--lora-run")
    ap.add_argument("--features")
    ap.add_argument("--pool")
    ap.add_argument("--teacher", help="teacher scores (.npz from sfs.teacher.run)")
    ap.add_argument("--variants", default="GT,GTKD,KD")
    ap.add_argument("--text-emb", help="exported caption / phrasing embeddings (instead of loading SigLIP 2)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    log = lambda msg: print(msg, flush=True)  # noqa: E731

    m = read_manifest(a.manifest)
    captions = load_captions(a.captions)
    classes = sorted(captions)
    lora = read_shard(a.lora_emb)
    if a.cmd == "pool":
        cap = _text_embeddings([t for c in classes for t in captions[c]], a.text_emb).reshape(len(classes), -1, 768)
        proto = cap.mean(1)
        proto /= np.linalg.norm(proto, axis=1, keepdims=True)
        pool = build_pool(m, SPLIT_COL[a.protocol], classes, proto, lora)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        pool.to_csv(a.out, index=False)
        log(f"wrote {a.out}: {len(pool):,} pairs ({int(pool.y.sum()):,} positive), "
            f"{len(teacher_requests(pool, classes, captions)):,} teacher requests")
        return 0

    from sfs.teacher.store import load_scores
    phr = load_phrasings(a.phrasings)
    feats = load_features(a.features)
    pool = pd.read_csv(a.pool)
    cap = _text_embeddings([t for c in classes for t in captions[c]], a.text_emb).reshape(len(classes), -1, 768)
    acts = sorted(phr)
    pe = _text_embeddings([t for x in acts for t in phr[x]], a.text_emb)
    phr_emb, k = {}, 0
    for x in acts:
        phr_emb[x] = pe[k:k + len(phr[x])]
        k += len(phr[x])
    teacher, teacher_id = (load_scores(a.teacher) if a.teacher else ({}, ""))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_index = {int(i): r for r, i in enumerate(feats["instance_id"])}
    variants = a.variants.split(",")
    for variant in variants:
        cfg = StudentConfig(protocol=a.protocol, variant=variant, lora_run=a.lora_run,
                            features_id=feats["meta"]["lora_encoder_id"],
                            teacher_id="" if variant == "GT" else teacher_id,
                            captions_sha=file_sha(a.captions), phrasings_sha=file_sha(a.phrasings),
                            pool_sha=file_sha(a.pool))
        validator = StudentValidator(m, SPLIT_COL[a.protocol], phr, phr_emb, lora, feat_index, cfg)
        train_student(cfg, feats, pool, cap, teacher, captions, classes, validator,
                      os.path.join(a.out, f"{a.protocol}-{variant}"), device, log)
    log(json.dumps(select(a.out, a.protocol, variants)[a.protocol]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
