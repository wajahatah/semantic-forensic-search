"""Text-drift check for the image + text LoRA (spec-lora-image-text §6 R2, D-054).

Does the adapted text side keep the relations between general wordings it was never trained on
(appearance, places, vehicles, ...)? For each adapter, the frozen sentences' embeddings are compared
with the adapted ones:
- rho:      Spearman correlation of the pairwise cosine similarities (the pass rule: >= RHO_PASS),
- self_cos: mean cosine between a sentence's frozen and adapted embedding,
- nn_agree: share of sentences whose nearest other sentence is the same under both.
The same numbers on the eval phrasings are given for scale. A guard against collapse, not accuracy.

    python scripts/text_drift_check.py --adapters A0=runs/lora_it/A-middle/best/adapter.pt,... \
        --sentences queries/appearance_sentences.csv --phrasings queries/phrasings.csv --out drift.json
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

RHO_PASS = 0.90
MODEL_ID = "google/siglip2-base-patch16-224"
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank), like scipy.stats.rankdata."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return sums[inv] / counts[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(_ranks(np.asarray(a, np.float64)), _ranks(np.asarray(b, np.float64)))[0, 1])


def drift(E0: np.ndarray, E1: np.ndarray) -> dict:
    """E0 frozen, E1 adapted; rows L2-normalised, same order."""
    iu = np.triu_indices(len(E0), 1)
    S0, S1 = E0 @ E0.T, E1 @ E1.T
    np.fill_diagonal(S0, -np.inf)
    np.fill_diagonal(S1, -np.inf)
    return {"rho": round(spearman(S0[iu], S1[iu]), 4),
            "self_cos": round(float((E0 * E1).sum(1).mean()), 4),
            "nn_agree": round(float((S0.argmax(1) == S1.argmax(1)).mean()), 4)}


def embed(text, tokens: dict) -> np.ndarray:
    import torch
    from sfs.train.lora import text_features
    text.eval()
    with torch.no_grad():
        return text_features(text, tokens).float().numpy()


def run(text, adapters: dict[str, dict], sets: dict[str, dict]) -> dict:
    """text: frozen text tower; adapters: name -> adapter payload (with "text_adapter", "config");
    sets: name -> tokens. Returns name -> set -> drift numbers."""
    from sfs.train.lora import load_adapter_state, wrap_lora
    frozen = {s: embed(text, t) for s, t in sets.items()}
    cfgs = {(p["config"]["lora_rank"], p["config"]["lora_alpha"]) for p in adapters.values()}
    if len(cfgs) != 1:
        raise ValueError(f"adapters differ in rank/alpha: {cfgs}")
    rank, alpha = cfgs.pop()
    wrapped = wrap_lora(text, rank, alpha, 0.0)          # dropout is off in eval mode anyway
    out = {}
    for name, payload in adapters.items():
        load_adapter_state(wrapped, payload["text_adapter"])
        out[name] = {s: drift(frozen[s], embed(wrapped, t)) for s, t in sets.items()}
        out[name]["pass"] = out[name]["sentences"]["rho"] >= RHO_PASS
    return out


def main(argv=None) -> int:
    import pandas as pd
    import torch
    from sfs.eval.queries import load_phrasings
    from sfs.train.lora import load_text_tower, tokenize

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapters", required=True, help="name=path,... (image + text runs' best/adapter.pt)")
    ap.add_argument("--sentences", default="queries/appearance_sentences.csv")
    ap.add_argument("--phrasings", default="queries/phrasings.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    sent = pd.read_csv(a.sentences, comment="#")
    phr = load_phrasings(a.phrasings)
    phr_list = [p for act in sorted(phr) for p in phr[act]]
    text, processor = load_text_tower(MODEL_ID, REVISION)
    sets = {"sentences": tokenize(processor, list(sent.sentence)), "phrasings": tokenize(processor, phr_list)}
    adapters, paths = {}, {}
    for kv in a.adapters.split(","):
        name, path = kv.split("=", 1)
        paths[name] = path
        adapters[name] = torch.load(path, map_location="cpu", weights_only=False)
        if "text_adapter" not in adapters[name]:
            raise ValueError(f"{path} has no text adapter (image-only run?)")
    res = run(text, adapters, sets)
    for name in res:
        res[name]["val_map"] = adapters[name]["val_map"]
    with open(a.sentences, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    out = {"rule": f"pass = sentences rho >= {RHO_PASS} for every adapter", "sentences_sha256": sha,
           "n_sentences": len(sent), "n_phrasings": len(phr_list), "model": MODEL_ID, "revision": REVISION,
           "adapters": paths,
           "results": res, "all_pass": all(r["pass"] for r in res.values())}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"{'adapter':<6} {'rho':>6} {'self_cos':>8} {'nn':>5} | phrasings {'rho':>6} {'self_cos':>8} {'nn':>5}  pass")
    for name, r in res.items():
        s, p = r["sentences"], r["phrasings"]
        print(f"{name:<6} {s['rho']:6.3f} {s['self_cos']:8.3f} {s['nn_agree']:5.2f} | phrasings "
              f"{p['rho']:6.3f} {p['self_cos']:8.3f} {p['nn_agree']:5.2f}  {r['pass']}")
    print(f"all pass: {out['all_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
