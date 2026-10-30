"""Frozen-encoder baseline evaluation (spec-crops-queries-baseline §5). Runs locally.

CLI:
    python -m sfs.eval.baseline --manifest manifest_v2.csv --embeddings embeddings_v2_siglip2.npz \
        --phrasings queries/phrasings.csv --out results/02_baseline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from sfs.eval.metrics import (average_precision, precision_at_k, r_precision, rank,
                              recall_at_k, reciprocal_rank, success_at_k)
from sfs.eval.queries import (build_q1, build_q2, check_phrasings, eligible_activities,
                              load_phrasings)
from sfs.eval.retrieval import InstanceIndex

PROTOCOLS = {"A": "split", "B": "split_b"}
METRIC_COLS = ["ap", "p10", "rprec", "r1", "r5", "r10", "rr", "s1", "s5", "s10"]


def score_query(rel: np.ndarray) -> dict:
    return {"ap": average_precision(rel), "p10": precision_at_k(rel, 10), "rprec": r_precision(rel),
            "r1": recall_at_k(rel, 1), "r5": recall_at_k(rel, 5), "r10": recall_at_k(rel, 10),
            "rr": reciprocal_rank(rel), "s1": success_at_k(rel, 1), "s5": success_at_k(rel, 5),
            "s10": success_at_k(rel, 10)}


def evaluate(queries, instance_ids: np.ndarray, scores: np.ndarray, text_col: dict) -> pd.DataFrame:
    """scores: (n_inst, n_texts) aligned with instance_ids; text_col: (activity, phrasing_idx) -> col."""
    pos = {int(i): r for r, i in enumerate(instance_ids)}
    rows = []
    for q in queries:
        ids = np.array(sorted(q.corpus), np.int64)
        sc = scores[[pos[i] for i in ids], text_col[(q.activity, q.phrasing_idx)]]
        rel = np.isin(rank(ids, sc), np.fromiter(q.relevant, np.int64))
        rows.append({"qid": q.qid, "kind": q.kind, "activity": q.activity,
                     "phrasing_idx": q.phrasing_idx, "n_rel": len(q.relevant),
                     "n_corpus": len(ids), **score_query(rel)})
    return pd.DataFrame(rows)


def random_scores(n_inst: int, n_texts: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((n_inst, n_texts), dtype=np.float32)


def summarise(df: pd.DataFrame) -> dict:
    """Macro = mean over activities of the per-activity mean (each activity counts equally)."""
    out = {}
    for kind, g in df.groupby("kind"):
        per_act = g.groupby("activity")[METRIC_COLS].mean()
        out[kind] = {"queries": len(g), "activities": len(per_act),
                     "macro": per_act.mean().round(4).to_dict(),
                     "query_mean": g[METRIC_COLS].mean().round(4).to_dict(),
                     "n_rel_median": float(g.n_rel.median()), "n_corpus_median": float(g.n_corpus.median())}
    return out


def measure_latency(encoder, index: InstanceIndex, text: str, reps: int = 20) -> dict:
    encoder.embed_texts([text])  # warm-up
    t_enc, t_score = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        q = encoder.embed_texts([text])
        t1 = time.perf_counter()
        s = index.score(q)
        rank(index.instance_ids, s[:, 0])
        t2 = time.perf_counter()
        t_enc.append(t1 - t0)
        t_score.append(t2 - t1)
    return {"text_encode_ms": round(1000 * float(np.median(t_enc)), 2),
            "score_and_rank_ms": round(1000 * float(np.median(t_score)), 2),
            "corpus_instances": len(index.instance_ids), "model_calls_per_query": 1,
            "vlm_calls_per_query": 0}


def main(argv=None) -> int:
    from sfs.data.manifest import read_manifest
    from sfs.embed.encoder import Encoder
    from sfs.store.shards import read_shard

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--phrasings", required=True)
    ap.add_argument("--encoder", default="siglip2")
    ap.add_argument("--views", default="tight,full")
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--random-seeds", type=int, default=20)
    ap.add_argument("--protocols", default="A,B",
                    help="an adapted model is evaluated only on its own protocol (spec-lora-finetune D8)")
    ap.add_argument("--out", required=True, help="output prefix; writes <out>.json and <out>_queries.csv")
    a = ap.parse_args(argv)

    m = read_manifest(a.manifest)
    acts = eligible_activities(m, "split")
    phr = load_phrasings(a.phrasings)
    check_phrasings(phr, acts)
    shard = read_shard(a.embeddings)
    if shard["meta"].get("encoder") != a.encoder:
        raise ValueError(f"embeddings are from {shard['meta'].get('encoder')!r}, not {a.encoder!r}")

    texts = [(act, j, t) for act in acts for j, t in enumerate(phr[act])]
    text_col = {(act, j): c for c, (act, j, _) in enumerate(texts)}
    enc = Encoder(a.encoder)
    T = enc.embed_texts([t for _, _, t in texts])

    result = {"encoder": a.encoder, "encoder_id": shard["meta"]["encoder_id"], "m": a.m,
              "activities": acts, "n_phrasings": len(texts), "protocols": {}}
    all_rows = []
    for proto in a.protocols.split(","):
        col = PROTOCOLS[proto]
        corpus = set(m.loc[m[col] == "eval", "instance_id"].astype(int))
        queries = build_q1(m, col, acts, phr) + build_q2(m, col, acts, phr)
        result["protocols"][proto] = {"corpus_instances": len(corpus), "systems": {}}
        for view in a.views.split(","):
            idx = InstanceIndex.from_shard(shard, view, corpus)
            if len(idx.instance_ids) != len(corpus):
                raise ValueError(f"{len(corpus) - len(idx.instance_ids)} corpus instances lack embeddings")
            df = evaluate(queries, idx.instance_ids, idx.score(T, a.m), text_col)
            df.insert(0, "system", f"{a.encoder}-{view}")
            df.insert(0, "protocol", proto)
            all_rows.append(df)
            result["protocols"][proto]["systems"][f"{a.encoder}-{view}"] = summarise(df)
            if view == a.views.split(",")[0]:
                result["protocols"][proto]["latency"] = measure_latency(enc, idx, texts[0][2])
        runs = [summarise(evaluate(queries, idx.instance_ids,
                                   random_scores(len(idx.instance_ids), len(texts), s), text_col))
                for s in range(a.random_seeds)]
        result["protocols"][proto]["systems"]["random"] = {
            k: {"macro": pd.DataFrame([r[k]["macro"] for r in runs]).mean().round(4).to_dict(),
                "query_mean": pd.DataFrame([r[k]["query_mean"] for r in runs]).mean().round(4).to_dict()}
            for k in runs[0]}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    pd.concat(all_rows).to_csv(a.out + "_queries.csv", index=False)

    for proto, r in result["protocols"].items():
        print(f"\n=== Protocol {proto}  ({r['corpus_instances']} eval instances) ===")
        for kind, cols in (("q1", ["ap", "p10", "rprec"]), ("q2", ["r1", "r10", "rr", "s1", "s5", "s10"])):
            table = {s: {c: v[kind]["macro"][c] for c in cols} for s, v in r["systems"].items()}
            print(f"[{kind} macro]\n" + pd.DataFrame(table).T.round(3).to_string())
        print("latency:", r["latency"])
    print(f"\nwrote {a.out}.json and {a.out}_queries.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
