"""Phase 4 of spec-rerank-cascade: eval-side VLM requests and the Result 04 report.

Order of use (each step reads only what earlier steps fixed on validation):
  1. `requests`  — after G1 and student selection: the (instance, text) pairs each VLM must score
                   on EVAL, plus the validation pairs used to choose the student->VLM weight.
  2. sfs.teacher.run on those request files (subset eval / val).
  3. `report`    — every system on eval: tables, raw files, the recall-vs-cost curve.

First stages: frozen SigLIP 2, LoRA, LoRA+student (z(student) + beta z(LoRA), beta from validation).
VLM stages: re-order the top k (k VLM calls) with gamma from validation (G1 for LoRA->VLM; the
student->VLM weight from the validation pairs requested here). "VLM on every candidate" is computed
for Q2 (`--everything`) and, since the owner's answer Q5 of 2026-11-03, for Q1 (`--everything-q1`,
Instruct only: ~1.5 h with the cached path). Q1-all pairs that were not scored for Result 04 are
requested separately (`--exclude-scored`) and scored into their own store, which `report` merges with
`--extra-scores`; already-scored pairs keep their Result 04 values (re-scoring them in a different
batch would move them by bf16 noise).

Merged-turn view (spec-vlm-followups E1): `--merge vehicle_turns_left,vehicle_turns_right=vehicle_turns`
relabels the manifest before any query is built, with `--phrasings queries/phrasings_merged_turns.csv`;
`--fixed-choices <Result 04 json>` reuses its fusion weights, so nothing is selected again.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from sfs.eval.cascade import System, evaluate_systems, top_needed

SPLIT_COL = {"A": "split", "B": "split_b"}
LORA_RUN = {"A": "A-middle", "B": "B-uniform"}
K_LORA = (5, 10, 20, 50)
K_STUDENT = (1, 3, 5, 10, 20)
VAL_STUDENT_K = 20


def load_protocol(protocol: str, manifest, phr: dict, phr_emb: dict, student_dir: str, device: str,
                  subset: str = "eval") -> dict:
    """Queries, text columns and first-stage score matrices over the protocol's `subset` corpus."""
    import torch
    from sfs.eval.queries import build_q1, build_q2, eligible_activities
    from sfs.eval.retrieval import InstanceIndex
    from sfs.rerank.features import load_features
    from sfs.rerank.model import TemporalReranker
    from sfs.rerank.train import score_matrix, zcols
    from sfs.store.shards import read_shard

    col = SPLIT_COL[protocol]
    if subset == "eval":
        acts = eligible_activities(manifest, "split")
        queries = build_q1(manifest, col, acts, phr) + build_q2(manifest, col, acts, phr)
    else:
        sub = manifest[manifest[col] == subset]
        counts = sub.activity.value_counts()
        acts = sorted(a for a in phr if counts.get(a, 0) >= 5)
        queries = build_q1(manifest, col, acts, phr, subset=subset)
    texts = [(a, j, phr[a][j]) for a in acts for j in range(len(phr[a]))]
    text_col = {(a, j): c for c, (a, j, _) in enumerate(texts)}
    T = np.stack([phr_emb[a][j] for a, j, _ in texts]).astype(np.float32)
    corpus = set(manifest.instance_id[manifest[col] == subset].astype(int))
    firsts = {}
    for name, path in (("frozen", "manifest_v2_drive/embeddings_v2_siglip2.npz"),
                       ("lora", f"data/embeddings_lora/{LORA_RUN[protocol]}/embeddings.npz")):
        idx = InstanceIndex.from_shard(read_shard(path), "tight", corpus)
        if len(idx.instance_ids) != len(corpus):
            raise ValueError(f"{name}: {len(corpus) - len(idx.instance_ids)} instances lack embeddings")
        ids = idx.instance_ids
        firsts[name] = idx.score(T, 3)
    selection = json.load(open(os.path.join(student_dir, "selection.json")))[protocol]
    sel = selection["selected"]
    feats = load_features(f"data/rerank/features_{LORA_RUN[protocol]}.npz")
    pos = {int(i): r for r, i in enumerate(feats["instance_id"])}
    rows = np.array([pos[int(i)] for i in ids])
    student_seconds, betas = None, {}
    # the selected student is "student"; the other trained variants are 0-call ablations (H3)
    for variant in selection["val_map"]:
        run = f"{protocol}-{variant}"
        best = torch.load(os.path.join(student_dir, run, "best.pt"), map_location="cpu", weights_only=False)
        cfg = best["config"]
        model = TemporalReranker(feats["feats"].shape[-1], T.shape[1], cfg["d"], cfg["layers"], cfg["heads"],
                                 feats["feats"].shape[1], cfg["dropout"]).to(device)
        model.load_state_dict(best["model"])
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        s = score_matrix(model, feats["feats"], feats["mask"], rows, T, device)
        if run == sel:
            student_seconds = (time.perf_counter() - t0) / T.shape[0]      # one text over the whole corpus
            firsts["student"] = zcols(s) + best["beta"] * zcols(firsts["lora"])
            firsts["student_only"] = s
        else:
            firsts[f"student_{variant}"] = zcols(s) + best["beta"] * zcols(firsts["lora"])
        betas[variant] = best["beta"]
    return {"queries": queries, "texts": texts, "text_col": text_col, "ids": ids, "firsts": firsts,
            "student": sel, "beta": betas[sel.split("-", 1)[1]], "betas": betas,
            "student_seconds_per_query": student_seconds}


def parse_merge(spec: str) -> dict[str, str]:
    """"a,b=new;c,d=new2" -> {"a": "new", "b": "new", "c": "new2", "d": "new2"}."""
    out = {}
    for group in filter(None, (g.strip() for g in spec.split(";"))):
        olds, new = group.split("=")
        for old in olds.split(","):
            out[old.strip()] = new.strip()
    return out


def parse_stores(spec: str) -> dict[str, list[str]]:
    """"model=path,model=path2,other=path3" -> {"model": [path, path2], "other": [path3]}."""
    out: dict[str, list[str]] = {}
    for kv in filter(None, spec.split(",")):
        model, path = kv.split("=", 1)
        out.setdefault(model, []).append(path)
    return out


def g0_files(spec: str, g1_path: str, models: list[str]) -> dict[str, str]:
    """G0 file per model, for the uncached seconds per VLM call. `spec` "model=path" names it; without
    one, the Result 04 layout `g0_<model>.json` next to the G1 file is used. A G0 file named in `spec`
    must exist (a bigger VLM must never be timed with the 2B's file)."""
    named = parse_stores(spec)
    out = {}
    for model in models:
        paths = named.get(model, [])
        if len(paths) > 1:
            raise ValueError(f"--g0 names {len(paths)} files for {model}")
        if paths and not os.path.exists(paths[0]):
            raise FileNotFoundError(paths[0])
        out[model] = paths[0] if paths else os.path.join(os.path.dirname(g1_path), f"g0_{model}.json")
    return out


def build_requests(data: dict, everything_q2: bool, k_q2: int, k_q1: int = max(K_LORA),
                   everything_q1: bool = False) -> set:
    need = set()
    q1 = [q for q in data["queries"] if q.kind == "q1"]
    q2 = [q for q in data["queries"] if q.kind == "q2"]
    for first in ("lora", "student"):
        need |= top_needed(q1, data["ids"], data["firsts"][first], data["text_col"], None if everything_q1 else k_q1)
        need |= top_needed(q2, data["ids"], data["firsts"][first], data["text_col"], None if everything_q2 else k_q2)
    return need


def systems_for(model: str, gamma_lora: float, gamma_student: float, k_max: int | None) -> list[System]:
    """VLM systems for one query kind; k_max None = the VLM scored every candidate, so the "-all" points too."""
    out = []
    for k in K_LORA:
        if k_max is None or k <= k_max:
            out.append(System(f"lora>{model}@{k}", "lora", k, gamma_lora))
    for k in K_STUDENT:
        if k_max is None or k <= k_max:
            out.append(System(f"student>{model}@{k}", "student", k, gamma_student))
    if k_max is None:
        out.append(System(f"{model}-all", "lora", None, 0.0))
        out.append(System(f"{model}-all+lora", "lora", None, gamma_lora))
    return out


def macro(df: pd.DataFrame, cols=("ap", "p10", "r1", "r10", "rr", "s1", "s5", "s10", "cost")) -> pd.DataFrame:
    per_act = df.groupby(["system", "activity"])[list(cols)].mean()
    return per_act.groupby("system").mean()


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    s = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((s[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def direction_test(queries, ids: np.ndarray, firsts: dict, text_col: dict, vlm: dict, activity_of: dict,
                   pair=("vehicle_turns_left", "vehicle_turns_right")) -> dict:
    """H5: inside each Q2 window holding both directions, does the score for a 'turn left' query rank
    left turns above right turns (and vice versa)? AUC per query, averaged; the VLM and every
    first-stage score in `firsts` are measured on exactly the same (instance, text) pairs."""
    pos = {int(i): n for n, i in enumerate(ids)}
    out = {"vlm": [], **{name: [] for name in firsts}}
    for q in queries:
        if q.kind != "q2" or q.activity not in pair:
            continue
        other = pair[1 - pair.index(q.activity)]
        cand = [int(i) for i in q.corpus if activity_of[int(i)] in pair and (int(i), q.text) in vlm]
        p = [i for i in cand if activity_of[i] == q.activity]
        n = [i for i in cand if activity_of[i] == other]
        if not p or not n:
            continue
        col = text_col[(q.activity, q.phrasing_idx)]
        out["vlm"].append(_auc(np.array([vlm[(i, q.text)] for i in p]), np.array([vlm[(i, q.text)] for i in n])))
        for name, sc in firsts.items():
            out[name].append(_auc(sc[[pos[i] for i in p], col], sc[[pos[i] for i in n], col]))
    return {"n_queries": len(out["vlm"]),
            **{f"{name}_auc": (float(np.mean(v)) if v else None) for name, v in out.items()}}


def main(argv=None) -> int:
    import torch
    from sfs.data.manifest import read_manifest
    from sfs.eval.queries import load_phrasings
    from sfs.teacher.gate import WEIGHTS, _phrasing_embeddings
    from sfs.teacher.store import load_scores, write_requests

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["requests", "report"])
    ap.add_argument("--manifest", default="manifest_v2_drive/manifest_v2.csv")
    ap.add_argument("--phrasings", default="queries/phrasings.csv")
    ap.add_argument("--students", default="runs/student")
    ap.add_argument("--g1", default="runs/cascade/g1.json")
    ap.add_argument("--models", default="instruct,reranker", help="model kinds, in G1 config-name order")
    ap.add_argument("--g0", default="",
                    help="report: model=g0.json per model (default g0_<model>.json next to --g1)")
    ap.add_argument("--everything", default="instruct", help="models scored on every Q2 candidate")
    ap.add_argument("--everything-q1", default="", help="models scored on every Q1 candidate (Q5, 2026-11-03)")
    ap.add_argument("--exclude-scored", default="",
                    help="requests: model=scores.npz; pairs already in that store are left out of the file")
    ap.add_argument("--extra-scores", default="",
                    help="report: model=dir[,model=dir]; eval stores merged in (existing pairs win)")
    ap.add_argument("--merge", default="", help="relabel activities first: old,old=new[;...] (E1)")
    ap.add_argument("--fixed-choices", default="",
                    help="report: a previous report json whose gamma_lora / gamma_student are reused")
    ap.add_argument("--k-q2", type=int, default=20, help="top-k scored on Q2 by the other models")
    ap.add_argument("--scores-root", default="data/teacher")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = read_manifest(a.manifest)
    if a.merge:
        m["activity"] = m.activity.replace(parse_merge(a.merge))
    phr = load_phrasings(a.phrasings)
    pe = _phrasing_embeddings(phr)
    models = a.models.split(",")
    everything = set(a.everything.split(",")) if a.everything else set()
    everything_q1 = set(a.everything_q1.split(",")) if a.everything_q1 else set()
    extra = parse_stores(a.extra_scores)
    fixed = json.load(open(a.fixed_choices))["choices"] if a.fixed_choices else {}

    if a.cmd == "requests":
        os.makedirs(a.out, exist_ok=True)
        exclude = parse_stores(a.exclude_scored)
        for model in models:
            need = set()
            for p in "AB":
                need |= build_requests(load_protocol(p, m, phr, pe, a.students, device), model in everything, a.k_q2,
                                       everything_q1=model in everything_q1)
            name = f"requests_eval_{model}.csv"
            for path in exclude.get(model, []):
                done, _ = load_scores(path, allow_eval=True)
                need -= set(done)
            if model in exclude:
                name = f"requests_eval_{model}_extra.csv"
            write_requests(os.path.join(a.out, name), sorted(need))
            print(f"{model}: {len(need):,} eval pairs over {len({i for i, _ in need}):,} instances -> {name}")
        val_need = set()
        for p in "AB":
            d = load_protocol(p, m, phr, pe, a.students, device, subset="val")
            val_need |= top_needed(d["queries"], d["ids"], d["firsts"]["student"], d["text_col"], VAL_STUDENT_K)
        write_requests(os.path.join(a.out, "requests_val_student.csv"), sorted(val_need))
        print(f"validation (student->VLM weight): {len(val_need):,} pairs")
        return 0

    # ------------------------------------------------------------ report
    g1 = json.load(open(a.g1))
    view_of = {}
    for model in models:
        cands = {n: c for n, c in g1["configs"].items() if n.startswith(model)}
        view_of[model] = max(cands, key=lambda n: cands[n]["mean_gain"])
    rows, choices, students, h5 = [], {}, {}, {}
    activity_of = dict(zip(m.instance_id.astype(int), m.activity))
    for p in "AB":
        val = load_protocol(p, m, phr, pe, a.students, device, subset="val")
        ev = load_protocol(p, m, phr, pe, a.students, device)
        base_sys = [System("frozen", "frozen"), System("lora", "lora"), System("student", "student"),
                    System("student-only", "student_only")]
        base_sys += [System(f"student[{k[8:]}]", k) for k in ev["firsts"] if k.startswith("student_")
                     and k != "student_only"]
        students[p] = {"selected": ev["student"], "betas": ev["betas"],
                       "seconds_per_query_whole_eval_corpus": ev["student_seconds_per_query"]}
        rows.append(evaluate_systems(ev["queries"], ev["ids"], ev["firsts"], ev["text_col"], {}, base_sys).assign(protocol=p))
        for model in models:
            cfg = view_of[model]
            gamma_lora = g1["configs"][cfg][p]["best_weight"]
            if f"{p}/{model}" in fixed:                      # E1: reuse, never re-select
                prev = fixed[f"{p}/{model}"]
                if prev["config"] != cfg or prev["gamma_lora"] != gamma_lora:
                    raise ValueError(f"{p}/{model}: fixed choices disagree with {a.g1}")
                best_g = prev["gamma_student"]
                choices[f"{p}/{model}"] = {**prev, "reused_from": a.fixed_choices}
            else:
                vval, _ = load_scores(os.path.join(a.scores_root, "val_student", cfg, "scores.npz"))
                vq = [q for q in val["queries"]]
                best_g, best_map = None, -1
                for g in WEIGHTS:
                    d = evaluate_systems(vq, val["ids"], val["firsts"], val["text_col"], vval,
                                         [System("s", "student", VAL_STUDENT_K, g)])
                    mp = d.groupby("activity").ap.mean().mean()
                    if mp > best_map + 1e-12:
                        best_g, best_map = g, mp
                choices[f"{p}/{model}"] = {"config": cfg, "gamma_lora": gamma_lora, "gamma_student": best_g,
                                           "student_val_map_with_vlm": best_map}
            vlm, _ = load_scores(os.path.join(a.scores_root, "eval", cfg, "scores.npz"), allow_eval=True)
            for path in extra.get(model, []):
                more, _ = load_scores(os.path.join(path, "scores.npz"), allow_eval=True)
                vlm = {**more, **vlm}
            h5_firsts = {k: ev["firsts"][k] for k in ("lora", "student_only", "student")}
            h5[f"{p}/{model}"] = direction_test(ev["queries"], ev["ids"], h5_firsts, ev["text_col"], vlm, activity_of)
            k_max = {"q1": None if model in everything_q1 else max(K_LORA),
                     "q2": None if model in everything else a.k_q2}
            for kind in ("q1", "q2"):
                qs = [q for q in ev["queries"] if q.kind == kind]
                sys_ = systems_for(model, gamma_lora, best_g, k_max[kind])
                rows.append(evaluate_systems(qs, ev["ids"], ev["firsts"], ev["text_col"], vlm, sys_).assign(protocol=p))
    df = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out + "_queries.csv", index=False)
    summary = {}
    for (p, kind), g in df.groupby(["protocol", "kind"]):
        summary[f"{p}/{kind}"] = macro(g).round(4).to_dict(orient="index")
    per_call = {}
    for model, g0_path in g0_files(a.g0, a.g1, models).items():
        if os.path.exists(g0_path):
            per_call[model] = json.load(open(g0_path))["seconds_per_pair_uncached"]
    json.dump({"choices": choices, "view_of": view_of, "students": students, "h5_direction": h5,
               "vlm_seconds_per_call_uncached": per_call, "summary": summary},
              open(a.out + ".json", "w"), indent=1)
    for key, tab in summary.items():
        print(f"\n=== {key} ===\n" + pd.DataFrame(tab).T.sort_values("cost").round(3).to_string())
    print(f"\nwrote {a.out}.json and {a.out}_queries.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
