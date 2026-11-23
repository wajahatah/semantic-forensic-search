"""Collect every number the final report quotes, from the raw result files, into report/numbers.json.

The report and deck builders read only that file, so no result number is typed by hand
(docs/planning/spec-report.md §1). Sources: docs/results/raw/* (Results 02–06) and, when present,
the local manifest for the dataset counts.

    python scripts/report_numbers.py            # -> report/numbers.json
"""
from __future__ import annotations

import csv
import json
import os
import sys

RAW = "docs/results/raw"
MANIFEST = "manifest_v2_drive/manifest_v2.csv"
OUT = "report/numbers.json"


def _load(raw: str, name: str):
    with open(os.path.join(raw, name), encoding="utf-8") as f:
        return json.load(f)


def _csv(raw: str, name: str) -> list[dict]:
    with open(os.path.join(raw, name), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _g1(d: dict) -> dict:
    """One G1 gate file -> per config: the per-protocol gains and the verdict."""
    out = {}
    for cfg, c in d["configs"].items():
        out[cfg] = {"mean_gain": c["mean_gain"], "pass": c["pass"]}
        for p in ("A", "B"):
            if p in c:
                out[cfg][p] = {k: c[p].get(k) for k in ("lora_val_map", "best_val_map", "vlm_only_val_map", "gain", "best_weight")}
    return {"k": d["k"], "min_gain": d["min_gain"], "teacher": d.get("teacher"), "pass": d["pass"], "configs": out}


def _g0(d: dict) -> dict:
    keep = ("model_id", "kind", "view", "n_frames", "peak_vram_gb", "max_vram_gb", "seconds_per_pair_uncached",
            "seconds_per_clip_cached_64", "determinism_max_abs_diff", "cached_vs_full_max_abs_diff",
            "cached_vs_full_spearman", "noise_floor_max_abs_diff", "shortcut_pass", "pass")
    return {k: d.get(k) for k in keep if k in d}


def dataset_counts(manifest: str = MANIFEST) -> dict | None:
    """Instances, cameras and activities per split, from the local manifest (None when it is not here)."""
    if not os.path.exists(manifest):
        return None
    rows = _csv(os.path.dirname(manifest), os.path.basename(manifest))
    cols = rows[0].keys() if rows else []
    out = {"instances": len(rows), "columns": sorted(cols)}
    for key in ("activity", "camera", "site", "split_a", "split_b", "split", "protocol_a", "protocol_b"):
        if key in cols:
            vals = {}
            for r in rows:
                vals[r[key]] = vals.get(r[key], 0) + 1
            out[key] = {"distinct": len(vals), "counts": vals if len(vals) <= 60 else None}
    return out


def clip_numbers(raw: str = RAW) -> dict:
    """Zero-shot encoder comparison (D-067): validation AUC / P@10 per encoder (Result 01 protocol) and the eval
    Q1 / Q2 macro metrics of CLIP and frozen SigLIP 2 per protocol (Result 02 protocol)."""
    v = _load(raw, "01b_clip_val.json")
    e = _load(raw, "02b_clip_eval.json")
    out = {"val": {"protocol": v["protocol"], "prompt": v["prompt"], "encoders": {}},
           "eval": {"protocol": e["protocol"], "protocols": {}}}
    for enc, r in v["encoders"].items():
        out["val"]["encoders"][enc] = {k: r[k] for k in ("n_instances", "n_activities", "auc", "p10", "chance_p10")}
        out["val"]["encoders"][enc]["per_activity_auc"] = {a: x["auc"] for a, x in r["per_activity"].items()}
    for p, blk in e["protocols"].items():
        out["eval"]["protocols"][p] = {"corpus": blk["corpus_instances"],
                                       "systems": {s: {"q1": x["q1"]["macro"], "q2": x["q2"]["macro"]}
                                                   for s, x in blk["systems"].items()}}
    return out


def collect(raw: str = RAW, manifest: str = MANIFEST) -> dict:
    b = _load(raw, "02_baseline.json")
    sel = _load(raw, "03_lora_selection.json")
    n: dict = {"sources": {}, "dataset": dataset_counts(manifest)}

    # ---- Result 02: frozen baseline (per protocol: corpus size, tight / full / random, Q1 and Q2 macro)
    n["baseline"] = {}
    for p, blk in b["protocols"].items():
        n["baseline"][p] = {"corpus": blk["corpus_instances"],
                            "systems": {s: {"q1": v["q1"]["macro"], "q2": v["q2"]["macro"]} for s, v in blk["systems"].items()}}
    n["baseline"]["encoder_id"] = b["encoder_id"]

    # ---- Result 03: LoRA (selected run per protocol + the ablation), validation selection, per activity
    n["lora"] = {"selection": sel, "runs": {}}
    for run in ("A-middle", "A-uniform", "B-uniform", "B-middle"):
        d = _load(raw, f"03_lora_{run}.json")
        p = run[0]
        sysm = d["protocols"][p]["systems"]["siglip2-tight"]
        n["lora"]["runs"][run] = {"protocol": p, "q1": sysm["q1"]["macro"], "q2": sysm["q2"]["macro"],
                                  "encoder_id": d["encoder_id"], "selected": sel[p]["selected"] == run}
    n["lora"]["per_activity"] = _csv(raw, "03_lora_per_activity.csv")
    for run in ("A-middle", "A-uniform", "B-uniform", "B-middle"):
        done = _load(raw, f"03_lora_{run}_done.json")
        n["lora"]["runs"][run]["training"] = {k: done.get(k) for k in ("best_epoch", "epochs_run", "best_val_map", "frozen_val_map") if k in done}

    # ---- Result 04 + 05: cascade summaries (every system x budget), students, latency, H5
    # 8B: the re-run with its own G0 file (D-048); 05_8b_cascade.json carries the 2B's time per call (summary identical)
    for key, name in (("2b", "04_cascade.json"), ("8b", "05_8b_cascade_g0fix.json"), ("32b", "05_32b_cascade.json"),
                      ("q1all", "05_q1all.json"), ("turns", "05_turns.json")):
        d = _load(raw, name)
        n.setdefault("cascade", {})[key] = {"summary": d["summary"], "students": d["students"],
                                           "vlm_seconds_per_call": d["vlm_seconds_per_call_uncached"],
                                           "h5_direction": d.get("h5_direction")}
    n["cascade"]["per_activity"] = _csv(raw, "04_cascade_per_activity.csv")
    n["cascade"]["turns_per_activity"] = [r for r in _csv(raw, "05_turns_per_activity.csv") if "turn" in r["activity"]]
    n["students"] = _load(raw, "04_student_selection.json")
    n["bootstrap"] = {"8b": _load(raw, "05_8b_bootstrap.json"), "32b": _load(raw, "05_32b_bootstrap.json")}

    # ---- gates: G0 (fit + speed) and G1 (does the VLM add information) for every VLM tried
    n["g0"] = {"qwen3vl_2b_instruct": _g0(_load(raw, "04_g0_instruct.json")),
               "qwen3vl_2b_reranker": _g0(_load(raw, "04_g0_reranker.json")),
               "qwen3vl_8b_fp8": _g0(_load(raw, "05_g0_8b.json")),
               "qwen3vl_32b_fp8": _g0(_load(raw, "05_g0_32b.json")),
               "gemma4_e4b": _g0(_load(raw, "05_g0_gemma_e4b.json")),
               "qwen25vl_3b": _g0(_load(raw, "05_g0_q25_3b.json")),
               "qwen25vl_7b_fp8": _g0(_load(raw, "05_g0_q25_7b.json"))}
    n["g1"] = {"2b_top50": _g1(_load(raw, "04_g1.json")),
               "2b_wholeval": _g1(_load(raw, "05_g1_wholeval.json")),
               "2b_16frames": _g1(_load(raw, "05_g1_16f.json")),
               "8b_top50": _g1(_load(raw, "05_g1_8b.json")),
               "32b_top50": _g1(_load(raw, "05_g1_32b.json")),
               "32b_wholeval": _g1(_load(raw, "05_g1_wholeval_32b.json")),
               "gemma_top50": _g1(_load(raw, "05_g1_gemma.json")),
               "qwen25_3b_top50": _g1(_load(raw, "05_g1_q25_3b.json")),
               "qwen25_7b_top50": _g1(_load(raw, "05_g1_q25_7b.json"))}
    diag = _load(raw, "04_g1_diagnostic.json")
    n["g1_auc"] = {"2b": {p: {k: v for k, v in diag[p].items() if k.startswith("auc") or k in ("pairs", "pos_rate", "random_order_map", "lora_map")}
                          for p in ("A", "B") if p in diag},
                   "8b": _load(raw, "05_g1_8b_auc.json"), "gemma": _load(raw, "05_g1_gemma_vs_qwen_auc.json")}

    # ---- Result 06: image + text LoRA and the text-drift guard
    e8 = _load(raw, "06_e8_report.json")
    n["e8"] = {k: e8[k] for k in ("seeds", "runs", "diffs", "verdict")}
    dr = _load(raw, "06_text_drift.json")
    n["drift"] = {k: dr[k] for k in ("rule", "n_sentences", "n_phrasings", "revision", "results", "all_pass")}

    # ---- Results 01/02 addenda (D-067): zero-shot CLIP ViT-B/16 next to SigLIP / SigLIP 2
    n["clip"] = clip_numbers(raw)

    n["sources"] = {"raw_dir": raw, "files": sorted(os.listdir(raw))}
    return n


def main(argv=None) -> int:
    out = argv[0] if argv else OUT
    n = collect()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(n, f, indent=1)
    a = n["cascade"]["2b"]["summary"]["A/q2"]
    print(f"{out}: baseline A Q1 mAP {n['baseline']['A']['systems']['siglip2-tight']['q1']['ap']:.3f}, "
          f"LoRA {n['lora']['runs']['A-middle']['q1']['ap']:.3f}, student Q2 S@1 {a['student']['s1']:.3f}; "
          f"systems in A/q2: {sorted(a)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
