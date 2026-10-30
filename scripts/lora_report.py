"""Result 03 tables: frozen baseline vs LoRA runs (spec-lora-finetune, docs/results/03-lora.md).

Run from the repo root after training, re-embedding, scripts/lora_val_check.py and the per-run eval
(`python -m sfs.eval.baseline ... --views tight --protocols <own> --out runs/lora/eval/<run>`).
Prints markdown tables; writes runs/lora/eval/compare_per_activity.csv.

- Each adapted model is evaluated on its own protocol only (spec D8).
- Macro = mean over activities. Bootstrap = paired over activities (10,000 resamples, seed 0).
- H2 grouping is fixed from the frozen baseline alone: "at chance" = baseline lift <= 2
  (lift = AP / share of the activity in the eval corpus), the rule used in Result 02.
"""
import json

import numpy as np
import pandas as pd

RUNS = ["A-middle", "A-uniform", "B-uniform", "B-middle"]
COLS = ["ap", "p10", "r1", "r10", "rr", "s1", "s5", "s10"]
SEL = json.load(open("runs/lora/selection.json"))
VAL = json.load(open("runs/lora/val_check.json"))
base_q = pd.read_csv("runs/baseline_successk/02_baseline_queries.csv")
base_j = json.load(open("runs/baseline_successk/02_baseline.json"))


def run_q(run):
    return pd.read_csv(f"runs/lora/eval/{run}_queries.csv")


def base_of(proto):
    return base_q[(base_q.protocol == proto) & (base_q.system == "siglip2-tight")]


def per_act(df, kind, col):
    return df[df.kind == kind].groupby("activity")[col].mean()


def boot(x, n=10000, seed=0):
    x = np.asarray(x)
    s = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)


def label(run):
    return f"{run} ({'selected' if SEL[run[0]]['selected'] == run else 'ablation'})"


# 1. headline
print("| Protocol | System | Q1 mAP | Q1 P@10 | Q2 R@1 | Q2 R@10 | Q2 MRR | Q2 S@1 | Q2 S@5 | Q2 S@10 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for proto in "AB":
    rnd = base_j["protocols"][proto]["systems"]["random"]
    systems = [("random", rnd["q1"]["macro"], rnd["q2"]["macro"]),
               ("baseline (frozen)", per_act(base_of(proto), "q1", COLS).mean(),
                per_act(base_of(proto), "q2", COLS).mean())]
    systems += [(label(r), per_act(run_q(r), "q1", COLS).mean(), per_act(run_q(r), "q2", COLS).mean())
                for r in RUNS if r[0] == proto]
    for s, q1, q2 in systems:
        print(f"| {proto} | {s} | {q1['ap']:.3f} | {q1['p10']:.3f} | {q2['r1']:.3f} | {q2['r10']:.3f} | "
              f"{q2['rr']:.3f} | {q2['s1']:.3f} | {q2['s5']:.3f} | {q2['s10']:.3f} |")

# 2. paired bootstrap: run - baseline, and selected - ablation
print("\n| Run | Q1 mAP Δ | Q2 S@1 Δ | Q2 S@10 Δ | Q2 MRR Δ |  (95% CI, paired over activities)")
print("|---|---|---|---|---|")
for r in RUNS:
    cells = []
    for kind, col in (("q1", "ap"), ("q2", "s1"), ("q2", "s10"), ("q2", "rr")):
        b = per_act(base_of(r[0]), kind, col)
        mu, lo, hi = boot((per_act(run_q(r), kind, col) - b).loc[b.index])
        cells.append(f"{mu:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print(f"| {label(r)} | " + " | ".join(cells) + " |")
for proto in "AB":
    s = SEL[proto]["selected"]
    o = next(r for r in RUNS if r[0] == proto and r != s)
    mu, lo, hi = boot(per_act(run_q(s), "q1", "ap") - per_act(run_q(o), "q1", "ap"))
    print(f"eval Q1 mAP {s} - {o}: {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]")

# 3. per activity (selected models)
rows = []
for proto in "AB":
    b, d = base_of(proto), run_q(SEL[proto]["selected"])
    for act, g in b[b.kind == "q1"].groupby("activity"):
        share = g.n_rel.iloc[0] / g.n_corpus.iloc[0]
        ap_b, ap_l = g.ap.mean(), d[(d.kind == "q1") & (d.activity == act)].ap.mean()
        rows.append({"protocol": proto, "activity": act, "n_rel": int(g.n_rel.iloc[0]), "share": share,
                     "ap_base": ap_b, "lift_base": ap_b / share, "ap_lora": ap_l,
                     "lift_lora": ap_l / share, "delta": ap_l - ap_b})
per = pd.DataFrame(rows)
per.to_csv("runs/lora/eval/compare_per_activity.csv", index=False)
w = per.pivot(index="activity", columns="protocol").sort_values(("lift_base", "A"), ascending=False)
print("\n| Activity | n eval (A) | AP A: base → LoRA | lift A | AP B: base → LoRA | lift B |")
print("|---|---|---|---|---|---|")
for act, r in w.iterrows():
    print(f"| {act} | {int(r[('n_rel', 'A')])} | {r[('ap_base', 'A')]:.3f} → {r[('ap_lora', 'A')]:.3f} | "
          f"{r[('lift_base', 'A')]:.1f} → {r[('lift_lora', 'A')]:.1f} | "
          f"{r[('ap_base', 'B')]:.3f} → {r[('ap_lora', 'B')]:.3f} | "
          f"{r[('lift_base', 'B')]:.1f} → {r[('lift_lora', 'B')]:.1f} |")

# 4. H2
for proto in "AB":
    p = per[per.protocol == proto]
    for name, g in (("at chance (baseline lift <= 2)", p[p.lift_base <= 2]), ("rest", p[p.lift_base > 2])):
        print(f"H2 {proto} {name}, n={len(g)}: AP {g.ap_base.mean():.3f} -> {g.ap_lora.mean():.3f} "
              f"(Δ {g.delta.mean():+.3f}); lift {g.lift_base.mean():.2f} -> {g.lift_lora.mean():.2f}; "
              f"{sorted(g.activity) if 'chance' in name else ''}")
    print(f"H2 {proto} got worse: {sorted(p[p.delta < 0].activity)}")

# 5. H4 (validation only)
for proto in "AB":
    mid = pd.Series(VAL["runs"][f"{proto}-middle"]["recorded_per_activity"])
    uni = pd.Series(VAL["runs"][f"{proto}-uniform"]["recorded_per_activity"])
    d = (mid - uni).sort_values()
    print(f"\nH4 {proto} (validation, {len(d)} activities): middle - uniform mean {d.mean():+.4f}")
    print(d.round(3).to_string())
