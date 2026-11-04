"""Result 04 tables and figure (spec-rerank-cascade D8), from the output of
`python -m sfs.eval.cascade_run report --out runs/cascade/04_cascade`.

Writes:
  <fig>        the recall-vs-cost curve: Q2 S@1 against VLM calls per query (log axis), one panel per
               protocol; 0-call systems are drawn at the "0" tick left of 1. Plain SVG (no plotting
               dependency).
  <tables>     markdown tables (Q2 primary, Q1, latency, H5) for docs/results/04-cascade.md.
  <per_act>    per-activity Q2 S@1 / Q1 AP for every system (CSV).
  <hyp>        (with --baseline-per-activity) H1-H4 statistics: paired bootstrap over activities
               (10,000 resamples, seed 0, as Result 03) and the motion group = frozen-baseline lift <= 2
               (Result 02 rule, from runs/lora/eval/compare_per_activity.csv).

Usage:
  python scripts/cascade_report.py --report runs/cascade/04_cascade --fig docs/results/04_cascade_curve.svg \
      --tables runs/cascade/04_tables.md --per-activity docs/results/raw/04_cascade_per_activity.csv       --baseline-per-activity runs/lora/eval/compare_per_activity.csv --hypotheses runs/cascade/04_hypotheses.md
"""
from __future__ import annotations

import argparse
import json
import math
import re

import numpy as np
import pandas as pd

ZERO_X = 0.5                     # where 0-call systems sit on the log axis (labelled "0")
SERIES = [                       # (name regex, label, colour, dashed)
    (r"^lora>instruct@", "LoRA → Instruct @k", "#2563eb", False),
    (r"^student>instruct@", "LoRA+student → Instruct @k", "#16a34a", False),
    (r"^lora>reranker@", "LoRA → Reranker @k", "#2563eb", True),
    (r"^student>reranker@", "LoRA+student → Reranker @k", "#16a34a", True),
]
POINTS = [("frozen", "frozen SigLIP 2", "#9ca3af"), ("lora", "LoRA", "#2563eb"), ("student", "LoRA+student", "#16a34a"),
          ("instruct-all", "Instruct on all", "#dc2626"), ("instruct-all+lora", "Instruct on all + LoRA", "#f59e0b")]


def _panel(tab: dict, x0: float, y0: float, w: float, h: float, title: str, ymax: float) -> list[str]:
    costs = [max(r["cost"], ZERO_X) for r in tab.values()]
    lx_min, lx_max = math.log10(ZERO_X) - 0.1, math.log10(max(costs)) + 0.1
    sx = lambda c: x0 + (math.log10(max(c, ZERO_X)) - lx_min) / (lx_max - lx_min) * w   # noqa: E731
    sy = lambda v: y0 + h - v / ymax * h                                               # noqa: E731
    out = [f"<text x='{x0 + w / 2}' y='{y0 - 10}' class='t' text-anchor='middle'>{title}</text>",
           f"<rect x='{x0}' y='{y0}' width='{w}' height='{h}' class='frame'/>"]
    for v in [i / 10 for i in range(0, int(ymax * 10) + 1, 2)]:
        out.append(f"<line x1='{x0}' x2='{x0 + w}' y1='{sy(v):.1f}' y2='{sy(v):.1f}' class='grid'/>"
                   f"<text x='{x0 - 6}' y='{sy(v) + 4:.1f}' class='a' text-anchor='end'>{v:.1f}</text>")
    ticks = [ZERO_X] + [c for c in (1, 3, 10, 30, 100) if math.log10(c) <= lx_max]
    for c in ticks:
        out.append(f"<line x1='{sx(c):.1f}' x2='{sx(c):.1f}' y1='{y0}' y2='{y0 + h}' class='grid'/>"
                   f"<text x='{sx(c):.1f}' y='{y0 + h + 16}' class='a' text-anchor='middle'>{0 if c == ZERO_X else c}</text>")
    for pat, _, col, dashed in SERIES:
        pts = sorted((r["cost"], r["s1"]) for n, r in tab.items() if re.match(pat, n))
        if pts:
            d = " ".join(f"{sx(c):.1f},{sy(v):.1f}" for c, v in pts)
            out.append(f"<polyline points='{d}' fill='none' stroke='{col}' stroke-width='2'"
                       f"{' stroke-dasharray=\"6 4\"' if dashed else ''}/>")
            out += [f"<circle cx='{sx(c):.1f}' cy='{sy(v):.1f}' r='3' fill='{col}'/>" for c, v in pts]
    for name, _, col in POINTS:
        if name in tab:
            r = tab[name]
            out.append(f"<rect x='{sx(r['cost']) - 5:.1f}' y='{sy(r['s1']) - 5:.1f}' width='10' height='10' "
                       f"fill='{col}' stroke='white'/>")
    return out


def figure(summary: dict) -> str:
    q2 = {p: summary[f"{p}/q2"] for p in "AB" if f"{p}/q2" in summary}
    ymax = min(1.0, math.ceil(max(r["s1"] for t in q2.values() for r in t.values()) * 10 + 1) / 10)
    W, H, pw, ph = 980, 460, 380, 300
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' font-family='system-ui,sans-serif'>",
             "<style>.t{font-size:15px;font-weight:600;fill:#111}.a{font-size:11px;fill:#444}"
             ".frame{fill:none;stroke:#999}.grid{stroke:#e5e7eb}.l{font-size:12px;fill:#222}</style>",
             f"<rect width='{W}' height='{H}' fill='white'/>"]
    for n, (p, tab) in enumerate(q2.items()):
        parts += _panel(tab, 60 + n * (pw + 70), 40, pw, ph, f"Protocol {p} — Q2 Success@1 (eval)", ymax)
    parts.append(f"<text x='{60 + pw + 35}' y='{40 + ph + 38}' class='a' text-anchor='middle'>"
                 "VLM calls per query (log scale; 0 = no VLM)</text>")
    lx, ly = 60, 40 + ph + 62
    for i, (_, label, col, dashed) in enumerate(SERIES):
        x = lx + (i % 2) * 300
        y = ly + (i // 2) * 18
        parts.append(f"<line x1='{x}' x2='{x + 26}' y1='{y}' y2='{y}' stroke='{col}' stroke-width='2'"
                     f"{' stroke-dasharray=\"6 4\"' if dashed else ''}/><text x='{x + 32}' y='{y + 4}' class='l'>{label}</text>")
    for i, (_, label, col) in enumerate(POINTS):
        x, y = lx + 620, ly - 10 + i * 15
        parts.append(f"<rect x='{x}' y='{y - 5}' width='10' height='10' fill='{col}'/>"
                     f"<text x='{x + 16}' y='{y + 4}' class='l'>{label}</text>")
    parts.append("</svg>")
    return "\n".join(parts)


def tables(rep: dict) -> str:
    out = []
    for kind, cols, title in (("q2", ["cost", "s1", "s5", "s10", "rr"], "Q2 (forensic windows)"),
                              ("q1", ["cost", "ap", "p10"], "Q1 (whole eval corpus)")):
        for p in "AB":
            tab = pd.DataFrame(rep["summary"][f"{p}/{kind}"]).T[cols].sort_values(["cost", cols[1]])
            out.append(f"\n**{title}, protocol {p}** (macro over activities; cost = VLM calls per query)\n")
            out.append("| system | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
            for name, r in tab.iterrows():
                out.append(f"| {name} | " + " | ".join(f"{r[c]:.1f}" if c == "cost" else f"{r[c]:.3f}" for c in cols) + " |")
    out.append("\n**Estimated latency per Q2 query** (VLM calls × uncached per-call time from G0 + student)\n")
    out.append("| protocol | system | calls | seconds |\n|---|---|---|---|")
    for p in "AB":
        st = rep["students"][p]["seconds_per_query_whole_eval_corpus"]
        for name, r in pd.DataFrame(rep["summary"][f"{p}/q2"]).T.sort_values("cost").iterrows():
            model = next((m for m in rep["vlm_seconds_per_call_uncached"] if m in name), None)
            sec = r["cost"] * (rep["vlm_seconds_per_call_uncached"][model] if model else 0.0)
            sec += st if name.startswith("student") else 0.0
            out.append(f"| {p} | {name} | {r['cost']:.1f} | {sec:.2f} |")
    out.append("\n**H5 — left vs right turns** (mean per-query AUC inside Q2 windows holding both directions; "
               "same pairs for every scorer)\n")
    out.append("| protocol / VLM | queries | VLM | LoRA | student alone | LoRA+student |\n|---|---|---|---|---|---|")
    f = lambda v: "—" if v is None else f"{v:.3f}"   # noqa: E731
    for key, r in rep["h5_direction"].items():
        out.append(f"| {key} | {r['n_queries']} | {f(r['vlm_auc'])} | {f(r['lora_auc'])} | "
                   f"{f(r['student_only_auc'])} | {f(r['student_auc'])} |")
    return "\n".join(out) + "\n"


def per_activity(queries_csv: str) -> pd.DataFrame:
    df = pd.read_csv(queries_csv)
    q2 = df[df.kind == "q2"].groupby(["protocol", "activity", "system"]).s1.mean().rename("q2_s1")
    q1 = df[df.kind == "q1"].groupby(["protocol", "activity", "system"]).ap.mean().rename("q1_ap")
    return pd.concat([q2, q1], axis=1).reset_index()


def boot(x, n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, np.float64)
    s = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def hypotheses(queries_csv: str, base_per_activity_csv: str) -> str:
    df = pd.read_csv(queries_csv)
    lift = pd.read_csv(base_per_activity_csv).set_index(["protocol", "activity"]).lift_base
    pa = lambda p, kind, sys_, col: (df[(df.protocol == p) & (df.kind == kind) & (df.system == sys_)]  # noqa: E731
                                     .groupby("activity")[col].mean())
    fmt = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"   # noqa: E731
    out = ["| comparison (eval, Δ macro over activities, 95 % CI) | A | B |", "|---|---|---|"]
    comps = [("H1 student − LoRA, Q2 MRR", "q2", "student", "lora", "rr"),
             ("H1 student − LoRA, Q1 mAP", "q1", "student", "lora", "ap"),
             ("H1 student − LoRA, Q2 S@1", "q2", "student", "lora", "s1"),
             ("H2 LoRA→Instruct@10 − LoRA, Q2 S@1", "q2", "lora>instruct@10", "lora", "s1"),
             ("H2 LoRA→Reranker@10 − LoRA, Q2 S@1", "q2", "lora>reranker@10", "lora", "s1"),
             ("H3 GT+KD student − GT student, Q1 mAP", "q1", "student[GTKD]", "student", "ap"),
             ("KD-only student − GT student, Q1 mAP", "q1", "student[KD]", "student", "ap"),
             ("student→Instruct@10 − student, Q2 S@1", "q2", "student>instruct@10", "student", "s1"),
             ("student→Reranker@10 − student, Q2 S@1", "q2", "student>reranker@10", "student", "s1"),
             ("Instruct on all − LoRA, Q2 S@1", "q2", "instruct-all", "lora", "s1")]
    for name, kind, a_sys, b_sys, col in comps:
        cells = []
        for p in "AB":
            a, b = pa(p, kind, a_sys, col), pa(p, kind, b_sys, col)
            cells.append(fmt(boot((a - b).loc[b.index])) if len(a) else "—")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    out += ["", "| H1 by group (baseline lift ≤ 2 = motion-defined) | n | Q1 AP: LoRA → student | Δ | Q2 MRR: LoRA → student | Δ |",
            "|---|---|---|---|---|---|"]
    for p in "AB":
        q1l, q1s = pa(p, "q1", "lora", "ap"), pa(p, "q1", "student", "ap")
        q2l, q2s = pa(p, "q2", "lora", "rr"), pa(p, "q2", "student", "rr")
        lp = lift.xs(p)
        for gname, acts in (("motion (lift ≤ 2)", lp[lp <= 2].index), ("rest", lp[lp > 2].index)):
            acts = [x for x in acts if x in q1l.index]
            out.append(f"| {p} {gname}: {', '.join(sorted(acts)) if 'motion' in gname else ''} | {len(acts)} | "
                       f"{q1l[acts].mean():.3f} → {q1s[acts].mean():.3f} | {(q1s[acts] - q1l[acts]).mean():+.3f} | "
                       f"{q2l[acts].mean():.3f} → {q2s[acts].mean():.3f} | {(q2s[acts] - q2l[acts]).mean():+.3f} |")
    out += ["", "| H4 | Q2 S@1 student→Instruct@5 | Q2 S@1 Instruct on all | ratio | calls @5 | calls all | call share |",
            "|---|---|---|---|---|---|---|"]
    for p in "AB":
        q = df[(df.protocol == p) & (df.kind == "q2")]
        m = q.groupby(["system", "activity"])[["s1", "cost"]].mean().groupby("system").mean()
        a, b = m.loc["student>instruct@5"], m.loc["instruct-all"]
        out.append(f"| {p} | {a.s1:.3f} | {b.s1:.3f} | {a.s1 / b.s1:.2f} | {a.cost:.1f} | {b.cost:.1f} | "
                   f"{a.cost / b.cost:.1%} |")
    out += ["", "| activity | A: lift base | A Q1 AP LoRA → student | B Q1 AP LoRA → student | A Q2 S@1 LoRA → student → +Instruct@10 |",
            "|---|---|---|---|---|"]
    acts = sorted(pa("A", "q1", "lora", "ap").index, key=lambda x: -lift.get(("A", x), 0))
    for x in acts:
        cell = lambda p: f"{pa(p, 'q1', 'lora', 'ap').get(x, np.nan):.3f} → {pa(p, 'q1', 'student', 'ap').get(x, np.nan):.3f}"  # noqa: E731
        s2 = [pa("A", "q2", sname, "s1").get(x, np.nan) for sname in ("lora", "student", "student>instruct@10")]
        out.append(f"| {x} | {lift.get(('A', x), np.nan):.1f} | {cell('A')} | {cell('B')} | "
                   f"{s2[0]:.2f} → {s2[1]:.2f} → {s2[2]:.2f} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", default="runs/cascade/04_cascade")
    ap.add_argument("--fig", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--per-activity", required=True)
    ap.add_argument("--baseline-per-activity", help="Result 03 per-activity CSV with the frozen-baseline lift")
    ap.add_argument("--hypotheses", help="where to write the H1-H4 statistics (needs --baseline-per-activity)")
    a = ap.parse_args(argv)
    rep = json.load(open(a.report + ".json"))
    with open(a.fig, "w", encoding="utf-8") as fh:
        fh.write(figure(rep["summary"]))
    with open(a.tables, "w", encoding="utf-8") as fh:
        fh.write(tables(rep))
    per_activity(a.report + "_queries.csv").round(4).to_csv(a.per_activity, index=False)
    if a.hypotheses:
        with open(a.hypotheses, "w", encoding="utf-8") as fh:
            fh.write(hypotheses(a.report + "_queries.csv", a.baseline_per_activity))
    print(f"wrote {a.fig}, {a.tables}, {a.per_activity}" + (f", {a.hypotheses}" if a.hypotheses else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
