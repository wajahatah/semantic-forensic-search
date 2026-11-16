"""Figures for the final report, drawn from report/numbers.json (docs/planning/spec-report.md §6).

    python scripts/report_figures.py            # -> report/figures/fig1_pipeline.png, fig2_cost_curve.png, fig3_per_activity.png

F1 is the pipeline as built; F2 the recall-vs-cost curve (Q2 Success@1 against VLM calls per query, both
protocols, 2B / 8B / 32B); F3 the per-activity Q1 AP (frozen -> LoRA -> student) that shows what single
frames cannot see.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

NUMBERS = "report/numbers.json"
OUT = "report/figures"
C = {"frozen": "#7f8c8d", "lora": "#2980b9", "student": "#c0392b", "2b": "#27ae60", "8b": "#8e44ad", "32b": "#d35400",
     "train": "#fdebd0", "data": "#eaf2f8", "online": "#e8f8f5", "alert": "#fef9e7", "edge": "#2c3e50"}


# ---------------------------------------------------------------- F1 pipeline
def _box(ax, x, y, w, h, text, fc, fs=8.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.6", fc=fc, ec=C["edge"], lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def _arrow(ax, p, q, text=None, ls="-"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=C["edge"], lw=0.9, ls=ls, shrinkA=1, shrinkB=1))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 1.2, text, ha="center", va="bottom", fontsize=7, color=C["edge"])


def fig_pipeline(path):
    fig, ax = plt.subplots(figsize=(10, 6.6), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")
    lane = lambda y, t: ax.text(0.5, y, t, fontsize=9, fontweight="bold", va="center", ha="left", color=C["edge"])  # noqa: E731
    for y0, y1, fc in ((50, 65, C["data"]), (34, 49, C["train"]), (18, 33, C["alert"]), (1, 17, C["online"])):
        ax.add_patch(FancyBboxPatch((0, y0), 100, y1 - y0, boxstyle="round,pad=0,rounding_size=0.8", fc=fc, ec="none", alpha=0.55))
    # lane A: offline indexing
    lane(63.5, "A · Offline indexing (once per clip)")
    bx = [(2, 53, 16, 7.5, "MEVA activity clips\n(annotated instances)"), (21, 53, 15, 7.5, "2 fps sampling\n≤ 16 frames"),
          (39, 53, 17, 7.5, "Actor-region crops\n(tight / context)"), (59, 53, 18, 7.5, "SigLIP 2 image encoder\n+ LoRA adapter"),
          (80, 53, 18, 7.5, "Clip vectors (exact index)\n+ metadata (SQLite)")]
    for x, y, w, h, t in bx:
        _box(ax, x, y, w, h, t, "white")
    for i in range(len(bx) - 1):
        _arrow(ax, (bx[i][0] + bx[i][3 - 1] if False else bx[i][0] + bx[i][2], 56.75), (bx[i + 1][0], 56.75))
    # lane B: training
    lane(47.5, "B · Training (offline, once; every choice made on validation)")
    _box(ax, 2, 36.5, 21, 8, "192 training captions\n↔ actor crops (train split)", "white")
    _box(ax, 26, 36.5, 22, 8, "LoRA fine-tuning\nrank 16, 1.2 M parameters\nsigmoid image–text loss", "#f9d9b8", fs=7.8, bold=True)
    _arrow(ax, (23, 40.5), (26, 40.5))
    _arrow(ax, (48, 44.5), (64, 53), None, "--")
    _box(ax, 52, 36.5, 21, 8, "Human labels of query–clip\npairs (+ VLM teacher labels,\nonly past gate G1)", "white")
    _box(ax, 75.5, 36.5, 23, 8, "Temporal student re-ranker\nframe vectors + box motion\ntargets: GT / GT+KD / KD", "#f9d9b8", fs=7.8, bold=True)
    _arrow(ax, (73, 40.5), (75.5, 40.5))
    # lane C: alert ingestion
    lane(31.5, "C · Alert ingestion (your own video)")
    cx = [(2, 21, 12, 7.5, "Any video\n(upload)"), (17, 21, 16, 7.5, "YOLOv8 + ByteTrack\nperson · vehicles · bag"),
          (36, 21, 19, 7.5, "Alert rule: 0.5 s confirm,\nhold until gone 2 s"), (58, 21, 15, 7.5, "5 s clip\n11 frames at 2 fps"),
          (76, 21, 22, 7.5, "Same crops, encoder and\nstudent features → store")]
    for x, y, w, h, t in cx:
        _box(ax, x, y, w, h, t, "white")
    for i in range(len(cx) - 1):
        _arrow(ax, (cx[i][0] + cx[i][2], 24.75), (cx[i + 1][0], 24.75))
    # lane D: query path
    lane(15.5, "D · Query path (online) — three cost levels")
    qx = [(2, 4, 13, 8, "Text query"), (18, 4, 17, 8, "SigLIP 2 text encoder\n(frozen)"),
          (38, 4, 18, 8, "FAST · exact search\nover every clip\n0 VLM calls · 0.04 s"),
          (59, 4, 18, 8, "STANDARD · + student\nre-rank of candidates\n0 VLM calls · ≈ 0.2 s"),
          (80, 4, 18, 8, "VERIFIED · VLM yes/no\non the top k only\nk VLM calls · ≈ 1–4 s")]
    for i, (x, y, w, h, t) in enumerate(qx):
        _box(ax, x, y, w, h, t, "white" if i < 2 else "#d4efdf", bold=i >= 2)
    for i in range(len(qx) - 1):
        _arrow(ax, (qx[i][0] + qx[i][2], 8), (qx[i + 1][0], 8))
    _arrow(ax, (89, 53), (89, 12.2), None, "--")
    _arrow(ax, (87, 21), (84, 12.2), None, "--")
    ax.text(50, 0.6, "Shaded boxes: components trained in this project. Dashed arrows: stored vectors feeding the query path.",
            ha="center", va="bottom", fontsize=7.5, style="italic", color=C["edge"])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- F2 cost curve
def fig_cost_curve(n, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200, sharey=True)
    for ax, p in zip(axes, ("A", "B")):
        s2, s8, s32 = (n["cascade"][k]["summary"][f"{p}/q2"] for k in ("2b", "8b", "32b"))
        q1all = n["cascade"]["q1all"]["summary"][f"{p}/q2"]
        for key, c, m, lab in (("frozen", C["frozen"], "s", "frozen SigLIP 2"), ("lora", C["lora"], "o", "LoRA"),
                               ("student", C["student"], "D", "LoRA + temporal student")):
            ax.scatter([s2[key]["cost"]], [s2[key]["s1"]], c=c, marker=m, s=46, zorder=5, label=lab)
        for blk, c, lab in ((s2, C["2b"], "… → Qwen3-VL-2B on top k"), (s8, C["8b"], "… → 8B on top k"), (s32, C["32b"], "… → 32B on top k")):
            ks = sorted((k for k in blk if k.startswith("student>instruct@")), key=lambda k: blk[k]["cost"])
            ax.plot([blk["student"]["cost"]] + [blk[k]["cost"] for k in ks], [blk["student"]["s1"]] + [blk[k]["s1"] for k in ks],
                    "-o", c=c, ms=4, lw=1.4, label=lab)
        ks = sorted((k for k in s2 if k.startswith("lora>instruct@")), key=lambda k: s2[k]["cost"])
        ax.plot([0] + [s2[k]["cost"] for k in ks], [s2["lora"]["s1"]] + [s2[k]["s1"] for k in ks], "--o", c=C["lora"], ms=3.5, lw=1,
                alpha=0.8, label="LoRA → 2B on top k (no student)")
        for key, lab, dy in (("instruct-all", "2B VLM on every candidate", -0.03), ("instruct-all+lora", "2B on all + LoRA", 0.02)):
            if key in q1all:
                ax.scatter([q1all[key]["cost"]], [q1all[key]["s1"]], c=C["2b"], marker="x", s=46, zorder=5)
                ax.annotate(lab, (q1all[key]["cost"], q1all[key]["s1"]), xytext=(-6, 12 if dy > 0 else -14), textcoords="offset points",
                            fontsize=7, ha="right", color=C["2b"])
        ax.set_xscale("symlog", linthresh=5)
        ax.set_xticks([0, 5, 10, 20, 50, round(q1all["instruct-all"]["cost"])])
        ax.set_xticklabels(["0", "5", "10", "20", "50", f"all\n(~{round(q1all['instruct-all']['cost'])})"], fontsize=8)
        ax.set_xlabel("VLM calls per forensic (Q2) query", fontsize=9)
        ax.set_title(f"Protocol {p}: " + ("official split" if p == "A" else "unseen cameras"), fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Q2 Success@1 (macro over 22 activities)", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.5, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- F3 per activity
def fig_per_activity(n, path):
    rows = [r for r in n["cascade"]["per_activity"] if r["protocol"] == "A" and r["system"] in ("frozen", "lora", "student")]
    ap = {}
    for r in rows:
        ap.setdefault(r["activity"], {})[r["system"]] = float(r["q1_ap"])
    acts = sorted(ap, key=lambda a: ap[a]["frozen"])
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=200)
    ys = range(len(acts))
    for y, a in zip(ys, acts):
        ax.plot([ap[a]["frozen"], ap[a]["student"]], [y, y], c="#bdc3c7", lw=2, zorder=1)
    for key, c, m, lab in (("frozen", C["frozen"], "s", "frozen SigLIP 2"), ("lora", C["lora"], "o", "LoRA"), ("student", C["student"], "D", "LoRA + student")):
        ax.scatter([ap[a][key] for a in acts], list(ys), c=c, marker=m, s=34, zorder=3, label=lab)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([a.replace("_", " ") for a in acts], fontsize=8)
    ax.set_xlabel("Q1 average precision, protocol A (eval split)", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    with open(NUMBERS, encoding="utf-8") as f:
        n = json.load(f)
    os.makedirs(OUT, exist_ok=True)
    fig_pipeline(os.path.join(OUT, "fig1_pipeline.png"))
    fig_cost_curve(n, os.path.join(OUT, "fig2_cost_curve.png"))
    fig_per_activity(n, os.path.join(OUT, "fig3_per_activity.png"))
    print("figures ->", OUT, sorted(os.listdir(OUT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
