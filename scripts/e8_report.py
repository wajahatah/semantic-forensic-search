"""E8 report: image + text vs image-only LoRA on validation, per seed (spec-lora-image-text §5, D-052).

Reads the copied run folders under --root:
    <root>/lora_{io,it}{,_s<seed>}/<run>/done.json      (seed 0 has no suffix)
    <root>/per_activity.json                             {"lora_io_s1/A-middle": {activity: AP}, ...}
and prints the comparison tables (Markdown) and writes the numbers to --out.

    python scripts/e8_report.py --root runs/e8 --seeds 0,1 --out docs/results/raw/06_e8_report.json

The reading (D-052, fixed before any seed-1 number): the §5 rule is applied to the per-protocol means
over seeds; an effect "repeats" when every seed gives the same sign and each is larger than REPEAT.
"""
from __future__ import annotations

import argparse
import json
import os

RUNS = ("A-middle", "B-uniform")
GAIN, MAX_DROP, REPEAT = 0.010, 0.005, 0.010


def run_dir(arm: str, seed: int) -> str:
    return f"lora_{arm}" + (f"_s{seed}" if seed else "")


def verdict(diffs: dict[str, list[float]]) -> dict:
    """diffs: run -> [image+text − image-only, one per seed]. The §5 rule on the seed means + repeats."""
    means = {r: sum(d) / len(d) for r, d in diffs.items()}
    gain = sum(means.values()) / len(means)
    repeats = {r: len(d) > 1 and (all(x > REPEAT for x in d) or all(x < -REPEAT for x in d))
               for r, d in diffs.items()}
    return {"mean_diff": means, "mean_gain": gain, "gain_ok": gain >= GAIN,
            "no_drop_ok": all(m >= -MAX_DROP for m in means.values()),
            "adopt": gain >= GAIN and all(m >= -MAX_DROP for m in means.values()), "repeats": repeats}


def activity_table(per_act: dict, run: str, seeds: list[int]) -> list[dict]:
    """Per activity: image+text − image-only for each seed, and whether the sign agrees across seeds."""
    rows = []
    for act in sorted(per_act[f"{run_dir('io', seeds[0])}/{run}"]):
        d = [per_act[f"{run_dir('it', s)}/{run}"][act] - per_act[f"{run_dir('io', s)}/{run}"][act]
             for s in seeds]
        rows.append({"activity": act, "diff": [round(x, 4) for x in d],
                     "agree": all(x > 0 for x in d) or all(x < 0 for x in d)})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="runs/e8")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]

    done = {}
    for arm in ("io", "it"):
        for s in seeds:
            for r in RUNS:
                with open(os.path.join(a.root, run_dir(arm, s), r, "done.json"), encoding="utf-8") as f:
                    done[(arm, s, r)] = json.load(f)
                if done[(arm, s, r)]["config"]["seed"] != s:
                    raise ValueError(f"{run_dir(arm, s)}/{r}: done.json records seed {done[(arm, s, r)]['config']['seed']}")
    diffs = {r: [done[("it", s, r)]["best_val_map"] - done[("io", s, r)]["best_val_map"] for s in seeds]
             for r in RUNS}
    v = verdict(diffs)

    print("| Protocol | Seed | Image only (best epoch) | Image + text (best epoch) | Difference |")
    print("|---|---|---|---|---|")
    for r in RUNS:
        for s in seeds:
            io, it = done[("io", s, r)], done[("it", s, r)]
            print(f"| {r} | {s} | {io['best_val_map']:.4f} ({io['best_epoch']}) | "
                  f"{it['best_val_map']:.4f} ({it['best_epoch']}) | {it['best_val_map'] - io['best_val_map']:+.4f} |")
        print(f"| {r} | mean | | | **{v['mean_diff'][r]:+.4f}** |")
    print(f"\nMean gain {v['mean_gain']:+.4f} (>= +{GAIN}: {v['gain_ok']}); no protocol below -{MAX_DROP}: "
          f"{v['no_drop_ok']} -> adopt: {v['adopt']}. Repeats: {v['repeats']}")

    acts = {}
    path = os.path.join(a.root, "per_activity.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            per_act = json.load(f)
        for r in RUNS:
            acts[r] = activity_table(per_act, r, seeds)
            agree = [x for x in acts[r] if x["agree"]]
            print(f"\n{r}: {len(agree)} of {len(acts[r])} activities move the same way in every seed "
                  f"({sum(x['diff'][0] > 0 for x in agree)} better, {sum(x['diff'][0] < 0 for x in agree)} worse)")
            for x in acts[r]:
                print(f"  {x['activity']:<40} {' '.join(f'{d:+.3f}' for d in x['diff'])}{'' if x['agree'] else '  (mixed)'}")

    if a.out:
        out = {"seeds": seeds, "runs": {f"{run_dir(arm, s)}/{r}": {k: done[(arm, s, r)][k] for k in
                                        ("best_val_map", "best_epoch", "epochs_run", "frozen_val_map", "config_hash")}
                                        for (arm, s, r) in done},
               "diffs": diffs, "verdict": v, "per_activity": acts}
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
