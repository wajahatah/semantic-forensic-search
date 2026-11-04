"""Owner feedback page (spec-rerank-cascade §10, O3): the open questions, gate outcomes and, per
activity, the VLM's most confident mistakes on VALIDATION with their frames — so the owner can
judge the teacher without the work stopping. Writes one self-contained local HTML file
(data/review/, git-ignored; frames embedded as JPEG).

CLI:
  python -m sfs.teacher.feedback --questions runs/cascade/questions.json --g0 runs/cascade/g0_*.json \
      --g1 runs/cascade/g1.json --scores data/teacher/g1/<teacher>/scores.npz       [--diagnostic runs/cascade/g1_diagnostic.json] --out data/review/vlm_feedback.html
The questions file holds the plain-language text (intro, reply_hint, story, glossary, questions with
lettered options and a recommendation); the gate numbers and the examples come from the run files.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import sys

import numpy as np

PER_KIND = 2          # examples per activity: most confident false positives / false negatives
THUMB = 112


def _jpeg(img: np.ndarray) -> str:
    import cv2
    h, w = img.shape[:2]
    s = THUMB / max(h, w)
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def mistakes(scores: dict, activity_of_text: dict, activity_of_inst: dict) -> dict:
    """Per activity: [(kind, instance, text, margin)] — confident false positives and false negatives."""
    out = {}
    for act in sorted(set(activity_of_text.values())):
        rows = [(i, t, m) for (i, t), m in scores.items() if activity_of_text[t] == act]
        fp = sorted([r for r in rows if activity_of_inst[r[0]] != act], key=lambda r: -r[2])[:PER_KIND]
        fn = sorted([r for r in rows if activity_of_inst[r[0]] == act], key=lambda r: r[2])[:PER_KIND]
        out[act] = [("said YES, wrong", *r) for r in fp] + [("said NO, wrong", *r) for r in fn]
    return out


def _sure(margin: float) -> str:
    """The VLM score in words (margin = log p(yes) - log p(no))."""
    side = "YES" if margin > 0 else "NO"
    return f"very sure {side}" if abs(margin) >= 5 else f"fairly sure {side}" if abs(margin) >= 2 else f"leaning {side}"


def _table(head: list[str], rows: list[list[str]]) -> str:
    e = html.escape
    return ("<table><tr>" + "".join(f"<th>{e(h)}</th>" for h in head) + "</tr>"
            + "".join("<tr>" + "".join(f"<td>{e(str(c))}</td>" for c in r) + "</tr>" for r in rows) + "</table>")


def _question(n: int, q: dict) -> str:
    e = html.escape
    out = [f"<div class='q'><div class='tag'>Q{n} · {e(q.get('importance', ''))}</div><h3>{e(q['title'])}</h3>",
           f"<p><b>What happened:</b> {e(q['what'])}</p>"]
    if q.get("why"):
        out.append(f"<p><b>Why:</b> {e(q['why'])}</p>")
    if q.get("table"):
        out.append(_table(q["table"]["head"], q["table"]["rows"]))
    out.append("<p><b>Your options:</b></p><ul class='opts'>")
    for o in q["options"]:
        rec = " <span class='rec'>my recommendation</span>" if o.get("recommended") else ""
        out.append(f"<li><b>{e(o['key'])}</b> — {e(o['text'])}{rec}</li>")
    out.append(f"</ul><p class='mut'><b>What I did meanwhile:</b> {e(q.get('did', ''))}</p>")
    if q.get("details"):
        out.append(f"<details><summary>Technical details</summary><p>{e(q['details'])}</p></details>")
    out.append("</div>")
    return "".join(out)


def render(page: dict, g0: list[dict], g1: dict, examples: dict, crops, activity_of_inst: dict,
           view: str, effect: dict | None = None) -> str:
    """`page`: the questions file (intro, reply_hint, story, glossary, questions); `effect`: optional
    {protocol: {activity: VLM-minus-LoRA validation AP}} for the teacher configuration (G1 diagnostic)."""
    e = html.escape
    effect = effect or {}
    parts = ["<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>VLM stage review</title><style>",
             ":root{--bg:#fff;--fg:#1d1d1f;--mut:#666;--line:#ddd;--warn:#b45309;--ok:#15803d;--card:#f6f6f7}",
             "@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#eee;--mut:#aaa;--line:#333;--warn:#f59e0b;--ok:#4ade80;--card:#222}}",
             "body{background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,sans-serif;margin:0 auto;max-width:900px;padding:16px}",
             "table{border-collapse:collapse;margin:8px 0;display:block;overflow-x:auto}",
             "td,th{border:1px solid var(--line);padding:5px 9px;text-align:left}",
             ".q{background:var(--card);border-left:4px solid var(--warn);padding:4px 16px 8px;margin:18px 0;border-radius:4px}",
             ".tag{color:var(--warn);font-weight:600;font-size:13px;margin-top:10px;text-transform:uppercase}",
             ".q h3{margin:4px 0 8px}.opts li{margin:4px 0}.rec{color:var(--ok);font-weight:600}",
             ".mut{color:var(--mut)}.reply{border:2px solid var(--ok);padding:8px 12px;border-radius:4px}",
             "details{margin:6px 0}summary{cursor:pointer;color:var(--mut)}",
             ".ex{margin:6px 0 16px}.ex img{margin-right:2px;border-radius:3px;max-width:11%}h3.act{margin-top:26px}",
             "</style></head><body><h1>VLM stage — questions for you</h1>"]
    parts += [f"<p>{e(t)}</p>" for t in page.get("intro", [])]
    parts.append(f"<p class='reply'><b>How to answer:</b> {e(page.get('reply_hint', ''))}</p>")
    story = page.get("story")
    if story:
        parts.append("<h2>The story so far</h2>")
        parts += [f"<p>{e(t)}</p>" for t in story["text"]]
        parts.append(_table(story["head"], story["rows"]))
        parts.append(f"<p class='mut'>{e(story.get('after', ''))}</p>")
    if page.get("glossary"):
        parts.append("<details><summary><b>Words used on this page</b> (open if a term is unclear)</summary>")
        parts.append(_table(["Term", "Meaning"], page["glossary"]) + "</details>")
    parts.append("<h2>Questions</h2>")
    parts += [_question(n, q) for n, q in enumerate(page["questions"], 1)]

    parts.append("<h2>Check results in numbers</h2><details><summary>G0 — can the laptop run the VLM?</summary>"
                 "<table><tr><th>model</th><th>GPU memory (GB of 8)</th><th>s per clip and question, slow way</th>"
                 "<th>s per clip for 64 questions, shortcut</th><th>passed</th></tr>")
    for r in g0:
        parts.append(f"<tr><td>{e(r['model_id'])}</td><td>{r['peak_vram_gb']:.2f}</td>"
                     f"<td>{r['seconds_per_pair_uncached']:.3f}</td><td>{r.get('seconds_per_clip_cached_64', float('nan')):.2f}</td>"
                     f"<td>{'yes' if r['pass'] else '<span class=tag>no — see Q4</span>'}</td></tr>")
    parts.append("</table></details><details><summary>G1 — does the VLM improve the ranking? (validation mAP)</summary>"
                 "<p class='mut'>Each cell: LoRA alone → LoRA + VLM (VLM alone). Needed: +0.02.</p>"
                 "<table><tr><th>set-up</th><th>Protocol A</th><th>Protocol B</th><th>passed</th></tr>")
    for name, c in g1["configs"].items():
        cell = lambda p: (f"{c[p]['lora_val_map']:.3f} → {c[p]['best_val_map']:.3f} "  # noqa: E731
                          f"({c[p]['vlm_only_val_map']:.3f})")
        parts.append(f"<tr><td>{e(name)}{' <b>(used)</b>' if name == g1['teacher'] else ''}</td>"
                     f"<td>{cell('A')}</td><td>{cell('B')}</td><td>{'yes' if c['pass'] else 'no — see Q2'}</td></tr>")
    parts.append("</table></details>")

    parts.append(f"<h2>Where the VLM goes wrong — examples (for Q2)</h2><p>Each row is one <b>validation</b> clip: "
                 f"8 frames, left to right in time, cropped around the person or vehicle ({e(view)} view). "
                 "<b>Said YES, wrong</b> = the clip shows a different activity, but the VLM was sure it matched. "
                 "<b>Said NO, wrong</b> = the clip does show this activity, but the VLM said no. "
                 "Activities where the VLM hurts the ranking most come first.</p>")

    def hurt(act):                                    # mean VLM-minus-LoRA AP over the protocols that have it
        vals = [effect[p][act] for p in effect if act in effect[p]]
        return sum(vals) / len(vals) if vals else 0.0
    for act in sorted(examples, key=hurt):
        note = ", ".join(f"{p}: {effect[p][act]:+.2f}" for p in sorted(effect) if act in effect[p])
        parts.append(f"<h3 class='act'>{e(act.replace('_', ' '))}</h3>")
        if note:
            parts.append(f"<p class='mut'>Adding the VLM changed validation AP for this activity by {e(note)}.</p>")
        for kind, iid, text, margin in examples[act]:
            frames, _ = crops.frames(iid, view)
            imgs = "".join(f"<img src='{_jpeg(f)}' alt=''>" for f in frames)
            parts.append(f"<div class='ex'><b>{e(kind)}</b>: query “{e(text)}” → VLM {_sure(margin)} "
                         f"(score {margin:+.1f}); the clip is really <b>{e(activity_of_inst[iid].replace('_', ' '))}</b> "
                         f"<span class='mut'>(instance {iid})</span><br>{imgs}</div>")
    parts.append("</body></html>")
    return "\n".join(parts)


def main(argv=None) -> int:
    from sfs.data.manifest import read_manifest
    from sfs.eval.queries import load_phrasings
    from sfs.teacher.frames import CropIndex
    from sfs.teacher.store import load_scores, read_scores

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="manifest_v2_drive/manifest_v2.csv")
    ap.add_argument("--phrasings", default="queries/phrasings.csv")
    ap.add_argument("--crops", default="data/crops_v2")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--g0", nargs="+", required=True)
    ap.add_argument("--g1", required=True)
    ap.add_argument("--scores", required=True, help="G1 validation scores of the teacher configuration")
    ap.add_argument("--diagnostic", help="G1 diagnostic json: per-activity VLM-minus-LoRA AP for the notes")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    m = read_manifest(a.manifest)
    phr = load_phrasings(a.phrasings)
    act_of_text = {t: act for act, ts in phr.items() for t in ts}
    act_of_inst = dict(zip(m.instance_id.astype(int), m.activity))
    scores, _ = load_scores(a.scores)
    view = read_scores(a.scores)[3]["view"]
    g1 = json.load(open(a.g1))
    effect = {}
    if a.diagnostic:
        diag = json.load(open(a.diagnostic))
        key = f"per_activity_{g1['teacher']}_minus_lora"
        effect = {p: d[key] for p, d in diag.items() if isinstance(d, dict) and key in d}
    page = render(json.load(open(a.questions, encoding="utf-8")), [json.load(open(p)) for p in a.g0], g1,
                  mistakes(scores, act_of_text, act_of_inst), CropIndex(a.crops), act_of_inst, view, effect)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
