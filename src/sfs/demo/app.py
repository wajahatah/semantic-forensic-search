"""Search demo UI (spec-search-demo §4.5): Gradio, local only.

    python -m sfs.demo.app                      # http://127.0.0.1:7860
    python -m sfs.demo.app --no-vlm             # without the Verified level (CPU or a busy GPU)

The handlers live in DemoApp (no Gradio types in or out except gallery items), so they are unit-tested
without a browser; build_ui only wires them to components.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from sfs.demo.search import LEVELS, SearchResult

LEVEL_LABELS = {"fast": "Fast: LoRA only (0 VLM calls)",
                "standard": "Standard: LoRA + student (0 VLM calls)",
                "verified": "Verified: + VLM checks the top k (k calls)"}
SHOWN = 20
FREE_EXAMPLES = ("someone carrying a box out of a building", "two people chatting near a car",
                 "a person on the phone at the bus stop")


def _fmt_s(x: float) -> str:
    if x < 1:
        return f"{x * 1000:.0f} ms"
    return f"{x:.1f} s" if x < 60 else f"{x / 60:.1f} min"


def cost_markdown(r: SearchResult) -> str:
    stages = [f"{name} {_fmt_s(sec)}" for name, sec in r.seconds.items()]
    total = sum(r.seconds.values())
    vlm_all = r.vlm_all_seconds
    lines = [f"**{len(r.ids)} shown of {r.n_searched:,} clips searched** · {LEVEL_LABELS[r.level].split(':')[0]}"
             + (" · frozen SigLIP 2 baseline" if r.system == "frozen" else ""),
             f"Time: {' · '.join(stages)} · **total {_fmt_s(total)}**",
             f"VLM calls: **{r.vlm_calls}** · a VLM on every searched clip would take "
             f"{r.n_searched:,} calls ≈ {_fmt_s(vlm_all)} of VLM time "
             f"({_fmt_s(r.vlm_seconds_per_call)} per call, {'measured now' if r.vlm_calls else 'Result 04'})"]
    return "  \n".join(lines)


class DemoApp:
    def __init__(self, searcher, previews, phrasings: dict[str, list[str]] | None = None):
        self.s, self.p = searcher, previews
        self.phr = phrasings or {}
        self.text_activity = {t: a for a, ts in self.phr.items() for t in ts}
        c = searcher.b.clips
        self.choices = {"site": sorted(c.site.unique()), "camera": sorted(c.camera.unique()),
                        "date": sorted(c.date.unique())}
        self.hours = (int(c.hour.min()), int(c.hour.max()))

    def examples(self) -> list[str]:
        return [ts[0] for _, ts in sorted(self.phr.items())] + list(FREE_EXAMPLES)

    def filters(self, sites, cameras, dates, hour_from, hour_to) -> dict:
        f = {"site": sites or None, "camera": cameras or None, "date": dates or None}
        if hour_from is not None and int(hour_from) > self.hours[0]:
            f["hour_from"] = int(hour_from)
        if hour_to is not None and int(hour_to) < self.hours[1]:
            f["hour_to"] = int(hour_to)
        return f

    def caption(self, r: SearchResult, n: int, show_labels: bool, query: str) -> str:
        i = int(r.ids[n])
        row = self.s.b.clips.iloc[self.s.pos[i]]
        cap = f"#{n + 1} · {row.camera} {row.site} · {row.date} {int(row.hour):02d}h"
        if r.level == "verified" and i in r.vlm_checked:
            cap += f" · VLM yes {1 / (1 + np.exp(-r.scores['vlm'][n])):.0%}"
        if show_labels:
            want = self.text_activity.get(query.strip())
            mark = "" if want is None else (" ✓" if row.activity == want else " ✗")
            cap += f" · [{row.activity}]{mark}"
        return cap

    def gallery(self, r: SearchResult, show_labels: bool, query: str) -> list[tuple[str, str]]:
        return [(self.p.path(int(i)), self.caption(r, n, show_labels, query)) for n, i in enumerate(r.ids[:SHOWN])]

    def correct_count(self, r: SearchResult, query: str) -> str:
        want = self.text_activity.get(query.strip())
        if want is None:
            return ""
        acts = self.s.b.clips.activity.to_numpy()[[self.s.pos[int(i)] for i in r.ids[:10]]]
        return f"  \nLabels: this is an evaluation phrasing for **{want}**; {int((acts == want).sum())} of the top 10 match."

    def run(self, query, level_label, k, sites, cameras, dates, hour_from, hour_to, show_labels):
        query = (query or "").strip()
        if not query:
            return [], "Type a description of what you are looking for.", None
        level = next(lv for lv, lab in LEVEL_LABELS.items() if lab == level_label)
        if level == "verified" and not self.s.vlm_available():
            return [], "The Verified level needs the VLM on a GPU; it is not available here.", None
        r = self.s.search(query, level=level, k=int(k), top=SHOWN,
                          filters=self.filters(sites, cameras, dates, hour_from, hour_to))
        md = cost_markdown(r) + (self.correct_count(r, query) if show_labels else "")
        return self.gallery(r, show_labels, query), md, {"query": query, "ids": [int(i) for i in r.ids],
                                                          "scores": {k_: [float(x) for x in v] for k_, v in r.scores.items()}}

    def compare(self, query, sites, cameras, dates, hour_from, hour_to, show_labels):
        query = (query or "").strip()
        if not query:
            return [], [], "Type a description first."
        f = self.filters(sites, cameras, dates, hour_from, hour_to)
        q = self.s.embed(query)
        base = self.s.search_vector(q, query, system="frozen", top=SHOWN, filters=f)
        final = self.s.search_vector(q, query, level="standard", top=SHOWN, filters=f)
        md = ("**Left:** frozen SigLIP 2 (Result 02 baseline). **Right:** final system without a VLM "
              "(LoRA + student, Results 03–04).")
        if show_labels:
            md += self.correct_count(base, query).replace("Labels:", "Left:") + self.correct_count(final, query).replace(
                "Labels:", "Right:")
        return self.gallery(base, show_labels, query), self.gallery(final, show_labels, query), md

    def detail(self, state, index: int):
        if not state or index is None or index >= len(state["ids"]):
            return [], ""
        i = state["ids"][index]
        row = self.s.b.clips.iloc[self.s.pos[i]]
        frames, nums = self.p.frames(i, "tight")
        items = [(f, f"frame {n}") for f, n in zip(frames, nums)]
        sc = {k: v[index] for k, v in state["scores"].items() if not np.isnan(v[index])}
        md = (f"**Clip {i}** · site {row.site} · camera {row.camera} · {row.date} {int(row.hour):02d}h · "
              f"video frames {row.frame_start}–{row.frame_end} (`{row.clip_stem}`)  \n"
              f"Annotated MEVA activity: `{row.activity}`  \n"
              + " · ".join(f"{k} {v:.3f}" for k, v in sc.items()))
        return items, md


def build_ui(app: DemoApp):
    import gradio as gr
    with gr.Blocks(title="Semantic Forensic Search — MEVA") as ui:
        gr.Markdown("## Semantic forensic search over MEVA surveillance clips\n"
                    f"Corpus: the protocol A **evaluation** split — {len(app.s.b.ids):,} annotated clips the "
                    "models never trained on. Type what you are looking for; choose how much checking to pay for.")
        with gr.Row():
            query = gr.Textbox(label="Query", placeholder="e.g. a person opening a car door", scale=4)
            example = gr.Dropdown(app.examples(), label="Examples", scale=2)
        with gr.Row():
            level = gr.Radio([LEVEL_LABELS[lv] for lv in LEVELS], value=LEVEL_LABELS["standard"], label="Cost level")
            k = gr.Slider(5, 20, value=10, step=5, label="k (Verified: clips the VLM checks)")
        with gr.Accordion("Filters (site / camera / day / hour)", open=False):
            with gr.Row():
                sites = gr.Dropdown(app.choices["site"], multiselect=True, label="Site")
                cameras = gr.Dropdown(app.choices["camera"], multiselect=True, label="Camera")
                dates = gr.Dropdown(app.choices["date"], multiselect=True, label="Day")
            with gr.Row():
                h0 = gr.Slider(*app.hours, value=app.hours[0], step=1, label="From hour")
                h1 = gr.Slider(*app.hours, value=app.hours[1], step=1, label="To hour")
        labels = gr.Checkbox(False, label="Show annotated labels (✓/✗ when the query is an evaluation phrasing)")
        flt = [sites, cameras, dates, h0, h1]
        with gr.Tab("Search"):
            go = gr.Button("Search", variant="primary")
            cost = gr.Markdown()
            results = gr.Gallery(label="Results (context view, 2 fps)", columns=5, height="auto",
                                 object_fit="contain", allow_preview=False)
            with gr.Row():
                frames = gr.Gallery(label="Selected clip: tight crops in time order", columns=8, height=160,
                                    allow_preview=True)
            info = gr.Markdown()
            state = gr.State(None)
        with gr.Tab("Compare with the baseline"):
            go_cmp = gr.Button("Compare", variant="primary")
            cmp_md = gr.Markdown()
            with gr.Row():
                left = gr.Gallery(label="Frozen SigLIP 2", columns=4, height="auto", allow_preview=False)
                right = gr.Gallery(label="Final system (LoRA + student)", columns=4, height="auto",
                                   allow_preview=False)

        example.change(lambda e: e or gr.update(), example, query)
        inputs = [query, level, k, *flt, labels]
        go.click(app.run, inputs, [results, cost, state])
        query.submit(app.run, inputs, [results, cost, state])

        def on_select(st, evt):
            return app.detail(st, evt.index)
        # a real class, not a string: this module uses postponed annotations, and Gradio injects the
        # click event only when it can see the SelectData type
        on_select.__annotations__ = {"evt": gr.SelectData}

        results.select(on_select, state, [frames, info])
        go_cmp.click(app.compare, [query, *flt, labels], [left, right, cmp_md])
    return ui


def main(argv=None) -> int:
    from sfs.demo.previews import Previews
    from sfs.demo.search import Bundle, Searcher
    from sfs.eval.queries import load_phrasings
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", default="data/demo/A")
    ap.add_argument("--crops", default="data/crops_v2")
    ap.add_argument("--phrasings", default="queries/phrasings.csv")
    ap.add_argument("--no-vlm", action="store_true", help="hide the Verified level's VLM")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args(argv)
    s = Searcher(Bundle(a.bundle), a.crops)
    if a.no_vlm:
        s.vlm_path = ""
    prev = Previews(s.crops, os.path.join(a.bundle, "previews"))
    n = prev.render_all(s.b.ids)
    if n:
        print(f"rendered {n:,} previews")
    s.embed("warm up")                                    # load every model before the first query
    _ = s.student
    if s.vlm_available():
        print("loading the VLM for the Verified level ...")
        _ = s.vlm
    app = DemoApp(s, prev, load_phrasings(a.phrasings))
    build_ui(app).launch(server_name="127.0.0.1", server_port=a.port, share=False,
                         allowed_paths=[os.path.abspath(os.path.join(a.bundle, "previews"))])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
