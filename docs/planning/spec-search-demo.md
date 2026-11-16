# Spec — Search demo on MEVA

**Date:** 2026-11-05 · **Status:** draft; **build started before approval at the owner's request**
(D-054: "write the demo spec and start working on it, I'll review the spec later"). Changes from the
review are applied to the build. · **Log:** D-054, D-056
**Scope:** MEVA only. The nexa_fight part of the plan (D-049) was dropped by the owner on 2026-11-05.
**Presentation:** a screen recording, not a live demo (owner, 2026-11-05, D-058).
**Time budget:** ~3–4 working days of the 13 left; the report and slides keep the rest.

---

## 1. What this change does

It turns the evaluated system (Results 03–05) into an interactive search tool that runs on the laptop.
A user types a free-text description ("a person opening a car door", "someone carrying a box out of a
building"). They can narrow the search to a site, camera, day or hour, and choose how much checking to
pay for. They get a ranked list of matching MEVA clips, shown as short animations, with the cost of the
search: time per stage and the number of VLM calls.

The demo is the project's headline idea made visible: **the same query at three cost levels**.
- **Fast:** LoRA only.
- **Standard:** LoRA + student.
- **Verified:** plus the VLM checks the top k.

The demo also shows each level's VLM calls and time against "VLM on everything".

## 2. State before

- All search pieces exist as evaluation code, not as a service:
  - frame-embedding index and mean-of-top-3 scoring: `sfs.eval.retrieval.InstanceIndex`;
  - LoRA-adapted embeddings: `data/embeddings_lora/A-middle/embeddings.npz`;
  - student features and the selected GT student: `data/rerank/features_A-middle.npz`, `runs/student/A-GT/best.pt`, β = 1;
  - fusion: `sfs.eval.cascade_run.load_protocol`;
  - VLM re-ordering: `sfs.eval.cascade.topk_rerank`, with γ = 2 from Result 04's validation choice;
  - the 2B Instruct scorer: `sfs.teacher.vlm`, weights in `data/models/Qwen3-VL-2B-Instruct`.
- Clips can only be seen as stored crops (`data/crops_v2`: tight PNG, context JPEG ≤ 448 px, ≤ 16
  frames per clip at 2 fps). The raw videos (~70 GB) are on Google Drive, not on the laptop.
- Every query must be one of the 66 fixed phrasings. There is no free-text path, no metadata filter
  outside the Q2 builder, and no UI.
- Installed on the laptop: `gradio 6.30`, `chromadb 1.5.9`, `fastapi`, `uvicorn`, `opencv`. No new
  download is needed.

## 3. State after

- New package `sfs.demo`:
  - `index.py`: builds a demo bundle once from the existing stores;
  - `search.py`: the search engine, no UI;
  - `app.py`: the Gradio UI.
- Command: `python -m sfs.demo.app` → `http://127.0.0.1:7860`, local only, no public share link.
- For an eval phrasing, the demo reproduces what the evaluation measured. A slow test pins this, so
  what the demo shows is what the report measured. **Measured 2026-11-05 on all 66 Q1 and 1,140 Q2
  queries × Fast/Standard:**
  - Success@1 and Success@10 are identical on every query.
  - AP differs by ≤ 0.0001 on Q1 and ≤ 0.012 on one Q2 window, with a mean ≤ 0.00002.
  - Scores differ by ≤ 1e-7 (LoRA) and ≤ 5e-6 (fused). The cause is float rounding: BLAS
    multiplies one query vector in a different order than the evaluation's 66 at once, so near-tied
    clips can swap places. Bit-identical rankings are therefore not possible for typed text.
- Docs: `docs/demo.md` (how to run it, what each control does, a scripted walk-through for the
  presentation) and a README section.

## 4. Design

### 4.1 Corpus (D1)
**The protocol A eval split: 3,726 clips, 117 videos, 24 cameras, 4 sites, 3 days.** The LoRA model
and the student never trained on these clips (A's train/val are disjoint from them), so the demo shows
honest generalisation. Indexing all 10,678 clips would mix in training clips, which the model has
memorised to some degree.

### 4.2 The bundle (`sfs.demo.index`, built once)
- `meta.sqlite`, table `clips`: instance id, site, camera, date, hour, start/end frame, video stem, the
  **annotated activity** (MEVA's human label, used only for the "show labels" view, §4.5), and the crop
  shard and rows of the clip's frames.
- `vectors.npz`: the clip index (LoRA tight frame embeddings + row offsets) and, for the comparison
  view, the frozen SigLIP 2 frame embeddings. Copied from the stores; ids and order checked.
- The student features for the corpus are sliced from `features_A-middle.npz`.
- `bundle.json`: sources with their hashes, encoder ids, β, γ, the student run, and the corpus split.
  The engine refuses a bundle whose sources do not match.
- Size: 263 MB for the bundle plus ~1.2 GB of preview animations (measured 2026-11-05). It goes under `data/demo/A/`, which git ignores.

### 4.3 Search engine (`sfs.demo.search.Searcher`)
`search(text, filters, level, k) → ranked clips + cost`:
1. **Text → vector:** the frozen SigLIP 2 text side, the same call the evaluation uses.
2. **Fast (LoRA):** mean of the top 3 frame similarities per clip, over the whole corpus.
3. **Standard (LoRA + student):** the student scores every clip (~0.18 s per query on the laptop GPU,
   Result 04), then the fused score is z(student) + β·z(LoRA), with β = 1.
4. **Verified:** the 2B Instruct VLM scores the top k of the Standard ranking (default k = 10, choices
   5/10/20). The top k is re-ordered by z(VLM) + γ·z(first stage), with γ = 2 (`topk_rerank`,
   unchanged). Cost = k calls, about 0.17 s each.
5. **Filters** (site, camera, day, hour range): applied **after** scoring the whole corpus, exactly as
   the Q2 evaluation does, so the z-normalisation and the scores are identical to the evaluated system.
6. Returns the top 50 with per-stage seconds, VLM calls, and "VLM on everything" for comparison
   (corpus size × 0.17 s).

The engine has no UI code and is unit-tested. Models load once, lazily: the VLM only loads the first
time Verified is used, about 5 GB of GPU memory. On a CPU-only machine, Verified is disabled with a
message.

### 4.4 Previews
- Each result shows its **context-view frames as a short animation** (2 fps source played at 4 fps,
  256 px), rendered once per clip into `<bundle>/previews/` (a disk cache, ~all 3,726 clips rendered
  at first start).
- The detail panel shows the frame strip with the tight crops.
- No raw video is needed. Full-frame clips would need the Drive videos, which are not on the laptop;
  that is optional decision D5.

### 4.5 UI (`sfs.demo.app`, Gradio)
- **Search:**
  - controls: query box, example-query menu, cost level (Fast / Standard / Verified + k), filters;
  - output: result gallery (top 20, animated) and a cost bar (seconds per stage, VLM calls, what
    "VLM on everything" would cost).
- **Compare:** the same query side by side, frozen SigLIP 2 vs the final system. This makes the
  fine-tuning gain visible.
- **Show labels** (toggle, off by default): each result's annotated MEVA activity. When the query is
  one of the eval phrasings, correct results are marked. This lets the audience check the
  answers live.
- **Detail:** clicking a result shows its frames, metadata, scores per stage, and for Verified the
  VLM's yes-probability.

### 4.6 Vector store (D2)
The proposal named ChromaDB + SQLite. **Recommended: exact in-memory search plus SQLite metadata.**
- The corpus is ~60k frame vectors, and an exact search takes milliseconds.
- ChromaDB's HNSW index is approximate. Its ranking could differ from the evaluated one, and the
  clip score (mean of a clip's top-3 frames) would need a second exact pass anyway.
- The final report states the change from the proposal and why.
- A ChromaDB back-end for stage 1, with a recall check against the exact search, would add ~0.5 day
  if the owner wants it.

## 5. Tests
- **Bundle:** built from a tiny fixture; ids, order and hashes are checked; the engine refuses a
  mismatched bundle.
- **Parity (the key test, slow, real data):** for every Q1 and Q2 eval query, Fast and Standard give
  the evaluated Success@1/@10 exactly. Scores match within 1e-5, and AP within float rounding (mean
  |ΔAP| < 1e-4).
- **Code-path test (fixture):** Fast / Standard / frozen equal the evaluation functions
  (`InstanceIndex.score`, `score_matrix`, `zcols`) exactly. Verified with a fake VLM equals
  `topk_rerank`.
- **Filters:** a Q2 window gives the Q2 corpus and order.
- **Cost:** VLM calls = min(k, clips left after filtering); VLM on CPU is refused cleanly.
- **Previews:** the animation has the clip's frames in time order; the cache stays bounded.
- **UI:** the Gradio app builds, and its handlers run against a fake engine (no browser).
- **Full suite** before handing over.

## 6. Risks
| Risk | Mitigation |
|---|---|
| The demo silently differs from the evaluated system | Parity test against `load_protocol`; the bundle records its sources |
| GPU memory on the laptop (8 GB) | VLM loads only on first Verified use; SigLIP + student < 1 GB |
| Free-text queries far from MEVA's 22 activities give poor results | Shown as they are; example queries cover the activities; the docs say what the model was trained for |
| Appearance words ("blue jacket") | The frozen text side is kept (D-055), but the image side was tuned for activities; the docs say so |
| Time | Engine + parity first (day 1), UI (day 2), compare/labels/docs (day 3); ChromaDB only if asked |

## 7. Decisions for the owner (defaults in **bold** are being built)
1. **D1 Corpus:** **protocol A eval split (3,726 clips)**, or all clips.
2. **D2 Vector store:** **exact + SQLite (report the change from the proposal)**, or add a ChromaDB
   back-end (+0.5 day).
3. **D3 VLM in Verified:** **local Qwen3-VL-2B Instruct (Result 04)**, or the 8B on the office server
   over Tailscale. The 8B is better on A (+0.058 S@1). Since the demo is recorded (D-058), a network failure no
   longer matters, but the server's GPU job has to be restarted for the recording (released 2026-11-05).
4. **D4 Show-labels toggle:** **included, off by default**, or left out.
5. **D5 Previews:** **stored crops as animations**, or fetch the source videos for a handful of
   showcase clips from Drive at the office for full-frame playback.
