# Spec — Offline Data Preparation & Embedding Generation

**Date:** 2026-09-28
**Status:** APPROVED 2026-10-21 — amendments in §8 and §9 supersede earlier sections.
**Related:** [`spec-retrieval-system.md`](spec-retrieval-system.md) (serving path), [`project-selection.md`](project-selection.md) Plan F

---

## 1. What this change covers

The first build step of the offline path: turning the MEVA subset on Drive into a frozen-encoder
embedding index that can be downloaded and used locally. Concretely:

1. **Manifest v2** — add train/eval split, deduplication and leakage checks to the manifest.
2. **Frame sampling policy** — how many frames per activity instance, and which.
3. **Region policy** — actor-region crops versus full frames.
4. **Encoder choice** — the frozen baseline encoder and its identity.
5. **Embedding job** — resumable, per-video processing on Colab.
6. **Storage format** — what gets downloaded to the local machine.
7. **Code structure** — move from notebook-cell strings to a tested Python package.

**Out of scope** (separate specs later): LoRA adaptation, VLM teacher labelling, distillation,
the serving path, the query set.

---

## 2. State before this change

Established by Cells 7–8 on 2026-09-28:

| Fact | Value | Implication |
|---|---|---|
| Annotated clips in repo | 2,212 activity files: **1,443** `kitware-meva-training`, **769** `kitware` | The repo already separates training from evaluation annotations |
| Activity instances in repo | 66,221 across 38 types | — |
| Video files on Drive | 677 (~70 GB) | — |
| Instances **with** matching video | **10,721** across **32** types | Sufficient to proceed; exceeds the "thousands" requirement |
| Instances without video | 55,500 | Recoverable by targeted download if needed |
| Class balance | 1,263 (`person_talks_to_person`) down to 31 (`person_reads_document`) within the top 25; 7 more types below 31 | ~40:1 imbalance or worse |
| Instance duration | min 8, p25 32, **median 67**, p75 144, mean 322, max 9,009 frames | Median ≈ 2.2 s; extreme right tail (max = 5 min) |
| Frame rate | **30 fps** in the main corpus (373 frames = 12.43 s) | Examples folder is 25 fps; proposal stated 25 fps. Always read fps from the container |
| Resolution | 1920×1072 | Not 1080; read from container |
| Vocabulary version | Repo uses `person_talks_to_person` style; examples folder uses `People_Talking` style | Examples folder cannot be mixed in without a mapping — use repo annotations only |
| Box annotations | `geom.yml`: one box per actor per frame, dense (max gap 1 frame) | Actor-region cropping is feasible |

**Gaps in the current manifest:**
- No `split` column — the train/eval distinction above is lost.
- No check for the same clip appearing in both groups (train→eval leakage).
- Per-split usable counts are unknown. If the downloaded hours are mostly training clips, the eval
  split may be too thin to measure anything.

---

## 3. Decisions

### D1 — Manifest v2

Add to each row: `split` (`train` from `kitware-meva-training`, `eval` from `kitware`),
`anno_file`, `fps`, `actor_ids`. Then:

- **Dedupe** on `(clip_stem, activity, frame_start, frame_end)`.
- **Leakage check:** fail if any `clip_stem` appears in both splits.
- **Camera overlap report:** list cameras present in both splits. Not a failure — the official split
  shares cameras — but reported, since scene familiarity inflates eval numbers.
- **Validation split:** hold out ~10% of *training clips* (by clip, never by instance) for model selection.
- **Print per-split counts** overall and per activity.

**Gate:** if the eval split has < 1,500 usable instances, or any activity type used for evaluation has
< 20 eval instances, run a *targeted* download of eval clips before embedding (see D2).

### D2 — Corpus scope: annotated clips only

The index contains **only clips that have annotations**. Unannotated clips on Drive are excluded,
because MEVA's annotations are exhaustive *within* an annotated clip but say nothing about
unannotated ones: an unannotated clip may well contain "person talks to person", and indexing it
would turn correct retrievals into false negatives, understating recall.

If more data is needed, download is **annotation-driven**: list annotated clips lacking video, map
each stem to its S3 key (`drops-123-r13/{date}/{hour}/{stem}.r13.avi`, verified against a bucket
listing), prioritise eval clips and tail activity types, and fetch only those.

### D3 — Frame sampling

```
N = clamp( ceil(duration_s × 2), 1, N_max=16 )      # 2 frames per second, capped
frames = uniform over [frame_start, frame_end], endpoints included when N >= 2
```

At 30 fps: p25 instance → 3 frames, median → 5, p75 → 10; the cap is reached at 8 s.
Rate-based sampling keeps the frame count proportional to evidence rather than fixed, and the cap
bounds the length bias described in `spec-retrieval-system.md` §5.2.

### D4 — Region policy: actor crop (primary) + full frame (ablation)

**Problem.** In the inspected example, the actor's box is 106×242 px in a 1920×1072 frame. A
full-frame resize to 224×224 reduces it to ~12×51 px with anisotropic distortion — too small for any
encoder to recognise an action. Full-frame embedding of MEVA will likely underperform badly for this
reason alone.

**Policy.** For each sampled frame, take the **union of the instance's actor boxes** at that frame,
pad by 25% per side, expand to a square, enforce a minimum side of 224 px, clip to image bounds, and
resize to 224×224. If no box exists at that frame, use the nearest frame's box within ±5 frames;
otherwise fall back to the full frame and count the fallback.

Embed **both** views (`crop`, `full`). The crop view is the primary index; the full view is the
ablation arm that quantifies how much the region policy matters — a result worth reporting.

**Serving consistency.** At alert time the detector (`nexa_fight`) supplies person boxes. The same
crop function must be used offline and online, or offline results will not transfer. This extends
the encoder-consistency contract: the crop policy is part of `encoder_id`.

### D5 — Encoder: SigLIP 2 ViT-B/16 (frozen, baseline)

`google/siglip2-base-patch16-224`, pinned to an exact revision. Apache-2.0. 768-dim embeddings.

| Reason | Detail |
|---|---|
| Retrieval quality | SigLIP 2 outperforms SigLIP and other open baselines on zero-shot retrieval, most noticeably at B size |
| Small-batch training | Sigmoid loss outperforms softmax contrastive loss at small batch sizes. This matters directly for LoRA fine-tuning on an 8 GB GPU, where large batches are impossible |
| Localisation | SigLIP 2 was trained with objectives targeting localisation and dense features, relevant to small subjects |

Fallback: `openai/clip-vit-base-patch16` (512-dim). Changing encoder is a full reindex (§D8).

**Consequence for the other spec:** `spec-retrieval-system.md` assumes 512-dim vectors. It must be
updated to 768 when this spec is approved.

### D6 — Embedding job

- **One pass per video.** Collect every frame index needed by every instance in the video, sort,
  read sequentially once. Per-instance seeking over the Drive mount would be far slower.
- **Resumable.** One output shard per video; skip if the shard exists and validates. Colab sessions
  time out; a restart must lose at most one video's work.
- **fp16 on GPU**, batch as large as memory allows.
- **Text side not needed yet.** Query embeddings are computed at evaluation time.

### D7 — Storage

Per-video shard `shards/{stem}.npz`:

| Array | Type | Shape |
|---|---|---|
| `emb` | float16, L2-normalised | n × 768 |
| `instance_id` | int32 | n |
| `frame_idx` | int32 | n |
| `view` | uint8 (0 = crop, 1 = full) | n |
| `fallback` | bool (crop fell back to full frame) | n |

Consolidated per split: `embeddings_{train,val,eval}.npz`, `instances_{split}.parquet`
(manifest rows), and `encoder.json`.

**Size estimate:** ~10.7k instances × ~6–8 frames × 2 views × 768 × 2 bytes ≈ **200–260 MB**.
This corrects the earlier "10–50 MB" figure, which assumed one 512-dim vector per clip; frame-level,
two-view, 768-dim storage is larger but still trivial to download.

### D8 — Encoder identity

`encoder.json` records model id, HF revision SHA, image preprocessing parameters, and the crop-policy
parameters (pad ratio, min side, fallback window). `encoder_id` = hash of that JSON. Every shard
carries it; consolidation fails if shards disagree.

### D9 — Code structure

Stop writing pipeline code as notebook-cell strings. Create a package, test it locally, and have
Colab install and run it:

```
pyproject.toml
src/sfs/
  data/kpf.py         parse activities.yml / geom.yml
  data/manifest.py    manifest v2 (supersedes scripts/colab_build_manifest.py logic)
  data/sampling.py    D3
  data/regions.py     D4
  embed/encoder.py    D5, D8
  embed/run.py        D6 — CLI: python -m sfs.embed.run --split ...
  store/shards.py     D7
tests/
  fixtures/           small activities/geom files from the local examples set
  test_kpf.py  test_sampling.py  test_regions.py  test_manifest.py  test_shards.py
  test_embed_smoke.py (slow; one local example clip, end to end)
scripts/colab_embed.py  thin runner: clone repo, pip install -e ., run CLI
```

Colab then runs **exactly the tested code** rather than a pasted copy that can drift.

---

## 4. Verification

| Level | Check | Where |
|---|---|---|
| Unit | KPF parsing on fixtures; sampling (N rule, clamping, spacing, endpoints); region (padding, squaring, clipping, min side, fallback); manifest (split assignment, dedupe, leakage failure); shard round-trip | Local, pytest |
| Smoke | Full pipeline on 2–3 local example clips (RTX 4060) | Local |
| Run integrity | vectors written = frames planned; no NaN/inf; norms ≈ 1; one `encoder_id` across shards; fallback rate reported | Colab, end of job |
| Sanity | For each activity, zero-shot text prompt → mean similarity to matching instances exceeds non-matching (per-activity AUC > 0.5). Computed for both views, which is also the first crop-vs-full signal | Local, on downloaded embeddings |

The sanity check is not the baseline evaluation (that needs the query set — separate spec). It
exists to catch a broken pipeline before any training is built on top of it.

---

## 5. State after this change

- `src/sfs` package with unit tests passing locally.
- Manifest v2 on Drive with splits, leakage check passed, per-split counts known.
- Frozen SigLIP 2 embeddings (crop + full views) for all usable instances, on Drive and downloaded
  locally (~200–260 MB), identified by a single `encoder_id`.
- Sanity AUCs recorded per activity and per view.
- `spec-retrieval-system.md` updated to 768-dim; README updated with corrected dataset figures
  (2,212 annotated clips, 66,221 instances, 38 types, 30 fps).

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Drive-mount video decoding is slow (~70 GB) | Sequential per-video reads; resumable shards; estimate 1–3 h on Colab |
| Eval split too thin after manifest v2 | Gate in D1 triggers targeted eval download |
| Many-actor instances produce a union box covering most of the frame | Acceptable — degrades gracefully to full-frame; reported via crop-area statistics |
| Boxes missing at sampled frames | Nearest-frame fallback, then full-frame fallback; rate reported |
| Local environment lacks torch | Setup step: venv + CUDA torch + transformers + pytest before any code |

---

## 7. Decisions requiring approval

1. **SigLIP 2 ViT-B/16** as the encoder (vs CLIP ViT-B/16).
2. **Actor-crop primary + full-frame ablation**, both embedded.
3. **2 fps sampling, cap 16.**
4. **Official split** (`kitware-meva-training` → train/val, `kitware` → eval) with camera-overlap reporting.
5. **Annotated clips only** in the index; annotation-driven download if the D1 gate fails.
6. **Move to a `src/sfs` package** with tests; Colab becomes a thin runner.

---

## 8. Review outcome and amendments (2026-10-21)

Where this section conflicts with §3, this section wins.

| # | Decision | Outcome |
|---|---|---|
| 1 | Encoder | **Amended.** Embed with **both SigLIP (`google/siglip-base-patch16-224`) and SigLIP 2 (`google/siglip2-base-patch16-224`)** in the same decoding pass. The winner is selected on the **validation** split, never on eval. Each has its own `encoder_id`; shards are kept separate per encoder. |
| 2 | Region policy | **Amended — three views** (rationale below). |
| 3 | Sampling | Approved. 2 fps, cap 16, both exposed as configuration rather than constants. |
| 4 | Split | **Amended.** Protocol A = official split (primary). Protocol B = **unseen-camera** split (eval cameras never appear in train), reported as a generalisation test. Protocol B is *reported alongside*, not chosen between — selecting a split by which gives the higher score would be selecting on the test set. |
| 5 | Corpus | Approved, with clarification: unannotated clips are **not** used for metrics (no ground truth exists for them). They may appear only in the qualitative demo. |
| 6 | Package + runner | Approved, with runner changed to the **VS Code Google Colab extension** on the Pro account (see "Data source" below). |

### D4 (amended) — three views

The original crop policy risked discarding context that distinguishes look-alike actions (e.g. a
fight and a hug share close proximity; setting and bystander reactions can separate them). Each
sampled frame is embedded three ways:

| View | Region | Purpose |
|---|---|---|
| `tight` | union of actor boxes, +25% pad per side, squared, min side 224 px | Maximum detail on the actors |
| `context` | union of actor boxes expanded to **3× width and height**, squared, min side 448 px, clipped to frame | Actors plus surroundings |
| `full` | whole frame | Ablation: what cropping buys at all |

A lone person in a large frame is handled by `context`: the box grows around them. The
tight-vs-context-vs-full comparison is a reported result. Per frame we also record the actor box
size in pixels, so accuracy can be analysed against subject size — that is where low image
quality will actually show up.

**Clarification on image quality.** A blurry or small crop does not corrupt the data; it produces a
weaker embedding for that frame. Upscaling a small crop adds no information but removes none
either. The real failure mode is subjects too small to resolve, which the box-size analysis
measures directly rather than assumes.

**Known limitation.** Embeddings are per frame; motion is only captured through aggregation over
sampled frames. Actions that differ mainly in motion (fight vs hug) may remain confusable — a
limitation to state in the report, and a candidate for future work with a video encoder.

Storage estimate rises to ~3 views × 2 encoders ≈ **750 MB–1.5 GB** total; still a reasonable download.

### Data source (amended)

The VS Code Colab extension does not support `drive.mount()` (per the extension's documentation
and setup guides as of late 2025; to be re-checked at setup), and the Pro account is a different
Google account from the one holding the Drive data. Therefore:

- **Video:** downloaded directly from S3 into the runtime's local disk at datacenter speed — only
  the annotated clips needed (a few tens of GB), not the 70 GB Drive copy.
- **Annotations:** cloned from `gitlab.kitware.com` into the runtime (reachable from Colab; ~30 s).
- **Outputs:** embedding shards written to runtime disk, then downloaded through VS Code's file
  explorer (≤1.5 GB). Shards are resumable, so a runtime reset costs one video's work.

Drive is no longer on the critical path. The existing Drive copy remains a backup.

---

## 9. Final amendment (2026-10-21) — runner and data source

Supersedes "Data source (amended)" in §8 and the runner in §8 row 6.

- **Runner:** browser Colab (Pro account). No VS Code extension. The repo is private, so code is
  **pasted**, not cloned. To prevent drift between tested code and pasted code, a build script
  generates the Colab cell from the `src/sfs` package — the pasted cell *is* the tested code.
- **Data source:** the existing Drive copy (`MyDrive/DL-Project/meva`), mounted with
  `drive.mount()` as before. If the Pro account is not the Drive owner, the `DL-Project` folder is
  shared with it. **Nothing from S3 is downloaded to the local machine.** The only artefacts that
  come to the local machine are the embedding shards and manifest (≤1.5 GB).
- **Split:** Protocol A (official) primary, Protocol B (unseen camera) reported — confirmed.
