# Experiment & Decision Log

The single record of **what was tried, what came out, and why each decision was taken**.
Append-only: entries are never rewritten; a later entry supersedes an earlier one explicitly.
Detailed result write-ups live in `docs/results/`; specs in `docs/planning/`.

Entry format: **Context → Experiment / evidence → Result → Decision → Basis.**

---

## Phase 0 — Problem and dataset selection (2026-09-12 → 2026-09-20)

### D-001 · Project direction (2026-09-12)
- **Context:** Course requires a deep-learning project on a public dataset that is not on Kaggle and not already used for the proposed task.
- **Considered:** fight/violence detection (reuse of the earlier `nexa_fight` work), beggar face classification, plant disease, Urdu OCR, oral-cancer screening, industrial anomaly detection, and natural-language search over surveillance alerts.
- **Decision:** natural-language forensic search over surveillance video, with a cost constraint (as few vision-language-model calls as possible).
- **Basis:** reuses the existing surveillance pipeline legitimately; it has a measurable, under-reported angle (accuracy vs cost); and it trains real deep-learning components (LoRA, distillation).
- **Rejected:** *beggar face classification* — the label is not visible in an image, there is no ground-truth source, and it would mean collecting biometric data from a vulnerable group without consent. *Violence detection on RWF-2000* — the dataset is on Kaggle and heavily published. Detail: `docs/planning/project-selection.md`.

### D-002 · PlantCity disqualified (2026-09-12)
- **Experiment:** novelty check (Kaggle, Papers With Code, Hugging Face, GitHub, citing papers) on every shortlisted dataset.
- **Result:** PlantCity passed Kaggle/PwC but had already been used in a July 2026 MDPI paper for transfer learning on the same crops.
- **Decision:** dropped, even though it had been the first recommendation.
- **Basis:** fails the "not already used for this task" requirement. Lesson recorded: run novelty checks *before* recommending.

### D-003 · MEVA chosen as the dataset (2026-09-19)
- **Experiment:** actually downloaded the data rather than trusting the documentation: anonymous S3 access, the 121-clip example set, the full annotation repository (via Colab), and decode/parse checks.
- **Result:** CC BY 4.0; 13,968 annotation files; 1920×1072 video at 30 fps; 36–38 activity types; per-frame actor boxes. Used for activity detection (TRECVID ActEV) but **never for natural-language retrieval**.
- **Decision:** MEVA, with a natural-language query set we construct ourselves.
- **Basis:** satisfies non-Kaggle, real-world, large and licensed; the task is new for this dataset. Claim stated precisely: "new task for this dataset", not "unused dataset".

### D-004 · Fallback proposal (2026-09-14)
- **Decision:** multi-source violence detection (AIRTLab + Bus Violence + SCFD, about 2,050 clips) with cross-dataset evaluation.
- **Basis:** AIRTLab alone has only 350 clips, below the "thousands of samples" rule; combining sources fixes the size and creates the novelty (train on one dataset, test on another).

### D-005 · Annotations fetched through Colab (2026-09-19)
- **Evidence:** `gitlab.kitware.com` times out from the local network (WinError 10060) but returns HTTP 200 from Colab.
- **Decision:** all annotation work happens in Colab; data lives on Google Drive.

### D-006 · Download a subset only (2026-09-19)
- **Evidence:** local S3 throughput is 0.47 MB/s (443 GB would take 11 days); video is read once to make embeddings; unannotated video cannot produce training pairs.
- **Decision:** a few site-hours of video (ended up as ~70 GB — my per-hour estimate was 9× too low), annotated clips only.

---

## Phase 1 — Data preparation (2026-09-28 → 2026-10-22)

### D-007 · Usable corpus (2026-09-28, Cells 7–8)
- **Result:** 2,212 annotated clips (1,443 training-group / 769 eval-group); 66,221 activity instances in total, of which **10,721 have downloaded video**; 32 activity types usable. Median activity length 67 frames (~2.2 s), maximum 9,009 frames.
- **Decision:** no more downloading; 10,721 instances is enough.

### D-008 · Spec decisions for offline embedding (approved 2026-10-21)
| # | Decision | Basis |
|---|---|---|
| 1 | Embed with **both SigLIP and SigLIP 2**; choose on validation | User request; reading the video is the expensive part, so a second encoder adds little cost |
| 2 | **Three views: tight, context, full** | People are ~12×51 px after a full-frame resize to 224; context added at the user's suggestion, because background can tell look-alike actions apart (e.g. fight vs hug) |
| 3 | Sample 2 frames per second, at most 16 per instance | Proportional to duration; the cap limits the bias that favours long alerts |
| 4 | Protocol A (official split) **+ Protocol B (unseen cameras)**, both reported | Choosing whichever split scores higher would be choosing on the test set (user agreed) |
| 5 | Annotated clips only; unannotated clips never used in metrics | An unannotated clip might contain the queried action, so a correct result would be counted as wrong |
| 6 | Tested `src/sfs` package; Colab runs a generated one-cell bundle | The repo is private, so code is pasted into Colab; generating the paste from the tested code prevents the two from drifting apart |

### D-009 · Manifest v2 (2026-10-21)
- **Result:** 10,678 instances after removing 43 duplicates; leakage check (same clip in train and eval) passed. Protocol A: train 6,213 / val 739 / eval 3,726. Protocol B: 8 held-out cameras, 3,938 eval instances.
- **Finding:** in Protocol A **all 24 eval cameras also appear in training**, so it measures "new footage from cameras the model has seen". Protocol B measures cameras the model has never seen.
- **Decision:** Protocol B is reported as the generalisation result.
- **Open:** 10 activity types have fewer than 20 eval instances (68 instances in total, 1.8%) — see D-012.

### D-010 · Bug: actor boxes not found (2026-10-22)
- **Evidence:** the first embedding run had **99.5% fallback to the full frame**; only 2 of 252 clips found any boxes. A diagnostic on Colab showed the repo's box files use two YAML styles (`{geom: {id1: …}}` and `{'geom': {'id1': …}}`); a text regex matched only the first.
- **Decision:** parse the box files with a real YAML parser; **stop the job** if a clip's actors have no boxes at all; add a pipeline version to the encoder id so old shards are recomputed instead of being skipped as finished.
- **Basis:** fix the root cause, not just this case. The failure had been printed on every line (`fallback_rate: 1.0`) but not acted on — which is why it is now an error rather than a printed number.

### D-011 · Encoder and view selection (2026-10-23) → `docs/results/01-zero-shot-validation.md`
- **Experiment:** frozen encoders, no training, **validation split only**; per-activity text prompts; instance score = mean of the top 3 frame similarities.
- **Result:** SigLIP 2 + tight crop is best (AUC 0.697, P@10 0.22 vs chance 0.065). Cropping doubles top-10 precision compared with the full frame (0.22 vs 0.10). Vehicle and door activities score AUC > 0.9; fine-grained human actions are at chance; `person_stands_up` scores 0.127 (inverted).
- **Decision:** **SigLIP 2 + tight view** is the base encoder (user approved, 2026-10-23).
- **Basis:** a clear margin on the selection split. The finding that matters most: a single frame carries no motion — which motivates LoRA and a re-ranker that looks at frames in order (temporal).

### D-012 · Rare activity types (2026-10-23)
- **Context:** 10 activity types have <20 eval instances (68 instances, 1.8% of eval); `person_purchases` has 0 training instances.
- **Options:** (A) exclude them from evaluation queries but keep them in the searched corpus as distractors; (B) targeted download of more clips.
- **Decision:** **A** (user, 2026-10-23).
- **Basis:** with <20 examples any per-activity metric is mostly noise; B costs a day with no guarantee more examples exist. Query set = the 22 types with ≥20 eval instances under Protocol A; the same 22 are used for Protocol B so the two protocols stay comparable.

### D-013 · Spec approved: crop store + query set + frozen baseline (2026-10-23)
- **Decision:** build `spec-crops-queries-baseline.md` exactly as written.
- **Basis:** scoped deliberately so its results decide the next spec (LoRA, re-ranker) — user's instruction to proceed in stages.

### D-014 · Crop store failed its integrity gate (2026-10-24)
- **Experiment:** spec Part 1 gate — re-embed 500 stored tight crops, compare with the original embeddings. Required mean cosine ≥ 0.98.
- **Result:** **0.964** (p01 0.921) → FAILED. The gate did its job: building on these crops would have mixed storage effects into later model comparisons.
- **Diagnosis experiment** (54 crops, one storage step added at a time):

  | Step | Mean cos | Small crops | Large crops |
  |---|---|---|---|
  | direct crop, no storage | 1.0000 | 1.0000 | 1.0000 |
  | + resize to 336 | 0.9682 | 0.9650 | 0.9842 |
  | + JPEG q95 | 0.9643 | 0.9631 | 0.9706 |
  | actual stored crop | 0.9643 | — | — |
- **Findings:** (1) GPU fp16 vs CPU fp32 makes **no** difference (row 1 = 1.0000) — the initial suspicion was wrong. (2) The **resize to 336 is the main cause**: the median tight crop is 224 px, so most crops were *upscaled* then downscaled again by the encoder. (3) JPEG adds a smaller loss (~0.4 points, ~1.4 on large crops). (4) Stored files are exactly what the code intended (row 4 = row 3).
- **Lesson:** the 336 px choice ("room for augmentation, more detail for the teacher") was wrong for crops mostly *smaller* than 336 — there was no extra detail to keep.
- **Proposed fix (pending approval):** tight crops stored at native resolution as lossless PNG (pixel-exact); context crops at native size capped to 448 px, JPEG q95 (teacher only). New test: stored tight crop must be pixel-identical to the direct crop.

---

## Open decisions

| ID | Question | Status |
|---|---|---|
| D-014 | Crop store format fix: tight = native PNG, context = native ≤448 px JPEG | **Pending approval** |

---

## Standing rules

- **No decision is ever taken on eval-split numbers.** Selection uses validation only; eval is reported, never used to choose.
- Every result records the data version (encoder id / pipeline version) it came from.
- Failed and negative results are logged with the same care as positive ones.
