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
- **Decision (user approved, 2026-10-24):** tight crops stored at native resolution as lossless PNG (pixel-exact); context crops at native size capped to 448 px, JPEG q95 (teacher only). New regression test: stored tight crop must be pixel-identical to the direct crop. Re-run into `crops_v2`.
- **Basis:** removes the cause found by the diagnosis instead of tuning a threshold; the gate stays at 0.98.

### D-015 · Integrity check crashed Colab RAM (2026-10-25)
- **Evidence:** with the PNG crop store, C2 (`verify`) crashed the free-tier runtime (~12 GB RAM) on every attempt. The JPEG store had not crashed.
- **Cause:** `verify` cached every crop shard it opened; the 500 sampled crops span up to ~250 shards, so the whole store (~10+ GB as PNG, ~4× the JPEG size) was held in memory. The cache was useless — the loop visits each shard once. Secondary: `build_index` loaded every shard's image bytes just to read its small index columns.
- **Decision:** one shard in memory at a time in `verify`; `build_index` reads only the row arrays (`read_crop_rows`, lazy npz members).
- **Basis:** fix the cause, not raise memory. Two regression tests: index building must never load image bytes; `verify` must hold at most one shard at a time.

### D-016 · Crop store v2 passes; heavy work moves off Colab (2026-10-27)
- **Context:** on Colab the integrity check never completed — the runtime disconnected while reading the ~14 GB store from Drive (repeated Drive-mount drops, see D-015 era).
- **Decision:** move all post-extraction work to the local machine. The crop store (`crops_v2`, 14 GB) was downloaded at the owner's office (300 Mbps) and copied to `data/crops_v2/`. Colab remains only for jobs that must read the raw video on Drive.
- **Basis:** video decoding — the only step that needed Colab — is finished; every later step reads crops or embeddings, both now local. The local RTX 4060 runs the check in 86 s.
- **Completeness check:** 252/252 shards valid, one store id (`84b0a9ac41c5`), 147,138 crops = index rows, 73,569 tight + 73,569 context (matches the embedding rows).
- **Integrity gate (full 2,000-crop sample, local GPU):** mean cosine **1.000**, p01 **0.9998** → **PASSED**.

  | Crop store | Mean cos | Gate |
  |---|---|---|
  | v1 — 336 px JPEG | 0.964 | failed |
  | v2 — native PNG | 1.000 | passed |
- **Outcome:** spec `spec-crops-queries-baseline` Part 1 complete. All later comparisons (baseline vs LoRA) can use the crop store with no storage effect.

### D-017 · Query phrasings approved and frozen (2026-10-27)
- **Context:** the owner pointed out the phrasings could not be judged from label names alone.
- **Review method:** a local review page (`data/review/phrasing_review.html`, git-ignored) showed, per activity, the 3 phrasings beside 4 real annotated examples as time-ordered filmstrips (tight crops + one context crop) — **training split only**, so eval stayed unseen. Official definitions: DIVA Annotation Guidelines v1.0 (extracted to `data/meva/docs/`).
- **Observation:** many scenes are crowded indoor gatherings; for some activities (e.g. `hand_interacts_with_person`) the decisive detail is small within a busy crop.
- **Decision:** phrasings approved as written (owner: activity names and descriptions correct; the three per activity are paraphrases). **Frozen** before any baseline result was seen.
- **Basis:** queries written and reviewed blind to retrieval results cannot be tuned to the model.

### D-018 · Frozen baseline on the eval split (2026-10-27) → `docs/results/02-frozen-baseline.md`
- **Experiment:** spec Part 3. SigLIP 2 frozen, tight vs full vs random; Q1 (66 category queries) and Q2 (380 / 222 forensic events × 3 phrasings); Protocols A and B. First eval-split numbers — reported, not used for any decision.
- **Result (Protocol A, macro):** tight Q1 mAP **0.179** (full 0.096, random 0.047); Q2 R@10 **0.510** (full 0.299, random 0.266), MRR 0.446. Protocol B: mAP 0.174, R@10 0.496. Cost: 1 encoder call, 0 VLM calls, ~40–60 ms per query.
- **Findings:** cropping holds on eval (~2×); A ≈ B as expected for a model never trained on these cameras; motion-defined activities (pick up, put down, transfer, stand up, sit down) at chance (lift 1–2); phone use and vehicle/door events found well.
- **Caveat:** Q2 Recall@K is capped when a window has more than K relevant instances; Success@K proposed for the next spec, defined before further results.
- **Outcome:** spec `spec-crops-queries-baseline` complete (Parts 1–3). This is the number every later system must beat.

### D-019 · Training caption bank approved (2026-10-30)
- **Context:** LoRA spec D1 — a training caption bank separate from, and disjoint with, the frozen evaluation phrasings (no exact match; same-activity similarity < 0.80; max 0.79).
- **Review:** first pass (2026-10-29) the owner questioned `hand_interacts_with_person` (looked like signalling) and `person_carries_heavy_object` (no heavy items visible) — crowded crops made it unclear which people were meant. A second review page (`data/review/caption_review_boxes.html`, git-ignored) drew the activity's actors and objects from MEVA's box annotations (2,090 crops, 0 placement mismatches) beside the official MEVA definition. Training split only.
- **Findings from the boxes:** `hand_interacts_with_person` is annotated from 1 s before contact to 1 s after, so edge frames show the approach; the carried "heavy object" (tracked as its own actor) is mostly folding chairs.
- **Decision:** owner approved the 192 captions as written; `queries/train_captions.csv` frozen (sha256 prefix `139d6fc19a02a9ad`).
- **Follow-ups:** edge-frame label noise proposed as a D4 addition; owner's "general phone-use caption" idea recorded in the spec (§10) for later, not adopted now.

### D-020 · LoRA fine-tuning spec approved for build (2026-10-30) → `docs/planning/spec-lora-finetune.md`
- **Approved as drafted:** D2 image-tower LoRA only (image + text LoRA later, separate spec), D3 label-aware sigmoid loss, D7 checkpoint selection on validation only, D8 one model per protocol, D11 Success@K (K = 1, 5, 10, defined before any adapted result).
- **Changed in review:** D5 augmentation — no horizontal flip; zoom in/out, stretch, resolution degradation, mild colour jitter (owner). D6 — up to 100 epochs, patience 5 (owner; was 15 / 3); LR schedule changed to warm-up + reduce-on-plateau, because a cosine laid over 100 epochs would barely decay before early stopping. D4 — train both uniform and middle-weighted frame sampling (edge frames include MEVA's ~1 s lead-in/out; D-019) → 4 runs; per protocol the variant is chosen on validation, eval reported for all four (H4 added).
- **Added by owner:** D13 crash-safe per-epoch checkpoints with exact, config-checked resume.
- **Basis:** owner review rounds 1–2 (2026-10-29/30).

### D-021 · LoRA fine-tuning results (2026-10-30) → `docs/results/03-lora.md`
- **Experiment:** spec `spec-lora-finetune` as approved (D-020). Four runs (A/B × uniform/middle-weighted), all stopped early (best epochs 28 / 16 / 10 / 10; ~1.5 h total on the RTX 4060, peak 5.0 GB).
- **Selection (validation only, recorded in `runs/lora/selection.json` before eval was read):** A → **A-middle** (val mAP 0.437 vs 0.418); B → **B-uniform** (0.3776 vs 0.3756, near-tie). Frozen val mAP: A 0.272, B 0.222.
- **Integrity:** val mAP recomputed from the re-embedded stores matches the trainer's record within 0.0004 for all runs; adapter-off re-embedding reproduces the baseline (cos min 0.99996).
- **Result (eval, macro, own protocol only):** Q1 mAP A 0.179 → **0.313** (+0.134, 95 % CI [+0.085, +0.194]); B 0.174 → **0.280** (+0.107 [+0.062, +0.158]). Q2 S@1 A 0.273 → 0.389, S@10 0.819 → 0.902. Cost unchanged: 1 encoder call, 0 VLM calls. A improved all 22 activities; B 20 of 22.
- **Hypotheses:** H1 confirmed. H2 rejected as stated — motion-defined activities (baseline lift ≤ 2) gained almost as much AP as the rest but remain the weakest (pick up / put down lift ≈ 2). H3 weak support — A–B gap 0.033 (frozen 0.005) but CIs overlap and the protocols differ in corpus and training cameras; ~80 % of the gain transfers to unseen cameras. H4 rejected — the A validation win comes almost entirely from one activity (`vehicle_turns_left`), `hand_interacts_with_person` moved the other way, and the variants tie on eval.
- **Caveats:** one seed per run; small validation (739 / 586 instances) makes epoch and variant choice noisy; model revision still "unpinned".
- **Outcome:** spec `spec-lora-finetune` complete. The adapted encoder is the new first-stage retriever. Remaining errors concentrate in motion-defined activities → motivates the VLM teacher / temporal re-ranker spec (next).

---

## Open decisions

| ID | Question | Status |
|---|---|---|
| — | none currently | — |

---

## Standing rules

- **No decision is ever taken on eval-split numbers.** Selection uses validation only; eval is reported, never used to choose.
- Every result records the data version (encoder id / pipeline version) it came from.
- Failed and negative results are logged with the same care as positive ones.
