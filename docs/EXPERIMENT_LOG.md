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

### D-022 · Re-ranking cascade spec approved for build (2026-10-31) → `docs/planning/spec-rerank-cascade.md`
- **Scope:** VLM scorer (yes/no margin) as the expensive stage and teacher; gates G0 (laptop feasibility) and G1 (VLM must add information on validation); offline teacher labels; temporal student re-ranker over frame embeddings + box motion, trained from human labels / teacher / both; recall-vs-VLM-call curve on eval; hypotheses H1–H5.
- **Owner answers:** O1 download now on the home connection — Hugging Face rate-limits it (429), so weights come from the official ModelScope mirror, sha256-verified, revision pinned. O2 both Qwen3-VL-2B-Instruct and Qwen3-VL-Reranker-2B evaluated; the better on validation is the teacher, both reported on eval. O3 gates never stop the work: failures are marked "needs owner feedback" and collected with example images in a review page. O4, O5 approved. Runs on the laptop.
- **Basis:** owner review 2026-10-31.

### D-023 · Gates G0 / G1 and teacher choice (2026-10-31) → `runs/cascade/g0_*.json`, `runs/cascade/g1.json`
- **G0 (laptop feasibility, 20 validation clips, context view):** Qwen3-VL-2B-Instruct 6.1 GB peak VRAM, 0.64 s per clip for 64 texts (cached), 0.17 s per pair uncached, deterministic; Qwen3-VL-Reranker-2B 4.4 GB, 0.16 s per pair, deterministic — **pass**. The Instruct cached-vs-full check gave 0.37 against the spec's 0.1: diagnosed as bf16 rounding, not a bug — two mathematically identical uncached runs (eager vs sdpa attention) differ by up to 0.90 (mean 0.27); cached vs full max 0.44, mean 0.07, rank correlation 0.995; margins span −13…+11; in fp32 the cached path is exact (`tests/test_teacher.py`). Cached path used; the threshold question is **marked for owner feedback** (`runs/cascade/g0_instruct_noise.json`).
- **G1 (VLM inside the LoRA top-50, validation Q1 mAP; pass = +0.02 on a protocol):** instruct-tight A 0.437→0.446, B 0.378→0.380; instruct-context −0.012 / +0.002; reranker-tight A 0.437→0.440, B 0.378→0.390; reranker-context −0.008 / +0.001. Every configuration chose w = 2 (largest LoRA weight). **G1 fails for all four** — marked for owner feedback; the work continues (O3).
- **Why it fails (validation diagnostic, `runs/cascade/g1_diagnostic.json`):** inside the top-50 the VLMs separate right from wrong clips with AUC 0.54–0.63 (LoRA 0.69–0.75); VLM-only ordering mAP 0.39 vs random 0.31 vs LoRA 0.44 (A). The signal is real but weaker than LoRA and only partly independent (correlation 0.14–0.41). Per activity the VLMs help on doors, texting, picking up, vehicle stops; they fail on vehicle turn left/right (A −0.54 AP), standing up and carrying heavy objects.
- **Teacher:** pre-set rule (best mean gain) → **Qwen3-VL-Reranker-2B, tight view**. Instruct-tight is reported as the ablation model on eval.
- **Basis:** validation only; eval not read.

### D-024 · Student selection and cascade results (2026-10-31) → `docs/results/04-cascade.md`
- **Teacher labels:** Qwen3-VL-Reranker-2B, tight, 41,266 train/val pairs (2 captions per class); AUC 0.62 on the training pools.
- **Student selection (validation Q1 mAP):** A GT 0.600 / GT+KD 0.594 / KD 0.424; B 0.566 / 0.539 / 0.377 (LoRA 0.437 / 0.378) → **GT selected for both**; β = 1. VLM fusion weights γ = 2 everywhere (largest first-stage weight).
- **Eval (read once, after all choices):** Q2 S@1 A 0.389 (LoRA) → **0.531 (LoRA + student, 0 VLM calls)** → 0.546 / 0.560 with Instruct @5 / @10; B 0.369 → **0.519** → 0.550 / 0.550. Q1 mAP A 0.313 → 0.476, B 0.280 → 0.442. Instruct on every Q2 candidate (124 / 308 calls) 0.360 / 0.367 S@1 — no better than LoRA.
- **Hypotheses:** H1 confirmed (+0.16 Q1 mAP, +0.13 Q2 MRR; motion-defined group +0.36 AP vs +0.12–0.13); H2 not tested (G1 failed; LoRA→Instruct@10 +0.033 / +0.024, CIs touch 0); H3 rejected (GT+KD −0.02, KD −0.19); H4 confirmed but trivial (VLM-on-all is weak); H5 rejected — VLM and student both at chance on left vs right turns (AUC 0.49–0.51).
- **Report scope completed before eval was read:** GT+KD / KD ablations, latency estimate, H5 measurement (`direction_test`), SVG figure, bootstrap CIs (`scripts/cascade_report.py`).
- **Basis:** spec D8/D9; selection on validation only.

### D-025 · Owner answers to the Result 04 questions (2026-11-03) → `docs/planning/spec-vlm-followups.md`
- **Review page** rewritten in plain language first (owner: too complex to answer); the old wording is kept in `runs/cascade/questions_v1_technical.json`.
- **Q1 next step:** merge `vehicle_turns_left` / `_right` into one "vehicle turns" activity instead of building direction features → spec E1 (an extra view; the frozen 22-activity set stays the main record).
- **Q2 weak VLM:** test the 2B VLM with 16 frames (E3) and a bigger VLM through an API (E4), so the report can state what was tried and what it showed.
- **Q3 distillation:** current results recorded as they are (GT kept); retry distillation with the bigger VLM (E4).
- **Q4 shortcut check:** cached path accepted; the check is to be restated against the bf16 noise floor (it is a code threshold in `gate.py`, 0.1, not spec text). The full uncached eval re-scoring (~11–12 h) is kept as a later option.
- **Q5:** run "Instruct on every Q1 candidate" on eval — started 2026-11-03: 213,090 missing pairs scored into `data/teacher/eval/instruct-tight-q1all`; the 244,620 Result 04 pairs keep their values (re-scoring them in other batches would move them by bf16 noise). Code: `cascade_run --everything-q1 / --exclude-scored / --extra-scores`.
- **Q6:** run G1 on the whole validation set once; if the outcome matches the top-50 test, keep the top 50 (E2).
- **Q7 mirror weights — verified 2026-11-03 without a download:** Hugging Face file metadata answered (no 429 this time). Every model file of both VLMs matches (LFS sha256 / git blob sha1); only `.gitattributes` differs (mirror repository settings, unused). **Correction:** the revisions recorded at download (`ae9985b2…`, `5b295e98…`) are ModelScope's repository revisions, not Hugging Face commits, as the review page wrongly said; the Hugging Face revisions are `89644892…` (Instruct) and `4bd860ac…` (Reranker), added as `hf_revision` in `data/models/*/_source.json`. Record: `runs/cascade/weights_hf_check.json`.
- **Basis:** owner answers in chat, 2026-11-03. Everything except Q5 and the Q7 check waits for approval of the new spec.

### D-026 · Spec decisions and E4 clarified (2026-11-03) → `docs/planning/spec-vlm-followups.md` §8
- **E4 clarified by the owner:** "API" means a bigger open model run on the **office server** and opened as our own API, not a paid service. Spec rewritten: Qwen3-VL-8B-Instruct (bf16) first, 32B if memory allows; served by our own scorer code behind a small HTTP service (same prompt, frames and score as the 2B runs); office network only, shared secret set by the owner.
- **Approved:** E1 as an extra merged view with no retraining, with the three proposed phrasings (`queries/phrasings_merged_turns.csv`); students retrained only if a VLM passes G1 (E3, E4).
- **Open:** E2 scope (Reranker-context), office server details (GPU memory, OS, disk, network), order, approval of the spec as a whole for E2–E4.
- **E1 started** on the owner's go-ahead: `cascade_run --merge / --fixed-choices` (fusion weights reused from Result 04, nothing re-selected).

### D-027 · E2 and E3 approved and started (2026-11-03) → `docs/planning/spec-vlm-followups.md`
- **Owner:** "start E2 and E3"; office-server details for E4 follow the next morning.
- **E2 scope (assumption, recommended default; owner did not object):** Instruct-tight, Instruct-context and Reranker-tight on every validation clip; Reranker-context skipped (worst G1 set-up). Decision rule fixed before running: same pass/fail and gain within ±0.01 of the top-50 gain → keep the top 50.
- **E3:** G0 at 16 frames (context view, as the original G0) for both models, then G1 (top 50, tight) at 16 frames for both. Students retrained only if G1 passes (D-026).
- **Shortcut check restated (Q4):** cached-vs-full largest difference ≤ the eager-vs-sdpa difference of two slow passes on the same 3 clips × 16 captions, and rank correlation ≥ 0.99 (`gate.shortcut_check`). Re-measured at 8 frames under the new rule for the record (`runs/cascade/g0_instruct_8f_noise_rule.json`); the original G0 file is kept unchanged.
- **Run order (one GPU job at a time, `runs/cascade/followups.log`):** E5 report → E1 scoring + report → E2 Instruct → E3 → E2 Reranker (~4 h) → E2 gate. Tests: 122 pass.

### D-028 · Follow-up results E5, E1, E2, E3 (2026-11-03) → `docs/results/05-vlm-followups.md`
- **Runs:** `runs/cascade/followups.log`, 03:07–08:39, every step exit 0. Raw copies `docs/results/raw/05_*`. Tests: 123 pass (new: `scripts/g1_frames_diagnostic.py` + test).
- **E5 (eval, Q1 Instruct on every candidate):** mAP 0.207 (A) / 0.212 (B), above frozen (0.179 / 0.174), far below LoRA (0.313 / 0.280) and LoRA + student (0.476 / 0.442), at 3,726 / 3,938 calls (~10 min) per query; fused with LoRA 0.306 / 0.287 (≈ LoRA). Result 04 numbers unchanged (0 differing entries); its table gets the Q1 cells, marked † and dated.
- **E1 (eval, merged "vehicle turns", extra view, no retraining):** turn-class Q1 AP for LoRA 0.31 / 0.32 (left/right mean) → 0.57 / 0.62 merged; chance 0.02 → 0.04 (A). The direction, not the turn, was the failure; the VLM also finds turns (Instruct on all 0.52 / 0.59) but not direction. LoRA → Instruct @50 lifts merged-turn Q2 S@1 by +0.18 / +0.13. The student (trained on left/right) is worse than LoRA on the merged class (0.44 / 0.39). Macro mAP +0.01 for every stage; the other 20 activities unchanged. Nothing decided on these eval numbers.
- **E2 (validation, G1 on every clip):** all three set-ups fail in both tests; the whole-validation gains are lower by 0.012–0.019 (Instruct-tight −0.006, Instruct-context −0.024, Reranker-tight −0.012 mean). **By the fixed rule (D-027): not the same.** Per the spec the owner decides E4's G1 depth (whole validation is ~63k pairs, 13× top 50). Recommendation recorded in Result 05: top 50 first, whole validation only if top 50 passes.
- **E3 (validation, 16 frames):** G0 Instruct at 16 frames **fails on memory** (7.80 GB > 7.5; run still fit on the 8 GB card, work continued under the standing gate rule); Reranker passes (4.60 GB). Shortcut check under the new rule passes at 8 frames (0.444 ≤ 0.899, ρ 0.995); at 16 frames its 3 clips hold ≤ 8 frames, so it re-tests the same inputs. **G1 at 16 frames fails** for both (mean gain +0.007 / +0.007 vs +0.005 / +0.007 at 8). 65 % of G1 clips hold ≤ 8 frames (identical scores, confirmed: largest difference 0.000); on the 353 longer clips 16 frames change AUC by −0.034 to +0.019, no consistent direction. **E3 stops by the spec rule** (no teacher labels, no retraining).
- **Basis:** spec-vlm-followups E1–E3/E5 rules fixed before the runs; choices reused from Result 04 (`--fixed-choices`).

### D-029 · E4 set-up: office server surveyed, model and gate depth chosen (2026-11-03) → spec "E4 revised after the server survey"
- **Server (`lt-ai-server`, read-only survey over SSH, key login added by the owner):** RTX 4070 Ti SUPER 16 GB, 30 GB RAM, Ubuntu 24.04, 62 GB disk free, ~15 MB/s to Hugging Face; shared team machine whose Ollama service holds ~6.5 GB of the GPU while loaded.
- **Owner decisions:** (1) model **Qwen3-VL-8B-Instruct-FP8** (official, rev `9cdc6310…`, 10.6 GB; language model FP8, vision tower bf16) — the 8B in bf16 (17.5 GB) does not fit; fallback 4B bf16; (2) **wait for free GPU** before loading, owner informs the team; (3) **G1 depth: top 50 first, whole validation only if it passes** (answers D-028's open row).
- **Design narrowed:** service bound to 127.0.0.1 on the server, reached through an SSH tunnel; shared secret kept (env var set by the owner). Private conda env in the owner's home (conda-forge only).
- **Not tested:** Qwen3-VL-Reranker-8B (bf16 only, does not fit). FP8-vs-bf16 quality is a stated limitation.

### D-030 · E4 code built and server set up (2026-11-03) → spec §4, "E4 revised after the server survey"
- **Code:** `sfs.teacher.serve` (HTTP service around `VLMScorer`: localhost only, shared secret, size limits, waits for free GPU memory, out-of-memory → 503), `sfs.teacher.remote.RemoteScorer` (same interface, back-off retries, size check), `run --remote`, `gate g0 --remote --max-vram-gb`, `gate remote-check`; `VLMScorer` gets device hooks and loads quantised checkpoints straight onto the device. Frames travel as lossless PNG of the decoded crops.
- **Tests:** `tests/test_remote.py` (end-to-end on a local port with the tiny Qwen3-VL: remote = local exactly, wrong secret refused without retries, 413/400/404, OOM retried with back-off, secret never logged, lossless frames, wait-for-GPU, G0 and remote-check through the hooks). Full suite 134 pass, slow 2 pass. Regression: the changed scorer re-scores 19 stored 2B G1 pairs exactly.
- **Found while testing:** the service replied to 404/401 before reading the request body; on Windows the client then saw a connection reset instead of the reply. Fixed at the root: the body is always read (bounded) before any reply.
- **Server:** private env `~/envs/sfs` (laptop's versions), Qwen3-VL-8B-Instruct-FP8 @ `9cdc6310` and Qwen3-VL-2B-Instruct @ `89644892` downloaded on the server (~15 MB/s) and verified file by file against Hugging Face; 38 GB disk left.
- **Note:** transformers 5.22 will change Qwen3-VL video preprocessing; versions stay pinned at 5.20.2.
- **Next:** owner creates the shared secret; remote-check with the 2B (fits beside Ollama); G0 and G1 (top 50) with the 8B once ~14 GB of GPU memory is free.

### D-031 · Gemma 4 (E6) and night-time runs proposed; server disk surveyed (2026-11-03) → spec E6, "Night-time runs from home"
- **Owner:** a colleague's Gemma 4 experiments look good and Qwen has not given the results wanted, so test Gemma too; work from home at night with access limited to us, revoked afterwards; check disk space held under the owner's name.
- **Facts (read-only):** gemma-4-12B-it (cached rev 707f0a3b; HF now 43c9d014) and gemma-4-E4B-it (ee0ef602) are already in the owner's cache; both accept video in transformers 5.20.2; Apache 2.0, not gated. 12B bf16 = 23 GB, E4B = 15 GB, 26B-A4B = 52 GB, so on 16 GB only an FP8 language model fits. The server is normally shut down around 20:00–21:00. The owner's account holds 196 GB (HF cache 156 GB, of which the two 26B-A4B Gemma copies are 98 GB and do not fit the GPU).
- **Proposed (waiting for approval):** E6 = same G1 test with gemma-4-12B-it (FP8 language model quantised at load) first, E4B as fallback; new `GemmaScorer`; night jobs run on the server itself with the crop store copied there over the LAN (home link only for SSH and small results).
- **Not done by Claude:** firewall / router / sshd / VPN changes (security settings, need IT and sudo). IT request written: `docs/planning/office-server-remote-access.md` (options VPN > mesh VPN > IP-restricted port-forward; revocation checklist).

### D-032 · Gemma 4 (E6) approved and started; remote access by the owner; data on the server (2026-11-03) → spec E6 revisions
- **Owner:** "start the experimentation with gemma4 that we have done with the qwen ... don't wait for the spec approval"; Gemma "first locally", then "can we run the gemma model beside ollama on the server?" — yes: the laptop has only ~4.5 GB RAM free (66 GB committed by open apps), the server has 26 GB RAM and ~9.8 GB GPU free next to Ollama.
- **Run:** gemma-4-E4B-it (rev `ee0ef602`; laptop copy verified against Hugging Face, 7 files; the server uses the same cached files through links), bf16, per-layer table and audio tower on the CPU, ~7.5 GB of weights on the GPU, the rest streamed per pass. G1 scoring on the 4,800 top-50 validation pairs (tight, 8 frames), then G0 (context, 20 clips, limit 9.5 GB beside Ollama). Same pass rule as Qwen; next steps only if G1 passes.
- **Built:** `sfs.teacher.gemma` (GemmaScorer, placement, `stream_weights`), `load_scorer` dispatch by config, `--gpu-budget-gb`; G0 records placement, shortcut and mean yes/no mass. Tests 140 pass (incl. a tiny Gemma 4 with KV sharing and the full E4B architecture on the meta device).
- **Library problems found and fixed at the root:** transformers 5.20.2 `Gemma4Model.forward` drops `num_frames_per_video` (scorer passes precomputed vision features through `mm_encoder_outputs`); accelerate's `cpu_offload` copies call arguments and breaks Gemma's in-place KV sharing (own weight-streaming hook); Gemma's audio tower has no `sdpa` (attention set per sub-model, audio eager); server env lacked pandas (installed at the laptop's version).
- **Remote access:** no IT person; the owner and a colleague set it up themselves. Recommended Tailscale in user mode (no open port, only the owner's devices, SSH only); step-by-step guide + revocation: `docs/planning/office-server-remote-access.md`. Claude does not set up remote-access software or change security settings.
- **Data:** crop store (14 GB), manifest, request and query files copied to the server over the LAN (`~/sfs/project`), so night runs need only SSH.
- **Open:** which Qwen weights to delete (server Qwen2.5-VL 23 GB / E4 Qwen3-VL 14 GB / laptop 2B 8.5 GB) — owner asked whether local Qwen3 helped (it did not: G1 failed, D-023/D-028).

### D-033 · E6 result: Gemma 4 E4B fails G1 (2026-11-03) → `docs/results/05-vlm-followups.md` §E6
- **Run:** office server beside Ollama, bf16, 30/42 layers on the GPU; G1 scoring 4,800 pairs (1.89 s per clip; this run streamed all layers because of a size-count bug, fixed; re-check after the fix: identical scores); G0 afterwards.
- **G0:** 7.75 GB peak, deterministic, 0.16 s per question, 99.95 % of probability on yes/no; fails only the 3 s per clip speed rule (no shortcut yet: 9.4 s).
- **G1:** gain A −0.015, B +0.011, mean −0.002 → **fail**. Inside the top-50 pools Gemma's AUC is 0.54 (A) / 0.59 (B) vs Qwen 2B Instruct 0.63 / 0.59 and LoRA 0.67 / 0.60; VLM-only val mAP 0.338 / 0.356 vs Qwen 0.394 / 0.353.
- **Rule applied (spec E6):** E4B stops; no whole-validation run, no teacher labels. Next question for the owner: the 12B on the office server, or stop the VLM line.

### D-034 · Gemma stopped and removed; back to the bigger Qwen (E4), run directly on the server (2026-11-03)
- **Owner:** "if gemma is not giving us what we want then stop it and remove it and lets go back to qwen bigger vlm model" → answers D-033 (no 12B).
- **Removed:** laptop `data/models/gemma-4-E4B-it/model.safetensors` (16 GB; config, processor and tokenizer files kept for the tests and the record, noted in its `_source.json`); server link folder `~/sfs/models/gemma-4-E4B-it`. The Gemma models in the owner's server cache (pre-existing, 136 GB) are left for the owner. Gemma code and tests stay (they document E6).
- **E4 design change (assumption, stated to the owner):** the crop store is now on the server, so E4 runs there directly (`sfs.teacher.run --model` on the server) instead of through the HTTP service; simpler, no secret, survives nights. The service stays for office use.
- **Set-up check replaces the remote-check:** the server's 2B (Hugging Face weights, rev `89644892`) scores all 4,800 G1 pairs; compared with the laptop's stored 2B scores. **Rule fixed before the result:** largest difference ≤ 0.899 (the laptop's eager-vs-sdpa noise floor, `05_g0_instruct_8f_noise_rule.json`) and rank correlation ≥ 0.99.
- **8B-FP8:** the server job waits for ≥ 14.5 GB free GPU memory (team's Ollama unloaded), then G0 (`--max-vram-gb 15`) and G1 scoring on the same 4,800 pairs. G1 rule (D-029): top 50 first, whole validation only if it passes.

### D-035 · E4 server set-up check: fails the pre-fixed rule, flagged; work continues (2026-11-03) → `docs/results/raw/05_e4_server_check.json`
- **Result:** the server's 2B re-scored all 4,800 G1 pairs (0.11 s per clip). Largest difference from the laptop 1.115 > 0.899 → **FAIL by the rule fixed in D-034.** Only 1 of 4,800 pairs is above 0.899; mean difference 0.126 (the laptop's own eager-vs-sdpa noise has mean 0.272); rank correlation 0.999; the difference does not grow with |margin|.
- **What it means for G1:** server 2B val mAP 0.4475 / 0.3795 vs laptop 0.4458 / 0.3797 (A / B), VLM-only 0.3941 / 0.3534 vs 0.3944 / 0.3533, same verdict (fail). The server reproduces the laptop's G1 within 0.002, ten times below the +0.02 the gate needs.
- **Cause of the rule failure:** the rule compares a maximum over 4,800 pairs with a noise floor measured on 48 pairs; a larger sample has a longer tail. Not changed after the fact: recorded as a fail, **flagged for owner feedback**, and the 8B continues under the standing gate rule (gates never stop the work).
- **8B:** Ollama had unloaded (16.0 GB free at 14:37); 8B-FP8 G0 started immediately.
- **8B first attempt failed in seconds:** transformers' FP8 path needs Hugging Face's `kernels` package (0.17.x). Installed `kernels==0.17.2` in `~/envs/sfs` (torch/transformers unchanged); the FP8 kernel is fetched once from Hugging Face `kernels-community` and cached. Test load: 10.6 GB on the GPU, 11.45 GB peak, language model float8_e4m3fn, vision bf16, ~0.33 s per clip. G0 and G1 scoring restarted (waits for 12.5 GB free).

### D-036 · E4 result: Qwen3-VL-8B FP8 narrowly fails G1 (2026-11-03) → `docs/results/05-vlm-followups.md` §E4
- **Runs (server):** first two 8B attempts stopped in seconds (`kernels` missing; then `kernels` cannot resolve the kernel version offline) → 8B jobs run with the Hub online, kernel commit `03ae3193` recorded. G0 50 s, G1 scoring 4,800 pairs in 4.8 min (0.28 s per clip).
- **G0: pass** (13.0 GB ≤ 15, 0.64 s per clip, deterministic, shortcut within noise floor).
- **G1: fail by 0.0009** — gain A +0.0066 (w = 1), B +0.0191 (w = 2); VLM-only val mAP 0.424 / 0.379 (LoRA 0.437 / 0.378; 2B 0.394 / 0.353); per-query AUC 0.661 / 0.627 (2B 0.628 / 0.590, LoRA 0.671 / 0.596).
- **Reading:** size helps clearly (the 8B nearly matches LoRA alone on A and beats it on B; first VLM to get real weight in the fusion), but the fusion gain stays just under +0.02; with ±0.002 server/laptop noise this is at the threshold.
- **Rule applied:** E4 stops (D-026/D-029: next steps only if G1 passes). **Flagged for owner feedback.**

### D-037 · E4 cascade rows with the 8B; GPU memory held during runs (2026-11-03)
- **Owner:** "if GPU is free lets start doing our experimentation on it and if someone calls to up the ollama block it".
- **Read as (assumption, stated to the owner):** run the recommended option of D-036, the 8B as the final cascade stage: fusion weight chosen on validation (1,920 val-student pairs), then the same eval top-k pairs the 2B scored in Result 04 (244,620 pairs, 6,935 clips). G1 verdict unchanged (fail); no whole-validation G1, teacher labels or distillation.
- **"Block Ollama":** not done as asked. Claude does not stop, reconfigure or disable the team's shared Ollama service (system setting, needs admin rights, would silently break colleagues' work). Instead each job holds the GPU memory it needs for its whole run (`sfs.teacher.run --reserve-gb 14.5`; PyTorch keeps the block cached), and waits for 14.8 GB free before starting; Ollama uses what is left. The owner informs the team.
- **Started 15:15** on the server (`runs/cascade/e4_cascade.log`); report: `cascade_run report --g1 runs/cascade/g1_8b.json --models instruct --scores-root data/teacher_8b`.

### D-038 · Qwen2.5-VL kept and tested (E7); team fine with slow Ollama (2026-11-03) → spec E7
- **Owner:** the team has no issue with Ollama running slowly while we hold the GPU; do not delete the Qwen2.5-VL models — test whether they contribute, delete only if not; asked what the "server set-up check" means (explained in plain words; acceptance still open, D-035).
- **Built:** `VLMScorer` loads Qwen-VL models through the auto class (`QWEN_TYPES`: qwen3_vl, qwen2_5_vl) and can store the language model in FP8 at load (`--fp8-at-load`, run name `+fp8load`). Tests 144 pass (tiny Qwen2.5-VL: cached = full); laptop 2B re-scores 19 stored pairs exactly.
- **Server:** Qwen2.5-VL-3B (rev 66285546) and 7B (rev cc594898) linked from the owner's cache and verified against Hugging Face (11 / 14 files). G0 + G1 scoring queued after the 8B cascade scoring (`runs/cascade/e7.log`).

### D-039 · E4 cascade rows: the 8B as final stage (eval, 2026-11-03) → Result 05 §E4
- **Run (server):** val-student 1,920 pairs (0.26 s per clip) and eval 244,620 pairs (0.49 s per clip, 56 min), GPU memory held 14.5 GB. γ chosen on validation (A 1 / 2, B 2 / 2), eval read once (`runs/cascade/05_8b`).
- **Q2 S@1:** A: LoRA + student 0.531 → 8B @5 0.574, @10 0.589, @20 0.596 (2B: 0.546 / 0.560 / 0.558); B: 0.519 → 0.560 / 0.554 / 0.564 (2B: 0.550 / 0.550 / 0.541). VLM on all: A 0.428, B 0.437 (2B 0.360 / 0.367), still below the 0-call system.
- **Bootstrap (22 activities):** A 8B@10 − student +0.058 [+0.020, +0.099]; A 8B@5 − 2B@5 +0.028 [+0.009, +0.046]; B differences not significant.
- **Reading:** reported, nothing selected on it. A bigger VLM is a clear gain as a short-list final stage on seen cameras; small on unseen cameras.

### D-040 · E7 Qwen2.5-VL-3B fails G1 (2026-11-03) → Result 05 §E7
- **3B (bf16):** G0 pass (8.5 GB, 0.37 s per clip, deterministic, shortcut within noise floor); G1 gain A +0.0013, B +0.0039 → **fail**; VLM-only val mAP 0.385 / 0.360 (≈ Qwen3-VL-2B). Not contributing.
- **7B (FP8 at load):** queued; waits for 14.8 GB free GPU memory (the team's Ollama reloaded 6.4 GB when the 3B finished).
- **7B result (16:46, after the owner unloaded Ollama):** G0 pass (11.4 GB, 0.64 s per clip); G1 gain A +0.0012, B +0.0137 → **fail**; VLM-only 0.380 / 0.361. Neither Qwen2.5-VL model contributes → deletion is the owner's call (D-038 rule).

### D-041 · GPU held for the whole experimentation; Qwen3-VL-32B FP8 next (2026-11-03)
- **Owner:** "why didn't u allow ollama to take the GPU?" → explained: the hold covered each job, not the gaps between jobs (Ollama reloaded after the 3B). Owner unloaded Ollama, then: "do not lose the memory till our whole experimentation is done"; "once qwen model is done do the test of the bigger model" → chose **Qwen3-VL-32B-Instruct-FP8** (rev `4bf2c2f3`, Apache 2.0, 35.5 GB).
- **Holder:** `sfs.teacher.queue --watch runs/queue --reserve-gb 14.5` on the server holds 14.5 GB from now until a STOP file at the end of the experimentation; jobs are dropped in as JSON files and run in one process (models released into the held cache, never back to the driver). Restarted twice (seconds) to load new code; memory regained each time.
- **Disk:** owner chose to delete the two **Gemma 4 26B-A4B** cache folders (98 GB; 24 → 120 GB free) instead of Qwen2.5-VL, because the owner's `nexa-vlm` project uses Qwen2.5-VL. Noted after deleting: `nexa-vlm/vlm_family.py` also lists `gemma-4-26B-A4B`; it would re-download if used.
- **Code:** `sfs.teacher.offload` (shared CPU/GPU split from E6: CpuLookup, stream_weights, place_layers; pinned memory only with ≥ 8 GB RAM to spare, else the OS file cache); `VLMScorer(gpu_budget_gb, clip_batch, pin)` with token embeddings on the CPU unless tied to lm_head (bug found by the test: the tiny/2B models tie them); `sfs.teacher.queue` (one-process queue + watch mode). Tests 150 pass.
- **Real-hardware check:** the 8B split (6 GB on the GPU, rest streamed) through the holder re-scores 19 G1 pairs **exactly** (max difference 0.0); 2.8 s per clip instead of 0.28.
- **32B plan:** G0 + G1 (top 50) through the holder, `gpu_budget_gb` 11, `clip_batch` 4, unpinned (~23 GB streamed from the file cache; the shared server has 30 GB RAM); expected ~3–4 h for G1.
- **E7 closed:** Qwen2.5-VL-7B fails G1 too (+0.001 / +0.014); both kept (nexa-vlm needs them), recorded as not contributing.

### D-042 · Qwen3-VL-32B FP8 passes G1 at top 50 (2026-11-03) → Result 05 §E4b
- **Run (holder, server):** G1 scoring 4,800 pairs 17:51–19:02 (4.24 s per clip; 16/64 layers on the GPU, 23.4 GB streamed unpinned, 4 questions per batch); G0 19:02–19:15.
- **G1: PASS** — gain A +0.0167 (w = 1), **B +0.0371** (w = 0.5); VLM-only val mAP 0.433 / 0.401 (LoRA 0.437 / 0.378). First VLM to pass; size trend 2B → 8B → 32B monotonic.
- **G0:** fails on speed only (25.4 s per clip for 64 questions); 12.3 GB peak, deterministic, shortcut within noise floor.
- **Next by the owner's rule (D-029):** whole-validation G1 (58,341 more pairs, 8 questions per batch for speed — batch size adds only bf16 noise; the top-50 pairs keep their scores), started 19:18; both must pass before teacher labels / distillation / cascade rows (D-026).

### D-043 · Qwen3-VL-32B FP8 passes G1 on the whole validation set (2026-11-03) → Result 05 §E4b
- **Run (holder, server):** 58,341 pairs over 1,325 clips, 19:18–23:26 (11.2 s per clip, 8 questions per batch); merged with the 4,800 top-50 pairs; gate `g1 --k 0` on the laptop.
- **G1 (whole validation): PASS** — gain A +0.0184, **B +0.0470** (both w = 2, the top of the pre-set grid; not extended); VLM-only val mAP 0.348 / 0.318 (LoRA 0.437 / 0.378). The 2B lost 0.012–0.019 going from top 50 to whole validation (E2); the 32B does not (top 50: +0.017 / +0.037).
- **Owner's rule (D-029) met:** passes at both depths → teacher labels, distillation and cascade rows are next (D-026).
- **Sizing (1.6 s per streamed pass, measured):** teacher labels 41,266 pairs ~8.6 h; validation student pairs ~0.6 h; eval short list ~5.6 h; optional eval "VLM on all" ~17.6 h. Not started: proposed to the owner first.

### D-044 · 32B teacher labels, validation and eval short list started (2026-11-04)
- **Owner:** "lets go ahead and do it.. I'll ask the team to let the system run" → the recommended plan: short list only, no eval "VLM on all" row.
- **Queued in the holder (`runs/queue/04_32b_teacher.json`, started 00:11):** (1) training-pool teacher labels `requests_trainval.csv` → `data/teacher_32b/trainval/instruct-tight`; (2) validation student pairs `requests_val_student.csv` → `.../val_student/...`; (3) eval short list `requests_eval_reranker.csv` (Q1 top-k + Q2 top-20, model-independent) → `.../eval/...`. Same settings as the whole-validation run (tight, 8 frames, 11 GB on the GPU, 8 questions per batch, unpinned). Request files checked identical to the laptop's (md5).
- **Then (laptop):** GTKD / KD students retrained on the 32B labels (GT unchanged), selection on validation, cascade report with the eval read once. If a new student is selected, the extra eval pairs it needs are scored with `--exclude-scored`.
- Scoring eval pairs is not reading eval results; no choice uses them.

### D-045 · Laptop steps after the 32B labels run unattended (2026-11-04)
- **Owner:** "schedule the script that will autostart right after this job complete so if we go offline we don't have to worry about it".
- **Script** `runs/student_32b/auto_after_labels.sh`, run by the Windows Task Scheduler task `SFS_after_32b_labels` (current user; once at creation 00:56 + at every log-on; no time limit; allowed on battery; one instance). It polls the server's holder log every 5 min over Tailscale (office address as fallback), copies the training-pool labels back and checks they cover all 41,266 pairs, trains GTKD + KD for A (A-middle) and B (B-uniform) into `runs/student_32b` (GT unchanged), then copies the validation (1,920) and eval short-list (38,202) stores. Every step is skipped if already done; it stops on a failed server job or a wrong pair count; a `STOP` file ends it; the task removes itself at the end. Log `runs/student_32b/auto.log`.
- **Stops before** student selection and the cascade report (by hand; eval read once after selection).
- **Checked:** wait + copy on the finished whole-validation job (file identical, 58,341 pairs); the training command re-trained Result 04's A-GTKD in a scratch folder with the same validation mAP (0.5943).
- **Needs:** the laptop on and awake (sleep pauses it; the owner's power setting is not changed).

### D-046 · The unattended chain moves to the server; laptop task removed (2026-11-04) — **replaces** D-045's laptop task
- **Owner:** "I want to schedule them on server not on laptop.. if left earlier tomorrow morning then you are not able to run the next job". The laptop task `SFS_after_32b_labels` was removed and its script stopped (it had only been waiting).
- **Server script** `~/sfs/after_labels.sh` (copy in `runs/student_32b/server_after_labels.sh`), started 02:18 under `setsid nohup`: (0) waits until the inputs from the laptop are complete (`inputs.sha256`, strict check); (1) reproducibility check: retrains Result 04's A-GTKD on the server GPU and compares the per-epoch log with the laptop's — identical → trains GTKD, KD only; different → also GT, so the three variants are compared on one machine; (2) waits for the 32B labels, checks 41,266 pairs; (3) trains A (A-middle) and B (B-uniform) into `runs/student_32b`. Students run as a separate process on the ~1.8 GB the holder leaves free (features stay on the CPU, only batches go to the GPU). Selection and the cascade report stay manual.
- **Code (small):** `sfs.rerank.train --text-emb` reads the caption / phrasing embeddings from a file made by `scripts/export_text_embeddings.py`, because the server has no SigLIP 2 weights (no download). Each group is embedded in the same single call training makes: a first version embedded all texts in one batch and changed the validation mAP in the 4th decimal (GPU results depend on the batch); the final version retrains Result 04's A-GTKD with every epoch's loss and validation mAP identical and the same weights. Test added; 151 pass.
- **Upload** over the home connection (0.19 MB/s measured): features (losslessly compressed 529 → 212 MB each), LoRA embeddings (103 MB each), pools, text embeddings, the 2B teacher store for the check; ~650 MB, ~1 h, sha256-checked per file. Large but needed; finishes long before the labels (~09:45).
- **Incident:** the first start passed the input check too early — Windows `sha256sum` prefixed 4 lines with a backslash and `sha256sum -c` skipped them as malformed. Training stopped at the first missing file; nothing was written. Fixed: hashes read from stdin, line count checked, `--strict`.

### D-047 · Distillation from the 32B does not help; GT students selected again (2026-11-04) → Result 05 §E4b

- **Labels:** 32B on 41,266 train/val pairs, 00:11–09:05 (holder log, `ok`). Teacher AUC on the training pools 0.686 / 0.695 (2B Reranker 0.613 / 0.615, same computation); mean soft target 0.49 for right clips, 0.27 for wrong (2B 0.46 / 0.41).
- **Reproducibility check (D-046 step 1):** Result 04's A-GTKD retrained on the server: best val mAP 0.6045 vs 0.5943 on the laptop (different GPU). By the pre-set rule all of GT, GTKD and KD were trained on the server (09:05–09:07, `runs/student_32b`) and selected among themselves.
- **Validation mAP (selection rule of Result 04, unchanged):** A GT 0.606 / GTKD 0.573 / KD 0.447; B GT 0.571 / GTKD 0.525 / KD 0.352 → **GT selected for both**. The GTKD drop (−0.033 / −0.046) is larger than the server-vs-laptop difference for the same GT student (+0.006 / +0.005).
- **Decision:** H3 stays rejected with the stronger teacher. The cascade rows use Result 04's laptop GT students (`runs/student/A-GT`, `B-GT`): the remaining 32B jobs score exactly their short lists (`requests_val_student.csv`, `requests_eval_reranker.csv`), so no extra eval pairs are needed and the 32B stage is the only change. No eval number was read.
- **Incident:** the upload script reported the four large files FAILED and re-sent each six times (finished 07:52, not ~03:10): the same backslash prefix in the local hash. The server copies match the laptop's (stdin hashes) and passed the 9-line strict check at 06:18; re-sends were atomic (`.part` + `mv`).
- **Raw:** `docs/results/raw/05_student32b_*`, `05_teacher32b_pool_quality.json`.

### D-048 · 32B cascade rows (eval, 2026-11-04) → Result 05 §E4b

- **Run:** server, val-student 1,920 pairs (09:05–09:43) and eval short list 38,202 pairs / 4,917 clips (09:43–15:30), all `ok`; scores copied back, sha256 identical. Report on the laptop with Result 04's GT students; γ chosen on validation (A: γ_LoRA 1 from `g1_32b.json` top 50, as for the 2B and 8B; γ_student 2. B: 0.5, 2); eval read once (`runs/cascade/05_32b`).
- **Q2 S@1:** A: LoRA + student 0.531 → 32B @5 0.573, @10 0.596, @20 0.605 (8B 0.574 / 0.589 / 0.596). B: 0.519 → 0.564 / 0.564 / 0.568 (8B 0.560 / 0.554 / 0.564).
- **Bootstrap (22 activities, `05_32b_bootstrap.json`):** 32B@10 − student A **+0.065 [+0.040, +0.095]**, B **+0.045 [+0.018, +0.073]**; 32B − 8B at @5 / @10: −0.001 / +0.007 (A), +0.004 / +0.009 (B), all intervals around 0.
- **Cost:** 1.74 s per uncached call (G0, 48 of 64 layers streamed) vs 0.13 s for the 8B.
- **Reading:** reported, nothing selected on it. The 32B is the first VLM with a significant final-stage gain on unseen cameras too, but it is not significantly better than the 8B, at ~13× the time per call on this GPU.
- **Bug found and fixed before this report:** `cascade_run report` timed every VLM with `g0_<kind>.json` next to the G1 file, so the 8B report (`05_8b_cascade.json`) recorded the 2B's 0.168 s per call instead of the 8B's 0.132 s. No number in Result 05's text came from that field. Fix: `--g0 model=path` (a named file must exist), test added, 152 pass. The 8B report was re-run with its own G0 file (`05_8b_g0fix`) to check that only that field changes.
- **Check result:** choices, per-query results (`_queries.csv` identical), summary and H5 identical; only the per-call time (0.168 → 0.132 s) and the student's measured wall-clock time per query (0.10 → 0.32 s, laptop load) differ. Corrected file `docs/results/raw/05_8b_cascade_g0fix.json`; the original is kept.

### D-049 · Next phase: two weeks to submission; demo (MEVA, then nexa_fight); image + text LoRA on the office server (2026-11-04)

- **Owner:** "we have two weeks left before submission"; demo "first do with meva then from the nexa_fight pipeline"; "finetune siglip on office system" → chosen: **image + text LoRA** (owner's request of 2026-10-30), and if it wins on validation **retrain the student only** (no new VLM scoring). The office GPU stays held for this.
- **Owner's question answered:** nexa_fight alerts are never trained on; they are embedded with the trained model and searched (proposal lanes B/C, unchanged). Caveats given: new cameras (measured by Protocol B), "fight" is outside MEVA's activities (retention check added to the spec), person boxes must come from the pipeline's detector.
- **Spec:** `docs/planning/spec-lora-image-text.md` (E8), waiting for approval; adoption rule and retention check fixed in it before any run. Demo spec to follow.

---

## Open decisions

| ID | Question | Status |
|---|---|---|
| D-025 | Approval of `spec-vlm-followups.md` (E1–E4; §8 lists five decisions) | E1, E2, E3 approved (D-026, D-027) and done (D-028); E4 waits for the office-server details |
| D-028 | E4's G1 depth | **Decided (D-029):** top 50 first, whole validation only if it passes |
| D-028 | Optional: retrain the student on the merged turn label | Open — owner |
| D-031 | Approve E6 and the night-run design | **Decided (D-032):** E6 approved (E4B first, run on the server beside Ollama); crop store copied |
| D-031 | Remote access route | **Decided (D-032):** Tailscale user mode, set up by the owner (no IT person) |
| D-032 | Which Qwen weights to delete for space | **Decided (D-038):** none now; Qwen2.5-VL tested first (E7), deleted only if it does not contribute |
| D-035 | Accept the E4 server set-up check despite the rule failure (1 of 4,800 pairs over the 48-pair noise floor; G1 identical within 0.002)? | Open — owner (recommended: accept) |
| D-036 | E4 8B fails G1 by 0.0009 | **Decided (D-037, as read):** cascade rows with the 8B; no distillation |
| D-033 | After Gemma E4B failed G1: 12B or stop | **Decided (D-034):** stop Gemma, remove it, back to Qwen 8B (E4) |
| D-043 | 32B passed G1 at both depths: start teacher labels → students → cascade rows; include the 17.6 h eval "VLM on all" row? | **Decided (D-044):** started, short list only |
| D-048 | 32B rows done; the 32B equals the 8B on eval at ~13× the time. Stop the VLM experiments and release the server GPU (STOP file), or test anything else on it first? | **Decided (D-049):** VLM experiments stop; GPU kept for the image + text LoRA |
| D-049 | Approve `spec-lora-image-text.md`; retention check R1 (nexa_fight labelled clips, if they exist) or R2; adoption bar | Open — owner |

---

## Standing rules

- **No decision is ever taken on eval-split numbers.** Selection uses validation only; eval is reported, never used to choose.
- Every result records the data version (encoder id / pipeline version) it came from.
- Failed and negative results are logged with the same care as positive ones.
