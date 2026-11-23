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

### D-050 · E8 approved and started; retention check dropped; all work committed and pushed (2026-11-04)

- **Owner:** approved E8 (image-only vs image + text LoRA, both protocols, server); retention check not needed (searches will be ordinary visual descriptions such as "a person in a blue jacket in the restaurant", not fights); "complete it asap".
- **Git (owner: "push all the things, repo is private"; no AI name):** 9 commits by concern (`b7c143c`..`b1724fe`): rerank, teacher core, remote service, queue, cascade runner, specs, Result 04, Result 05, docs. Every commit passes its tests in a clean checkout. Pushed with the 9 earlier commits: `origin/main` = `b1724fe`. Not pushed: `CLAUDE.md`, `manifest_v2_drive/`, data, runs, the backup branch. Snapshot before committing: `../snapshots/before_commits_2026-11-04.tgz`.
- **Code (E8):** `sfs.train.run --text-lora` (image-only config hashes unchanged, test), `sfs.train.lora` text tower helpers, queue job type `module` (whitelisted entry points run inside the GPU-holding process; `empty_cache` disabled during the job). 158 tests pass.
- **Server:** `peft 0.21.2` installed (`--no-deps`), SigLIP 2 at `75de2d55…` (sha256 identical), packed file identical; holder restarted with the new code (PID 1343756, 14.5 GB) at 20:19; queue file `05_e8_lora.json`: smoke run, image + text A-middle / B-uniform, image-only A-middle / B-uniform.

### D-051 · E8 result: image + text LoRA fails the adoption rule (validation, 2026-11-04)

- **Runs (server, same GPU, seed 0, 20:20–20:46, all `ok`):** best val mAP — image-only A-middle **0.4359** (epoch 12), B-uniform **0.3691** (epoch 6); image + text A-middle **0.4075** (epoch 8), B-uniform **0.4198** (epoch 24). Server image-only reproduces the laptop runs within 0.001 (A) / 0.007 (B) (laptop 0.4371 / 0.3758). Smoke run epoch 0 = 0.2717, identical to the laptop.
- **Rule (spec §5, fixed before the runs):** (1) mean gain (−0.0284 + 0.0507) / 2 = **+0.011 ≥ +0.010: pass**; (2) A drops by 0.028 > 0.005: **fail** → **not adopted**; the image-only model stays the final model.
- **Per activity (validation):** A: 6 better / 6 worse (largest: opens vehicle door −0.24, enters vehicle −0.19, turns left +0.11); B: 12 better / 4 worse (opens facility door +0.29, vehicle stops +0.16, carries heavy object −0.23). Single activities move by up to ±0.3, so single-seed noise is large.
- **Reading:** text adaptation helps on unseen cameras (B, +0.05) and hurts on seen cameras (A, −0.03). With one seed this may be partly noise. A second seed (~30 min) would show it; that is a new check, not part of the fixed rule → owner's call. No eval number read.
- **Raw:** `docs/results/raw/06_*`.

### D-052 · E8 second seed started; reading fixed before any seed-1 result (2026-11-04, 23:19)

- **Owner:** "Lets go with option 2, do the experiment and let me know the results. And also logs each and every result."
- **Code:** `sfs.train.run --seed` (default 0). It sets `TrainConfig.seed` (sampler, augmentation, caption draws) and now also seeds torch before the LoRA adapters are created (their initial values + dropout). Before this, the adapter init came from torch's unseeded process state. Seed-0 config hashes are unchanged (seed was already 0 in the config); `tests/test_train.py` 19 pass.
- **Runs (server, same GPU, holder restarted 23:18 to load the new code, new PID 1847435, Ollama not loaded at the time):** queue file `runs/queue/06_e8_seed1.json`, four jobs in this order: image-only A-middle, image + text A-middle, image-only B-uniform, image + text B-uniform, all `--seed 1`, outputs `runs/lora_{io,it}_s1/<run>`. Settings otherwise identical to D-051.
- **Reading fixed now, before any seed-1 number:**
  1. Per protocol, report each seed's difference (image + text − image-only) and the 2-seed mean.
  2. **The §5 rule is applied to the 2-seed means:** mean of A and B ≥ +0.010 and neither protocol's 2-seed mean drops by more than 0.005 → recommend adopting image + text.
  3. A protocol's effect "repeats" if both seeds give the same sign and each is larger than 0.010 in size; otherwise it is called noise.
  4. If the rule fails but B's gain and A's loss both repeat, the result is protocol-dependent: the owner may choose image + text for the nexa_fight demo (unseen cameras, protocol B). That would be reported as a choice made after seeing the results, not as a pass of the rule.
  5. Still validation only; no eval number is read for this decision.
- **Each run's result is appended below as it finishes.**
- **Run 1/4, image-only A-middle seed 1 (23:19–23:24):** best val mAP **0.4203** (epoch 7 of 12, stopped: no gain for 5 epochs); epoch 0 = 0.2717 (same frozen start). Seed 0 gave 0.4359 → the seed alone moves this run by 0.016.
- **Run 2/4, image + text A-middle seed 1 (23:24–23:29):** best val mAP **0.4112** (epoch 7 of 12). Difference to image-only seed 1: **−0.0091** (seed 0: −0.0284). A's 2-seed mean difference **−0.0188**; same sign in both seeds but seed 1 is below the 0.010 size bar → by rule 3, A's loss does not count as "repeats".
- **Run 3/4, image-only B-uniform seed 1 (23:29–23:35):** best val mAP **0.3858** (epoch 14 of 19); frozen start 0.2222. Seed 0 gave 0.3691 → the seed moves this run by 0.017 (laptop seed-0 run: 0.3758).
- **Run 4/4, image + text B-uniform seed 1 (23:35–23:44):** best val mAP **0.4297** (epoch 18). Difference to image-only seed 1: **+0.0439** (seed 0: +0.0507). Queue file `06_e8_seed1` finished, all four jobs `ok` (271 / 290 / 392 / 525 s).

### D-053 · E8 second-seed result: image + text still not adopted; its gain on unseen cameras repeats (validation, 2026-11-04)

Read with the rule fixed in D-052 (`scripts/e8_report.py --root runs/e8 --seeds 0,1`, raw `docs/results/raw/06_e8_report.json`):

| Protocol | Seed | Image only | Image + text | Difference |
|---|---|---|---|---|
| A-middle | 0 | 0.4359 | 0.4075 | −0.0284 |
| A-middle | 1 | 0.4203 | 0.4112 | −0.0090 |
| A-middle | mean | 0.4281 | 0.4094 | **−0.0187** |
| B-uniform | 0 | 0.3691 | 0.4198 | +0.0507 |
| B-uniform | 1 | 0.3858 | 0.4297 | +0.0439 |
| B-uniform | mean | 0.3775 | 0.4248 | **+0.0473** |

- **Rule (§5 on the 2-seed means):** mean gain **+0.0143 ≥ +0.010: pass**; A's mean drops by 0.019 > 0.005: **fail → not adopted.** The image-only model stays the final model (Results 03–05 unchanged).
- **Repeats (rule 3):** **B's gain repeats** (+0.051, +0.044; both > 0.010, about 3× the seed-to-seed spread of a single run, 0.016–0.017). A's loss has the same sign in both seeds but seed 1 (−0.009) is below the 0.010 bar → by the fixed reading it does **not** count as repeated.
- **Rule 4 (protocol-dependent choice) is therefore not triggered as written:** it needed both effects to repeat. What is established: text adaptation reliably helps unseen cameras; on seen cameras it is probably harmful (mean −0.019, both seeds negative) but not confirmed by the bar.
- **Per activity (sign agrees in both seeds):** B 14 of 17 agree, 12 better / 2 worse (consistently better: opens facility door +0.29/+0.19, vehicle stops +0.16/+0.12, talks on phone +0.15/+0.09, turns left +0.09/+0.09; consistently worse: carries heavy object −0.23/−0.10, exits through structure −0.04/−0.06). A 9 of 15 agree, 4 better / 5 worse (consistently worse: opens vehicle door −0.24/−0.13, enters vehicle −0.19/−0.11, texts on phone −0.10/−0.17, turns right −0.12/−0.08, stands up). On A, the vehicle-door and phone activities, where the camera views were seen in training, lose most.
- **Correction to D-051's counts:** "A 6 better / 6 worse, B 12 / 4" ignored changes smaller than 0.01 without saying so; strictly (seed 0) A is 6 / 9 and B 12 / 5.
- **Seed-to-seed spread of the same set-up:** image-only A 0.4359 / 0.4203, B 0.3691 / 0.3858. Differences under ~0.02 between single runs are within seed noise.
- **What using image + text anyway would take:** using it for the nexa_fight demo (unseen cameras = protocol B) would be a choice made after seeing these results and must be reported that way. It needs the steps spec §3/§8 left unbuilt because the model was not adopted: an export of the adapted text side's query embeddings, re-embedding with the new image adapter, retraining the student, and `--text-emb` in the eval. Estimate: ~0.5 day of code + ~1.5 h of runs. The owner decides; no eval number was read.
- **Tests:** 161 pass (new `tests/test_e8_report.py`).

### D-054 · Demo on MEVA only; text-drift check run; demo built before its spec is approved (2026-11-05)

- **Owner (2026-11-05):** "Leave the nexa fight we will give demo on the meva videos"; on the recommended text-drift check: "nice recommendation, lets try this and do the experiment"; "write the demo spec and start working on it, I'll review the spec later and will let u know if need to change anything". So the owner **explicitly waives the wait for spec approval** for the demo (as for E6, D-032). The spec is still written first, and the build follows it, so changes after the review stay cheap.
- **Correction recorded:** my earlier reason against image + text for nexa_fight ("a choice after seeing results") was wrong. Choosing on validation is allowed. The real gap was that no test covered appearance / place wordings, which the text-drift check now addresses.
- **Text-drift check (spec-lora-image-text §6 R2, fixed now, before any measurement, 02:23):**
  - Sentences: `queries/appearance_sentences.csv`, 40 appearance / carried-object / people / place / vehicle / scene wordings outside the MEVA activity vocabulary, sha256 `62e1703a2fb72420…`. None is closer than 0.80 (character similarity) to a training caption (max 0.793, shared words only).
  - For each of the four image + text adapters (A-middle and B-uniform, seeds 0 and 1): embed the sentences with the frozen text side and with the adapted text side.
  - **Primary measure, the pass rule:** Spearman ρ between the two sets of pairwise cosine similarities (780 pairs). This measures whether the adapted text side keeps the *relations* between general wordings. **Pass = ρ ≥ 0.90 for every adapter** (spec §6 R2 bar, unchanged).
  - **Described, not gated:** mean cosine between each sentence's frozen and adapted embedding; nearest-neighbour agreement; the same measures on the 66 eval phrasings for scale (where the change is known to help on B).
  - **What it means:** this guards against collapse; it is not an accuracy measure. Pass → image + text may be offered in the demo as a model choice. Fail → the demo uses the frozen text side only.
  - Runs on the server (adapters, weights and code are there; CPU, the GPU holder untouched).

### D-055 · Text-drift check fails: the adapted text side changes how general wordings relate (2026-11-05, 02:40)

`scripts/text_drift_check.py` (tests `tests/test_text_drift.py`, 5 pass), server CPU, 16 s. Sentence file
hash on the server matches the frozen one (`62e1703a…`). Raw: `docs/results/raw/06_text_drift.json`.

| Adapter | 40 appearance sentences: ρ | self-cos | NN agree | 66 eval phrasings: ρ | self-cos | NN agree | Pass (ρ ≥ 0.90) |
|---|---|---|---|---|---|---|---|
| A-middle seed 0 | 0.813 | 0.938 | 0.60 | 0.628 | 0.906 | 0.82 | no |
| A-middle seed 1 | 0.825 | 0.943 | 0.62 | 0.658 | 0.915 | 0.77 | no |
| B-uniform seed 0 | 0.776 | 0.931 | 0.60 | 0.556 | 0.888 | 0.73 | no |
| B-uniform seed 1 | 0.767 | 0.933 | 0.55 | 0.639 | 0.893 | 0.77 | no |

- **Rule: fail for every adapter → the demo uses the frozen text side** (image-only LoRA model), as fixed in D-054.
- **Reading:** the adapted text side moves general wordings less than the activity phrasings it was trained towards (ρ 0.77–0.83 vs 0.56–0.66), but enough that for ~40 % of the sentences the closest other sentence changes. The B adapters, the ones that helped on unseen cameras, drift the most. This says the relations change; it does **not** measure whether appearance search gets worse or better (there are no appearance labels in MEVA). The 0.90 bar was set in the spec without a calibration run.
- E8 is closed: image-only LoRA is the final model for the report and the demo.

### D-056 · Search demo spec written; build started with the defaults (2026-11-05)

- Spec `docs/planning/spec-search-demo.md`. The build follows its defaults until the owner reviews it (D-054 waiver):
  - corpus = protocol A eval split (3,726 clips);
  - exact in-memory search + SQLite metadata. ChromaDB, named in the proposal, is not used: it is approximate and would make the demo differ from the evaluated numbers. This is reported as a change;
  - local 2B Instruct for the Verified level (k VLM calls, γ = 2 and β = 1 from Result 04);
  - show-labels toggle, off by default;
  - stored-crop animations, no raw video.
- **Contract:** for an eval phrasing's own text vector, the demo's Fast / Standard rankings equal the evaluation's score columns exactly (parity test).

### D-057 · Search demo built and checked (2026-11-05, 03:30)

- **Code:** `src/sfs/demo/` (`index`, `search`, `previews`, `app`); `sfs.teacher.frames.CropIndex(names=…)` (optional subset of shards; the default is unchanged). Docs: `docs/demo.md`, README section, spec updated.
- **Parity, measured (slow test, real data):** the contract of D-056 cannot hold bit-exactly. Scores differ by ≤ 1e-7 (LoRA) and ≤ 5e-6 (fused), because BLAS adds up one query vector in a different order than 66 vectors at once, so near-ties swap places. Over all 66 Q1 and 1,140 Q2 queries × Fast/Standard:
  - Success@1 and Success@10 are **identical on every query**;
  - |ΔAP| ≤ 0.0001 (Q1) and ≤ 0.012 (one Q2 window), with a mean ≤ 0.00002.

  The spec was restated to this. `tests/test_demo.py::test_demo_reproduces_the_evaluated_metrics` (slow) passes in 7 min.
- **Bugs found and fixed while building:**
  - the exact-hour filter was missing (found by the Q2 parity run);
  - Gradio would not have passed the result-click event (postponed annotations hid `gr.SelectData`);
  - the VLM's loading time was counted as search time.
- **Live check (app at 127.0.0.1:7860, driven through `gradio_client`), "a person opening a car door":**

  | Level | Total time | VLM calls | Labelled matches in the top 10 |
  |---|---|---|---|
  | Fast | 0.2 s | 0 | 4 / 10 |
  | Standard | 0.8 s | 0 | 6 / 10 |
  | Verified, k = 10 | 3.2–4.3 s | 10 (0.27–0.36 s per call measured) | 6 / 10 |

  - The previews are served as the identical animated GIFs. The bundle takes 263 MB, plus 1.2 GB of previews (~30 min to render once).
  - The 2B VLM often gives low yes-probabilities even to correct clips (Result 04: 2B adds ≤ +0.03), which is visible in the captions.
- **Tests:** 177 pass (+3 slow deselected); demo tests 11 + 1 slow.

### D-058 · Demo presented as a recorded video; server GPU released (2026-11-05, 05:10)

- **Owner (2026-11-05):** "No we are not going to give live demo, I'll record the video and present it there.. if there is no experimentation remaining and we don't need the server to record the video we can release it."
- **Checked before release:** no experiment remains (E1–E8, the 8B/32B runs, the drift check are all logged; D-055 closed E8). The demo runs entirely on the laptop with the local 2B (spec D3 default), so the recording needs no server.
- **Release:** holder PID 1847435 (`sfs.teacher.queue --watch runs/queue`, up 5 h 52 min) had no queued jobs and 0 % use; `runs/queue/STOP` created at 05:10. The process exited by itself; the GPU went from 14,263 MiB to 61 MiB in use, with no processes on it.
- **Effect on the demo:** the "network failure during the presentation" argument against the 8B in Verified (spec D3) no longer applies to a recording. The 2B stays the default; the 8B would need the holder restarted (`~/sfs/start_holder.sh`) and is the owner's call. `docs/demo.md` now describes a recording walk-through.
- Still on the server, owner's call: Qwen2.5-VL 3B/7B (D-040), our env `~/envs/sfs` and data under `~/sfs`, the Tailscale node.

### D-059 · Alerts from your own videos + new dashboard; sign-in dropped; Docker later (2026-11-07)

- **Owner (2026-11-07), after testing the demo:**
  - **Alerts:** run YOLO on any video (YouTube or another dataset). Each detection is an alert named by its class, with 5 frames before and 5 after; the clip is embedded and found from the search.
  - **Dashboard:** two tabs, Detect (upload → alerts) and Search (query → alert clips), with a better design. The owner picks the design from options.
  - **Sign-in:** skipped.
  - **Docker:** after these two.
  - **Scope:** only what works with the current models; no new experiments.
- **Feasibility checked before the spec:**
  - The crops (`sfs.data.regions`) and the re-ranker inputs (`box_features`: LoRA tight + frozen context embeddings + box motion) can all be computed from YOLO tracks with the existing functions.
  - `yolov8m.pt` (COCO) is already on the laptop (drink_sprinkle_detection project).
  - Missing: `ultralytics` and `lap`. Node 24 is installed. Docker Desktop is installed, but its WSL engine is stopped.
- **Spec:** `docs/planning/spec-video-alerts-dashboard.md`, waiting for approval. Two points where the spec recommends a change to the owner's wording:
  - **D1:** an alert fires when an object's track is confirmed (≥ 3 frames) and repeats every 5.5 s while the object stays. Otherwise YOLO's per-frame detections give an alert every frame, or one alert per object, which misses later actions.
  - **D2:** the 5 + 5 frames are taken at 2 fps (5 s clip, the sampling the models were trained on), not as raw frames (0.17 s at 30 fps).
- **Confidence band:** relative to the MEVA corpus (High above the 99th percentile, Medium above the 95th), labelled as not a probability. Calibrated probabilities would need labelled videos, which are out of scope.
- **Design options:** three directions (dark console, light analytics, clip-map-first), published as a private design page for the owner to choose from.

### D-060 · Owner answers D1–D6 of the alerts + dashboard spec; build started (2026-11-07)

- **Owner:** "agree with both of your points"; "if person is detected then hold it till he is gone and detected again"; "I want the alert clip should be of 5 sec, not more not less"; "D3 – person, vehicles and one more common object"; "D4 – control room style"; "D5 – okay lets try"; "D6 – okay install it in dlproj".
- **Built as:**
  - **D1:** one alert per object, held while it stays; a new alert only after it was gone for 2 s. The repeat interval exists but is off.
  - **D2:** 11 frames at 2 fps; the window shifts at the video's start or end, so the clip is always 5.0 s (a video shorter than 5 s is flagged).
  - **D3:** backpack is my pick for the common object; handbag or suitcase are drop-in swaps.
  - **D4:** design A. **D5:** three.js PCA map, an illustration only.
- **Installs:**
  - `ultralytics 8.4.204` + `lap 0.5.14` into `dlproj`. A dry run first showed that torch 2.14.1, numpy 2.5.2 and OpenCV 5.0 stay as they are; confirmed after the install. The package list from before was saved in the scratchpad.
  - npm: react 18.3.1, react-dom 18.3.1, three 0.170.0, vite 5.4.11, @vitejs/plugin-react 4.3.4 (`web/node_modules`, git-ignored).
- **YOLO weights:** `yolov8m.pt` (COCO) copied from the owner's drink-spiking project into `data/models/yolo/`, sha256 `5d4a90cdc7a2…`; no download.

### D-061 · First run on real footage: alert flood and slow clips found and fixed (2026-11-07)

- **Test footage:** a local smoke test on the owner's own test clips (`drink_sprinkling_testing/videotest{1,3,6}.mp4`). It used a scratch alert store in the session scratchpad, never the demo store; nothing left the laptop.
- **Alert flood:** `videotest3` (14 s, crowded, at most 6 people at once) gave **38 alerts** with the draft rule (confirmation after 3 frames). The causes were boxes with confidence 0.26–0.45, and ByteTrack giving an occluded person a new id (42 track ids). Measured offline on the same detections:

  | Rule | Alerts (videotest3) |
  |---|---|
  | conf ≥ 0, confirm 3 frames (draft) | 38 |
  | conf ≥ 0.5, confirm 3 frames | 26 |
  | conf ≥ 0.5, confirm 15 frames (0.5 s) | 20 |
  | **+ track stitching** | **6** |

  - With track stitching on all three clips: 20 → 6, 7 → 3 and 4 → 2 alerts, against at most 6, 4 and 2 people at once.
  - **Adopted:** confirmation = 0.5 s with conf ≥ 0.5, plus stitching. A new track next to a just-lost track of the same class (≤ 2 s) keeps that object's alert.
  - This is the owner's "hold it till he is gone" rule, made robust to tracker id switches. Spec §4.2 amended.
- **Slow clips:**
  - The whole run took 303 s for 14 s of video (436 MB).
  - Profiled: detection + tracking take 8.9 s; one VP8 WebM per alert takes 4.1 s each, so 38 alerts × 4.1 s ≈ 156 s.
  - **Changed:** the whole video is written once with every tracked box drawn (`annotated.webm`), and each alert's clip is its 5 s segment of it. The same clip then took **50 s, including ~20 s of model loading, and 61 MB**. Spec §4.5 amended.
- **Embedding parity (slow test):** stored MEVA tight crops through the alert path give back the evaluated embeddings: cosine **1.00000** (LoRA) / **0.99999** (frozen), over 89 frames of 10 clips.
- **Back-end bugs found by the API tests and fixed:**
  - FastAPI could not see the route types. Postponed annotations made the upload and body types forward references; `sfs.web.server` no longer postpones them, and the request models are module-level.
  - SQLite connections were committed but never closed. They are now a context manager that commits or rolls back and always closes.

### D-062 · Alerts + dashboard built and checked (2026-11-07)

- **Code:**
  - `src/sfs/alerts/`: detect, rule, clip, embed, ingest, store.
  - `sfs.demo.search`: collections `meva` / `alerts` / `both`, alert filters, confidence bands, Verified on alert crops.
  - `src/sfs/web/`: FastAPI server, ingest job runner, 3D clip map.
  - `web/`: React front end, design A.
- **The MEVA results stay as they were:**
  - MEVA is scored in full, and its mean and spread are the reference for the alerts' z-scores.
  - With alerts in the store, MEVA rankings and scores are bit-identical (tested); the existing parity tests still pass.
- **Live check** on the laptop (RTX 4060, all models loaded, 5.7 GB):

  | Search | Clips | Time | VLM calls |
  |---|---|---|---|
  | Standard over MEVA + 9 alerts | 3,732–3,735 | 0.6–0.8 s | 0 |
  | Fast | 3,732 | 0.04 s | 0 |
  | Verified on alerts, k = 5 | 6 | 3.5 s | 5 |

  - **Upload over HTTP:** a 10 s 720p video gave 3 alerts, each a 5.0 s clip, in 22 s including loading YOLO. Thumbnails, the annotated WebM, the detail frames and the 3D map were served.
  - **Pages:** checked from headless-Chrome screenshots (Search, Detect, 3D map, Compare).
  - **3D map:** re-framed after the first screenshot (the cloud was off-centre and the points too small).
- **Honest reading for the demo:**
  - The owner's test footage (indoor, close-up) ranks "Low" against MEVA's best clips and sits in its own region of the map.
  - Videos unlike MEVA are a demonstration, not a measured result (stated in `docs/demo.md`).
- **Tests:** 197 pass (+20: `tests/test_alerts.py`, `tests/test_web.py`, new cases in `tests/test_demo.py`); slow: alert embedding parity passes.
- **Docs:** `docs/demo.md` rewritten (run, controls, how a video becomes searchable, recording walk-through), README section and layout, spec amended.
- **Kept:** the Gradio page (`sfs.demo.app`) still works. Removing it is the owner's call.

### D-063 · Docker Compose spec drafted (2026-11-07)

- **Owner:** tested the dashboard ("working fine"); "now lets work on the docker compose file".
- **Spec:** `docs/planning/spec-docker.md`, **waiting for approval**. One service, one command
  (`docker compose up` → http://127.0.0.1:8000), same server, same models.
- **Facts the design rests on (checked, not assumed):**
  - Docker Desktop 29.1.3 + Compose v2.40.3 installed (WSL2 backend), engine not running; driver 610.62.
  - The Hugging Face cache keeps SigLIP 2 as two-level Windows symlinks (Xet layout), unreadable
    through a Linux bind mount → a plain-file copy of the snapshot (1.5 GB, local, no download) is
    mounted as the container's cache. That also pins the real revision (`75de2d55…`), an open follow-up.
  - 61 GB free on C: → data and weights (~20 GB) are mounted read-only, never baked into the image.
  - `torch 2.14.1+cu126` exists on the cu126 index (`pip index versions`, no download).
- **Pre-fixed parity rule (§5):** 66 phrasings at Fast + Standard (top 50) and 5 at Verified k = 10,
  container vs laptop: Success@1/@10 identical on every query, scores within 1e-3, order may swap
  only across gaps < 1e-3. Alerts: same video → same alerts, frames, windows; embeddings cosine ≥ 0.9999.
- **Decisions for the owner (§7):** D1 build at the office (recommended; ~3–3.5 GB download);
  D2 alerts in a Docker volume (recommended) or the shared folder; D3 no CPU variant now;
  D4 test video; D5 purpose = reproducible demo on this laptop, not an image for the instructor.
- **Files written the same day** (owner chose "both in parallel" with the report, so the recommended
  defaults were taken: D2 volume, D3 no CPU variant, D5 a reproducible demo on this laptop; D1 the build
  at the office; D4 still to name): `Dockerfile` (web / runtime / test stages, non-root, offline at run
  time), `compose.yaml` (GPU, read-only data mounts, `alerts` volume, health check, loopback-only port),
  `.dockerignore`, `docker/requirements.txt` + `constraints.txt` (the `dlproj` pip freeze as constraints,
  so Linux resolves to the tested versions) + `requirements-test.txt`, `scripts/prepare_docker_models.py`
  (SigLIP 2 snapshot → plain files, hash-checked, pins commit `75de2d55…`), `scripts/docker_parity.py`,
  `docs/docker.md`; `sfs.web.server --host` via a new `build_parser()`.
- **Checked:** `docker compose config` valid; `tests/test_docker.py` (7 tests: parser, cache copy incl.
  corruption and symlinks, parity rule) + `tests/test_web.py` pass; full suite: 204 passed, 4 deselected, 11 warnings in 81.92s (0:01:21).
- **Rule refined before any measurement:** Verified parity gates only the set of clips the VLM checked
  (bf16 noise 0.37 in the raw logit, Result 04); Fast/Standard keep the 1e-3 score bar.
- **Not done:** the build itself and every check of spec §5 (office connection; Docker engine not running).

### D-064 · Final report and presentation: plan drafted (2026-11-07)

- **Owner:** "write the report with proper research, literature review, what was missing, what we claimed
  to do, what we experiment, what we achieve and make the presentation." Answers to four questions:
  proposal template style, ≤ 10 pages; Word + PDF report, PowerPoint deck; 10 minutes (~12 slides);
  report and Docker in parallel.
- **Spec:** `docs/planning/spec-report.md`, **waiting for approval**: outline with a page budget, the
  claims-vs-delivered table (§4.4, the spine of the report), the verified literature list, figures from the
  raw result files through `report/numbers.json`, a 12-slide deck with timings, decisions D1–D6.
- **Research done:** the submitted proposal re-read from its Word source (3 contributions, 3 hypotheses,
  10 references, a tools list that the work partly departed from); Result files 01–06 and their hypotheses
  sections; 14 web lookups. **Verified online:** Tran et al. CVPRW 2025 (arXiv 2504.08384); ForeSea
  (arXiv 2603.22872, Park, Li et al., Qualcomm, March 2026); SigLIP (2303.15343); SigLIP 2 (2502.14786);
  Qwen3-VL technical report (2511.21631); UCA (Yuan et al., CVPR 2024, 2309.13925); QASIR (2312.00414);
  ByteTrack (2110.06864); MEVA (2012.00914). **SurveiLens** [4] exists (IEEE Xplore 11400480; Owais,
  Siddiqui, Alam, Ali; preprint Sept 2025): the proposal cited it without authors; the venue name is
  still to be read off the IEEE page, which refuses automated reading.
- **The gap re-checked:** MEVA appears only in TRECVID ActEV (activity detection); no language-retrieval
  use of it was found, as in D-003.
- **Claims vs delivered, in one line each:** adaptation **confirmed**; distillation **not as claimed** (the
  temporal student trained on human labels is the gain, every teacher failed to add); the cost frontier
  **delivered** with a different shape (2B VLM on all < LoRA alone; 8B/32B +0.06 S@1 at 10 calls);
  ChromaDB / W&B / Optuna / ONNX / torch.compile **not used**, with reasons; alert ingestion + interface
  **delivered**; image + text LoRA **measured, not adopted**.
- **Report written the same day.** Owner: "write the report I'll read and let u know if need to change
  anything"; "be technical but don't be over complicated"; add the teacher failures (every VLM tested, the one
  that passed and added little). ChromaDB question answered: installed, never used; exact search was chosen
  in D-056 and the report says so. The §9 recommendations were taken as assumptions (completed-study
  framing, negatives in the main text, one-page appendix, author block from the proposal; cover date
  "November 2026" until the submission date is known).
- **Built:** `report/Ahmed_Wajahat_DLReport.docx` + `.pdf` — 12 pages: 10 of body including 5 figures,
  references on p. 11, appendix on p. 12 — by `scripts/report_build.js` (docx-js under `report/build/`,
  git-ignored) from `report/numbers.json` (`scripts/report_numbers.py`, checked by
  `tests/test_report_numbers.py` against the Result files) and `scripts/report_figures.py` (pipeline, cost
  curve, per-activity); dashboard screenshots from the running server (MEVA query; the owner's garage CCTV
  upload, 22 alerts). PDF exported with Word; every page checked by eye from PyMuPDF renders. Installs:
  `docx` (npm, `report/build/`), `pymupdf` into `dlproj` (page renders only).
- **Content:** 20 references (RANKVIDEO and Vote-in-Context left out: authors not verified); §6.4 lists all
  seven VLMs with G0/G1 and the AUC reading; §6.5 includes the 32B students; §4 states each proposal tool
  not used, with the reason. The confidence intervals and the 32B student table are quoted from Results
  03–05 (marked in the builder); everything else comes from the raw files.
- **Full suite:** 207 passed, 4 deselected, 12 warnings in 45.75s.
- **Deck built the same day:** `report/Ahmed_Wajahat_DLPresentation.pptx`, 12 slides for 10 minutes with
  timed speaker notes, by `scripts/deck_build.js` (pptxgenjs under `report/build/`; theme = the dashboard's
  control-room look, spec §9 D3 default) from `report/numbers.json` and the report figures; two native
  charts (LoRA bars, the cost curve as a line chart over the call budget), the VLM gate table, the demo
  screenshot, a claims scorecard. Checked: the skill's validator passes; every slide exported through
  PowerPoint and inspected (first pass found six layout defects — overflowing cards, a table running into
  its paragraph, cramped stats, three two-line titles — all fixed in the builder and re-checked). Installs:
  `pptxgenjs` (npm, `report/build/`), `defusedxml` + `python-pptx` into `dlproj` (the validator only).
- **Owner's first review (2026-11-07 evening):** the literature review must be prose, as in a paper (who did the
  work, what they claimed, what they measured, then the citation), not a table. Done in the builder: §3 is now
  five paragraphs by research line (encoders; corpus moment retrieval and cost; surveillance; adaptation and
  distillation; backbones, dataset, tools), each work with its method and its reported results, followed by the
  "what was missing" paragraph; the old Table 2 removed and the later tables renumbered; Table 1's MEVA row
  now quotes the paper's release figures (144 h annotated at the 2021 release). Figures 3 and 4 slightly
  smaller so the references still start on page 11. The Word file was open on the owner's machine (lock), so
  the build went to the scratchpad first (`REPORT_OUT`) and is copied into `report/` once the file is closed.
- **Next:** the owner reads both; changes come back as edits to the builders, never by hand in the files.


---

### D-065 · Docker image built and checked at the office; two platform differences measured (2026-11-08)

- **Owner:** at the office: "shall I run the docker compose command or anything need to do before?"; later asked
  for the commands and stopped the native server (PID 50608) themselves once the laptop-side parity recording
  was in.
- **Before the build:** Docker Desktop was not running (owner started it); `scripts/prepare_docker_models.py`
  copied the SigLIP 2 snapshot (6 files, 1.54 GB, every hash ok, commit `75de2d55…`) into `data/hf-cache`; a
  Linux container sees the RTX 4060 (driver 610.62); all six mounted folders present; C: had 49 GB free
  (Docker's own store holds 66 GB of images from other projects, 50 GB reclaimable — the owner's to prune).
- **First build failed** at the pip step: the Dockerfile copied `docker/requirements.txt` alone, and that file
  (and `requirements-test.txt`) pulls `docker/constraints.txt` in with `-c`. Fixed
  (`COPY docker/requirements.txt docker/constraints.txt docker/`) and a test added that parses the Dockerfile
  and the requirement files and fails on exactly that omission. Second build ok: pip step 414 s, `pip check`
  clean; `sfs-dashboard:0.1` id `77659e1a0b46`, 12.3 GB on disk / 4.17 GB content; base images pinned by
  digest (node 24.12.0-alpine `c921b97d…`, python 3.12.15-slim-bookworm `6a0e06f5…`).
- **Test image:** the first run failed only on the new Dockerfile test (the Dockerfile was not in the image) →
  the test stage copies it in; `pytest -q -rs`: 190 passed, 18 skipped (Qwen / Gemma processor files and raw
  result files are not in the image; 1 needs a GPU), 4 deselected, 24 s.
- **Container run:** healthy 116 s after the first `up` (91 s after the restart), 3,726 MEVA clips, cuda, VLM
  loaded, 0 alerts in the fresh volume `project_alerts`.
- **§5.2 parity (rule fixed in D-063):** laptop recording 137 searches / 73.8 s (conda server), container
  95.4 s. Success@1 and Success@10 identical on all 137 queries; 11 top-10 swaps, all across gaps < 1e-3;
  Verified: the same 5 sets. Fast scores within 5.6e-5. **Standard scores up to 5.42e-3 apart on 36 queries
  (median 1.06e-3) → the 1e-3 bar fails.** Median seconds Fast 0.068 → 0.090, Standard 0.260 → 0.268,
  Verified 4.12 → 7.86 (crops read through the Windows file share).
- **§5.3 alerts:** the owner's garage clip (the only video in the laptop store; D4 option A in effect) →
  **22 alerts on the laptop, 21 in the container** (same sha256, 3,806 frames, `yolov8m.pt:5d4a90cd`). The 9
  person alerts and the two opening alerts identical; car alerts start 1–4 frames apart (0.08–0.16 s); one
  borderline car track (66.72–71.72 s) only on the laptop; one laptop track is two in the container; peak
  confidences differ by ≤ 0.017.
- **Cause, measured** (`scripts/docker_platform_diag.py`, recordings in `runs/docker/diag/`; the same versions
  on both sides — torch 2.14.1+cu126, cuDNN 9.10.2, OpenCV 5.0.0, ultralytics 8.4.204 — on the same GPU):
  the student's logits with the same input are **bit-identical** (max 0.0); the query embeddings from the
  float16 text tower differ by max 3.5e-4 per component (cosine ≥ 0.9999996), which the student turns into
  max 0.054 logits (median 4.5e-4; the clips' logit spread is 2.8) — the whole Standard difference, and the
  5.6e-5 of Fast; the decoded frames differ on 32–35 % of pixels (mean 0.4, max 7–21 levels) although both
  report the FFMPEG back-end — the wheels bundle different FFmpeg builds — which moves YOLO's confidences
  across the thresholded, stateful alert rule.
- **Determinism controls (same clip ingested a second time on each side):** container run 2 = run 1, 21 alerts
  identical in class, start, end, confidence and time; laptop run 2 (`sfs.alerts.ingest` into a scratch
  store, container stopped meanwhile) = run 1, 22 alerts identical in class, time and confidence. The
  21-vs-22 is between the platforms, not noise within one.
- **§5.4 persistence:** `down` / `up`: 21 alerts identical before and after. **§5.5:** healthy; the log shows
  only model loading, no download. **§5.6:** full local suite 210 passed, 4 deselected, 55 s.
- **Decision for the owner (spec D6; recommendation A):** accept and document — rankings, Verified decisions
  and person alerts are identical; what differs is the third decimal of Standard scores and borderline vehicle
  tracks, traced to OS builds of libraries we pin by version but cannot pin by build — or B: the text tower in
  float32 on both platforms (changes the laptop's numbers; re-opens D-057's demo-vs-evaluation parity) plus one
  video-decoder build on both (PyAV with a pinned FFmpeg; changes the alert pipeline), about a day and still
  not guaranteed bit-exact. The rules themselves are not changed after the fact.
- **Files:** `Dockerfile` (constraints `COPY`, Dockerfile in the test stage, `-rs`), `scripts/docker_platform_diag.py`
  (new), `tests/test_docker.py` (+3 tests: Dockerfile `COPY` check, diag compare, diag frames; 10 in all),
  `docs/docker.md` (results table, "why the two platforms differ"), `docs/planning/spec-docker.md` (results,
  D1/D4 taken, D6), `README.md`, `runs/docker/` (recordings; untracked).
- **State:** the container is up on 127.0.0.1:8000 with the GPU; the conda way needs `docker compose stop`
  first (same port, same GPU).

### D-066 · Push of all pending work; submission package, CLIP baseline and 6-slide deck specified (2026-11-17)

- **Push (owner: "push with proper git history without adding on name"):** 8 commits by concern,
  `b1724fe..5be5f35`, each passing its tests in a clean checkout (140 → 196 passed, data-dependent tests skipped
  there). `manifest_v2_drive/*.npz` (4 × 342 MB, over GitHub's 100 MB limit) and the byte-identical
  `manifest_v2-1.csv` ignored, not pushed. **No further pushes until the owner asks.**
- **Owner's capability question** (own video: two people together, phone, leaving a building): answered from the
  code and Result 04's per-activity rows — an alert's actors are its object plus the tracked objects next to it, so
  multi-person scenes are embedded like MEVA's two-person activities (A, student: talks to person S@1 0.98, talks on
  phone 0.77, exits through structure 0.46); limits: the clip is the 5 s around the object's first appearance
  (repeat off, D-060), one alert per person, tracker boxes, unmeasured on own footage. "Walking side by side" is not
  a MEVA activity.
- **Owner's answers:** ship the complete system in `submission/` (models, scripts, Docker files, guide, report,
  presentation, test videos, a removable MEVA folder; MEVA data moved, not duplicated); Python on GPU or CPU;
  Google Drive link; run CLIP now; a 6-slide presentation to the owner's outline; report after the professor's
  template.
- **Spec:** `docs/planning/spec-submission.md`, **waiting for approval** (D1–D5: what "MEVA videos" means, code
  scope, junctions at the old paths, without-MEVA mode, CLIP download place).
- **CLIP weights** (`openai/clip-vit-base-patch16`, ~600 MB) downloaded at the office on the owner's go, before the
  spec approval (download only; no code). Main branch has only `pytorch_model.bin` (the safetensors copies are
  unmerged bot PRs), so the official file was taken at revision `57c21647…` (598,641,023 bytes, sha256 `ec89c7b0…`,
  equal to Hugging Face's LFS record), loaded with `weights_only` (no code execution); 149.6 M parameters; 3 min.

## D-067 — 2026-11-17 — Zero-shot CLIP ViT-B/16 measured (spec-submission §2)

- **Basis:** owner: "start testing and evaluating clip model while I am reviewing the spec" — an explicit go for
  spec §2 before the spec's approval (as D-032 / D-054). The submission folder (§1) and the 6-slide deck (§3)
  still wait for the approval.
- **Rule fixed before any CLIP number (spec §2):** no decision depends on it; whatever CLIP scores, the system does
  not change. Eval is therefore read once for a reported comparison.
- **Code:** `scripts/clip_baseline.py` (`embed` / `val` / `eval`), `tests/test_clip_baseline.py` (7 tests, tiny
  random CLIP, no download). CLIP uses its own preprocessing (shortest side 224 bicubic, centre crop) and tokenizer;
  float16 on the laptop GPU; the 73,569 packed tight crops embedded in 6 min (`data/embeddings_clip/`, 76 MB, rows
  aligned with the SigLIP 2 baseline by (instance, frame)).
- **Harness check first:** Result 01's validation code had run on Colab and was not in the repo, so it was re-written;
  re-run on the stored SigLIP / SigLIP 2 embeddings it reproduces Result 01 exactly (0.663 / 0.120, 0.697 / 0.220).
  The eval job re-ran SigLIP 2 beside CLIP and equals `02_baseline.json` on every stored metric (max diff 0.0).
- **Validation (Result 01 protocol):** CLIP AUC 0.726, P@10 0.207 (SigLIP 2 0.697 / 0.220). The AUC lead is
  `person_stands_up` alone (0.625 vs 0.127); over the other 14 types CLIP 0.733, SigLIP 2 0.738.
- **Eval (Result 02 protocol, tight):** Q1 mAP A 0.142 vs 0.179, B 0.138 vs 0.174; Q2 S@1 A 0.237 vs 0.273,
  B 0.261 vs 0.275; S@10 A 0.712 vs 0.820, B 0.754 vs 0.849. SigLIP 2 better on 18 of 22 activities in both
  protocols (largest: talks on phone, enters vehicle).
- **Result:** SigLIP 2 is the better zero-shot starting point, consistent with D-011. Recorded as addenda to
  Results 01 and 02; raw `docs/results/raw/01b_clip_val.json`, `02b_clip_eval.json`, `02b_clip_eval_queries.csv.gz`.
  `report/numbers.json` is extended when the 6-slide deck is built (after the spec approval).

## D-068 — 2026-11-18 — Embedder checked on the owner's own test videos (no code change)

- **Basis:** owner: "test them whether our embedder is working on them or not" — 5 videos from
  `C:/wajahat/nexa/nexa_fight/cctv_videos` (CCTV People Demo 1–3, the iProx retail-store clip, james-street 4K).
- **How:** the dashboard's own ingest (`python -m sfs.alerts.ingest --store runs/test_videos/alerts`, a separate store;
  `data/alerts` untouched), then `runs/test_videos/check.py` through the real `Searcher` (Fast, Standard; MEVA bundle as
  the score reference). Results `runs/test_videos/check.json`, top-5 contact sheets `runs/test_videos/sheets/`.
- **Ingest:** 686 alerts (9 / 92 / 28 / 54 / 503); 106–318 s per 720p clip; **james-street (3840×2160, 600 s) took
  3,017 s** — 4K decoding on the CPU is the bottleneck (GPU ~5 % busy).
- **Integrity:** every alert 11 frames, LoRA and frozen vectors unit length (±2.4e-4), all finite, mask = frames,
  93 % of frames with a box.
- **Meaning (YOLO class as weak ground truth):** vehicle queries rank vehicle alerts above person alerts with AUC 0.96
  (Fast and Standard); person queries 0.74–0.90. The single bicycle alert (a delivery rider) is **rank 1 of 686 at
  Fast** for "a person riding a bicycle" and "a bicycle", rank 8 / 11 at Standard.
- **Visual check of top 5:** cars, people walking through a shop door, pairs of people walking together: right.
  Phone: 2 of 5 plausible (hand at the ear). Bag: 1–2 of 5. Shop counter: not convincing.
- **Weak spots found:** (1) a "hub" — a person cut off at the frame edge gives a tight crop of background clutter
  (retail store, 37.7 s) that ranks near the top for almost every query; (2) umbrella-only crops; (3) the student
  (Standard), trained on MEVA's 22 activities, demotes objects outside them (bicycle 1 → 8); (4) near-camera people
  seen from above at the bottom edge of the 4K street camera, unlike MEVA's views. No fix made; each would need a spec.

## D-069 — 2026-11-20 — Submission spec approved with the recommended answers

- **Basis:** after the list of remaining delivery work, which ended "a 'go with the recommendations' covers all
  four", the owner: "okay good to go".
- **Decided:** D1 the MEVA folder = search index + crop store + 121 example clips (raw videos stay on Drive);
  D2 the whole `sfs` package goes in; D3 directory junctions at the old paths; D4 the without-MEVA mode is built now.
  D5 (CLIP download) was already done at the office (D-067).
- **Order:** the 6-slide deck first (slide 5 uses the CLIP numbers of D-067), then the submission folder; the owner
  records the demo video; report changes once the owner has read it or the template arrives; upload from the office.
- **Not covered by this go:** deleting anything on the office server (needs its own explicit confirmation) and any
  commit or push.

## D-070 — 2026-11-20 — 6-slide deck built; 8B time per call corrected in the numbers file

- **Basis:** spec-submission §3, approved in D-069.
- **Deck:** `report/Ahmed_Wajahat_DLPresentation_6.pptx`, built by `scripts/deck6_build.js` to the owner's outline:
  (1) introduction, (2) project description + MEVA + the two query types, (3) pipeline and the training pipeline,
  (4) architecture and why (SigLIP 2 + LoRA, temporal student, VLM on the top k; alternatives rejected on evidence),
  (5) results: zero-shot CLIP → frozen SigLIP 2 → LoRA → + student, Q1 mAP and Q2 Success@1, both protocols,
  (6) the VLM gates and the cost curve, takeaway, thanks and questions. Speaker notes timed for about 7 minutes
  (assumed; the talk length was not given). Validator pass; every slide exported through PowerPoint and inspected.
- **Shared builder code:** theme, layouts, helpers and number accessors moved from `scripts/deck_build.js` to
  `scripts/deck_common.js`. Check: the 12-slide deck rebuilt with the refactored builder equals the committed file in
  every part except random section ids and timestamps (112 parts compared); the committed file was put back.
- **Numbers file:** `scripts/report_numbers.py` gains a `clip` block (D-067; validation AUC / P@10 per encoder,
  eval Q1 / Q2 of CLIP and frozen SigLIP 2). Every other block unchanged.
- **Issue found and fixed:** the 8B block of `report/numbers.json` carried the 2B's time per call (0.168 s instead of
  0.132 s) because `report_numbers.py` read `05_8b_cascade.json`, the file written before the D-048 timing fix, not
  the corrected `05_8b_cascade_g0fix.json`. Found when the first build of slide 6 said "10×" for the 32B against the
  8B instead of 13×. The report was not affected (it reads the 8B time from the G0 file); nothing else reads the
  changed fields (8B time per call, the 8B run's student wall-clock). Fix: read the corrected file; a test pins the
  8B time to its G0 measurement.
- **Tests:** `tests/test_report_numbers.py` 4 pass (CLIP block, 8B time).

## D-071 — 2026-11-21 — 6-slide deck: architecture slide redrawn as a flow diagram with a worked example

- **Basis:** owner: the architecture slide had too much text; wanted a flow diagram (SigLIP 2, LoRA with its layers,
  the 2-layer 2.2 M transformer, inputs and outputs) and below it example frames passed through each block.
- **Done:** slide 4 of `report/Ahmed_Wajahat_DLPresentation_6.pptx` rebuilt in `scripts/deck6_build.js`. LoRA is drawn
  inside the SigLIP 2 block, not after it (it adds trained matrices to the encoder's attention layers; the owner's
  outline put it as a separate step, corrected in the answer).
- **Example:** `scripts/deck_arch_example.py` → `report/figures/arch_example.json` + frame PNGs. Final rule: query
  "a person picking up an object" (first frozen phrasing of a motion-defined activity), the top-ranked clip at
  Standard with that label and an actor box in every frame. Clip 9850 (2 frames): rank 286 (frozen) → 126 (LoRA) →
  1 (+ student). Two earlier rules were dropped before the slide was built: "opening a car door" gave crops of cars
  (the box covers the vehicle), and a ≥ 4-frame minimum pushed the choice to a crowded clip at rank 283 (pick-up clips
  are mostly 2–3 frames). An illustration, not a measurement; the slide says "one example".

## D-072 — 2026-11-21 — 6-slide deck: slide 3 replaced by a three-stage methodology flow

- **Basis:** owner: the pipeline figure was too complex to explain; wanted a simple flow: baseline (CLIP, SigLIP,
  SigLIP 2 → chosen model), LoRA fine-tuning with validation → results, the student on both protocols → results.
- **Done:** slide 3 = three stage boxes (baseline / LoRA / temporal student), each with an output box below it; numbers
  from `report/numbers.json`. CLIP is marked "reference, measured afterwards" (it was measured on 2026-11-17, after
  SigLIP 2 was chosen on 2026-10-23), so the slide does not claim it took part in the choice.
- **Builder:** `DECK_OUT` writes the deck to another path (Windows locks a deck open in PowerPoint).
- **Redrawn the same day** on the owner's feedback ("detailed, but in a simple way"): one row per stage, five short steps each (input → steps → output), every box a bold name + one line.

## D-073 — 2026-11-21 — 6-slide deck: less text, protocols on the methodology slide

- **Basis:** owner: the protocols were missing from the methodology slide; too much text across the deck; make it attractive.
- **Done:** slide 2 = three numbered points + the MEVA result strip + three big numbers + Q1/Q2 cards; slide 3 gains a
  Protocol A / B strip; slide 4 keeps the diagram with shorter labels and the example as big ranks (286 / 126 / 1);
  slide 5 replaces the paragraph with three callouts (0.179 → 0.476; ~80 % of LoRA's gain on unseen cameras;
  SigLIP 2 > CLIP on every test metric); slide 6 note cut to one line. Validator pass; slides 2–6 inspected in PowerPoint.

## D-074 — 2026-11-21 — 6-slide deck: visual redesign (slides show, the speaker tells)

- **Basis:** owner: make it more visual and professional for a master's presentation; not everything written out.
- **Done:** cover = title left, the dashboard screenshot right (`report/figures/cover_dash.png`, a crop of
  `fig4a_search.png`); slide 2 = "you type → you get" picture + three number tiles (≈ 10 min / 0.2 s / dataset);
  slide 4 = inputs and outputs written on the arrows, larger example frames and ranks; slide 6 = the 7-row gate
  table replaced by a bar chart of what each VLM adds (best of A / B, as the gate rule reads; only the 32B passes
  +0.02). Slides 3 and 5 unchanged. Detail moved to the speaker notes. Validator pass; all slides inspected.

## D-075 — 2026-11-21 — Deck: owner's hand edits kept; thanks moved to a 7th slide

- **Basis:** owner edited the deck in PowerPoint (cover: pill removed, author lines shortened; slide 2: third tile
  removed, two tiles centred) and asked for "thank you / any questions" on a 7th slide.
- **Done:** edited the owner's file directly with python-pptx (a rebuild would discard the hand edits): the thanks
  line removed from slide 6, takeaway re-centred, new slide 7 "Thank you / Any questions?" on the TITLE layout, added
  to the last section. Validator pass; opened and inspected in PowerPoint. Backup of the owner's 6-slide file in
  `report/backup/`.
- **Consequence:** `report/Ahmed_Wajahat_DLPresentation_6_new.pptx` is now hand-edited; `scripts/deck6_build.js`
  no longer reproduces it exactly (the cover screenshot swap of the same day is in the builder; the owner's edits
  and slide 7 are not). Port them into the builder before any rebuild.

## D-076 — 2026-11-21 — Deck slide 5: borders around the bars

- **Basis:** owner asked for borders between the columns of the two slide-5 charts, and nothing else changed.
- **Done:** a 1.5 pt outline in the slide background colour on both series of `chart1.xml` and `chart2.xml`, written
  straight into the hand-edited deck (no rebuild). Validator pass; slide exported and inspected. Backup:
  `report/backup/DLPresentation_6_new_before_bar_borders.pptx`. Not in `scripts/deck6_build.js` (see D-075).

## D-077 — 2026-11-23 — Submission folder built (spec-submission §1, approved D-069)

- **Basis:** owner: "where is the submission folder?" — approved 2026-11-20, not yet started.
- **Code (§1.3, §1.5):** `sfs.web.server` gains `--vlm` and `--previews` (defaults unchanged) and a without-MEVA
  mode: when the previews or the crop store are missing the MEVA collection is hidden (status lists `alerts`
  only, MEVA searches → 409, clip/frame/map requests → 404). Deviation from §1.5: no `meva_stats.json`; the
  0.26 GB MEVA index stays in `models/index` (it holds the temporal transformer anyway), and the 'without MEVA'
  part is the 15 GB of media. Front end offers only the collections the server lists. `.gitignore`
  `/submission/`, `.dockerignore` `submission`. Tests: 3 new web tests (scores bit-identical with and
  without MEVA), `tests/test_submission.py` (6).
- **Builder:** `scripts/build_submission.py assemble | move | check`, templates in `packaging/submission/`
  (guide `README.md`, `run.bat`, `run.sh`, `requirements.txt`, hand-in `Dockerfile` + `compose.yaml`);
  `requirements-cpu.txt` derived from the tested pins (torch `+cpu`). Layout differs from §1.2 in one point:
  the MEVA index is `models/index` and its previews `meva/previews`.
- **Moved, not copied (D3):** 7 folders renamed into `submission/` with junctions left at the old paths, the
  4 index files moved with hard links left; file counts identical before and after (e.g. crops 253, previews
  3,726). Size: models 5.71 GB, meva 16.02 GB (crops 13.96, previews 1.19, example videos 0.87), total 21.74 GB.
- **Checks:** laptop server through the junctions vs `runs/docker/laptop.json`: 137 queries, max score diff
  0.0, 0 swaps, the 5 Verified sets identical (`runs/docker/laptop_after_move.json`). Hand-in `run.bat`:
  starts, MEVA search + previews + front end; garage clip (2.5 min, 480p) uploaded → 22 alerts in 623 s,
  the laptop's count; without MEVA: same 22 alerts, Fast and Standard scores and bands identical on 4
  queries. CPU only (GPU hidden): start ≈ 1.5 min, Fast 0.3 s, Standard 2.4 s per search, Verified hidden; a 15 s clip took 495 s to
  ingest (2 alerts), an alert search 6.0 s. Full suite: 227 passed, 4 deselected.
- **Also in the folder:** report PDF + Word (7 Nov), the owner's current deck (`report/DLPresentation.pptx`,
  8 slides), the test clip; `demo_video/` empty until the owner records it. Upload to Drive = owner, at the office.

## D-078 — 2026-11-24 — MEVA data not handed in; local MEVA media deleted (owner)

- **Basis:** the professor asked for the dataset link, not the data ("don't share the data, share the link, I'll
  download it"); the owner: "delete the MEVA dataset as presentation was also done". The professor's hand-in list:
  report in the provided template, presentation (PDF), short intro video (MP4 or link), code with the DL model
  (link if large).
- **Done:** the three junctions `data/crops_v2`, `data/meva/examples/videos`, `data/demo/A/previews` removed (link
  only), then `submission/meva` deleted (crops 13.96 GB / 253 files, previews 1.19 GB / 3,726, example videos
  0.87 GB / 121); 16 GB freed. Copies remain on Drive (crop store) and on the office server (`~/sfs/project/data`).
  Kept: the MEVA index in `models/index` (embeddings + the temporal transformer), the manifest and annotations.
- **Code:** `build_submission.py move` no longer moves the MEVA media unless `--with-meva`; `check` no longer
  expects them; guide rewritten for a hand-in without MEVA (dataset link https://mevadata.org, how to rebuild the
  media). The laptop dashboard now starts in the without-MEVA mode (D-077). Full suite after the deletion: 227
  passed, 4 deselected.

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
| D-049 | Approve `spec-lora-image-text.md`; retention check R1 (nexa_fight labelled clips, if they exist) or R2; adoption bar | **Decided (D-050):** approved; no retention check; bar as written |
| D-051 | E8 fails the rule (A −0.028, B +0.051): accept, or run a second seed of all four runs? | **Decided (D-052):** second seed; reading fixed before the results |
| D-053 | Image + text not adopted (A mean −0.019); its B gain repeats (+0.047). Keep image-only everywhere, or use image + text for the nexa_fight demo as a post-hoc choice (~0.5 day + 1.5 h)? | **Decided (D-054, D-055):** nexa_fight demo dropped; drift check failed; image-only everywhere |
| D-058 | Verified level in the recorded demo: local 2B (default) or the 8B on the server (holder restart needed)? | Open — owner (recommended: 2B; the 8B adds +0.058 S@1 on A but means restarting the server job) |
| D-059 | Approve `spec-video-alerts-dashboard.md`: alert rule (D1), 2 fps clip (D2), alert classes (D3), design (D4), 3D clip map (D5), installs (D6) | **Decided (D-060):** all answered; built (D-062) |
| D-062 | Keep or remove the Gradio page (`sfs.demo.app`) now that the dashboard exists; backpack as the common object (or handbag / suitcase) | Open — owner |
| D-063 | `spec-docker.md`: confirm the defaults taken (D2 volume, D3 no CPU variant, D5 laptop demo), name the test video (D4); build at the office (D1) | **Built 2026-11-08 (D-065):** D1 at the office; D2/D3/D5 defaults kept; D4 = the owner's garage clip already in the store |
| D-064 | Report written (owner's instruction); owner to read `report/Ahmed_Wajahat_DLReport.pdf` and send changes; submission date for the cover; deck next | Open — owner |
| D-065 | Docker §5.2 / §5.3 fail their pre-fixed rules for measured platform reasons (float16 text tower on the Linux CUDA libraries; OpenCV wheels' FFmpeg builds decode differently): accept and document (recommended) or make the platforms bit-identical (~1 day)? | Open — owner |
| D-066 | Approve `spec-submission.md` (submission folder, CLIP baseline, 6-slide deck; D1–D5) | **Decided (D-069):** approved with the recommended answers; build started |

---

## Standing rules

- **No decision is ever taken on eval-split numbers.** Selection uses validation only; eval is reported, never used to choose.
- Every result records the data version (encoder id / pipeline version) it came from.
- Failed and negative results are logged with the same care as positive ones.
