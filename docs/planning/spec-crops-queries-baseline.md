# Spec — Crop Store, Query Set & Frozen Baseline Evaluation

**Date:** 2026-10-23
**Status:** APPROVED 2026-10-23 as written; D-012 = option A
**Depends on:** `spec-offline-embedding.md` (approved), `docs/results/01-zero-shot-validation.md`
**Log:** decision D-013 in `docs/EXPERIMENT_LOG.md`

---

## 1. What this change covers

Three parts, deliberately scoped so their results decide what comes next:

1. **Crop store** — save the sampled crops as images on Drive, once, so later training and the
   VLM teacher never have to read the 70 GB of video again.
2. **Query set** — natural-language queries with ground truth, built from MEVA's human labels.
3. **Frozen baseline evaluation** — retrieval accuracy and cost of SigLIP 2 + tight crops with no
   training, on both evaluation protocols. This is the number everything later must beat.

**Out of scope** (separate specs, decided after these results): LoRA, the VLM teacher, the
temporal student re-ranker, the demo.

---

## 2. State before this change

- Frozen embeddings exist for all 10,678 instances, all 3 views, both encoders (pipeline v2).
- SigLIP 2 + tight selected on validation (D-011).
- **No images are stored.** LoRA and the VLM teacher both need pixels, so without a crop store every
  epoch would have to decode the video again.
- **No query set exists**, so retrieval accuracy (Recall@K, mAP) has never been measured — only
  the per-activity AUC sanity check.

---

## 3. Part 1 — Crop store

### What gets stored
| View | Stored size | Used by |
|---|---|---|
| `tight` | 336×336 JPEG, quality 95 | LoRA training (random crop to 224 = augmentation), re-embedding, teacher |
| `context` | 336×336 JPEG, quality 95 | Teacher (surroundings help it judge); future ablations |

`full` is not stored: it is never trained on, and its embeddings already exist.

**Why 336 rather than 224:** gives room for random-resized-crop augmentation during LoRA, and
gives the VLM teacher more detail. The encoder resizes to 224 anyway.

### Format — one packed file per video, not one file per crop
About 147,000 small files on a Drive mount would be very slow to write and to list. So each video gets one
`crops/{stem}.npz` containing a single `uint8` byte buffer with every JPEG concatenated, plus
`offsets`, `instance_id`, `frame_idx` and `view`. No pickle. A global `crops_index.csv` maps
`(instance_id, frame_idx, view) → (file, row)`.

### Code
- Reuses the **existing tested** `plan()`, `regions_for()` and `iter_frames()`: the crops are
  guaranteed to be exactly the regions that were embedded.
- New `sfs/store/crops.py` (pack/unpack) and CLI `sfs.data.extract_crops`. Resumable per video,
  and it stops on missing boxes exactly like the embedding job.

### Integrity check (must pass before Parts 2–3 rely on it)
Re-embed a random 2,000 stored tight crops with SigLIP 2 and compare them with the existing v2
vectors for the same `(instance, frame)`. **Required: mean cosine ≥ 0.98.** This proves that
storing and reloading (JPEG + resize) does not change what the model sees. If it passes, every
later comparison — baseline vs LoRA — can use the crop store with nothing changed between them.

Estimated size: ~147k crops × ~25 KB ≈ **3.7 GB on Drive**. Runs on Colab (T4), ~1–2 h.

---

## 4. Part 2 — Query set

Two query sets, answering two different questions.

### Q1 — Category queries: "find every instance of this action"
- **3 hand-written phrasings per activity type**, e.g. `person_opens_vehicle_door` →
  "a person opening a car door" · "someone opens the door of a parked vehicle" ·
  "person pulling a vehicle door open".
- Relevant set = every instance of that activity in the split being searched.
- Written by Claude, **reviewed and corrected by the user** — this is the "hand-verified" step
  the proposal promised. The full list is a small file (`queries/phrasings.csv` (moved out of `data/`, which git ignores — reviewed phrasings must be versioned)) that you
  read in a few minutes.
- Phrasings are written **without** looking at any retrieval result, so they cannot be tuned to
  the model.

### Q2 — Forensic queries: "find the event, given where and roughly when"
- This mirrors real use: an investigator knows the camera and the hour, but not the moment.
- One query for each (activity, camera, date-hour) group present in the split. The text is a
  phrasing from Q1, and a **metadata filter** restricts the search to that camera and hour.
- Relevant set = the instances of that activity inside the filtered window (usually 1–5).
- This exercises the system's hybrid search (vector similarity + metadata filter), as in
  `spec-retrieval-system.md`.

### Which activities are queried (D-012 — needs your decision)
- **(A, recommended)** Only activity types with ≥ 20 eval instances (22 types, 98.2% of eval).
  The 10 rare types stay in the searched corpus as distractors, but are never queried.
- **(B)** A targeted download of clips containing the rare types first. Costs about a day, and may
  not help: `person_purchases` has 0 training instances in any annotated clip seen so far.

### Why no machine-generated captions
The course forbids synthetic data as the primary dataset. Ground truth comes **only** from MEVA's
human activity labels; the phrasings are hand-written and user-reviewed query wordings.

---

## 5. Part 3 — Frozen baseline evaluation

### Systems evaluated
| System | Role |
|---|---|
| **SigLIP 2, tight, frozen** | **The baseline** every later method must beat |
| SigLIP 2, full frame, frozen | Ablation: confirms on eval that cropping helps (decided before any eval number was seen) |
| Random ranking | Floor |

### Scoring
Instance score = mean of the top-m frame similarities (m = 3), as specified. Query score =
average over its 3 phrasings.

### Metrics
| Query set | Metrics | Why these |
|---|---|---|
| Q1 category | **mAP**, P@10, R-precision — macro (each activity counts equally) and micro | Each query has hundreds of relevant instances, so classic Recall@10 would be tiny and uninformative |
| Q2 forensic | **Recall@1, @5, @10, MRR** | Few relevant instances per query, so these are the meaningful ones |
| Both | Per-activity breakdown | So that strong vehicle/door results cannot hide weak human-action results |
| Cost | Query latency (ms), model calls per query | The first point on the accuracy-vs-cost curve |

### Splits
Protocol A eval and Protocol B eval, reported side by side. **These are the first eval-split
numbers in the project.** They are reported, never used to choose anything (standing rule in the
log). The same eval queries are reused unchanged for every later system.

### Code
`sfs/eval/metrics.py` (AP, P@K, R@K, MRR), `sfs/eval/queries.py` (build Q1/Q2 from the manifest
and phrasings), `sfs/eval/retrieval.py` (aggregate frame scores to instances, apply filters),
CLI `sfs.eval.baseline`. **Runs locally** on the downloaded embeddings — no Colab needed.

---

## 6. Verification

| Level | Check |
|---|---|
| Unit | Metrics against hand-computed examples (incl. edge cases: no relevant items, ties); crop pack/unpack round trip; query builder (filter windows, relevant sets, rare-type exclusion) |
| Integration | Synthetic end-to-end: extract crops from the synthetic test video and check pixel-exact regions |
| Data | Crop-store re-embedding cosine ≥ 0.98 (Part 1 gate); every query has ≥ 1 relevant item |
| Sanity | Baseline mAP > random on both protocols; per-activity pattern consistent with validation (vehicles high, fine-grained actions low) |

---

## 7. State after this change

- Crop store on Drive (~3.7 GB), integrity check passed.
- `data/queries/`: user-reviewed phrasings, Q1 and Q2 query files with ground truth.
- `docs/results/02-frozen-baseline.md`: baseline numbers for both protocols, per activity, with cost.
- Experiment log updated with D-012, D-013 and the baseline result.
- The decisions for the next spec (LoRA, re-ranker) are taken **from these results**.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| JPEG storage changes the embeddings | The 0.98 cosine gate; if it fails, store PNG or a higher JPEG quality |
| Q2 windows contain many instances of the same activity, making the task trivial or ill-defined | Report the distribution of relevant-set sizes; queries with large relevant sets are reported separately |
| User review of phrasings delays the work | The list is small (22 types × 3); the code is built in parallel and evaluation waits only on the final list |
| Colab session limits | The crop job is resumable per video, like the embedding job |

---

## 9. Decisions requiring approval

1. **Crop store** as specified (tight + context at 336 px, packed per video, 0.98 integrity gate).
2. **Q1 + Q2 query sets**, with phrasings written by Claude and reviewed by you.
3. **D-012: A or B** for the rare activity types.
4. **Baseline evaluation** as specified (systems, metrics, both protocols, run locally).
