# Spec — LoRA Fine-Tuning of the SigLIP 2 Image Encoder

**Date:** 2026-10-28
**Status:** **Approved for build 2026-10-30** (§11)
**Depends on:** `spec-crops-queries-baseline.md` (complete), `docs/results/02-frozen-baseline.md`
**Log:** caption bank D-019; spec approval D-020

---

## 1. What this change covers

Adapt SigLIP 2's image encoder to MEVA surveillance footage with LoRA, re-embed the corpus with the
adapted encoder, and measure the change against the frozen baseline with the same queries.

Concretely:
1. A **training caption bank**, separate from the evaluation phrasings.
2. A **packed, memory-mapped tight-crop file** for fast random access during training.
3. **LoRA training** of the image encoder with a label-aware sigmoid loss.
4. **Two trained models**, one per evaluation protocol (see D8 — this is required, not optional).
5. **Re-embedding** of all tight crops with each adapted encoder.
6. **Evaluation** with the existing baseline code, plus **Success@K**, defined here before any new
   result is seen.

**Out of scope** (later specs): VLM teacher, temporal student re-ranker, demo, text-tower training.

---

## 2. State before this change

- Frozen baseline (D-018), Protocol A: Q1 mAP **0.179**, Q2 R@10 **0.510**. Protocol B: 0.174 / 0.496.
- Activities defined by **motion** (pick up, put down, transfer, stand up, sit down) are at chance
  (lift ≈ 1–2); appearance-defined ones (phone use, doors, vehicle turns) are found well.
- Crop store local and verified (cos 1.000): 73,569 tight crops, native PNG, in 252 per-video shards.
- No training code exists. `peft` is listed in `requirements.txt`.

---

## 3. Decisions

### D1 — Training captions: a separate, disjoint caption bank
Training learns from (image, sentence) pairs. **Using the 66 evaluation phrasings as training text
would teach the model the exact test wording and inflate the result.**

- A new bank, `queries/train_captions.csv`: **6 captions per activity** for every activity with
  training instances (≈ 31 types — the 22 evaluated plus rare ones with train data; rare types add
  negatives and context even though they are never queried).
- Written by Claude, **reviewed by you** (same review page approach as D-017).
- **Enforced disjointness:** a test fails if any training caption equals an evaluation phrasing,
  or is a near-duplicate of one (normalised text similarity above a fixed threshold).
- Ground truth still comes only from MEVA human labels; captions are wording, not data.

### D2 — What is trained: image tower only (LoRA); text tower frozen
| Option | For | Against |
|---|---|---|
| **Image LoRA only (chosen)** | Fits 8 GB comfortably; caption embeddings computed once; the text side keeps its open-vocabulary understanding, which test queries rely on | Less capacity |
| Image + text LoRA | More capacity | Text drift towards the 6 training captions can *hurt* unseen query wordings; more memory |

Text-tower LoRA is noted as a possible later ablation, not part of this spec.

LoRA settings: rank 16, alpha 32, dropout 0.1, applied to the vision tower's attention projections
(q, k, v, out). Everything else frozen. Trainable parameters ≈ 1–2 % of the image tower.

### D3 — Loss: label-aware sigmoid loss against caption prototypes
Plain SigLIP training treats every other image–text pair in a batch as a negative. Here many images
in a batch share an activity, so that would push **correct** pairs apart (false negatives).

Instead, each step:
- samples one caption per activity → C caption embeddings (frozen text tower);
- computes logits = `t · (image_emb · caption_embᵀ) + b` for a batch of B images (B × C);
- target = 1 where the image's activity equals the caption's activity, else 0;
- loss = binary cross-entropy (sigmoid), the same form as SigLIP's own loss. `t` and `b` start from
  the pretrained values.

Resampling the caption each step stops the model from overfitting to one wording.

### D4 — Sampling: by instance, square-root class balancing
- One **training sample = one random sampled frame of one instance**; every epoch visits each
  instance once. Long instances (16 frames) therefore do not outweigh short ones (1 frame).
- Activity frequencies range 1,263 : 4. Instances are drawn with probability ∝ 1/√(class count) —
  a middle ground between natural frequency (common classes dominate) and full balancing (rare
  classes overfit).
- **Leakage guard:** the sampler asserts that it never yields a val or eval instance.

### D5 — Augmentation (and one deliberate exclusion)
- Random resized crop from the native tight crop to 224 px, scale 0.7–1.0, aspect 0.9–1.1.
- Mild colour jitter (brightness/contrast ±0.15) — surveillance lighting varies.
- **No horizontal flip.** Flipping turns `vehicle_turns_left` into `vehicle_turns_right` and would
  corrupt those labels.
- Validation and evaluation use **no augmentation** — exactly the processor's resize, as in the baseline.

### D6 — Training settings
AdamW, lr 1e-4, weight decay 0.01, cosine schedule with 5 % warm-up, batch 64 with fp16 autocast
(gradient checkpointing if memory requires), up to 15 epochs, **early stopping on validation**
(patience 3). Seed fixed; one run per protocol.

### D7 — Model selection: validation only
Checkpoint chosen by **validation Q1 macro mAP**, using the frozen evaluation phrasings as queries
over the **validation** split (activities with ≥ 5 val instances). The eval split is touched once,
after training, to report. Standing rule: no decision on eval numbers.

Validation is small (739 instances in Protocol A); selection will be noisy. Validation loss is
logged as a second signal.

### D8 — Two models, one per protocol (required)
Protocol B's eval instances (8 held-out cameras) include clips that are in Protocol A's *training*
split. A model trained on Protocol A therefore **must not** be evaluated on Protocol B — it would
have trained on part of B's test data.

- **Model A:** trained on `split == train`, selected on `split == val`, reported on Protocol A eval.
- **Model B:** trained on `split_b == train`, selected on `split_b == val`, reported on Protocol B eval.

The A–B comparison after training then measures how much the adapted model relies on having seen
the cameras (D-018 finding 3).

### D9 — Data access: a packed, memory-mapped tight-crop file
Random access across 252 shards (14 GB with context crops) is slow and memory-hungry. A one-time
step writes **tight crops only** into `data/train_tight.bin` (raw PNG bytes) + an offsets index,
opened with `numpy.memmap` — random access with no RAM pressure. Built from the verified crop store,
so it inherits the cos-1.000 guarantee; a test checks bytes are identical to the shard bytes.

### D10 — Re-embedding and identity
After training, every tight crop is re-embedded with the adapted encoder into new shards. The
encoder id now includes the **adapter weight hash and training config**, so adapted vectors can
never be mixed with baseline vectors (existing consolidation check enforces this).
Sanity check: with the adapter **disabled**, re-embedding reproduces the baseline vectors (cos ≈ 1.000).

### D11 — New metric, defined now: Success@K
**Success@K = 1 if at least one relevant instance is in the top K, else 0**, averaged over queries.
It is the natural measure for forensic queries ("did the investigator find the event?") and does
not suffer the cap that Recall@K has when a window holds more than K relevant instances (D-018
caveat). Added for K = 1, 5, 10 and reported for **both** the baseline and the adapted models.
Defined here, before any adapted result exists.

### D12 — Hypotheses (falsifiable)
- **H1:** Model A improves Protocol A Q1 macro mAP over the frozen baseline (0.179).
- **H2:** Gains concentrate on **appearance-defined** activities; **motion-defined** ones stay near
  chance, because training is still per-frame. (If H2 holds, it motivates the temporal re-ranker.)
- **H3:** The adapted models show an A–B gap; its size measures camera familiarity.

Each is reported whether confirmed or rejected.

---

## 4. Code

| File | Purpose |
|---|---|
| `src/sfs/train/data.py` | Packed tight-crop file (build + memmap reader), instance sampler with leakage guard, augmentation |
| `src/sfs/train/loss.py` | Label-aware sigmoid loss |
| `src/sfs/train/lora.py` | Wrap the vision tower with LoRA; trainable-parameter report; adapter save/load + hash |
| `src/sfs/train/run.py` | CLI: train one protocol, early stopping, checkpoints, training log |
| `src/sfs/embed/reembed.py` | Re-embed tight crops from the crop store with a given (adapted) encoder |
| `src/sfs/eval/metrics.py` | + `success_at_k` |
| `queries/train_captions.csv` | Training caption bank (owner-reviewed) |

Runs **locally** on the RTX 4060. Estimated: building the packed file ~10 min; training ~1–3 min
per epoch (≈ 6k instances per epoch); re-embedding 73,569 crops ~10 min per model.

---

## 5. Verification

| Level | Check |
|---|---|
| Unit | Loss against a hand-computed example; sampler never yields val/eval instances and follows √-balancing; augmentation never flips; packed-file bytes identical to shard bytes; LoRA wrap leaves base weights frozen and unchanged; caption bank disjoint from eval phrasings; Success@K hand-computed |
| Integration | Adapter disabled → re-embedding equals baseline vectors (cos ≈ 1.000); 50-step smoke run on a small subset: loss decreases, nothing NaN |
| Regression | Full existing suite still passes (baseline metrics unchanged by the metrics.py addition) |
| Result | Same eval queries, same code path as the baseline; only the embeddings differ |

---

## 6. State after this change

- `queries/train_captions.csv` reviewed; disjointness enforced by tests.
- Two LoRA adapters (A, B) with configs, training curves and hashes (weights git-ignored).
- Re-embedded tight-crop shards for each model.
- `docs/results/03-lora.md`: baseline vs adapted, both protocols, per activity, Success@K, the A–B
  gap, and H1–H3 confirmed or rejected.
- Experiment log and `CLAUDE.md` updated.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Overfitting (6k training instances, ~31 classes) | LoRA (few parameters), augmentation, caption resampling, early stopping on validation |
| Validation too small for reliable selection | Log val loss as well; report the selected epoch and the curve, so a noisy pick is visible |
| Catastrophic forgetting of open-vocabulary ability | Text tower frozen; small LoRA rank; evaluation uses unseen query phrasings, which would expose it |
| Label noise (other activities visible in a crop) | Accepted and stated; per-instance labels are what MEVA provides |
| 8 GB memory | Image-only LoRA, fp16, gradient checkpointing fallback, batch size reduced if needed |
| Motion-defined activities do not improve | Expected (H2) — that is the temporal re-ranker's job; reported, not hidden |

---

## 8. Decisions requiring your approval

1. **D1** separate caption bank (6 per activity, reviewed by you, disjoint from eval phrasings).
2. **D2** image-tower LoRA only, text tower frozen.
3. **D3** label-aware sigmoid loss with caption prototypes.
4. **D4–D6** sampling, augmentation (no flips), training settings.
5. **D8** two models, one per protocol.
6. **D11** Success@K as an additional metric, reported for baseline and adapted models.

---

## 9. Review round 1 (2026-10-29)

| Decision | Outcome |
|---|---|
| **D1** caption bank | **Approved in principle.** Draft written: `queries/train_captions.csv`, 192 captions (32 activities × 6, incl. 10 rare types that are trained on but never queried). Awaiting owner review via `data/review/caption_review.html`. Disjointness rule refined below |
| **D2** image-tower LoRA only | **Approved.** Owner wants **image + text LoRA as a later experiment** (separate spec, compared against this one) |
| **D3** label-aware sigmoid loss | **Approved** |
| **D5** augmentation | **Revised at owner's request** — no flipping; zoom in, zoom out, stretch and other suitable transforms (below) |
| **D8** two models, one per protocol | **Approved** |
| D4, D6, D7, D11 | **Not yet answered** |

### D1 (refined) — what "disjoint" means
The first check flagged 5 pairs and showed two different cases:
- **Different activity, similar wording** — e.g. training "a person *shutting* the door of a building"
  (`person_closes_facility_door`) vs test "a person *opening* the door of a building"
  (`person_opens_facility_door`). **Allowed and useful**: it teaches the model that opening and
  closing differ (a hard negative). Not leakage.
- **Same activity, near-identical wording** — e.g. "a person *coming* out of a car" vs test "a person
  *getting* out of a car". **Forbidden**: the model would practise the test sentence.

Rule enforced by test: **no exact match with any test phrasing; within the same activity,
similarity < 0.80** (max of character-sequence ratio and word-set Jaccard on normalised text).
Eight captions were rewritten to satisfy it; current maximum is 0.79.

### D5 (revised) — augmentation (training only; validation/eval untouched)
| Transform | Setting | Why it fits surveillance |
|---|---|---|
| **Zoom in** | random crop covering 60–100 % of the tight crop, resized to 224 | Actors appear at many distances; also teaches partial views |
| **Zoom out** | p = 0.25: shrink to 75–100 % and pad the border by reflection | Subjects are often smaller than in their crop. Reflection padding invents some border content — kept mild and infrequent for that reason |
| **Stretch** | aspect ratio 0.8–1.25 | Box shapes vary with pose and camera angle |
| **Resolution degradation** | p = 0.3: downscale to 50–100 %, then back up | Distant subjects and compressed CCTV are low resolution — the main real-world variation here |
| **Colour jitter** | brightness / contrast ±0.15 | Indoor/outdoor lighting changes |
| **Never** | horizontal flip | Swaps `vehicle_turns_left` ↔ `vehicle_turns_right` |

Not used: rotation (cameras are fixed and level), cutout/erasing (could erase the very hand or
object that defines the action).

*Alternative considered for zoom out:* take real surroundings from the stored context crop instead
of reflection padding. It needs each crop's exact frame coordinates, which requires the box
annotations locally — available since 2026-10-29 (`data/meva/annotations_local`); possible later if reflection padding proves harmful.

---

## 10. Review round 2 (2026-10-30)

| Item | Outcome |
|---|---|
| **D1** caption bank | **Approved** as written (owner reviewed `data/review/caption_review_boxes.html`: actor/object boxes from the geom annotations + official MEVA definitions). `queries/train_captions.csv` frozen, sha256 prefix `139d6fc19a02a9ad`. Open note: in sampled examples the "heavy object" is mostly folding chairs, which no caption names; owner kept the general wording |
| **D4** sampling | **Approved, with both edge-frame variants trained and compared** (below) |
| **D5** revised augmentation (§9) | **Approved** |
| **D6** training settings | **Changed by owner:** up to **100 epochs**, early-stopping **patience 5** (was 15 / 3). LR-schedule consequence below — awaiting owner choice |
| **D7** validation-only selection | **Approved** — also used to choose between the D4 variants |
| **D11** Success@K | **Approved** |

### D4 (final) — two sampling variants
- **(a) uniform:** the frame of an instance is drawn uniformly (original D4).
- **(b) middle-weighted:** frame drawn with a triangular weight over the instance's sampled frames
  (middle frame weight 1, first/last frame weight 1/3 of it; single-frame instances unaffected).

Both variants trained for both protocols → **4 runs** (A-a, A-b, B-a, B-b). Per protocol, the
variant with the higher **validation** Q1 macro mAP is the protocol's model; the choice is recorded
before eval is touched. Eval is then reported for all four runs, labelled "selected" / "ablation",
so the comparison is visible but cannot influence the choice.
- **H4 (added):** (b) beats (a) on validation, with the gain concentrated on activities annotated
  with lead-in / lead-out (e.g. `hand_interacts_with_person`).

### D6 (revised) — what 100 epochs means here
One epoch = one frame per training instance (~6k samples, ~94 steps at batch 64, ~1 min), so 15
epochs was only ~1,400 steps; 100 is a reasonable cap. Early stopping (patience 5 on validation Q1
mAP) ends most runs well before 100. Worst case ~1.7 h per run, ~7 h for all four.

**Consequence for the learning-rate schedule (needs a choice).** The original cosine schedule is
laid out over the maximum epochs. Over 100 epochs, a run stopped at epoch ~20 would still be at
~90 % of peak LR — the cosine decay would never happen. Options:
1. **Recommended:** warm-up 2 epochs, then constant LR, **halved when validation mAP has not
   improved for 2 epochs** (reduce-on-plateau); stop after 5 epochs without improvement; restore
   the best checkpoint.
2. Cosine over the full 100 epochs (as written; decay mostly unused if stopping is early).
3. Cosine over a shorter fixed horizon (e.g. 30 epochs) with 100 as a hard cap.

**Owner choice (2026-10-30): option 1** — warm-up 2 epochs, constant LR, halved after 2 epochs
without validation improvement, stop after 5, restore the best checkpoint.

### D13 (added by owner, 2026-10-30) — crash-safe checkpoints and resume
A power cut, shutdown or crash must never lose more than the epoch in progress (~1 min).
- After **every epoch**, `last.pt` is written **atomically** (temporary file, then rename — a crash
  mid-write leaves the previous checkpoint intact; the one before is kept as `prev.pt`). It holds
  everything needed to continue exactly: LoRA weights, optimizer, LR-scheduler (plateau) state,
  fp16 grad-scaler, epoch, step, best validation mAP, epochs since improvement, all RNG states
  (Python, NumPy, torch CPU + CUDA, sampler), and a hash of the training config.
- Whenever validation improves, the adapter is also saved to `best/` (this is the model used).
- The per-epoch training log (`train_log.csv`) is appended and flushed every epoch.
- Re-running the same command **resumes automatically** from `last.pt`; it refuses to resume if the
  config hash differs (no silent mixing of settings). A run that already finished is skipped, so
  the 4-run driver can simply be restarted after any interruption.
- **Test:** training 3 epochs straight and training 2, "crashing", then resuming 1 give identical
  weights (tiny model, CPU).
- Re-embedding (D10) is resumable per shard, as the embedding job already is.

## 11. Approval
**Approved for build by the owner on 2026-10-30** (all decisions §3, §9, §10 incl. D13). Recorded as D-020.

### D4 — proposed addition (awaiting decision): edge frames
Found during the caption review. Several MEVA activities are annotated with a lead-in and lead-out —
e.g. `hand_interacts_with_person` **starts 1 s before contact and ends 1 s after**. The first and
last sampled frames of such an instance may show the approach, not the action: per-frame label
noise. Options: (a) ignore (current D4: uniform frame choice); (b) sample middle frames more often
(e.g. a triangular weight over the instance's frames). (b) is cheap; whether it helps is decided on
validation only.

### Idea for later (owner, 2026-10-30): general "phone use" captions
Proposal: describe `person_talks_on_phone` and `person_texts_on_phone` with general captions
("a person using a phone") instead of separate call / message captions. Not adopted for this run,
because the evaluation queries (frozen, D-017) ask for each activity separately, and the loss treats
the other class's prototype as a negative — shared captions would pull the two prototypes together
and blur exactly the distinction being scored. Compatible variant to consider later: keep separate
captions and add a **hierarchy** — the two phone classes are not used as negatives for each other,
and a "phone use" query (both classes relevant) is added as an extra evaluation, defined before
results are seen.

## 12. Implementation notes (2026-10-30)

Choices made while building, within the approved design — listed so they can be reviewed:

| Point | What was done | Why |
|---|---|---|
| D4 epoch definition | One epoch = as many draws as training instances (6,213 A / 6,154 B), drawn **with replacement**, instance weight ∝ 1/√(class size), then one frame by the frame mode | The draft said both "each instance once" and "√-balancing"; they conflict. The intent of "once" was one sample per instance, not per frame — kept; balancing kept |
| D3 logit scale / bias | Start from the pretrained values (scale = log 112.6, bias −16.8) and are **trained** (no weight decay) | Pretrained bias is tuned for 1 positive in a batch of thousands; here 1 of 32 captions is positive. Monotone — cannot change any ranking at evaluation |
| D7 epoch 0 | The untrained model is scored on validation first and is itself a candidate for "best" | If training never beats the frozen model, the selected adapter is the identity — selection stays honest |
| D7 validation queries | The 22 queried activities with ≥ 5 validation instances (Protocol A: 739 val instances, 4,722 crops); val loss uses each class's mean caption embedding | Deterministic validation signal |
| Preprocessing | Resize = torchvision bilinear + antialias on uint8, exactly as the SigLIP processor in transformers 5 (a PIL resize differed by one level on ~0.3 % of pixels). Test: pixel-identical to the processor | Training/validation/re-embedding see the same pixels as the baseline |
| LoRA placement | 48 projections (q, k, v, out × 12 layers); 1,179,648 trainable of 94.1 M vision parameters (1.25 %) | D2 |
| Packed file | `data/packed_tight/` — 73,569 tight crops, 8.5 GB, copied byte-for-byte from `crops_v2` (store `84b0a9ac41c5`) | D9 |

**Verification done before training:**
- Unit/integration tests: 87 pass (15 new in `tests/test_train.py`), incl. a simulated power cut
  after epoch 2 → resumed run gives **bit-identical** weights to an uninterrupted run (CPU).
- Adapter off vs baseline vectors, 2,000 random crops: **cos min 0.99996, mean 1.00000**.
- Smoke run (real model, 2 × 30 steps): loss 6.82 → 4.69, no NaN, **peak GPU 5.0 GB** at batch 64
  (no gradient checkpointing needed), ~1 min per full epoch incl. 16 s validation.
- Regression: baseline re-evaluated with Success@K added — all previously reported per-query metrics
  identical (3,876 rows).

**Baseline Success@K (D11), recorded before any adapted eval result** (Q2, macro, tight):
Protocol A S@1 0.273 · S@5 0.655 · S@10 0.820; Protocol B 0.275 · 0.703 · 0.849
(random A: 0.116 · 0.385 · 0.567).

