# Result 05 — VLM follow-ups: Q1 VLM-on-all, merged "vehicle turns", G1 depth, 16 frames (draft)

**Date:** 2026-11-03 · **Spec:** `docs/planning/spec-vlm-followups.md` (E1, E2, E3, E5) · **Log:** D-025..D-028
**Status:** draft. E1–E7 complete. E4: 8B narrowly fails G1; **32B passes G1 at top 50 and on the whole
validation set** (D-043); distillation from the 32B does not help (D-047); **32B cascade rows (D-048):
significant gain over the 0-call system on both protocols, but not significantly better than the 8B**.
**Inputs:** as Result 04 (crop store `data/crops_v2`, `manifest_v2.csv`, frozen phrasings
`queries/phrasings.csv` sha256 `bfd6456a…`, LoRA runs A-middle / B-uniform, students A-GT / B-GT,
Qwen3-VL-2B-Instruct and Qwen3-VL-Reranker-2B; weights verified against Hugging Face, D-025). E1 adds
`queries/phrasings_merged_turns.csv` (sha256 `eca12ae7…`).
**Raw output:** `docs/results/raw/05_*` — E5 report (`05_q1all*`), E1 report (`05_turns*`), E2 gate
(`05_g1_wholeval.json`), E3 gates and diagnostic (`05_g0_*`, `05_g1_16f*.json`), E4/E4b gates
(`05_g0_8b.json`, `05_g1_8b*.json`, `05_g0_32b.json`, `05_g1_32b.json`, `05_g1_wholeval_32b.json`);
32B students (`05_student32b_*`, `05_teacher32b_pool_quality.json`); 32B cascade (`05_32b_cascade.json`,
`05_32b_cascade_queries.csv.gz`, `05_32b_bootstrap.json`).
**Run log:** `runs/cascade/followups.log` (every step exit 0, 2026-11-03 03:07–08:39).

Nothing was re-selected. E5 and E1 reuse Result 04's fusion weights (`--fixed-choices`); E2 and E3
are validation-only gate runs with decision rules fixed before they ran (spec §3, D-027).

---

## Summary

| Exp. | Question | Answer |
|---|---|---|
| E5 | Q1: does the 2B VLM on **every** candidate beat the cheap stages? | **No.** Instruct alone 0.207 / 0.212 mAP (A / B): above frozen SigLIP 2, far below LoRA (0.313 / 0.280) and LoRA + student (0.476 / 0.442), at ~3,700–3,900 calls (~10 min) per query. Fused with LoRA it equals LoRA. |
| E1 | Does one "vehicle turns" class work where left / right failed? | **Yes for retrieval.** Turn AP 0.31–0.32 → 0.57–0.62 for LoRA (chance 0.02–0.04). Turning is found; the direction was the problem. Macro mAP +0.01. The student, trained on left / right and not retrained, is worse than LoRA on the merged class. |
| E2 | Does G1 on the whole validation set agree with the top-50 G1? | **Same verdict, different gains.** All set-ups fail in both; the whole-validation gains are 0.012–0.019 lower. By the fixed rule this is **"not the same"** → owner decision for E4 (below). |
| E3 | Do 16 frames rescue the 2B VLM? | **No.** G1 still fails (+0.007 for both models, vs +0.005 / +0.007 at 8 frames). On the clips that really got more frames, the separation does not improve. E3 stops (spec rule). |

---

## E5 — Q1 "Instruct on every candidate" (eval)

213,090 missing pairs scored (`data/teacher/eval/instruct-tight-q1all`, 0.63 s per clip cached);
the 244,620 Result 04 pairs keep their values. Every Result 04 number is unchanged in this report
(0 differing entries). Fusion weight for "+ LoRA" = Result 04's γ (2.0).

| Prot. | System | VLM calls / Q1 query | Q1 mAP | Q1 P@10 | Q1 S@1 |
|---|---|---|---|---|---|
| A | frozen SigLIP 2 | 0 | 0.179 | 0.259 | 0.333 |
| A | LoRA | 0 | 0.313 | 0.408 | 0.485 |
| A | **LoRA + student** | **0** | **0.476** | **0.527** | **0.561** |
| A | LoRA → Instruct @50 | 50 | 0.315 | 0.408 | 0.485 |
| A | Instruct on all | 3,726 | 0.207 | 0.312 | 0.394 |
| A | Instruct on all + LoRA | 3,726 | 0.306 | 0.389 | 0.515 |
| B | frozen SigLIP 2 | 0 | 0.174 | 0.274 | 0.288 |
| B | LoRA | 0 | 0.280 | 0.362 | 0.439 |
| B | **LoRA + student** | **0** | **0.442** | **0.521** | **0.606** |
| B | LoRA → Instruct @50 | 50 | 0.284 | 0.367 | 0.439 |
| B | Instruct on all | 3,938 | 0.212 | 0.312 | 0.424 |
| B | Instruct on all + LoRA | 3,938 | 0.287 | 0.386 | 0.500 |

Latency of "Instruct on all": 3,726 × 0.17 s ≈ 10.4 min (A), ≈ 11 min (B) per query uncached, against
≈ 0.2 s for LoRA + student. This completes the Q1 side of Result 04's cost curve and repeats its Q2
finding: a 2B VLM over everything is worse than the cheap learned stages.

## E1 — merged "vehicle turns" view (eval, extra view)

`vehicle_turns_left` and `vehicle_turns_right` relabelled to `vehicle_turns` before queries are built;
the 6 turn phrasings replaced by 3 approved ones; 21 activities. No model retrained; Result 04's
fusion weights reused. New VLM pairs: 20,805 Instruct, 1,887 Reranker. The frozen 22-activity set
stays the main record.

**Macro over activities (21 vs 22):**

| Prot. | System | Q1 mAP 22 → 21 | Q2 S@1 22 → 21 |
|---|---|---|---|
| A | frozen SigLIP 2 | 0.179 → 0.186 | 0.273 → 0.279 |
| A | LoRA | 0.313 → 0.326 | 0.389 → 0.393 |
| A | LoRA + student | 0.476 → 0.490 | 0.531 → 0.548 |
| A | LoRA + student → Instruct @10 | 0.477 → 0.492 | 0.560 → 0.578 |
| A | Instruct on all | 0.207 → 0.218 | 0.360 → 0.374 |
| B | frozen SigLIP 2 | 0.174 → 0.183 | 0.275 → 0.293 |
| B | LoRA | 0.280 → 0.292 | 0.369 → 0.382 |
| B | LoRA + student | 0.442 → 0.452 | 0.519 → 0.533 |
| B | LoRA + student → Instruct @10 | 0.443 → 0.453 | 0.550 → 0.569 |
| B | Instruct on all | 0.212 → 0.224 | 0.367 → 0.384 |

The other 20 activities are unchanged (AP equal to 4 decimals); the macro gain comes only from the
turn class.

**The turn class itself** (left / right = mean of the two old classes):

| Prot. | System | Q1 AP left/right → merged | Q2 S@1 left/right → merged |
|---|---|---|---|
| A | chance (share of correct clips) | 0.02 → 0.04 | — |
| A | frozen SigLIP 2 | 0.223 → 0.420 | 0.352 → 0.561 |
| A | LoRA | 0.306 → **0.572** | 0.435 → 0.576 |
| A | LoRA + student | 0.310 → 0.438 | 0.380 → 0.591 |
| A | LoRA → Instruct @50 | 0.308 → 0.576 | 0.472 → **0.758** |
| A | Instruct on all | 0.254 → 0.523 | 0.361 → 0.651 |
| A | Instruct on all + LoRA | 0.310 → **0.594** | 0.491 → 0.742 |
| B | chance | 0.01 → 0.02 | — |
| B | frozen SigLIP 2 | 0.185 → 0.392 | 0.186 → 0.479 |
| B | LoRA | 0.320 → **0.615** | 0.316 → 0.542 |
| B | LoRA + student | 0.315 → 0.394 | 0.342 → 0.458 |
| B | LoRA → Instruct @50 | 0.323 → 0.624 | 0.402 → **0.667** |
| B | Instruct on all | 0.273 → 0.585 | 0.331 → 0.646 |
| B | Instruct on all + LoRA | 0.294 → **0.650** | 0.389 → 0.667 |

Findings:
- **The direction, not the turn, was the failure.** Merging doubles the share of correct clips
  (chance AP 0.02 → 0.04), but LoRA's AP rises by 0.27–0.30, far more than that. Every stage, the VLM
  included, finds turning vehicles; none tells left from right (consistent with H5 rejected, D-024).
- **The VLM helps on the merged class:** LoRA → Instruct @50 lifts Q2 S@1 by +0.18 (A) / +0.13 (B).
- **The student hurts the merged class** (Q1 AP 0.44 / 0.39 vs LoRA 0.57 / 0.62). It was trained on
  left / right captions and is not retrained here (E1 is "no retraining" by design). Retraining it on
  the merged label is a possible next step; it is not approved.
- These are eval numbers of an extra view; nothing is decided on them.

## E2 — G1 on the whole validation set (validation)

The VLM scored every validation clip for every query: 58,341 new pairs per set-up, plus the 4,800
top-50 pairs reused (Instruct ~15 min each, Reranker 3 h 20 min). Reranker-context skipped (D-027).

| Set-up | top-50 gain A / B / mean | whole-val gain A / B / mean | pass (top-50 → whole) | Δ mean |
|---|---|---|---|---|
| Instruct, tight | +0.009 / +0.002 / +0.005 | −0.009 / −0.004 / −0.006 | fail → fail | −0.012 |
| Instruct, context | −0.012 / +0.002 / −0.005 | −0.042 / −0.007 / −0.024 | fail → fail | −0.019 |
| Reranker, tight | +0.003 / +0.012 / +0.007 | −0.019 / −0.005 / −0.012 | fail → fail | −0.019 |

LoRA-only validation mAP is 0.437 (A) / 0.378 (B). The best fusion weight is w = 2 (the most LoRA,
grid edge) in every case.

**Rule (fixed before the run):** "the same" = same pass/fail **and** gain within ±0.01 of the top-50
gain, for every set-up. Pass/fail agrees everywhere; the gain is outside ±0.01 for all three set-ups.
**Verdict: not the same.** Per the spec, later G1 runs (E4) would use the whole validation set, and
the owner is asked first because it makes E4's G1 about 13× larger (63k instead of 4,800 pairs).

What the difference most likely means: with every clip in play, the VLM's mistakes on clips LoRA had put
low pull them up. The top-50 test is therefore the **more favourable** one for the VLM. In this run,
no set-up that failed at top 50 did better on the whole set. The cascade only ever shows the VLM the
top k (≤ 50), so the top-50 test is also the one that matches how the VLM is used.

## E3 — 2B VLMs with 16 frames (validation)

**G0 at 16 frames** (20 validation clips, context view, as the original G0):

| Model | Peak VRAM | Cached s / clip (64 texts) | Uncached s / pair | Deterministic | Shortcut check | G0 |
|---|---|---|---|---|---|---|
| Instruct, 8 frames (new rule, for the record) | 6.13 GB | 0.64 | 0.17 | yes | pass (0.444 ≤ 0.899; ρ 0.995) | **pass** |
| Instruct, 16 frames | **7.80 GB** | 0.64 | 0.17 | yes | pass (same 3 clips) | **fail (VRAM > 7.5)** |
| Reranker, 16 frames | 4.60 GB | — | 0.17 | yes | — | pass |

The Instruct run at 16 frames still fit on the 8 GB card. Under the standing rule (gates never stop
the work), E3 continued. The shortcut check's 3 clips hold only 7, 3 and 3 frames, so at 16 frames it
re-tests the 8-frame inputs. The cached path itself does not depend on the frame count (fp32 test in
`tests/test_teacher.py`). The per-clip speeds are means over 20 clips, of which only 4 have 16 frames.

**G1 at 16 frames** (top 50, tight):

| Set-up | 8 frames: gain A / B / mean | 16 frames: gain A / B / mean | pass |
|---|---|---|---|
| Instruct, tight | +0.009 / +0.002 / +0.005 | +0.006 / +0.009 / +0.007 | fail |
| Reranker, tight | +0.003 / +0.012 / +0.007 | +0.003 / +0.012 / +0.007 | fail |

**Where 16 frames can matter.** The crop store samples 2 fps, capped at 16 frames, so 65 % of the
1,003 G1 clips hold ≤ 8 frames (≤ 4 s). They receive identical input at 8 and 16 frames. Their scores
are identical (largest difference 0.000), which confirms the pipeline. On the 353 longer clips the
scores do change (mean |Δ| 1.04 Instruct, 0.10 Reranker). Inside the top-50 pools, per-query AUC of
the VLM score (`scripts/g1_frames_diagnostic.py`, `05_g1_16f_diagnostic.json`):

| Set-up | long clips, 8 → 16 frames | short clips (unchanged) |
|---|---|---|
| A, Instruct | 0.743 → 0.739 | 0.612 |
| B, Instruct | 0.647 → 0.666 | 0.563 |
| A, Reranker | 0.713 → 0.715 | 0.591 |
| B, Reranker | 0.667 → 0.633 | 0.543 |

16 frames move the long-clip AUC by −0.034 to +0.019, with no consistent direction. **E3 verdict:**
more frames do not fix the 2B VLM. By the spec rule, no teacher labels are made and no students are
retrained. Side finding: the VLMs separate long clips much better than short ones (0.65–0.74 vs
0.54–0.61). The short clips are where the VLM is weakest, and more frames cannot help there.

## E4 — Qwen3-VL-8B-Instruct-FP8 on the office server (validation, 2026-11-03)

Qwen's official FP8 checkpoint (rev `9cdc6310`; language model FP8, vision tower bf16), run directly
on the office server (RTX 4070 Ti SUPER 16 GB) with the copied crop store; FP8 kernel from Hugging
Face `kernels-community/finegrained-fp8` (cached commit `03ae3193`). Same frames, prompt, score and
video-once shortcut as the 2B.

**Server set-up check** (`05_e4_server_check.json`): the server's 2B re-scored all 4,800 G1 pairs.
Fails the rule fixed beforehand (largest difference 1.115 > 0.899; 1 of 4,800 pairs), but the mean
difference is 0.126, rank correlation 0.999, and G1 is reproduced within 0.002 (val mAP 0.4475 /
0.3795 vs laptop 0.4458 / 0.3797). Flagged for owner feedback; the rule compares a 4,800-pair maximum
with a 48-pair noise floor.

**G0** (20 clips, context, limit 15 GB): peak 13.0 GB, 0.64 s per clip for 64 questions (cached),
0.13 s per uncached pair, deterministic, shortcut within the noise floor (0.875 ≤ 1.250, ρ 0.995) →
**pass**.

**G1** (top 50, tight, 8 frames, same 4,800 pairs):

| Set-up | gain A | gain B | best w (A / B) | VLM-only val mAP A / B | pass |
|---|---|---|---|---|---|
| Qwen3-VL-2B Instruct (laptop) | +0.009 | +0.002 | 2 / 2 | 0.394 / 0.353 | fail |
| Gemma 4 E4B | −0.015 | +0.011 | 2 / 2 | 0.338 / 0.356 | fail |
| **Qwen3-VL-8B FP8** | **+0.007** | **+0.019** | **1** / 2 | **0.424 / 0.379** | **fail (by 0.0009)** |

LoRA alone: 0.437 / 0.378. Per-query AUC inside the pools (`05_g1_8b_auc.json`): 8B 0.661 / 0.627,
2B 0.628 / 0.590, LoRA 0.671 / 0.596. Fused val mAP by weight w (A): 0.424 (VLM only), 0.438, 0.443,
**0.444**, 0.442; (B): 0.379, 0.387, 0.394, 0.396, **0.397**.

**E4 verdict:** the bigger Qwen is clearly better than the 2B: its own ranking nearly matches LoRA on
A and beats it on B, and on A the best fusion gives it real weight for the first time. But the
fusion gain stays below the +0.02 bar (B +0.0191; the server-vs-laptop difference is about ±0.002,
so this is at the threshold, not clearly above it). By the spec rule G1 fails and E4 stops (no
whole-validation run, no teacher labels, no distillation) unless the owner decides otherwise.

**E4 cascade rows (eval; owner, D-037).** The 8B as the final stage, exactly as the 2B in Result 04:
γ chosen on validation (A: γ_LoRA 1, γ_student 2; B: 2, 2), the same eval top-k pairs (244,620 pairs,
6,935 clips; 0.49 s per clip on the server). Eval read once (`05_8b_cascade.json`). G1's verdict is
unchanged; no distillation.

| Prot. | System | VLM calls / Q2 query | Q2 S@1 2B | Q2 S@1 8B | Q2 MRR 2B | Q2 MRR 8B |
|---|---|---|---|---|---|---|
| A | LoRA + student | 0 | 0.531 | 0.531 | 0.690 | 0.690 |
| A | … → VLM @5 | 5.0 | 0.546 | **0.574** | 0.702 | 0.718 |
| A | … → VLM @10 | 9.8 | 0.560 | **0.589** | 0.710 | 0.728 |
| A | … → VLM @20 | 18.6 | 0.558 | **0.596** | 0.710 | 0.737 |
| A | LoRA → VLM @10 | 9.8 | 0.421 | 0.461 | 0.586 | 0.615 |
| A | VLM on all candidates | 124.5 | 0.360 | 0.428 | 0.523 | 0.576 |
| B | LoRA + student | 0 | 0.519 | 0.519 | 0.663 | 0.663 |
| B | … → VLM @5 | 4.9 | 0.550 | **0.560** | 0.685 | 0.691 |
| B | … → VLM @10 | 9.7 | 0.550 | 0.554 | 0.684 | 0.692 |
| B | … → VLM @20 | 18.1 | 0.541 | **0.564** | 0.680 | 0.696 |
| B | LoRA → VLM @10 | 9.7 | 0.393 | 0.395 | 0.548 | 0.548 |
| B | VLM on all candidates | 307.8 | 0.367 | 0.437 | 0.531 | 0.589 |

Q1 mAP barely moves (the VLM re-ranks only the top k): A 0.477 → 0.477 at @10, B 0.443 → 0.443.

Paired bootstrap over the 22 activities (10,000 resamples, seed 0; `05_8b_bootstrap.json`), Q2 S@1:

| Comparison | A | B |
|---|---|---|
| 8B @10 − LoRA + student | **+0.058 [+0.020, +0.099]** | +0.036 [−0.002, +0.074] |
| 8B @5 − 2B @5 | **+0.028 [+0.009, +0.046]** | +0.011 [−0.010, +0.034] |
| 8B @10 − 2B @10 | +0.030 [−0.000, +0.058] | +0.004 [−0.029, +0.041] |

**Reading:** on Protocol A the 8B as a 5–10-call final stage is a clear improvement (+0.058 over the
0-call system, about twice the 2B's +0.03), at about 0.3 s per call on the office GPU. On unseen
cameras (B) the gains are smaller and not significant. "VLM on everything" is still worse than the
0-call LoRA + student, as in Result 04. This is consistent with G1: the 8B narrowly misses the
validation bar (B +0.019), and helps most where it is used, on a short list.

### E4b — Qwen3-VL-32B-Instruct-FP8 (validation, 2026-11-03): **first VLM to pass G1**

Official FP8 checkpoint (rev `4bf2c2f3`, Apache 2.0, 35.5 GB; files verified). On the 16 GB GPU with
the CPU/GPU split of `sfs.teacher.offload` (exact; 8B split check identical): 16 of 64 decoder layers
and the vision tower on the GPU (10.6 GB), 48 layers (23.4 GB) streamed per pass from the OS file
cache (unpinned, shared server), token embeddings on the CPU; 4 questions per cached batch. Run
inside the GPU holder (D-041).

**G1 (top 50, tight, 8 frames, same 4,800 pairs): PASS.**

| Model | gain A | gain B | best w (A / B) | VLM-only val mAP A / B |
|---|---|---|---|---|
| Qwen3-VL-2B | +0.009 | +0.002 | 2 / 2 | 0.394 / 0.353 |
| Qwen3-VL-8B FP8 | +0.007 | +0.019 | 1 / 2 | 0.424 / 0.379 |
| **Qwen3-VL-32B FP8** | **+0.017** | **+0.037** | **1 / 0.5** | **0.433 / 0.401** |

LoRA alone 0.437 / 0.378. Fused val mAP by w (A): 0.433, 0.444, 0.449, **0.454**, 0.449; (B): 0.401,
0.413, **0.415**, 0.414, 0.401. Size helps monotonically (2B → 8B → 32B); on unseen cameras (B) the 32B
alone ranks better than LoRA and the best fusion weights it above LoRA.

**G0 (context, 20 clips): fails on speed only** — 25.4 s per clip for 64 questions, 1.74 s per
uncached pair (rule: 3 s per clip), because 23 GB are streamed per pass; peak 12.3 GB, deterministic,
shortcut within the noise floor (1.036 ≤ 1.145).

**G1 on the whole validation set (owner's rule, D-029): PASS** (`05_g1_wholeval_32b.json`). The
58,341 extra pairs (1,325 clips, 19:18–23:26, 11.2 s per clip, 8 questions per batch) merged with the
4,800 top-50 pairs; same gate, `--k 0`.

| Model | gain A | gain B | best w (A / B) | VLM-only val mAP A / B |
|---|---|---|---|---|
| Qwen3-VL-2B (E2) | −0.009 | −0.004 | 2 / 2 | 0.292 / 0.282 |
| **Qwen3-VL-32B FP8** | **+0.018** | **+0.047** | **2 / 2** | **0.348 / 0.318** |

LoRA alone 0.437 / 0.378. Fused val mAP by w (A): 0.348, 0.395, 0.418, 0.440, **0.455**; (B): 0.318,
0.359, 0.390, 0.413, **0.425**. Unlike the 2B, the 32B's gains do not shrink on the whole validation
set (top 50: +0.017 / +0.037). On all clips the VLM alone is far below LoRA, so the best weight moves to
the top of the pre-set grid (w = 2, LoRA weighted most); the grid was fixed before the run and is not
extended.

**Verdict: the 32B passes G1 at both depths** → by D-026 the next steps are teacher labels with the
32B, the GTKD / KD students retrained on them, and cascade rows. Estimated server time (from this run:
1.6 s per streamed pass; one pass for the video plus one per 8 questions):

| Step | Pairs / clips | Server time |
|---|---|---|
| Teacher labels for the training pools (`requests_trainval.csv`) | 41,266 / 9,362 | ~8.6 h |
| Validation student pairs (γ choice) | ~1,900 / ~660 | ~0.6 h |
| Eval short list (Q1 top-k + Q2 top-20; Reranker-sized request) | ~38,000 / ~4,900 | ~5.6 h |
| *Optional:* eval "VLM on all candidates" (Q2) | 244,620 / 6,935 | ~17.6 h |

**Started 2026-11-04 00:11 (owner, D-044):** the first three rows, without the optional one.

#### Students distilled from the 32B (validation, 2026-11-04, D-047)

Labels: 41,266 pairs, 00:11–09:05 on the server. The three students were retrained on the server
(`runs/student_32b`), because its GPU does not reproduce the laptop's training exactly: Result 04's
A-GTKD retrained there reaches 0.6045 instead of 0.5943. So GT, GTKD and KD are compared on one
machine. Settings, pools, features and the selection rule are those of Result 04.

| Protocol | GT | GT+KD (32B) | KD (32B) | Selected | *Result 04, laptop: GT / GT+KD (2B) / KD (2B)* |
|---|---|---|---|---|---|
| A | **0.606** | 0.573 | 0.447 | GT | *0.600 / 0.594 / 0.424* |
| B | **0.571** | 0.525 | 0.352 | GT | *0.566 / 0.539 / 0.377* |

The 32B is a clearly better teacher than the 2B on the training pairs (`05_teacher32b_pool_quality.json`):

| Teacher | AUC (A / B) | mean soft target, right clips | wrong clips |
|---|---|---|---|
| Qwen3-VL-Reranker-2B | 0.613 / 0.615 | 0.46 | 0.41 |
| Qwen3-VL-32B-Instruct-FP8 | 0.686 / 0.695 | 0.49 | 0.27 |

Still, GT+KD falls further below GT than it did with the 2B: −0.033 / −0.046, against a machine-to-machine
difference of about 0.006 for the same student. KD alone improves on A and gets worse on B. A likely
reading: at AUC 0.69 the teacher is still no better than LoRA's own ranking, and its soft target for a
right clip averages 0.49. Mixed half-and-half with the human label of 1, this pulls right clips down. So
the student learns more from the human labels alone.

**Verdict:** GT is selected again on both protocols, and distillation does not help with the 32B either.
H3 stays rejected. The cascade rows use Result 04's GT students (laptop, `runs/student`), whose short lists
the remaining 32B jobs score, so the 32B stage is the only change against Result 04.

#### 32B cascade rows (eval, 2026-11-04, D-048)

The 32B as the final stage, exactly as the 2B (Result 04) and the 8B (§E4): the same LoRA + GT
students (laptop, `runs/student`), γ chosen on validation (A: γ_LoRA 1 from the top-50 G1, γ_student 2;
B: 0.5, 2), and the same eval short list (Q1 top 50, Q2 top 20; 38,202 pairs, 4,917 clips, 09:43–15:30
on the server). No "VLM on all candidates" row (owner, D-044). Eval read once (`05_32b_cascade.json`).

| Prot. | System | VLM calls / Q2 query | Q2 S@1 2B | Q2 S@1 8B | **Q2 S@1 32B** | Q2 MRR 32B |
|---|---|---|---|---|---|---|
| A | LoRA + student | 0 | 0.531 | 0.531 | 0.531 | 0.690 |
| A | … → VLM @5 | 5.0 | 0.546 | 0.574 | **0.573** | 0.717 |
| A | … → VLM @10 | 9.8 | 0.560 | 0.589 | **0.596** | 0.732 |
| A | … → VLM @20 | 18.6 | 0.558 | 0.596 | **0.605** | 0.740 |
| A | LoRA → VLM @10 | 9.8 | 0.421 | 0.461 | 0.448 | 0.612 |
| B | LoRA + student | 0 | 0.519 | 0.519 | 0.519 | 0.663 |
| B | … → VLM @5 | 4.9 | 0.550 | 0.560 | **0.564** | 0.693 |
| B | … → VLM @10 | 9.7 | 0.550 | 0.554 | **0.564** | 0.692 |
| B | … → VLM @20 | 18.1 | 0.541 | 0.564 | **0.568** | 0.695 |
| B | LoRA → VLM @10 | 9.7 | 0.393 | 0.395 | 0.415 | 0.566 |

Q1 mAP barely moves (A 0.476 → 0.478 at @10, B 0.442 → 0.444), as for the smaller VLMs.

Paired bootstrap over the 22 activities (10,000 resamples, seed 0; `05_32b_bootstrap.json`), Q2 S@1.
The same code reproduces the 8B's recorded differences (+0.058 / +0.036); its interval edges differ
by ≤ 0.002 from `05_8b_bootstrap.json`.

| Comparison | A | B |
|---|---|---|
| 32B @10 − LoRA + student | **+0.065 [+0.040, +0.095]** | **+0.045 [+0.018, +0.073]** |
| 32B @20 − LoRA + student | **+0.074 [+0.047, +0.103]** | **+0.049 [+0.021, +0.079]** |
| 32B @5 − 2B @5 | **+0.027 [+0.010, +0.047]** | +0.015 [−0.003, +0.033] |
| 32B @10 − 2B @10 | **+0.037 [+0.022, +0.052]** | +0.013 [−0.014, +0.041] |
| 32B @5 − 8B @5 | −0.001 [−0.018, +0.015] | +0.004 [−0.010, +0.017] |
| 32B @10 − 8B @10 | +0.007 [−0.020, +0.034] | +0.009 [−0.018, +0.035] |

**Cost:** 1.74 s per uncached call for the 32B on the office server (G0; 48 of 64 layers streamed from
CPU memory), against 0.13 s for the 8B (`g0_8b.json`) — `05_8b_cascade.json` records the 2B's 0.17 s by a report bug, fixed in
`05_8b_cascade_g0fix.json` (otherwise identical, D-048) and 0.17 s for the 2B on the laptop. Ten calls
per query: ~17 s for the 32B, ~1.3 s for the 8B.

**Reading:**
- The 32B is the first VLM whose gain as a final stage is significant on **both** protocols, including
  unseen cameras (B: +0.045 at 10 calls; the 8B's +0.036 was not significant).
- But it is **not significantly better than the 8B** on either protocol (differences ≤ 0.009, intervals
  around 0), at ~13× the time per call on this GPU. Its clear G1 pass (validation) shows up on eval
  only as a small, uncertain edge over the 8B.
- On this hardware the 8B is the better cost point. The 32B's advantage is that its gain holds on
  unseen cameras with statistical support.
- "LoRA → VLM" stays far below "LoRA + student → VLM" for every VLM: the student is still the most
  valuable stage per unit of cost.
- Distillation from the 32B did not help either (above, D-047). H3 stays rejected.

## E7 — Qwen2.5-VL from the owner's server cache (validation, 2026-11-03)

Same frames, prompt, score and shortcut; server; same 4,800 top-50 pairs.

| Model | G0 (peak, s per clip / 64 q) | G1 gain A | G1 gain B | VLM-only val mAP A / B | G1 |
|---|---|---|---|---|---|
| Qwen2.5-VL-3B (bf16) | pass (8.5 GB, 0.37 s) | +0.001 | +0.004 | 0.385 / 0.360 | **fail** |
| Qwen2.5-VL-7B (FP8 at load) | pass (11.4 GB, 0.64 s) | +0.001 | +0.014 | 0.380 / 0.361 | **fail** |

For reference: Qwen3-VL-2B +0.009 / +0.002 (0.394 / 0.353), Qwen3-VL-8B +0.007 / +0.019 (0.424 /
0.379), LoRA alone 0.437 / 0.378. **Neither Qwen2.5-VL model contributes** (owner's rule, D-038:
delete if not); both are below the Qwen3-VL-8B.

## E6 — Gemma 4 E4B (validation, 2026-11-03)

gemma-4-E4B-it (rev `ee0ef602`, Apache 2.0) on the office server beside the team's Ollama: bf16, no
quantisation; per-layer embedding table and audio tower in CPU memory, 30 of 42 decoder layers on
the GPU, 12 streamed per pass. Same frames, question and yes/no score as Qwen; Gemma's own processor
(66 tokens per frame). Scores were re-checked after a placement fix: identical (19 pairs, max
difference 0).

**G0** (20 validation clips, context view): peak 7.75 GB (limit 9.5 GB beside Ollama), deterministic
(difference 0), 0.16 s per question, **99.95 % of the probability on yes/no** (the model answers in
the expected form). **Fails on speed only:** no video-once shortcut yet, so 64 questions take 9.4 s
per clip (rule: 3 s).

**G1** (top 50, tight, 8 frames; same 4,800 pairs as Qwen):

| Set-up | gain A | gain B | mean | pass |
|---|---|---|---|---|
| Qwen3-VL-2B Instruct (Result 04) | +0.009 | +0.002 | +0.005 | fail |
| Qwen3-VL-2B Reranker (Result 04) | +0.003 | +0.012 | +0.007 | fail |
| **Gemma 4 E4B** | **−0.015** | **+0.011** | **−0.002** | **fail** |

How well each model alone separates correct from wrong clips inside the same pools
(`05_g1_gemma_vs_qwen_auc.json`):

| | Qwen 2B Instruct | Qwen 2B Reranker | Gemma 4 E4B | LoRA |
|---|---|---|---|---|
| A: per-query AUC | 0.628 | 0.615 | 0.542 | 0.671 |
| B: per-query AUC | 0.590 | 0.596 | 0.586 | 0.596 |
| A: VLM-only val mAP | 0.394 | 0.388 | 0.338 | 0.437 |
| B: VLM-only val mAP | 0.353 | 0.349 | 0.356 | 0.378 |

**E6 verdict (E4B):** Gemma 4 E4B does not help; on Protocol A it is weaker than the 2B Qwen
models, on B about equal. Like every Qwen set-up, the best fusion gives the VLM the least weight
(w = 2). By the spec rule E6 stops for the E4B: no whole-validation run, no teacher labels. The 12B
(office server, whole GPU, FP8 language model) is the owner's call.

---

## Limitations

- **E1** reuses Result 04's fusion weights and a student trained on left / right. A retrained student
  could change the merged-class numbers. The merged phrasings were approved before eval was read
  but were not tested on training data.
- **E2** skipped Reranker-context (the worst top-50 set-up).
- **E3** changes only 35 % of the G1 clips. A test on long clips only would need its own query set.
  The 16-frame shortcut check re-tests ≤ 8-frame inputs (see above).
- **All:** 2B models, default prompts, a single seed for the students, CIs not recomputed for these
  extra views.

## Open for the owner

*Updated 2026-11-04.* Earlier items 1 (G1 depth: decided (c), D-029) and 2 (server details: received,
D-029/D-030) are closed.

1. **Stop or continue the VLM experiments (D-048).** Every planned VLM step is done. The server GPU is
   still held for us (14.5 GB) until a STOP file is written. Recommended: stop and release it, since the
   32B gives no significant gain over the 8B.
2. **Accept the E4 server set-up check (D-035)** despite its rule failure (1 of 4,800 pairs; G1
   reproduced within 0.002). Recommended: accept.
3. **E1 follow-up (optional):** retrain the student on the merged turn label (~1 h on the laptop,
   needs a spec note). Only if "vehicle turns" will be the demo's activity.
4. **Delete Qwen2.5-VL 3B / 7B** on the server (failed G1, D-040). Kept for nexa-vlm so far.
