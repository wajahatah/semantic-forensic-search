# Result 01 — Zero-shot encoder and view selection (validation split)

**Date:** 2026-10-23 · **Data:** `embeddings_v2_{siglip,siglip2}.npz` (pipeline v2, 220,707 vectors each)
**Split:** validation only (739 instances, 15 activity types with ≥5 examples). Eval untouched.
**Decision:** **SigLIP 2 + tight view** selected as the base encoder (spec-offline-embedding §8, row 1).

## Method

Frozen encoders, no training. For each activity type, text prompt
`"a surveillance camera photo of a <activity words>"`. Instance score = mean of the top-3 frame
similarities (spec-retrieval-system §5.2). Per activity: AUC (matching vs non-matching instances)
and P@10 (share of correct instances among the 10 highest-scored). Macro-averaged over activities.

## Results

| View | SigLIP AUC | SigLIP 2 AUC | SigLIP P@10 | SigLIP 2 P@10 |
|---|---|---|---|---|
| full | 0.661 | 0.639 | 0.087 | 0.10 |
| context | 0.670 | 0.691 | 0.120 | 0.14 |
| **tight** | 0.663 | **0.697** | 0.120 | **0.22** |

Chance: AUC 0.500; P@10 0.065.

Per activity (SigLIP 2, tight):

| AUC > 0.9 | ≈ chance or below |
|---|---|
| person_enters_vehicle 0.996, vehicle_turns_left 0.977, vehicle_turns_right 0.976, person_opens_vehicle_door 0.962, person_closes_vehicle_door 0.961, person_opens_facility_door 0.922 | person_carries_heavy_object 0.491, person_puts_down_object 0.475, person_texts_on_phone 0.471, person_picks_up_object 0.462, hand_interacts_with_person 0.373, **person_stands_up 0.127** |

## Findings

1. **Actor-region cropping doubles top-10 precision** for SigLIP 2 (0.22 vs 0.10 full frame).
2. **The encoder ranking depends on the view.** SigLIP is slightly better on full frames; SigLIP 2 wins once cropped.
3. **Zero-shot fails on fine-grained human actions.** Large, distinctive vehicle/door events are found
   almost perfectly; object handling, phone use and posture changes are at or below chance.
   `person_stands_up` scores 0.127 — systematically inverted — because a single frame of someone
   standing up resembles someone seated: per-frame embeddings carry no motion.
4. This is the gap domain adaptation (LoRA) and a temporal re-ranker are meant to close.

## Caveats

Validation is small; several activity types have 6–11 positives. Differences of ~0.01 AUC are
within noise. The view effect (0.22 vs 0.10) and the activity-group split are not.

## Bug found on the way (pipeline v1 → v2)

The first embedding run found actor boxes in only 2 of 252 clips: the MEVA repo mixes two YAML
styles for `geom.yml` and a text regex matched only one. 99.5% of crops silently fell back to the
full frame. Fixed by parsing with a real YAML parser; the job now refuses to run on a clip whose
actors have no boxes, and a pipeline version in the encoder id invalidates stale output.

## Addendum 2026-11-17 — zero-shot CLIP ViT-B/16 (D-067)

Added for the presentation's encoder comparison (spec-submission §2). **No decision depends on it**: the
encoder was chosen on 2026-10-23, and the rule fixed before any CLIP number was that the system does not
change whatever CLIP scores. `openai/clip-vit-base-patch16` (revision `57c21647…`, same ViT-B/16 at 224 px
as our SigLIP 2), tight view, its own preprocessing and tokenizer, embedded from the local packed tight crops
(73,569 crops, store `84b0a9ac41c5`). Code: `scripts/clip_baseline.py` (test `tests/test_clip_baseline.py`).
The same code first re-ran this result's SigLIP and SigLIP 2 rows from their stored embeddings and
reproduced them exactly (0.663 / 0.120, 0.697 / 0.220); this protocol had run on Colab, so its code was
re-written for this check. Raw: `docs/results/raw/01b_clip_val.json`.

| Encoder (tight) | AUC | P@10 |
|---|---|---|
| SigLIP | 0.663 | 0.120 |
| SigLIP 2 | 0.697 | **0.220** |
| CLIP ViT-B/16 | **0.726** | 0.207 |

CLIP's higher AUC comes almost entirely from `person_stands_up` (CLIP 0.625, SigLIP 2 0.127 — the type
SigLIP 2 ranks inverted). Without it the two are level (mean AUC over the other 14: CLIP 0.733, SigLIP 2
0.738). Top-10 precision, the measure closest to search, is lower for CLIP.
