# Result 06 — Image + text LoRA vs image-only LoRA (E8), validation

**Date:** 2026-11-04 · **Spec:** `docs/planning/spec-lora-image-text.md` · **Log:** D-050 – D-053
**Status:** complete — image + text **not adopted**
**Inputs:** packed tight crops `data/packed_tight` built on the office server (byte-identical to the
laptop's, store id `84b0a9ac41c5`), `manifest_v2.csv`, caption bank `queries/train_captions.csv`
(`139d6fc19a02a9ad`, D-019), frozen phrasings `queries/phrasings.csv` (`331bb6bba25a73bd`, D-017),
SigLIP 2 base at revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`.
**Raw output:** `docs/results/raw/06_*` (per run `done.json` + training log, per-activity AP,
`06_e8_report.json`).
**Reproduce the tables:** `python scripts/e8_report.py --root runs/e8 --seeds 0,1 --out docs/results/raw/06_e8_report.json`.

## Setup

- **Arms:** *image-only* = Result 03's set-up (LoRA rank 16 on the image side, text side frozen);
  *image + text* = the same plus LoRA rank 16 on the text side's attention projections (48 projections,
  1,179,648 parameters, the same count as the image side). In image + text, the captions drawn each step
  go through the adapted text side with gradients, and the 66 eval phrasings are re-embedded every epoch
  for validation.
- **Runs:** the two runs Result 03 selected, A-middle and B-uniform, each in both arms. All on the same
  office GPU (RTX 4070 Ti SUPER, inside the GPU holder), so the comparison has no machine difference.
- **Everything else as Result 03:** label-aware sigmoid loss, batch 64, √-balanced sampling, augmentation
  without flip, warm-up + reduce-on-plateau, early stop after 5 epochs without a validation gain, ≤ 100 epochs.
- **Metric:** validation Q1 macro mAP over the protocol's validation activities (A 15, B 17), with the
  frozen eval phrasings as queries. The phrasings never appear in training (checked), so the image + text
  arm is measured on wordings its text side never saw.
- **Adoption rule (spec §5, fixed before any run):** mean gain over A and B ≥ +0.010 **and** neither
  protocol drops by more than 0.005.
- **Second seed (D-052):** the rule is applied to the per-protocol means over seeds 0 and 1; an effect
  "repeats" when both seeds give the same sign and each is larger than 0.010. Fixed before any seed-1 number.
- **Seeds:** seed 0 = sampler / augmentation / caption draws seeded, LoRA init from torch's process state
  (as Result 03). Seed 1 = all of these plus the LoRA init seeded (`--seed 1`).
- **No eval-split number is read in this result.**

## Results

### Validation mAP per run

| Protocol | Seed | Image only (best epoch) | Image + text (best epoch) | Difference |
|---|---|---|---|---|
| A-middle | 0 | 0.4359 (12) | 0.4075 (8) | −0.0284 |
| A-middle | 1 | 0.4203 (7) | 0.4112 (7) | −0.0090 |
| A-middle | **mean** | 0.4281 | 0.4094 | **−0.0187** |
| B-uniform | 0 | 0.3691 (6) | 0.4198 (24) | +0.0507 |
| B-uniform | 1 | 0.3858 (14) | 0.4297 (18) | +0.0439 |
| B-uniform | **mean** | 0.3775 | 0.4248 | **+0.0473** |

Frozen start (epoch 0): A 0.2717, B 0.2222, identical in every run. The seed-0 image-only runs reproduce
the laptop's Result 03 runs (A 0.4371, B 0.3758) within seed noise.

### Decision by the pre-set rule

- **Gain:** mean over A and B = **+0.014 ≥ +0.010 → pass.**
- **No protocol loses:** A's 2-seed mean drops by **0.019 > 0.005 → fail.**
- **→ Not adopted.** The image-only model stays the final model; Results 03–05 are unchanged.

### How solid each effect is

| | Seed 0 | Seed 1 | Same sign | Both > 0.010 | Reading |
|---|---|---|---|---|---|
| B (unseen cameras) | +0.051 | +0.044 | yes | yes | **repeats**: text adaptation helps |
| A (seen cameras) | −0.028 | −0.009 | yes | no | probably harmful, not confirmed by the bar |

For scale, the same image-only set-up moves by 0.016 (A) and 0.017 (B) between seeds alone. B's gain
is about three times that.

### Per activity (sign agreeing in both seeds)

- **B:** 14 of 17 agree, **12 better / 2 worse.** Consistently better: opens facility door
  (+0.29 / +0.19), vehicle stops (+0.16 / +0.12), talks on phone (+0.15 / +0.09), vehicle turns left
  (+0.09 / +0.09), hand interacts with person (+0.06 / +0.09). Consistently worse: carries heavy object
  (−0.23 / −0.10), exits through structure (−0.04 / −0.06).
- **A:** 9 of 15 agree, **4 better / 5 worse.** Consistently worse: opens vehicle door (−0.24 / −0.13),
  enters vehicle (−0.19 / −0.11), texts on phone (−0.10 / −0.17), turns right (−0.12 / −0.08), stands
  up (−0.02 / −0.03). Consistently better: opens facility door, talks to person, turns left, closes
  vehicle door.
- Full table: `docs/results/raw/06_e8_report.json` (`per_activity`).

## Reading

1. **Adapting the text side helps on cameras the model has never seen** (+0.047 val mAP, both seeds).
   A possible explanation (a hypothesis, not tested here): with the text side fixed, the image side has to move its
   embeddings towards the fixed text embeddings, which ties it to the training views. When the text
   side can also move, the two sides meet in between, and that transfers better to new views.
2. **On seen cameras it costs accuracy** (−0.019, both seeds negative but one below the bar). The
   losses concentrate on fine-grained vehicle-door and phone activities, where the image-only model is
   already strong on familiar views.
3. **The text side trains on 192 captions only.** The validation queries are 66 different wordings
   it never saw, so the B gain is not caption memorisation. General language beyond MEVA was not
   tested (the retention check was dropped, D-050).
4. **Seed noise is large for single runs** (±0.017). Any LoRA comparison in this project resting on
   one seed and a difference below ~0.02 should be read with that in mind.

## Text-drift check (D-054, D-055)

Run after the owner moved the demo to MEVA and asked whether the image + text model suits
free-form appearance queries. Rule fixed before measuring: for each image + text adapter, the Spearman
ρ between the frozen and adapted pairwise similarities of 40 appearance / place / vehicle sentences
(`queries/appearance_sentences.csv`, none in training) must be ≥ 0.90.

| Adapter | Sentences ρ | Self-cos | NN agree | Eval phrasings ρ (scale) | Pass |
|---|---|---|---|---|---|
| A-middle seed 0 | 0.813 | 0.938 | 0.60 | 0.628 | no |
| A-middle seed 1 | 0.825 | 0.943 | 0.62 | 0.658 | no |
| B-uniform seed 0 | 0.776 | 0.931 | 0.60 | 0.556 | no |
| B-uniform seed 1 | 0.767 | 0.933 | 0.55 | 0.639 | no |

**Fail for every adapter.** The adapted text side changes how general wordings relate to one another,
less than it changes the activity phrasings but clearly (the nearest neighbour changes for ~40 % of the
sentences), and most for the B adapters. This is a guard, not an accuracy measure: MEVA has no
appearance labels, so whether appearance search gets better or worse is not known. By the pre-set rule
the demo uses the frozen text side, i.e. the image-only model. Raw: `docs/results/raw/06_text_drift.json`.

## What is not claimed

- No eval-split number was read; this is a validation-only comparison.
- The image + text model was not carried into the student / cascade, so its effect on Q2 forensic
  queries is unknown.
- Using the image + text model for unseen-camera footage would be a choice made after seeing these
  results, not a pass of the rule (D-053). It was not made: the nexa_fight demo was dropped and the
  text-drift check failed (D-054, D-055).
