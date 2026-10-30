# Result 03 — LoRA-adapted SigLIP 2 vs frozen baseline, eval split

**Date:** 2026-10-30 · **Spec:** `docs/planning/spec-lora-finetune.md` · **Log:** D-021
**Inputs:** packed tight crops `data/packed_tight` (store id `84b0a9ac41c5`), `manifest_v2.csv`,
caption bank `queries/train_captions.csv` (sha256 prefix `139d6fc19a02a9ad`, D-019), frozen eval
phrasings `queries/phrasings.csv` (`331bb6bba25a73bd`, D-017). Baseline: Result 02, re-scored with
Success@K (all earlier metrics unchanged).
**Raw output:** `docs/results/raw/03_lora_*` (per-run eval JSON + per-query CSV, training logs,
`done.json`, `selection.json`, validation check, per-activity table).
**Reproduce the tables:** `python scripts/lora_val_check.py A-middle A-uniform B-uniform B-middle`,
then `python scripts/lora_report.py`.

## Setup

- **Model:** SigLIP 2 base, LoRA rank 16 on the vision tower's attention projections (48 modules,
  1.18 M trainable parameters = 1.25 %). The text tower is frozen, so query encoding is unchanged.
- **Training:** label-aware sigmoid loss over the 192-caption bank, √-balanced instance sampling,
  augmentation without flip, warm-up + reduce-on-plateau, early stop after 5 epochs without a
  validation gain (D-020).
- **Four runs:** protocol A or B × frame sampling uniform or middle-weighted. Each model is
  evaluated **only on its own protocol** (D8: A's training clips overlap B's eval).
- **Selection:** per protocol, by validation Q1 macro mAP. `selection.json` was written before the
  eval split was read. Eval numbers below are reported, not used for any choice.
- **Metrics:** as Result 02, plus Success@K (D11, defined before any adapted result). Macro = mean
  over the 22 query activities. CIs: paired bootstrap over activities, 10,000 resamples.

## Training and selection (validation only)

| Run | Epochs run | Best epoch | Val mAP frozen → best | Role |
|---|---|---|---|---|
| A-uniform | 33 | 28 | 0.272 → 0.418 | ablation |
| **A-middle** | 21 | 16 | 0.272 → **0.437** | **selected (A)** |
| **B-uniform** | 15 | 10 | 0.222 → **0.3776** | **selected (B)** |
| B-middle | 15 | 10 | 0.222 → 0.3756 | ablation |

All runs stopped early; none came near the 100-epoch cap. ~57 s per epoch on the RTX 4060
(5.0 GB peak GPU memory), ~1.5 h for all four runs. The B choice is a near-tie (0.002).

**Check that the evaluated vectors are the selected models:** validation mAP recomputed from the
re-embedded stores matches the value the trainer recorded for the best epoch to within 0.0004
(fp16 rounding) for all four runs. With the adapter disabled, re-embedding reproduces the baseline
vectors (cos min 0.99996 over 2,000 crops).

## Headline (eval split, macro over activities)

| Protocol | System | Q1 mAP | Q1 P@10 | Q2 R@1 | Q2 R@10 | Q2 MRR | Q2 S@1 | Q2 S@5 | Q2 S@10 |
|---|---|---|---|---|---|---|---|---|---|
| A | random | 0.047 | 0.044 | 0.033 | 0.267 | 0.252 | 0.116 | 0.385 | 0.567 |
| A | baseline (frozen) | 0.179 | 0.259 | 0.101 | 0.511 | 0.446 | 0.273 | 0.655 | 0.819 |
| A | **A-middle (selected)** | **0.313** | **0.408** | **0.145** | **0.597** | **0.558** | **0.389** | **0.797** | **0.902** |
| A | A-uniform (ablation) | 0.314 | 0.386 | 0.138 | 0.606 | 0.547 | 0.372 | 0.783 | 0.900 |
| B | random | 0.046 | 0.044 | 0.036 | 0.275 | 0.243 | 0.109 | 0.370 | 0.569 |
| B | baseline (frozen) | 0.174 | 0.274 | 0.099 | 0.496 | 0.460 | 0.275 | 0.703 | 0.849 |
| B | **B-uniform (selected)** | **0.280** | **0.362** | **0.135** | **0.545** | **0.528** | **0.369** | **0.723** | **0.893** |
| B | B-middle (ablation) | 0.275 | 0.338 | 0.126 | 0.546 | 0.533 | 0.366 | 0.749 | 0.911 |

**Improvement over the frozen baseline** (95 % CI, paired over activities):

| Run | Q1 mAP Δ | Q2 S@1 Δ | Q2 S@10 Δ | Q2 MRR Δ |
|---|---|---|---|---|
| A-middle (selected) | +0.134 [+0.085, +0.194] | +0.115 [+0.065, +0.168] | +0.083 [+0.035, +0.134] | +0.111 [+0.078, +0.146] |
| A-uniform (ablation) | +0.136 [+0.086, +0.196] | +0.099 [+0.052, +0.147] | +0.081 [+0.040, +0.125] | +0.101 [+0.068, +0.135] |
| B-uniform (selected) | +0.107 [+0.062, +0.158] | +0.094 [+0.035, +0.151] | +0.044 [+0.005, +0.081] | +0.068 [+0.021, +0.112] |
| B-middle (ablation) | +0.101 [+0.058, +0.152] | +0.091 [+0.025, +0.154] | +0.062 [+0.022, +0.102] | +0.073 [+0.023, +0.122] |

**Cost:** unchanged — 1 text-encoder call, 0 VLM calls per query (~7 ms text encoding + ~30 ms to
score and rank ~3,800 instances). The adapter only changes the image embeddings, which are
computed offline once per alert.

## Per activity — Q1 AP, selected models

*Lift* = AP ÷ share of the activity in the eval corpus (1.0 = chance). Sorted by baseline lift (A).

| Activity | n eval (A) | AP A: base → LoRA | lift A | AP B: base → LoRA | lift B |
|---|---|---|---|---|---|
| person_enters_vehicle | 32 | 0.187 → 0.248 | 21.8 → 28.8 | 0.227 → 0.177 | 37.3 → 29.0 |
| person_talks_on_phone | 66 | 0.365 → 0.687 | 20.6 → 38.8 | 0.244 → 0.498 | 25.2 → 51.6 |
| vehicle_reverses | 21 | 0.107 → 0.121 | 19.0 → 21.5 | 0.093 → 0.125 | 19.4 → 26.0 |
| vehicle_turns_left | 78 | 0.262 → 0.327 | 12.5 → 15.6 | 0.188 → 0.345 | 15.8 → 28.9 |
| hand_interacts_with_person | 46 | 0.128 → 0.198 | 10.3 → 16.1 | 0.099 → 0.185 | 3.6 → 6.7 |
| vehicle_turns_right | 68 | 0.184 → 0.285 | 10.1 → 15.6 | 0.182 → 0.294 | 15.3 → 24.6 |
| person_exits_vehicle | 51 | 0.133 → 0.183 | 9.7 → 13.3 | 0.108 → 0.232 | 14.7 → 31.6 |
| person_closes_vehicle_door | 75 | 0.185 → 0.190 | 9.2 → 9.4 | 0.161 → 0.197 | 12.4 → 15.2 |
| person_opens_vehicle_door | 79 | 0.172 → 0.256 | 8.1 → 12.1 | 0.166 → 0.245 | 13.1 → 19.3 |
| vehicle_stops | 72 | 0.123 → 0.184 | 6.4 → 9.5 | 0.104 → 0.122 | 10.8 → 12.7 |
| vehicle_starts | 63 | 0.108 → 0.154 | 6.4 → 9.1 | 0.123 → 0.147 | 12.8 → 15.2 |
| person_opens_facility_door | 171 | 0.276 → 0.343 | 6.0 → 7.5 | 0.271 → 0.220 | 8.5 → 6.9 |
| person_texts_on_phone | 220 | 0.311 → 0.718 | 5.3 → 12.2 | 0.224 → 0.633 | 3.1 → 8.8 |
| person_enters_scene_through_structure | 447 | 0.311 → 0.437 | 2.6 → 3.6 | 0.241 → 0.304 | 4.9 → 6.2 |
| person_exits_scene_through_structure | 435 | 0.300 → 0.369 | 2.6 → 3.2 | 0.319 → 0.356 | 5.2 → 5.8 |
| person_talks_to_person | 394 | 0.238 → 0.744 | 2.3 → 7.0 | 0.291 → 0.654 | 2.5 → 5.7 |
| person_transfers_object | 68 | 0.040 → 0.130 | 2.2 → 7.1 | 0.047 → 0.094 | 2.0 → 4.0 |
| person_carries_heavy_object | 334 | 0.192 → 0.552 | 2.1 → 6.2 | 0.259 → 0.512 | 2.7 → 5.3 |
| person_sits_down | 114 | 0.064 → 0.181 | 2.1 → 5.9 | 0.084 → 0.175 | 1.1 → 2.4 |
| person_stands_up | 76 | 0.033 → 0.098 | 1.6 → 4.8 | 0.073 → 0.205 | 1.0 → 2.8 |
| person_puts_down_object | 393 | 0.114 → 0.249 | 1.1 → 2.4 | 0.164 → 0.241 | 1.3 → 1.9 |
| person_picks_up_object | 355 | 0.097 → 0.232 | 1.0 → 2.4 | 0.149 → 0.205 | 1.2 → 1.6 |

Model A improved all 22 activities. Model B improved 20 and lost on two
(`person_enters_vehicle` −0.050 on a small class; `person_opens_facility_door` −0.051).

## Hypotheses (spec D12 + H4)

**H1 — Model A improves Protocol A Q1 macro mAP over 0.179: confirmed.** 0.179 → 0.313
(+0.134, CI [+0.085, +0.194]; 1.75×). Every reported metric improves, Q2 included: an
investigator now finds a correct event at rank 1 for 39 % of forensic queries (was 27 %) and in
the top 10 for 90 % (was 82 %).

**H2 — gains concentrate on appearance-defined activities; motion-defined stay near chance:
rejected as stated.** The group was fixed from the baseline alone (lift ≤ 2, Result 02 rule):
pick up, put down, stand up (A; plus sit down in B). Their AP rose almost as much as the rest
(A: +0.112 vs +0.138; B: +0.089 vs +0.111), and their lift roughly tripled. But they remain the
**weakest activities by far** — pick up / put down reach lift ≈ 2 (AP ≈ 0.2–0.25) while
appearance-led activities reach lift 6–50. Per-frame training lifted them off chance but did not
make them findable; the case for a temporal component still stands, on these numbers rather than
on "stuck at chance".

**H3 — the adapted models show an A–B gap measuring camera familiarity: weak support.** After
LoRA, Q1 mAP is 0.313 (A) vs 0.280 (B), a 0.033 gap where the frozen model had 0.005. The gain on
unseen cameras is ~80 % of the gain on seen cameras (+0.107 vs +0.134). The two gains' CIs
overlap heavily and each protocol has its own eval corpus (A: 3,726 instances / 24 cameras;
B: 3,938 / 8 cameras) and its own training set (23 vs 16 cameras), so with one seed the gap
cannot be attributed to camera familiarity alone. The main reading: **most of the adaptation
transfers to cameras the model has never seen.**

**H4 — middle-weighted sampling beats uniform on validation, gain concentrated on activities with
lead-in/lead-out (e.g. `hand_interacts_with_person`): rejected.** On validation, A favours middle
(+0.019) but almost all of that is one activity (`vehicle_turns_left` +0.231; without it the mean
is +0.004), and `hand_interacts_with_person` moved the *other* way (−0.069). B is a tie (−0.002).
On eval (reported only) the variants are indistinguishable: A −0.001 [−0.012, +0.009],
B +0.005 [−0.001, +0.013]. Edge-frame weighting makes no measurable difference here.

## Other findings

1. **Biggest gains are on people-interaction and object-carrying activities the frozen model
   barely separated:** `person_talks_to_person` 0.238 → 0.744, `person_carries_heavy_object`
   0.192 → 0.552, `person_texts_on_phone` 0.311 → 0.718. The same activities gain on unseen
   cameras (B), so this is not camera memorisation. The training captions are disjoint from the
   eval phrasings (no exact match, same-activity similarity < 0.80), so it is not query leakage.
2. **Validation selection is noisy at this size** (739 / 586 val instances, 15 / 17 activities
   with ≥ 5 val instances). The A choice rested mostly on one activity, and the B choice on a
   0.002 margin; on eval both pairs tie. The selected models are therefore representative of the
   method, not a lucky pick — and the comparison between sampling variants is inconclusive.
3. **Training is short and cheap.** Best epochs 10–28 (~1,000–2,700 steps); validation gains are
   largest in the first ~6 epochs.

## Limitations

- One seed per run; no variance across training seeds. CIs above are over activities, not seeds.
- Validation is small (see finding 2), so the best epoch is a noisy pick.
- Model revision is "unpinned" (transformers 5 does not expose the commit hash through the
  current loader); the exact SigLIP 2 weights should be pinned before the final report.
- Per-frame training only: the model cannot see motion, by design (that is the next stage).

## What this means for the next stage

The adapted encoder is the new first-stage retriever: Protocol A Q2 S@10 = 0.90 at zero VLM cost.
The remaining errors are concentrated in motion-defined activities (pick up, put down, stand up,
sit down, transfer), which is where a VLM teacher / temporal re-ranker over the top-K candidates
has the most to add. The recall-vs-cost curve now has two zero-VLM points (frozen, adapted).
