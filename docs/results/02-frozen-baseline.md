# Result 02 — Frozen baseline (SigLIP 2, no training), eval split

**Date:** 2026-10-27 · **Spec:** `spec-crops-queries-baseline.md` Part 3 · **Log:** D-018
**Inputs:** `embeddings_v2_siglip2.npz` (encoder id `8356d9b38d31`, pipeline v2), `manifest_v2.csv`,
`queries/phrasings.csv` (approved and frozen before this run, D-017; sha256 prefix `bfd6456afafa6bb7` at run time). After the run only the `# Status` comment line was updated (file was locked by Excel during the run); the 66 phrasings are unchanged — current sha256 prefix `331bb6bba25a73bd`.
**Raw output:** `docs/results/raw/02_baseline.json`, `docs/results/raw/02_baseline_queries.csv`.
**First eval-split numbers in the project.** Reported only — no decision was taken from them.

## Setup

- **Query activities:** 22 (≥ 20 eval instances, D-012). 3 phrasings each = 66 phrasings.
- **Q1 category** ("find every instance"): 66 queries per protocol, searched over the whole eval split.
- **Q2 forensic** ("find it on camera C in hour H"): 380 events (Protocol A) / 222 (Protocol B),
  × 3 phrasings. Median 3 relevant instances in a window of ~56.
- **Instance score:** mean of top-3 frame similarities. **Random:** mean of 20 seeds.

## Headline (macro over activities)

| | Q1 mAP | Q1 P@10 | Q2 R@1 | Q2 R@10 | Q2 MRR |
|---|---|---|---|---|---|
| **A — tight (baseline)** | **0.179** | **0.259** | **0.101** | **0.510** | **0.446** |
| A — full frame | 0.096 | 0.127 | 0.043 | 0.299 | 0.266 |
| A — random | 0.047 | 0.044 | 0.033 | 0.266 | 0.252 |
| **B — tight (baseline)** | **0.174** | **0.274** | **0.099** | **0.496** | **0.460** |
| B — full frame | 0.088 | 0.097 | 0.053 | 0.299 | 0.242 |
| B — random | 0.046 | 0.044 | 0.036 | 0.275 | 0.243 |

**Cost (the first point on the accuracy-vs-cost curve):** 1 text-encoder call per query, 0 VLM
calls; ~7–26 ms text encoding + ~30 ms to score and rank ~3,800 instances on the RTX 4060.

## Findings

1. **Cropping holds up on eval.** Tight crops roughly double every metric over full frames
   (Q1 mAP 0.179 vs 0.096), confirming the validation-based choice (D-011) on unseen data.
2. **Well above chance, far from good.** Q1 mAP is 3.8× random, but 0.18 means most of each
   ranked list is wrong. This is the gap the trained components must close.
3. **Protocol A ≈ Protocol B (0.179 vs 0.174) — as expected for a frozen model.** It has never
   trained on any of these cameras, so camera familiarity cannot help it. This confirms the two
   protocols are comparable; any A–B gap that appears **after** LoRA will measure how much the
   trained model relies on having seen the cameras.
4. **Per-activity: what the frozen model can and cannot find** (Protocol A, tight; *lift* =
   AP ÷ share of the activity in the corpus, so 1.0 = chance):

   | Activity | AP | lift | P@10 |
   |---|---|---|---|
   | person_talks_on_phone | 0.365 | 20.6 | 0.87 |
   | person_texts_on_phone | 0.311 | 5.3 | 0.67 |
   | person_opens_facility_door | 0.276 | 6.0 | 0.37 |
   | vehicle_turns_left | 0.262 | 12.5 | 0.40 |
   | … | | | |
   | person_picks_up_object | 0.097 | 1.0 | 0.07 |
   | person_puts_down_object | 0.114 | 1.1 | 0.20 |
   | person_transfers_object | 0.040 | 2.2 | 0.07 |
   | person_stands_up | 0.033 | 1.6 | 0.00 |

   **Object handling and posture changes are at chance** (lift ≈ 1–2): they are defined by
   *motion*, which single-frame embeddings do not see. Same pattern as validation (Result 01).
   Phone use and distinctive vehicle/door events are found well. Full table: raw CSV.
5. **Wording matters, but moderately.** AP differs by 0.044 on average between the three
   phrasings of the same activity.
6. **Natural phrasings beat the template used for validation.** `person_texts_on_phone` was at
   chance in Result 01 (template prompt "a surveillance camera photo of a person texts on phone",
   validation split) but has lift 5.3 here. Different splits, so this is indicative only — but it
   suggests the template prompt understated the model.

## Caveat — Q2 Recall@K is capped when a query has many relevant instances

Q2 Recall@K = fraction of *all* relevant instances found in the top K. When a window holds more
than K relevant instances, Recall@K cannot reach 1:

| Q2 relevant instances | events | R@1 | R@10 |
|---|---|---|---|
| 1 | 82 | 0.244 | 0.801 |
| 2–5 | 162 | 0.136 | 0.648 |
| ≥ 6 | 136 | 0.034 | 0.287 |

The low numbers in the last row are partly the cap, not failure. For forensic use, the natural
measure is **Success@K** (at least one correct result in the top K); MRR already reflects the
first correct hit. Success@K was **not** pre-registered for this run, so it is not reported here;
it is proposed for the next spec, defined before any further results are seen.

## Addendum 2026-11-17 — zero-shot CLIP ViT-B/16 on the eval split (D-067)

A reported comparison only (spec-submission §2; no decision depends on it, so reading the eval split once
for it is allowed). CLIP ViT-B/16 (Result 01 addendum for the set-up), tight view, the 66 frozen phrasings,
the same `sfs.eval.baseline` functions. The SigLIP 2 rows were re-run in the same job from the stored
embeddings and equal this result's raw file on every stored metric (max difference 0.0). Success@1/@10 are
shown because later results use them; they were computed for both encoders in the same run.
Raw: `docs/results/raw/02b_clip_eval.json`, `02b_clip_eval_queries.csv.gz`.

| Protocol · encoder (tight) | Q1 mAP | Q1 P@10 | Q2 S@1 | Q2 S@10 | Q2 MRR |
|---|---|---|---|---|---|
| A · SigLIP 2 | **0.179** | **0.259** | **0.273** | **0.820** | **0.447** |
| A · CLIP ViT-B/16 | 0.142 | 0.189 | 0.237 | 0.712 | 0.391 |
| B · SigLIP 2 | **0.174** | **0.274** | **0.275** | **0.849** | **0.460** |
| B · CLIP ViT-B/16 | 0.138 | 0.192 | 0.261 | 0.754 | 0.409 |

SigLIP 2 is ahead on every measure in both protocols and on 18 of the 22 activities (Q1 AP). The
largest gaps are `person_talks_on_phone` (A 0.365 vs 0.106) and `person_enters_vehicle` (A 0.187 vs
0.055); CLIP is slightly ahead on `person_texts_on_phone`, `person_talks_to_person` and `vehicle_stops`.
On validation (Result 01 addendum) the two looked close on AUC; on the natural phrasings and the larger
eval split, SigLIP 2 is clearly the better zero-shot starting point, consistent with the choice of D-011.
