# Semantic Forensic Search

Cost-bounded natural-language search over surveillance video, using a distilled
vision-language re-ranker.

Deep Learning graduate project — NED University.

---

## Problem

A surveillance installation produces thousands of automated alerts per day. When an incident is
reported after the fact, locating the relevant footage means manual review, because no index exists
over what actually happens inside the frames.

This project indexes that footage so it can be searched in plain language — *"a person taking a bag
from a parked car"* — and returns ranked clips with their camera and timestamp.

**The constraint is cost.** A vision-language model (VLM) applied to every candidate clip gives good
relevance judgements but is infeasible at archive scale, since cost grows linearly in the number of
query–clip pairs scored. The research question is:

> How much of the retrieval accuracy of an exhaustive VLM system can be retained while reducing VLM
> invocations by one to two orders of magnitude?

The headline result is a **recall-versus-cost Pareto frontier** — an axis the surveillance-retrieval
literature reports almost exclusively in terms of accuracy.

---

## Approach

Three stages, in which the expensive component is progressively removed rather than merely reduced.

| Stage | Component | Trained? |
|---|---|---|
| Index | Dual encoder (ViT), LoRA-adapted to surveillance imagery | **yes** — contrastive fine-tuning |
| Re-rank | Small cross-encoder | **yes** — distilled from a VLM teacher |
| Verify | VLM, on a small escalated subset only | no — frozen, and used mainly offline as teacher |

The VLM teacher runs **once, offline**, to generate relevance labels. At query time it is replaced
by the distilled student. That substitution is the cost reduction being measured.

See [`proposals/system_diagram.png`](proposals/system_diagram.png) for the full architecture.

---

## Dataset

**MEVA** — Multiview Extended Video with Activities (Kitware / IARPA DIVA).

| | |
|---|---|
| Licence | CC BY 4.0 |
| Video | ~328 h, 4,259 clips, multi-camera, real cameras at a physical site |
| Annotations | 13,968 KPF files, ~184 h, 37 activity types with frame-level timespans |
| Access | Public, anonymous — no account or approval form |
| Source | [mevadata.org](https://mevadata.org) · [AWS Open Data](https://registry.opendata.aws/mevadata/) |

MEVA has been used for **activity detection** (NIST TRECVID ActEV). It has **not** been used for
natural-language moment retrieval — that literature runs on DiDeMo, Charades-STA and QVHighlights.
No natural-language query benchmark exists for MEVA; this project constructs one from the existing
human activity annotations.

Data is **not committed to this repository** (see `.gitignore`). It is reproducible via the
acquisition scripts below.

---

## Repository layout

```
src/sfs/                      the package (tested locally, run on Colab via the bundle)
  data/kpf.py                 MEVA KPF annotation parsing (activities, per-frame boxes)
  data/manifest.py            manifest v2: splits (official + unseen-camera), dedupe, leakage check
  data/sampling.py            frames per activity instance (2 fps, cap 16)
  data/regions.py             tight / context / full crop views
  data/extract_crops.py       crop store job + integrity gate (Colab)
  embed/encoder.py            SigLIP and SigLIP 2 wrappers, encoder_id
  embed/run.py                resumable per-video embedding job
  embed/reembed.py            re-embed all tight crops with a trained LoRA adapter (resumable)
  store/shards.py             embedding shard format
  store/crops.py              packed crop store (tight = native PNG, context = JPEG)
  eval/queries.py             Q1 category + Q2 forensic query sets
  eval/metrics.py             AP, P@K, R-precision, Recall@K, MRR, Success@K
  eval/retrieval.py           frame -> instance scoring (mean of top-m)
  eval/baseline.py            retrieval evaluation of any embeddings file (local)
  train/data.py               packed tight-crop file, instance sampler, augmentation
  train/captions.py           training caption bank + disjointness rule vs eval phrasings
  train/loss.py               label-aware sigmoid loss
  train/lora.py               LoRA on the SigLIP 2 vision tower
  train/run.py                training loop: validation-only selection, crash-safe resume
  teacher/frames.py           an instance's tight/context crops for the VLM, in time order
  teacher/vlm.py              VLM scorer (yes/no margin); KV-cache reuse for the instruct model
  teacher/run.py              resumable VLM scoring job over a request file
  teacher/gate.py             gates G0 (laptop feasibility) and G1 (does the VLM add information?)
  teacher/feedback.py         owner review page: plain-language questions with options + the VLM's mistakes with frames
  teacher/serve.py            office-server scoring service: our VLMScorer behind HTTP (localhost + SSH tunnel, shared secret)
  teacher/remote.py           RemoteScorer: the laptop-side client, same interface as VLMScorer (retries, size limit)
  teacher/gemma.py            Gemma 4 scorer (E6; stopped)
  teacher/offload.py          CPU/GPU split for models that do not fit the GPU (exact; streamed decoder layers)
  teacher/queue.py            one-process job queue; `--watch` holds the GPU memory until a STOP file
  rerank/features.py          per-instance frame sequences: LoRA + context embeddings + box motion
  rerank/model.py             student: small transformer over the frame sequence
  rerank/train.py             student training (GT / GT+KD / KD), validation selection, resume
  eval/cascade.py             top-k re-ranking over cached scores, exact cost counting
  eval/cascade_run.py         eval requests and the recall-vs-cost report (with GT+KD / KD ablations, H5, latency)
queries/phrasings.csv         reviewed natural-language query phrasings (3 per activity) — EVAL ONLY
queries/train_captions.csv    reviewed training captions (6 per activity), disjoint from the above
tests/                        pytest suite (`pytest`; real-model tests: `pytest -m slow`)
colab/
  sfs_bundle.py               GENERATED one-cell copy of src/sfs for Colab
  RUN_EMBEDDING.md            Colab runbook: manifest v2 + embedding job
docs/EXPERIMENT_LOG.md        every experiment, result and decision, with its basis
docs/results/                 result write-ups
docs/planning/                specs (spec-vlm-followups.md = latest; E4 open)
docs/learning/                concepts guide
scripts/
  build_colab_bundle.py       regenerates colab/sfs_bundle.py
  colab_meva_setup.py         corpus + annotation acquisition (Colab -> Drive)
  colab_build_manifest.py     v1 manifest (superseded by sfs.data.manifest)
  lora_val_check.py           re-embedded stores reproduce the recorded validation mAP
  lora_report.py              Result 03 tables (baseline vs LoRA, bootstrap CIs, H2/H4)
  fetch_vlm_weights.py        VLM weights from the ModelScope mirror, resumable, sha256-checked
  cascade_report.py           Result 04 figure (SVG, no plotting dependency), tables, per-activity CSV
  g1_frames_diagnostic.py     Result 05 (E3): 8 vs 16 frames, VLM AUC on long / short clips (validation)
proposals/                    submitted proposal + system diagram
```

---

## Getting the data

Annotations are hosted on `gitlab.kitware.com`, which is not reachable from every network. The
acquisition scripts therefore run in Google Colab, which also avoids pulling tens of gigabytes over
a home connection.

1. Open a Colab notebook, `Runtime > Change runtime type > GPU`
2. Run `python scripts/colab_meva_setup.py` locally to print the cells, then paste each into Colab
3. Run the cells from `scripts/colab_build_manifest.py` to build `manifest.csv`

**Do not download the full corpus.** Video is read once to compute embeddings; after that the
pipeline only touches vectors, and the entire retrieval index is a few megabytes. A working subset
of a few site-hours is sufficient.

---

## Local environment

Python 3.12, CUDA-enabled PyTorch (developed on an RTX 4060 Laptop, 8 GB).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Heavy jobs (embedding the corpus) run on Colab with data mounted from Google Drive. Only embedding
shards and the manifest are brought to the local machine.

---

## LoRA fine-tuning (local)

Spec: `docs/planning/spec-lora-finetune.md`. Runs on the laptop GPU (peak 5 GB at batch 64).

```bash
# 1. once: pack the tight crops of the verified crop store (8.5 GB, ~1 min)
python -c "from sfs.train.data import build_packed; build_packed('data/crops_v2', 'data/packed_tight')"

# 2. train all four runs (protocol A/B x uniform/middle-weighted frame sampling)
python -m sfs.train.run --runs A-uniform,A-middle,B-uniform,B-middle --out runs/lora     --packed data/packed_tight --manifest manifest_v2_drive/manifest_v2.csv     --captions queries/train_captions.csv --phrasings queries/phrasings.csv

# 3. re-embed with the selected adapter, then evaluate on that protocol only
python -m sfs.embed.reembed --packed data/packed_tight --baseline manifest_v2_drive/embeddings_v2_siglip2.npz     --adapter runs/lora/A-middle/best/adapter.pt --out data/embeddings_lora/A-middle
python -m sfs.eval.baseline --manifest manifest_v2_drive/manifest_v2.csv     --embeddings data/embeddings_lora/A-middle/embeddings.npz --phrasings queries/phrasings.csv     --views tight --protocols A --out runs/lora/eval/A-middle

# 4. result tables (docs/results/03-lora.md)
python scripts/lora_val_check.py A-middle A-uniform B-uniform B-middle
python scripts/lora_report.py
```

**Interruptions are safe.** After every epoch the full training state is written atomically to
`last.pt` (previous one kept as `prev.pt`). If the laptop shuts down or the job is stopped, run the
same command again: it resumes after the last finished epoch (at most ~1 min lost), skips runs that
already finished, and refuses to resume if any training setting changed. The best validation epoch
is kept in `best/adapter.pt`; `train_log.csv` has one row per epoch (epoch 0 = untrained model).
Re-embedding resumes per 4,096-crop part. Check before trusting re-embedded vectors:
`python -m sfs.embed.reembed ... --check-baseline 2000` (adapter off must reproduce the baseline).

---

## Re-ranking cascade: VLM teacher, student, cost curve (local)

Spec: `docs/planning/spec-rerank-cascade.md`. Laptop GPU (8 GB). Run every command with
`HF_HUB_OFFLINE=1` (Hugging Face rate-limits this connection; all models are local).

```bash
# 0. weights (once; resumable, sha256-checked): data/models/<model>/
python scripts/fetch_vlm_weights.py Qwen/Qwen3-VL-2B-Instruct Qwen/Qwen3-VL-Reranker-2B

# 1. student inputs (once per LoRA run) and pair pools (once per protocol)
python -m sfs.rerank.features --manifest manifest_v2_drive/manifest_v2.csv     --lora data/embeddings_lora/A-middle/embeddings.npz --frozen manifest_v2_drive/embeddings_v2_siglip2.npz     --out data/rerank/features_A-middle.npz
python -m sfs.rerank.train pool --protocol A --manifest manifest_v2_drive/manifest_v2.csv     --lora-emb data/embeddings_lora/A-middle/embeddings.npz --out data/rerank/pool_A.csv

# 2. gates: G0 per model, then G1 on validation (score the requests with each model x view)
python -m sfs.teacher.gate g0 --model data/models/Qwen3-VL-2B-Instruct --kind instruct --out runs/cascade/g0_instruct.json
python -m sfs.teacher.gate g1-requests --out data/teacher/requests_g1.csv
python -m sfs.teacher.run --model data/models/Qwen3-VL-2B-Instruct --kind instruct --view context     --requests data/teacher/requests_g1.csv --subset val --out data/teacher/g1/instruct-context
python -m sfs.teacher.gate g1 --scores data/teacher/g1/*/scores.npz --out runs/cascade/g1.json

# 3. teacher labels for the pools, then the students (GT, GT+KD, KD) per protocol
python -m sfs.teacher.run ... --requests data/teacher/requests_trainval.csv --subset trainval --out data/teacher/trainval/<teacher>
python -m sfs.rerank.train train --protocol A --lora-run A-middle --manifest manifest_v2_drive/manifest_v2.csv     --lora-emb data/embeddings_lora/A-middle/embeddings.npz --features data/rerank/features_A-middle.npz     --pool data/rerank/pool_A.csv --teacher data/teacher/trainval/<teacher>/scores.npz --out runs/student
# on a machine without the SigLIP 2 weights (office server): export the text embeddings once, then add
#   --text-emb data/rerank/text_emb_siglip2.npz to the train command (bit-identical results)
python scripts/export_text_embeddings.py --out data/rerank/text_emb_siglip2.npz

# 4. eval requests, VLM scoring (subset eval / val), report
python -m sfs.eval.cascade_run requests --out data/teacher
python -m sfs.teacher.run ... --requests data/teacher/requests_eval_<model>.csv --subset eval --out data/teacher/eval/<model>-<view>
python -m sfs.teacher.run ... --requests data/teacher/requests_val_student.csv --subset val --out data/teacher/val_student/<model>-<view>
python -m sfs.eval.cascade_run report --out runs/cascade/04_cascade
python scripts/cascade_report.py --report runs/cascade/04_cascade --fig docs/results/04_cascade_curve.svg \r
    --tables runs/cascade/04_tables.md --per-activity docs/results/raw/04_cascade_per_activity.csv

# 5. follow-ups (spec-vlm-followups.md). E5: Instruct on every Q1 candidate; only pairs missing from
#    the Result 04 store are scored, into their own store, and merged by the report
python -m sfs.eval.cascade_run requests --models instruct --everything-q1 instruct \r
    --exclude-scored instruct=data/teacher/eval/instruct-tight/scores.npz --out data/teacher/q1all
python -m sfs.teacher.run ... --requests data/teacher/q1all/requests_eval_instruct_extra.csv --subset eval --out data/teacher/eval/instruct-tight-q1all
python -m sfs.eval.cascade_run report --everything-q1 instruct --extra-scores instruct=data/teacher/eval/instruct-tight-q1all \r
    --fixed-choices runs/cascade/04_cascade.json --out runs/cascade/05_q1all
# E1: merged-turn view (extra view; the frozen 22-activity set stays the main record)
python -m sfs.eval.cascade_run requests --merge vehicle_turns_left,vehicle_turns_right=vehicle_turns \r
    --phrasings queries/phrasings_merged_turns.csv --everything-q1 instruct \r
    --exclude-scored instruct=<each instruct eval store>,reranker=data/teacher/eval/reranker-tight/scores.npz --out data/teacher/turns
python -m sfs.eval.cascade_run report --merge ... --phrasings queries/phrasings_merged_turns.csv --everything-q1 instruct \r
    --extra-scores instruct=<dir>,reranker=<dir> --fixed-choices runs/cascade/04_cascade.json --out runs/cascade/05_turns
# E2: G1 on every validation clip (--k 0); only pairs missing from the top-50 store are scored
python -m sfs.teacher.gate g1-requests --k 0 --exclude-scored data/teacher/g1/instruct-tight/scores.npz --out data/teacher/requests_g1_wholeval.csv
python -m sfs.teacher.run ... --requests data/teacher/requests_g1_wholeval.csv --subset val --out data/teacher/g1_wholeval/<model>-<view>
python -m sfs.teacher.gate g1 --k 0 --scores data/teacher/g1_wholeval/<cfg>/scores.npz+data/teacher/g1/<cfg>/scores.npz ... --out runs/cascade/g1_wholeval.json
# E3: 16 frames (G0 checks the shortcut against the bf16 noise floor), then G1 at 16 frames
python -m sfs.teacher.gate g0 --model data/models/Qwen3-VL-2B-Instruct --kind instruct --frames 16 --out runs/cascade/g0_instruct_16f.json
python -m sfs.teacher.run ... --frames 16 --requests data/teacher/requests_g1.csv --subset val --out data/teacher/g1_16f/<model>-tight
python -m sfs.teacher.gate g1 --scores data/teacher/g1_16f/instruct-tight/scores.npz data/teacher/g1_16f/reranker-tight/scores.npz     --out runs/cascade/g1_16f.json
python scripts/g1_frames_diagnostic.py --out runs/cascade/g1_16f_diagnostic.json
# tables / per-activity for E5 and E1 (Result 05)
python scripts/cascade_report.py --report runs/cascade/05_turns --fig runs/cascade/05_turns_curve.svg     --tables runs/cascade/05_turns_tables.md --per-activity runs/cascade/05_turns_per_activity.csv
```

### Office server (E4, spec-vlm-followups D-029)

A bigger VLM (Qwen3-VL-8B-Instruct-FP8) runs on the office server `lt-ai-server` (16 GB GPU, shared
with the team's Ollama). It is served by our own scorer code, so frames, prompt and score are the
same as the 2B runs. The service listens on the server's localhost only; the laptop reaches it
through an SSH tunnel.

```bash
# one-time, by the owner: the shared secret, created on the laptop and copied to the server
# (never printed; both sides read ~/.config/sfs/secret, or the SFS_SERVE_SECRET variable)
# server, one-time set-up (private env ~/envs/sfs, pinned weights, integrity check):
#   ~/sfs/setup.sh ; ~/envs/sfs/bin/python ~/sfs/verify_weights.py
# server: start the service (waits until 14 GB of GPU memory is free, then loads)
PYTHONNOUSERSITE=1 PYTHONPATH=~/sfs/code/src ~/envs/sfs/bin/python -m sfs.teacher.serve \
    --model ~/sfs/models/Qwen3-VL-8B-Instruct-FP8 --kind instruct --port 8765 --min-free-gb 14
# laptop: tunnel, then the usual commands with --remote instead of --model
ssh -N -L 8765:127.0.0.1:8765 lt_internal_server
python -m sfs.teacher.gate remote-check --model data/models/Qwen3-VL-2B-Instruct --remote http://127.0.0.1:8765 \
    --out runs/cascade/e4_remote_check.json            # with the 2B served: the API adds no error
python -m sfs.teacher.gate g0 --remote http://127.0.0.1:8765 --kind instruct --max-vram-gb 15 --out runs/cascade/g0_8b.json
python -m sfs.teacher.run --remote http://127.0.0.1:8765 --kind instruct --view tight \
    --requests data/teacher/requests_g1.csv --subset val --out data/teacher/g1_8b/instruct-tight
python -m sfs.teacher.gate g1 --scores data/teacher/g1_8b/instruct-tight/scores.npz --out runs/cascade/g1_8b.json
# shared GPU: hold the memory a long job needs (other users get what is left; their services are not touched)
python -m sfs.teacher.run --model ... --reserve-gb 14.5 ...
```

Every long job resumes when re-run with the same command (VLM scoring per 100-clip part; student
training per epoch, refused if a setting changed). Training code cannot open eval-split VLM scores.

---

## Status

Embeddings done (SigLIP 2 + tight crops, `docs/results/01`). Frozen baseline on the eval split
done (`docs/results/02`: Q1 mAP 0.179, Q2 R@10 0.510). LoRA fine-tuning done
(`docs/results/03`: Q1 mAP 0.179 → 0.313 on Protocol A, 0.174 → 0.280 on unseen cameras; Q2 S@10
0.82 → 0.90). Re-ranking cascade done (`docs/results/04`): a temporal student re-ranker over
embeddings + box motion lifts Q2 S@1 0.39 → 0.53 (A) / 0.37 → 0.52 (B) and Q1 mAP to 0.476 / 0.442 at
0 VLM calls; a 2B VLM on the top 5–10 adds ≤ +0.03 S@1; the VLM gate G1 failed and distillation from
the 2B teacher did not help. Follow-ups (`docs/results/05`, draft): the 2B VLM on every Q1 candidate
reaches only 0.207 / 0.212 mAP at ~10 min per query; one merged "vehicle turns" class is found well
(LoRA AP 0.57 / 0.62), so the turn direction was the failure; 16 frames do not rescue the 2B VLM.
Open: a bigger VLM served from the office server (E4).
