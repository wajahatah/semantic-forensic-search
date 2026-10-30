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
  store/shards.py             embedding shard format
  store/crops.py              packed crop store (tight = native PNG, context = JPEG)
  eval/queries.py             Q1 category + Q2 forensic query sets
  eval/metrics.py             AP, P@K, R-precision, Recall@K, MRR, Success@K
  eval/retrieval.py           frame -> instance scoring (mean of top-m)
  eval/baseline.py            retrieval evaluation of any embeddings file (local)
queries/phrasings.csv         reviewed natural-language query phrasings (3 per activity)
tests/                        pytest suite (`pytest`; real-model tests: `pytest -m slow`)
colab/
  sfs_bundle.py               GENERATED one-cell copy of src/sfs for Colab
  RUN_EMBEDDING.md            Colab runbook: manifest v2 + embedding job
docs/EXPERIMENT_LOG.md        every experiment, result and decision, with its basis
docs/results/                 result write-ups
docs/planning/                specs (spec-crops-queries-baseline.md = current build)
docs/learning/                concepts guide
scripts/
  build_colab_bundle.py       regenerates colab/sfs_bundle.py
  colab_meva_setup.py         corpus + annotation acquisition (Colab -> Drive)
  colab_build_manifest.py     v1 manifest (superseded by sfs.data.manifest)
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

## Status

Embeddings done (SigLIP 2 + tight crops selected on validation, `docs/results/01`). Crop store,
query set and baseline evaluation implemented (`spec-crops-queries-baseline.md`); awaiting the
Colab crop run and review of `queries/phrasings.csv` before the baseline is run.
