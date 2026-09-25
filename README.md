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
docs/planning/
  project-selection.md        dataset evaluation, candidate comparison, novelty verification
  spec-retrieval-system.md    system spec: alert schema, storage design, query flow
scripts/
  colab_meva_setup.py         corpus + annotation acquisition (Colab -> Drive)
  colab_build_manifest.py     annotation inventory, video alignment, query-set manifest
proposals/
  Ahmed_Wajahat_DLProposal.*  submitted proposal
  system_diagram.png          system architecture diagram
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

## Status

Research and design phase. Proposal submitted; implementation not yet begun.
