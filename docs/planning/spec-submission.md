# Spec — Submission package, CLIP baseline, 6-slide presentation

**Date:** 2026-11-17 · **Status:** approved 2026-11-20 with the recommended answers to D1–D4 (D-069); D5 done (D-067) · **Owner's answers this spec rests on**
(2026-11-17): ship the complete system as it runs on the laptop, in a `submission/` folder holding the
models, the scripts, the Docker files, a guide, the presentation, the report, a test-videos folder and a MEVA
folder that can be deleted or moved if the professor does not want MEVA; **move** the MEVA data into it, do not
duplicate it; the professor runs it with Python on GPU or CPU; delivery by a Google Drive link; run CLIP now; the
presentation is 6 slides (owner's outline, §3). The report waits for the professor's template.

---

## 1. Submission folder

### 1.1 What it is for

The professor can (a) read the report and the slides, (b) watch the recorded demo, (c) install and run the
same dashboard on his own machine — upload his own videos (or the ones we give) and search them, and, if the
MEVA folder is kept, search the 3,726 MEVA clips at all three levels exactly as on the laptop.

### 1.2 Layout (target, inside the project folder; git-ignored, never pushed)

```
submission/
  README.md                    the guide: what is here, install, run (GPU / CPU / Docker), test, troubleshooting
  code/                        everything needed to run (and to read how it was trained):
    sfs/                       the Python package (copy of src/sfs, ~1 MB)
    web/dist/                  the built dashboard front end (no Node needed)
    queries/phrasings.csv      the frozen query phrasings (the dashboard's examples)
    requirements.txt           pinned for Python 3.12 (CPU and CUDA install lines in the guide)
    run.bat / run.sh           start the dashboard with the paths below
    Dockerfile, compose.yaml, docker/   the Docker way (NVIDIA GPU only)
  models/                      MOVED from the project, junctions left behind (§1.4)
    siglip2/                   SigLIP 2 base, plain files (data/hf-cache, 1.5 GB)
    lora_adapter/              our LoRA adapter (runs/lora/A-middle/best, ~5 MB)
    yolo/yolov8m.pt            the detector (data/models/yolo, 50 MB)
    qwen3-vl-2b/               the Verified-level VLM (data/models/Qwen3-VL-2B-Instruct, 4.0 GB)
  meva/                        MOVED; delete this folder to ship without MEVA (§1.5)
    index/                     the MEVA search index + temporal transformer + previews (data/demo/A, 1.5 GB)
    crops/                     the crop store for Verified and frame details (data/crops_v2, 14 GB)
    example_videos/            121 short MEVA example clips (data/meva/examples/videos, 0.9 GB) — real MEVA
                               footage the professor can upload as test videos
  test_videos/                 the garage CCTV clip (copied from data/alerts, 8 MB) + any videos the owner adds
  report/                      final report (after the template arrives)
  presentation/                the 6-slide deck (§3)
  demo_video/                  the owner's screen recording
```

Total ≈ 22 GB with MEVA, ≈ 5.6 GB without it. **"MEVA videos"** is read as: the MEVA index, its crop store and
the 121 MEVA example clips. The raw MEVA videos (~70 GB) are on Drive only, the dashboard does not use them,
and they are not shipped. **The owner to confirm.**

### 1.3 Code changes (small, tested)

- `sfs.web.server`: a `--vlm` path option (today the VLM path is fixed to `data/models/Qwen3-VL-2B-Instruct`).
  Default unchanged.
- SigLIP 2 location: the server already follows `HF_HOME`; `run.bat`/`run.sh` set it to `models/siglip2`
  (the plain-file copy `prepare_docker_models.py` made). Nothing else changes.
- **CPU path, checked, not assumed:** with no GPU the encoders run in float32 and Verified is hidden (the 2B VLM
  on a CPU takes minutes per clip); the guide says so. Measured on this laptop with the GPU hidden
  (`CUDA_VISIBLE_DEVICES=""`): server start, one Fast/Standard search, one 15 s upload — times go in the guide.
- `.dockerignore` gains `submission` (otherwise a Docker build would send 22 GB as build context);
  `.gitignore` gains `/submission/`.

### 1.4 Moving without breaking the laptop (owner: "move, don't duplicate")

The dashboard, the Docker setup and the tests read the data from `data/…` and `runs/…`. Each moved folder gets a
**directory junction** at its old place (Windows `mklink /J`, no admin rights needed), so everything on the laptop
keeps working and nothing is stored twice. Checked after the move:
- conda dashboard starts and answers the 137 parity queries identically to `runs/docker/laptop.json`;
- the Docker container starts and its mounts see the data through the junctions (if Docker Desktop cannot follow a
  junction, `compose.yaml` is pointed at the `submission/` paths instead — reported, not hidden);
- the full test suite passes.

Deleting `submission/meva/` later leaves dangling junctions: the laptop dashboard then has no MEVA (it says so at
start) until the folder is moved back. The guide's last section explains this to the owner.

### 1.5 Without MEVA

If `meva/` is removed, the dashboard must still run for uploaded videos. Today alert scores are standardised with
MEVA's score statistics, so the server needs the MEVA index. Change: the index's statistics (a few kB) are saved to
`models/meva_stats.json`; with no MEVA index the server uses that file, searches only the uploaded videos, and the
MEVA collection is hidden. Test: the same alert scores with the full index and with the stats file only.

### 1.6 Delivery

The owner uploads `submission/` to Google Drive and shares the link — **at the office** (22 GB; the home connection
rule). The guide states the size of each folder so the professor can download only what he needs.

---

## 2. CLIP baseline (owner: "Run CLIP now")

- **Model:** `openai/clip-vit-base-patch16` — same ViT-B/16 at 224 px as our SigLIP 2, so the comparison is fair.
  ~600 MB download through the library (pinned revision recorded). **Downloaded at the office.** Runs on the laptop
  GPU; no server.
- **Measured exactly as the frozen baselines were**, from the local crop store, tight view (the selected view):
  1. validation, Result 01 protocol: per-activity AUC and P@10, prompt `"a surveillance camera photo of a <activity
     words>"`, instance score = mean of the top-3 frames — next to SigLIP and SigLIP 2;
  2. eval, Result 02 protocol: Q1 mAP and Q2 Success@1 with the 66 frozen phrasings, Protocols A and B — next to
     the frozen SigLIP 2 baseline (0.179 / 0.174).
- **No decision depends on it** (the encoder was chosen on 2026-10-23); it is a reported comparison, so reading the
  eval split once for it is allowed. Rule fixed now: whatever CLIP scores, the system does not change.
- **Code:** `scripts/clip_baseline.py` (embeds the crops with CLIP, then the existing evaluation functions); a unit
  test on a tiny random CLIP built from a config (no download). Result appended to Result 02 as a dated row and to
  `report/numbers.json`. Time: ~2–3 h.

## 3. Presentation — 6 slides (owner's outline)

Built with the same builder style as the 12-slide deck (new file `report/Ahmed_Wajahat_DLPresentation_6.pptx`;
the 12-slide deck is kept), every number from `report/numbers.json`, speaker notes with timings.

1. **Introduction** — you (Wajahat Ahmed, CS-141/2026, CS-5138 Deep Learning, NED) and the project in one line.
2. **Project description** — the problem (search hours of CCTV by a sentence), why VLM calls are the cost, MEVA.
3. **Pipeline + how the training pipeline was built** — offline (crops → LoRA → embeddings → temporal
   transformer) and online (query → Fast → Standard → Verified); the training pipeline on the same slide.
4. **Model architecture, and why** — SigLIP 2 + LoRA (what is trained), the temporal transformer (frame tokens +
   query token), the VLM on the top 10; compared with the alternatives we tried (full frame, frozen, VLM on all).
5. **Results and comparison** — zero-shot CLIP vs SigLIP vs SigLIP 2; frozen → LoRA → temporal transformer (Q1
   mAP, Q2 Success@1, both protocols); what each VLM contributed.
6. **The VLM part + thanks + questions** — the gate, 7 VLMs, only the large ones help at the top 10, the cost
   curve's message; thanks to the professor and the audience; questions.

Slide 5 waits for the CLIP numbers; slides 1–4 and 6 can be built first.

**§1 built 2026-11-23 (D-077):** `python scripts/build_submission.py assemble | move | check`; guide and run
scripts in `packaging/submission/`. Differences from §1.2/§1.5: the MEVA index is `models/index` (needed in both
modes) and its previews `meva/previews`; the without-MEVA mode keeps the index instead of a `meva_stats.json`.

**Built 2026-11-20 (D-070):** `scripts/deck6_build.js` → `report/Ahmed_Wajahat_DLPresentation_6.pptx`, notes timed
for about 7 minutes; theme, layouts and helpers moved to `scripts/deck_common.js`, which the 12-slide builder now
uses too (its rebuilt deck is identical to the committed one apart from section ids and timestamps). Slide 5 reads
the new `clip` block of `report/numbers.json`.

## 4. Before / after

| | Before | After |
|---|---|---|
| Hand-in | report + 12-slide deck in `report/`; data spread over `data/`, `runs/` | one `submission/` folder with guide, code, models, MEVA, test videos, report, 6-slide deck, demo video |
| Laptop runs | data read from `data/…` | same paths, now junctions into `submission/` |
| Dashboard without MEVA | not possible | runs on uploaded videos with `models/meva_stats.json` |
| Encoder comparison | SigLIP vs SigLIP 2 | + CLIP ViT-B/16, validation and eval |
| Git | pushed `5be5f35` | code/doc changes stay local until the owner asks to commit/push; `submission/` never in git |

## 5. Decisions for the owner

- **D1 · "MEVA videos" = index + crop store + 121 example clips (not the 70 GB raw videos on Drive).** Recommended.
- **D2 · Code in the submission:** the whole `sfs` package (runtime + training + evaluation, ~1 MB) so the
  professor can read how the models were trained (recommended), or only what the dashboard needs.
- **D3 · Junctions at the old paths** so the laptop keeps working (recommended), or point the repo's defaults at
  `submission/` directly.
- **D4 · Without-MEVA mode (§1.5)** — build it now (recommended; ~1 h) or only if the professor declines MEVA.
- **D5 · CLIP download** at the office (recommended) or on the home connection (~600 MB, 6–20 min).
