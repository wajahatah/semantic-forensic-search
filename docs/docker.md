# Running the dashboard in Docker

Spec: `docs/planning/spec-docker.md` (D-063). Files: `Dockerfile`, `compose.yaml`, `.dockerignore`,
`docker/requirements.txt` + `docker/constraints.txt` (+ `requirements-test.txt` for the test stage),
`scripts/prepare_docker_models.py`, `scripts/docker_parity.py`. Tests: `tests/test_docker.py`.

**Status (2026-11-08):** built at the office and checked (results in the table below and in
`runs/docker/`). The container serves the same 3,726 MEVA clips with the GPU and the VLM, is healthy
91–116 s after `up`, keeps its alerts across `down`/`up`, and passes the unit suite in Linux. Two checks
**fail their pre-fixed rule** and are explained to the last digit by the platform builds of two libraries,
not by our code: Standard scores differ from the laptop by up to 5.4e-3 (rule: 1e-3) because the float16
SigLIP 2 text tower gives slightly different embeddings on the Linux CUDA libraries, and the same video
gives 21 alerts instead of 22 because the Linux OpenCV wheel decodes the frames to slightly different
pixels. Rankings are identical on every query. Whether to accept this or change the two components is the
owner's decision (experiment log D-065; see "Why the two platforms differ" below).

The conda way (`python -m sfs.web.server`, `docs/demo.md`) keeps working unchanged. The two are
alternatives: both use port 8000 and the laptop's 8 GB GPU holds one copy of the models (5.7 GB).

## What is in the image, and what is not

- **In the image:** `src/`, the built front end (`web/dist`, built by the Node stage), the Python packages
  at the exact versions of the tested `dlproj` env, `queries/phrasings.csv`. 12.3 GB on disk (4.2 GB of
  compressed content), mostly torch and its CUDA libraries.
- **Mounted read-only from this folder** (see `compose.yaml`): the MEVA demo bundle `data/demo/A`, the
  crop store `data/crops_v2`, the LoRA adapter, `data/models/yolo`, `data/models/Qwen3-VL-2B-Instruct`,
  and `data/hf-cache` (below). Nothing large is copied into the image, and nothing is written to these
  folders.
- **Your alerts** live in the Docker volume `alerts` (spec D2), not in `data/alerts`: alerts made in
  Docker are not visible to the conda run and the other way round. They survive `docker compose down`;
  `docker compose down -v` deletes them.
- **Offline:** `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `YOLO_OFFLINE=1`. The container never
  downloads anything after the build.
- **Non-root:** the server runs as user `sfs` (uid 1000).

## First time

1. **Docker Desktop** must be running (WSL 2 backend; GPU support is on by default with an NVIDIA
   driver installed on Windows). Check: `docker info` answers, and
   `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi` shows the RTX 4060.
2. **SigLIP 2 for the container** (once, ~1.5 GB copied locally, nothing downloaded):

   ```bash
   python scripts/prepare_docker_models.py
   ```

   The laptop's Hugging Face cache stores the weights as Windows symbolic links that a Linux container
   cannot follow. The script writes real files in the same layout under `data/hf-cache`, checking every
   file's hash against its blob name, and records the commit it copied in `data/hf-cache/prepared.json`.
   That commit (`75de2d55…`) is the SigLIP 2 revision the container runs. `--check` verifies a copy.
3. **Build and run** (the first build took about 9 minutes at the office, 414 s of it the pip step; it is
   cached afterwards):

   ```bash
   docker compose up            # builds on the first run; open http://127.0.0.1:8000 once "open http://…" appears
   ```

   The models take about a minute to load; `docker compose ps` shows `healthy` when `/api/status` answers.
   `docker compose down` stops it.

## Checks after the first build (spec §5) — results of 2026-11-08

| Check | Command | Result |
|---|---|---|
| Compose file valid | `docker compose config -q` | **pass** |
| Unit suite inside Linux | `docker build --target test -t sfs-test .` | **pass** — 190 passed, 18 skipped (VLM / Gemma processor files and raw result files are not in the image; 1 needs a GPU), 4 deselected, 24 s; `pip check` clean in both stages |
| Search parity (laptop → container) | `python scripts/docker_parity.py record --out runs/docker/laptop.json` with the conda server; then the container and `… --out runs/docker/docker.json`; then `python scripts/docker_parity.py compare runs/docker/laptop.json runs/docker/docker.json` | **rankings pass, Standard scores fail the 1e-3 bar.** 137 queries: Success@1 and Success@10 identical on every one; 11 top-10 swaps, all across gaps < 1e-3; Verified checked the same 5 sets. Fast scores within 5.6e-5. Standard scores up to 5.4e-3 apart on 36 of 66 queries (median 1.1e-3). Cause measured below. Median seconds Fast 0.068 → 0.090, Standard 0.260 → 0.268, Verified 4.1 → 7.9 (crops read through the Windows file share) |
| Same alerts from the same video | the owner's garage clip (`data/alerts/media/1/video.mp4`, 152 s, already in the laptop store) uploaded to the container; `/api/videos/{id}/alerts` compared | **fail: 22 on the laptop, 21 in the container.** Same file (sha256), same frame count, same detector weights. The 9 person alerts and the two opening alerts are identical; car alerts start 1–4 frames apart (0.08–0.16 s), one borderline car track (66.7–71.7 s) is confirmed only on the laptop, one laptop track is two in the container; peak confidences differ by ≤ 0.017. Cause measured below |
| Alerts survive a restart | `docker compose down && docker compose up`, alerts still listed | **pass** — 21 alerts, identical before and after; server ready 91 s after `up` |
| Health + offline | `docker compose ps`; container log | **pass** — `healthy`; the log shows only model loading, no download attempt (offline variables set) |
| Image size and digests | `docker image ls sfs-dashboard`, `docker image inspect` | `sfs-dashboard:0.1` id `77659e1a0b46`, 12.3 GB on disk / 4.17 GB content; `sfs-test` id `3fec1b1e9c87`, 12.5 GB / 4.22 GB. Base images pinned by digest in the build: `node:24.12.0-alpine@sha256:c921b97d…`, `python:3.12.15-slim-bookworm@sha256:6a0e06f5…` |

The recordings behind the table are in `runs/docker/`: `laptop.json`, `docker.json`, `compare.json`
(parity), `laptop_alerts.json`, `docker_alerts.json`, `docker_alerts_rerun.json` (alerts), `diag/`
(the platform diagnostic below).

**Parity rule** (fixed before any measurement, `scripts/docker_parity.py`): all 66 frozen phrasings at
Fast and Standard over MEVA (top 50), the first 5 at Verified k = 10. Per query, Success@1 and
Success@10 identical; Fast/Standard scores within 1e-3, top-10 order may swap only across a score gap
below 1e-3; Verified: the VLM checked the same set of clips.

## Why the two platforms differ (measured, not assumed)

`scripts/docker_platform_diag.py record` saves, on one platform, the decoded pixels of four frames of
the test video, the SigLIP 2 embeddings of the 66 phrasings, and the student's logits for every MEVA
clip — once with the platform's own embeddings and once with the other platform's (`--ref`), so that
`compare` can split the difference into its parts. Both sides run the same versions (Python 3.12.15,
torch 2.14.1+cu126, cuDNN 9.10.2, OpenCV 5.0.0, ultralytics 8.4.204) on the same GPU; what differs is
the OS build of the libraries. Result (`runs/docker/diag/compare.json`):

| Part | Laptop vs container | What it means |
|---|---|---|
| Student logits, **same input** | identical to the bit (max 0.0) | the student's GPU arithmetic is the same on both |
| Query embeddings (float16 text tower) | max 3.5e-4 per component, cosine ≥ 0.9999996 | the Windows and Linux CUDA libraries round the float16 text tower differently |
| Student logits, each platform's own embeddings | max 0.054, median 4.5e-4 (the clips' logit spread is 2.8) | the student is a sharp function of the query, so the tiny embedding difference becomes the 5.4e-3 seen in the Standard scores (z-scored, top 50). Fast uses the embedding directly: 5.6e-5 |
| Decoded frames (both report the FFMPEG back-end) | not identical: 32–35 % of pixels differ, mean 0.4, max 7–21 grey levels | the FFmpeg builds bundled in the two OpenCV wheels decode or colour-convert differently; YOLO's confidences move by up to 0.017, and the alert rule (conf ≥ 0.5 held for 0.5 s, then tracking) is thresholded and stateful, so borderline tracks differ |

Each platform is deterministic on its own: a second ingest of the same clip gave the container the same
21 alerts and the laptop the same 22, identical in class, time and confidence (experiment log, D-065).
The two differences are therefore between the platforms, not noise within one.

**What this means for the demo.** The search results a viewer sees are the same in Docker as in the
conda run (same clips, same order except near-ties, same Verified decisions). Your own videos produce
the same person alerts and nearly the same vehicle alerts, with borderline vehicle tracks possibly
differing by one. Making the two bit-identical would need (a) the text tower in float32 on both
platforms, which changes the laptop's numbers too and re-opens the demo-vs-evaluation parity of D-057,
and (b) one video decoder build on both platforms (e.g. PyAV with one pinned FFmpeg), which changes the
alert pipeline. Neither is needed for a recorded demo; both are the owner's call (D-065).

## Development

- `docker build --target test .` runs the unit suite in Linux (`pytest -q -rs`, so the build log lists the
  tests that skip there); the build fails if a test fails. The test stage copies the `Dockerfile` in so
  that `tests/test_docker.py` can check it from inside the image too.
- `docker/requirements*.txt` pull in `docker/constraints.txt` with `-c`; every file they name must be
  `COPY`ed into the image (a test checks this — the first build failed on exactly that).
- Changing `src/` or `web/` needs `docker compose build` (or `up --build`). Changing data needs nothing.
- A CPU-only variant (no GPU, `--no-vlm`) was not made (spec D3); add a `compose.cpu.yaml` override if
  you ever need one.

## Things to know

- **File sharing speed.** The data folders are shared from Windows into the Linux VM, which is slower
  than a Linux disk: start-up (the 4 GB VLM is read once) and the detail views may be slower than the
  conda run. The parity script reports the median seconds per search for both runs.
- **VP8 writing.** The annotated WebM of an upload is written by OpenCV's bundled FFmpeg. The Linux wheel
  is checked with an upload at the first build; the Windows wheel prints a harmless
  `tag 'VP80' is not supported` warning (`docs/demo.md`).
- **GPU not found** (`could not select device driver "nvidia"`): Docker Desktop → Settings → Resources
  → WSL integration, and an up-to-date NVIDIA Windows driver. These are the owner's settings.
- **Port busy:** stop the conda server first (both use 8000), or set `ports: ["127.0.0.1:8001:8000"]`.
