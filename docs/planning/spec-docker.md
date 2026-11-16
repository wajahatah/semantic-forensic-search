# Spec — Docker Compose for the search dashboard

**Date:** 2026-11-07 · **Built and checked at the office 2026-11-08** (results: `docs/docker.md`, log D-065;
§5.2 and §5.3 fail their pre-fixed rules for measured platform reasons — owner's decision D6 below) ·
**Earlier status:** files written and unit-tested 2026-11-07 (owner: "both in parallel" with
the report, so the recommended defaults of §7 were taken and are open to overwrite); **the build and the
checks of §5 wait for the office connection and for the owner's answers to §7** · **Log:** D-063
**Scope (owner, 2026-11-07):** "now lets work on the docker compose file", after the alerts +
dashboard work was tested by the owner (D-062).
**Time budget:** about 1 day, out of the ~11 left before submission.
**No training, no new experiments, no change to any model or result.**

---

## 1. What this change does

The dashboard (`python -m sfs.web.server`) becomes runnable with one command:

```bash
docker compose up        # → http://127.0.0.1:8000
```

New files:

| File | What it is |
|---|---|
| `Dockerfile` | Multi-stage build: (1) Node builds the front end, (2) Python runtime with the pinned packages and `src/`, (3) a `test` stage that also runs the unit tests |
| `compose.yaml` | One service, `dashboard`: GPU, data mounts, the alert-store volume, port, health check |
| `.dockerignore` | Keeps the build context to code only (no `data/`, `runs/`, `manifest_v2_drive/`, `.git`, `web/node_modules`) |
| `docker/requirements.txt` | Exact versions for the server, taken from the tested `dlproj` env (§4.2) |
| `scripts/prepare_docker_models.py` | One-time copy of the SigLIP 2 snapshot into a plain-file Hugging Face cache the container can read (§4.4) |
| `scripts/docker_parity.py` | Checks that the container gives the same search results as the laptop run (§5) |
| `docs/docker.md` | How to build, run, check and stop it |

Code change (small): `sfs.web.server` gets a `--host` option (default `127.0.0.1`, unchanged). Inside
the container the server must listen on `0.0.0.0`; Compose publishes it **on the laptop's loopback
only** (`127.0.0.1:8000`), so the dashboard is still reachable from this laptop only, as now.

Docs updated in the same change: `README.md` (Docker section + layout), `requirements.txt` (pointer to
the pinned file), `docs/demo.md` (Docker as a second way to run it), experiment log, `CLAUDE.md`.

## 2. The system before

- The dashboard runs from the `dlproj` conda env on Windows (Python 3.12.15, torch 2.14.1+cu126,
  transformers 5.20.2, ultralytics 8.4.204). Setup is manual: conda env, `pip install`, `npm run build`.
- The server loads, from local paths:

  | What | Where | Size |
  |---|---|---|
  | MEVA demo bundle (vectors, metadata, student) | `data/demo/A` | 263 MB + 1.2 GB previews |
  | Crop store (Verified level, detail view) | `data/crops_v2` | 14 GB |
  | SigLIP 2 base | Hugging Face cache, `google/siglip2-base-patch16-224` rev `75de2d55…` | 1.5 GB |
  | LoRA adapter | `runs/lora/A-middle/best/adapter.pt` | 4.6 MB |
  | Qwen3-VL-2B (Verified) | `data/models/Qwen3-VL-2B-Instruct` | 4.0 GB |
  | YOLO | `data/models/yolo/yolov8m.pt` | 50 MB |
  | Query phrasings (examples) | `queries/phrasings.csv` | small |
  | Alert store (read + write) | `data/alerts` | grows with uploads |

- **Measured facts that shape the design:**
  - Docker Desktop 29.1.3 + Compose v2.40.3 are installed, WSL2 backend; the engine is **not
    running** now. NVIDIA driver 610.62 (RTX 4060 8 GB).
  - The Hugging Face cache stores SigLIP 2 as **Windows symlinks, two levels deep** (Xet layout).
    A Linux container cannot follow them through a bind mount.
  - The cu126 index has `torch 2.14.1+cu126` (checked with `pip index versions`, no download).
  - Disk: **61 GB free** on C: (not the 155 GB in older notes). The image (~8 GB) fits; copying the
    data into Docker volumes (~20 GB) would be tight, so the data is mounted, not copied.

## 3. The system after

- `docker compose up` builds (first time) and starts the same server, with the same models and data,
  at http://127.0.0.1:8000. The conda run keeps working unchanged; the two are alternatives, not run
  at the same time (one port, and the 8 GB GPU holds one copy of the models: 5.7 GB).
- The image contains **code + front end + Python packages only**. Data and model weights are mounted
  **read-only** from the project folder; nothing large is baked in. Reason: 20 GB of data, and the
  demo is recorded on this laptop (D-058).
- Alerts made inside Docker live in a Docker volume (D2) and survive `docker compose down` / `up`.
- Search results equal the laptop run (rule in §5).

## 4. Design

### 4.1 Images

- **Front-end stage:** `node:24.12.0-alpine` (matches the laptop's Node 24.12.0), `npm ci` from the
  committed `web/package-lock.json`, `npm run build` → `web/dist`. Only `dist/` is copied onward.
- **Runtime stage:** `python:3.12.15-slim-bookworm` (matches the laptop's Python).
  - torch + torchvision from the PyTorch cu126 index (the wheels bring their own CUDA libraries; the
    GPU driver comes from Windows through Docker Desktop, so no CUDA base image is needed).
  - apt: only the shared libraries OpenCV needs (`libgl1`, `libglib2.0-0`), `--no-install-recommends`.
  - `pip check` runs in the build and fails it on any conflict.
  - Runs as a **non-root user** (uid 1000). `PYTHONPATH=/app/src`, `PYTHONUNBUFFERED=1`.
  - **Offline at runtime:** `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `YOLO_OFFLINE=1`, and
    Ultralytics' settings dir in a writable temp folder. The container never downloads anything after
    the build.
- **Test stage:** runtime + `pytest` + `tests/`; `docker build --target test` runs the unit suite
  (`-m "not slow"`) inside Linux. Not part of the image that runs.
- Pinned by tag; the digests pulled at the office are recorded in `docs/docker.md` after the first
  build.

### 4.2 Pinned packages (`docker/requirements.txt`)

Exact versions from `pip freeze` of `dlproj` (2026-11-07), for what the server imports only:
torch 2.14.1+cu126, torchvision 0.29.1+cu126, transformers 5.20.2, tokenizers 0.23.3,
safetensors 0.8.0, huggingface_hub 2.2.0, accelerate 1.15.0, peft 0.21.2, sentencepiece 0.2.2,
protobuf 7.36.2, numpy 2.5.2, pandas 3.0.6, pyarrow 26.0.0, PyYAML 6.0.3, opencv-python 5.0.0.93,
pillow 12.3.0, tqdm 4.70.1, ultralytics 8.4.204, lap 0.5.14, fastapi 0.156.2, uvicorn 0.54.0,
python-multipart 0.0.32. Not included: gradio (old page), chromadb (unused), pytest (test stage only).

### 4.3 `compose.yaml`

```yaml
services:
  dashboard:
    build: .
    image: sfs-dashboard:0.1
    ports: ["127.0.0.1:8000:8000"]          # this laptop only
    command: ["python", "-m", "sfs.web.server", "--host", "0.0.0.0", "--alerts", "/alerts"]
    working_dir: /app
    volumes:
      - ./data/demo/A:/app/data/demo/A:ro
      - ./data/crops_v2:/app/data/crops_v2:ro
      - ./data/models/Qwen3-VL-2B-Instruct:/app/data/models/Qwen3-VL-2B-Instruct:ro
      - ./data/models/yolo:/app/data/models/yolo:ro
      - ./runs/lora/A-middle/best:/app/runs/lora/A-middle/best:ro
      - ./data/hf-cache:/hf-cache:ro          # §4.4; HF_HOME=/hf-cache
      - alerts:/alerts                        # D2
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]
    healthcheck:                              # GET /api/status; models take ~1 min to load
      start_period: 180s
    restart: "no"                             # a demo, not a service that must come back by itself
volumes:
  alerts:
```

(Sketch; the real file adds the environment block and the health-check command.) Paths inside the
container equal the server's defaults, so no other argument changes.

### 4.4 SigLIP 2 weights

`scripts/prepare_docker_models.py` copies the snapshot, following the symlinks, into
`data/hf-cache/hub/models--google--siglip2-base-patch16-224/` with `refs/main` and
`snapshots/75de2d55…/` holding real files (1.5 GB, local copy, no download). Hugging Face reads a
plain-file cache like this offline, so the model id and the encoder id stay
`google/siglip2-base-patch16-224` and nothing else changes. The script checks each copied file's
sha256 against the blob name. Side effect worth having: the SigLIP 2 revision is now pinned to a real
commit (open follow-up "pin the real revision").

### 4.5 Known risks (checked during the build, not assumed)

1. **VP8 writing in Linux OpenCV.** The Windows wheel writes the annotated WebM; the Linux wheel's
   bundled FFmpeg must too. Checked by an upload in the container. If it fails, I stop and bring
   options to the owner (no silent codec swap).
2. **Reading mounted data from Windows is slower** than from the Linux disk (Docker Desktop file
   sharing). Expected effect: slower start-up (the 4 GB VLM is read once) and detail views. Start-up
   time and a search's time are measured and recorded; if a search is clearly slower (> 1.5× the
   laptop run), I report it before going on.
3. **Linux vs Windows GPU kernels** can change float rounding, so near-tied clips may swap. The parity
   rule (§5) is written for this, as in the demo's parity contract (D-056).

## 5. Verification (fixed before any measurement)

1. `docker compose config` is valid; `docker build --target test` passes the unit suite in Linux;
   `pip check` clean; image size recorded.
2. **Search parity, container vs laptop run.** `scripts/docker_parity.py` sends the same queries to
   a running server and saves the answers (JSON): first the laptop run, then (laptop server stopped)
   the container.
   - Queries: all 66 frozen phrasings at **Fast** and **Standard** over MEVA (top 50), and 5 of them
     at **Verified, k = 10**.
   - **Pass rule:** for every query, Success@1 and Success@10 (from the MEVA labels) are identical.
     Fast and Standard: every returned score is within **1e-3** of the laptop's, and the top-10 order
     may differ only where the laptop's score gap is < 1e-3 (near-ties). Verified: the set of the k
     clips the VLM checked is identical; their order and the VLM probabilities are reported, not gated
     (*refined 2026-11-07, before any measurement:* the bf16 VLM's score noise is 0.37 in the raw
     logit, Result 04, so a 1e-3 bar on it would measure the GPU, not the container).
   - The script has a unit test with fake answers (pass, near-tie pass, real-difference fail).
3. **Alerts.** The same test video uploaded to both runs: same number of alerts, same classes, same
   alert frames and clip windows; alert embeddings cosine ≥ 0.9999. Which video: D4.
4. **Persistence.** `docker compose down`, `up`: the alerts are still listed and searchable.
5. **Health + offline.** The health check turns healthy; the container log shows no network access
   attempts (offline variables set).
6. Full local suite (`pytest`) still passes after the `--host` change, plus a test for it.

**Results (2026-11-08, office build; details and numbers in `docs/docker.md`):** 1 pass (190 passed,
18 skipped for files not in the image, `pip check` clean). 2: Success@1/@10 identical on all 137 queries,
top-10 swaps only across near-ties, Verified sets identical, Fast scores within 5.6e-5 — but **Standard
scores up to 5.4e-3 apart (rule 1e-3): fail**; cause measured with `scripts/docker_platform_diag.py`:
the float16 text tower differs by ≤ 3.5e-4 per component between the Windows and Linux CUDA libraries,
the student's own arithmetic is bit-identical. 3: **22 alerts on the laptop, 21 in the container: fail**;
cause measured: the OpenCV wheels' FFmpeg builds decode the same video to pixels that differ on a third
of the positions (up to 21 levels), moving YOLO's confidences by ≤ 0.017 across the thresholded rule;
person alerts identical, vehicle alerts shift by 1–4 frames, one borderline track differs. Each platform
is deterministic on its own. 4 pass (21 alerts identical after `down`/`up`). 5 pass (healthy in 91–116 s,
no download in the log). 6 pass (full suite: 210 passed, 4 deselected). First build attempt failed because `docker/constraints.txt`
was not copied into the image; fixed in the Dockerfile with a test for it.

## 6. What does not change

Models, data, results, the search code, the alert rule, the front end's behaviour, and the conda way
of running the dashboard. No system or Docker Desktop settings are changed by me: starting Docker
Desktop and its GPU support are the owner's settings (Docker Desktop's WSL2 backend supports NVIDIA
GPUs by default).

## 7. Decisions for the owner (recommendation first)

- **D1 · Where to build the image (~3–3.5 GB download: Python packages with CUDA, base images).**
  - **A (recommended): at the office** (300 Mbps, a few minutes). The project rule keeps the home
    connection off large downloads (9 h for 1.6 GB earlier).
  - B: at home anyway, if you accept 35 min – 2 h at 0.5–1.7 MB/s.
  - Everything up to the build (files, parity script, tests, the SigLIP copy) can be written at home
    now; only `docker build` waits.
- **D2 · Where alerts made inside Docker are stored.**
  - **A (recommended): a Docker volume.** SQLite on a Windows folder shared into Linux is a known
    source of locking and corruption problems; a volume lives on the Linux disk.
  - B: the project's `data/alerts` folder, shared with the conda run (both see the same alerts).
- **D3 · A CPU-only variant** (no GPU; Verified hidden; slow uploads).
  - **A (recommended): not now.** The demo is recorded on this laptop with its GPU.
  - B: add `compose.cpu.yaml`, an override that drops the GPU and adds `--no-vlm` (~1 h).
- **D4 · Test video for the alert check (§5.3).** It is stored only in a throwaway Docker volume and
  the laptop run's store, and deleted after the check.
  - **A (recommended):** a video you choose for the recording.
  - B: your drink-spiking test clip `videotest1.mp4`, as in the earlier smoke test.
  - *Taken 2026-11-08:* the garage CCTV clip the owner had already uploaded to the laptop store (the only
    video there, 152 s, 22 alerts) — option A in effect. It stays in the Docker volume `project_alerts`
    (21 alerts) for the demo; `docker compose down -v` removes it.
- **D1 · taken:** built at the office 2026-11-08 (A). **D2, D3, D5:** defaults A kept, unchanged by the build.
- **D6 · The two platform differences (§5 results). Accept, or make the platforms bit-identical?**
  - **A (recommended): accept and document.** Rankings, Verified decisions and person alerts are
    identical; what differs is the 3rd decimal of Standard scores and borderline vehicle tracks, both
    traced to OS builds of libraries we pin by version but cannot pin by build. The demo is recorded on
    the laptop; Docker is the reproducible packaging of the same code.
  - B: text tower in float32 on both platforms (changes the laptop's numbers; the demo-vs-evaluation
    parity of D-057 must be re-measured) and one video-decoder build on both (PyAV with a pinned
    FFmpeg; changes the alert pipeline and its tests). About a day, and still not guaranteed bit-exact.
- **D5 · Purpose check.** I assume Docker is for running the demo reproducibly on this laptop (and
  showing the setup in the report), **not** for the instructor to run: they would need the 20 GB of
  MEVA data and models. If the instructor should run it, the design changes (baked-in models, no
  crop store, smaller demo bundle) and needs a new estimate.
