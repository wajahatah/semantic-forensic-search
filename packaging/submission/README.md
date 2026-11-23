# Semantic Forensic Search — code and models

**Wajahat Ahmed · CS-141/2026 · CS-5138 Deep Learning (Graduate) · NED University**

You type what happened ("a person opening a car door") and get back the matching CCTV clips, ranked, with
almost no calls to a large vision-language model (VLM). This folder holds the complete working system: the code,
the trained models and a dashboard you can run on your own computer to upload videos and search them.

The MEVA dataset used for training and evaluation is **not included**; it is public at
**https://mevadata.org** (Kitware / IARPA DIVA, CC BY 4.0). See §6.

---

## 1. What is in this folder

| Folder | What it holds | Size |
|---|---|---|
| `code/` | the Python package `sfs` (training, evaluation and the dashboard), the built web page, run scripts, Docker files | < 0.1 GB |
| `models/` | everything the dashboard loads | 5.7 GB |
| `test_videos/` | a short CCTV clip to try the upload with | < 0.1 GB |
| `my_alerts/` | that clip already processed (22 alerts), so a search works straight away | 0.1 GB |

Inside `models/`:

| | What it is | Size |
|---|---|---|
| `siglip2/` | Google's SigLIP 2 base model (image and text encoder), unchanged | 1.4 GB |
| `lora_adapter/adapter.pt` | **our** LoRA adapter for SigLIP 2's image encoder (1.18 M trained weights) | < 0.1 GB |
| `index/student.pt` | **our** temporal transformer (2.2 M trained weights) | < 0.1 GB |
| `index/vectors.npz`, `meta.sqlite` | the embeddings of the 3,726 MEVA test clips; uploaded videos are scored relative to them | 0.3 GB |
| `yolo/yolov8m.pt` | the YOLOv8m detector that turns an uploaded video into alerts | < 0.1 GB |
| `qwen3-vl-2b/` | the Qwen3-VL-2B VLM used only by the "Verified" level | 4.0 GB |

## 2. What you need

- **Python 3.12** (3.12 exactly: the package versions are pinned for it).
- **An NVIDIA GPU with at least 8 GB** is recommended (tested on an RTX 4060 Laptop, 8 GB, Windows 11). The
  NVIDIA driver must support CUDA 12.6 (driver 560 or newer). Without a GPU the dashboard still runs, more
  slowly and without the Verified level (§5).
- About 6 GB of disk for the Python packages, plus this folder.
- No internet at run time: everything loads from this folder.

## 3. Install (once)

Open a terminal **in the `code` folder**.

Windows:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux / macOS:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Without an NVIDIA GPU, install `requirements-cpu.txt` instead of `requirements.txt` (same versions, CPU build
of PyTorch). The GPU install downloads about 3 GB.

## 4. Run

With the environment active, still in `code`:

```bat
run.bat            (Windows)
./run.sh           (Linux / macOS)
```

The first line says "MEVA media not found … the MEVA collection is hidden": expected, the dataset is not in this
folder (§6). Wait for `open http://127.0.0.1:8000` (about a minute while the models load), then open that address
in a browser. Stop it with Ctrl+C. Another port: `run.bat --port 8010`.

### Using the dashboard

**Detect tab — your videos.** Upload a video (`test_videos/garage_cctv.mp4` or your own MP4/MOV/AVI/MKV/WebM).
YOLO finds people, vehicles and backpacks; each object that stays for 0.5 s becomes an alert with a 5-second clip
around it, and its crops are embedded with the same models the system was trained and evaluated with. Processing
takes about 4× the video's length on a laptop GPU (the 2.5-minute test clip: 10 minutes); the page shows the
progress. Alerts are kept in `my_alerts/` and survive a restart; the test clip is already there.

**Search tab.** Type what happened and choose a **level**:
- **Fast** — the LoRA-tuned SigLIP 2: cosine similarity between the sentence and each frame, 0 VLM calls.
- **Standard** — Fast + our temporal transformer, which reads the frames in order (motion), 0 VLM calls.
  This is the main result of the project.
- **Verified** — Standard, then the VLM checks only the top k clips (k VLM calls).

The bar under the search shows the time of each stage and what a VLM on every clip would have cost. Results can
be narrowed by object class and video; "Map" shows every clip as a point in 3D (the embeddings reduced to 3
dimensions) with the query's results lit up.

Example queries: "a person walking past a parked car", "a person getting into a vehicle", "two people talking",
"a person carrying a bag", "a person riding a bicycle".

## 5. Without a GPU

Install `requirements-cpu.txt` (§3) and run as in §4. The models run on the processor in full precision; search
works and **the Verified level is greyed out**, because the VLM would take minutes per clip on a processor.
Measured on the laptop with its GPU switched off: start about 1.5 minutes; a Standard search 2–6 s; **uploading a
15-second clip took about 8 minutes** (YOLO on the processor), so try short clips, or search the test clip that
is already processed.

## 6. The MEVA dataset

The models were trained and evaluated on MEVA (https://mevadata.org): 10,678 annotated activity instances from
24 fixed cameras. The dataset is not redistributed here. With it, the dashboard also searches the 3,726 MEVA test
clips (with site / camera / date / hour filters and the annotated labels); that needs two folders derived from the
dataset, placed next to `code` as `meva/crops` (person crops per clip, 14 GB) and `meva/previews` (one
animation per clip, 1.2 GB). They are built from the downloaded videos and annotations by the pipeline in the
report: `sfs.data.manifest` (clip list, Protocols A and B) → `sfs.data.extract_crops` (crop store) →
`sfs.demo.previews` (animations). Without them everything else works as described.

## 7. Run with Docker (NVIDIA GPU only)

Needs Docker with GPU support (Linux: NVIDIA Container Toolkit; Windows: Docker Desktop with the WSL 2 back end).
In `code`:

```sh
mkdir ..\meva        (Windows; Linux: mkdir -p ../meva)
docker compose up
```

The first run builds the image (about 12 GB, downloads the Python packages), then the dashboard is at
http://127.0.0.1:8000. `models/` is mounted read-only; alerts are kept in a Docker volume. `docker compose down`
stops it.

On the same data the Docker image returns the same rankings as the Windows run; the Standard scores can differ in
the third decimal and one borderline alert can appear or not, because the Linux builds of PyTorch's CUDA libraries
and of OpenCV's video decoder round differently. Both were measured; neither changes a ranking.

## 8. The code

`code/sfs/` is the whole project package, including how the models were trained and evaluated:

| Module | What it does |
|---|---|
| `sfs.data` | MEVA annotations → clip list, the two protocols (A: official split; B: 8 cameras never seen in training), frame sampling (2 fps, up to 16 frames), person crops |
| `sfs.embed` | SigLIP / SigLIP 2 embeddings of the crops |
| `sfs.train` | LoRA fine-tuning of SigLIP 2's image encoder |
| `sfs.rerank` | the temporal transformer ("student") and its training |
| `sfs.teacher` | the VLM scorers (Qwen3-VL 2B / 8B / 32B and the others tested) |
| `sfs.eval` | Q1 (mAP) and Q2 (Success@1) evaluation, the cost cascade |
| `sfs.alerts` | YOLO + tracking → alerts → embeddings, for uploaded videos |
| `sfs.demo`, `sfs.web` | the search engine and the dashboard server |

`code/queries/phrasings.csv` holds the 66 test sentences (22 activities × 3 wordings), frozen before any result.

## 9. If something goes wrong

| What you see | What to do |
|---|---|
| `python` not found, or a version other than 3.12 | install Python 3.12; on Windows use `py -3.12` as in §3 |
| `No matching distribution found for torch==2.14.1+cu126` | the Python is not 3.12, or the platform is macOS (use `requirements-cpu.txt`) |
| `CUDA out of memory` | close other programs using the GPU, or run with `--no-vlm` (frees about 4 GB) |
| "CPU only" in the dashboard although there is a GPU | the CPU requirements were installed, or the NVIDIA driver is older than CUDA 12.6 needs |
| Verified is greyed out | no GPU, `--no-vlm`, or `models/qwen3-vl-2b` is missing |
| Port 8000 in use | `run.bat --port 8010`, then open http://127.0.0.1:8010 |
| An upload stays at "processing" for long | a long or 4K video: decoding runs on the processor (a 4K minute takes several minutes) |
