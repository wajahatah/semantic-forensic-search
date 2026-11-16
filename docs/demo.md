# Search dashboard — how to run and record it

Specs:
- `docs/planning/spec-video-alerts-dashboard.md`: alerts from your videos, and the dashboard;
- `docs/planning/spec-search-demo.md`: the search engine.

Code: `src/sfs/alerts/`, `src/sfs/demo/search.py`, `src/sfs/web/`, `web/` (front end).
Tests: `tests/test_alerts.py`, `tests/test_web.py`, `tests/test_demo.py`.

The dashboard has two tabs:
- **Detect:** you upload a video. YOLO finds the people, vehicles and backpacks, and each one becomes an
  alert named after its class, with a 5-second clip.
- **Search:** you describe what you are looking for and get the matching clips. The search covers:
  - your alerts;
  - the **protocol A evaluation split of MEVA**: 3,726 annotated clips that the LoRA model and the
    student never trained on;
  - or both together.

It runs on the laptop only (127.0.0.1) and has no sign-in (owner decision, 2026-11-07).

## Run it

```bash
# once
python -m sfs.demo.index --protocol A        # the MEVA bundle (~10 s, 263 MB in data/demo/A)
cd web && npm install && npm run build       # the front end -> web/dist (needs Node; ~2 s to build)

# every time
python -m sfs.web.server                     # open http://127.0.0.1:8000
python -m sfs.web.server --no-vlm            # without the Verified level (CPU, or to save GPU memory)
```

**In Docker:** `docker compose up` runs the same server in a container (`docs/docker.md`; files written
2026-11-07, first build at the office).

**Startup:** the server loads every model at start (SigLIP 2, the LoRA adapter, the student, and the
2B VLM when a GPU is present). With all of them loaded, the laptop GPU holds about 5.7 GB. YOLO loads
with the first upload.

**Data the laptop needs:**
- `data/demo/A` (and its `previews/`, rendered once by `python -m sfs.demo.app`, ~30 min);
- `data/crops_v2`;
- `runs/lora/A-middle/best/adapter.pt`;
- `data/models/yolo/yolov8m.pt` (COCO, sha256 `5d4a90cd…`, copied from the drink-spiking project);
- `data/models/Qwen3-VL-2B-Instruct`, for Verified only.

**Where your alerts are kept:** `data/alerts/`, which git ignores:
- `alerts.sqlite`;
- `vectors/`: one file per video;
- `media/`: the uploaded video, its annotated WebM, and the frames, crops and thumbnail of each alert.

**Development:** `cd web && npm run dev` serves the UI on :5173 with hot reload and forwards the API to
the running server.

## How a video becomes searchable

1. **Detect + track.** YOLO (`yolov8m.pt`) with ByteTrack runs on every frame. Only these classes raise
   alerts: person, bicycle, car, motorcycle, bus, truck, backpack. You can untick classes before
   uploading.
2. **Alert rule** (`sfs.alerts.rule`):
   - An object raises an alert once it has been seen for **0.5 s** with YOLO confidence **≥ 0.5**.
   - The alert is **held** while the object stays in view.
   - The same object raises a new alert only after it was **gone for 2 s** and is detected again.
   - A new track that appears where a track of the same class was lost within those 2 s is treated as
     the same object (track stitching).
   - Other objects next to it are recorded ("person · near car").
3. **Clip** (`sfs.alerts.clip`): 5 frames before + the alert frame + 5 after, at 2 frames per second.
   That is 11 frames over **exactly 5 seconds**. At the start or end of the video the window shifts to
   stay inside it.
4. **Crops, embeddings and re-ranker features:** the same code as for MEVA.
   - The tight and context regions come from `sfs.data.regions`.
   - The LoRA and frozen SigLIP 2 embeddings come from `sfs.alerts.embed`.
   - The box-motion features come from `sfs.rerank.features`.
   - A slow test runs stored MEVA crops through this path and gets the evaluated embeddings back
     (cosine 1.0000 LoRA / 0.99999 frozen).
5. **Annotated video:** the whole video is written once as a WebM with every tracked box drawn. Each
   alert's clip is its 5-second segment of that video.
6. **Searchable:** the alert is stored and the Search tab finds it at once.

**Speed and size:** detection is about real time on the laptop. A 10 s 720p video was searchable 22 s
after upload, including loading YOLO.

OpenCV prints `FFMPEG: tag 0x30385056/'VP80' is not supported …` for each WebM. This is FFmpeg's
warning about its tag table; the file is written as VP8 and plays in the browser (checked by reading
it back).

## Controls

| Control | What it does |
|---|---|
| **Detect · drop zone / Detect alerts** | Upload a video (MP4, MOV, AVI, MKV, WebM, ≤ 4 GB). Progress runs through three stages: detect + track, cut clips + embed, searchable |
| **Detect · objects that raise an alert** | Which YOLO classes create alerts for this upload |
| **Detect · video player + timeline** | The annotated video. Each mark on the timeline is an alert; clicking it (or a table row) loops that alert's 5 s clip |
| **Detect · Search these alerts →** | Opens the Search tab limited to this video's alerts |
| **Detect · Delete** | Removes the video and all its alerts (asks first) |
| **Search · Collection** | Your videos / MEVA eval split / both |
| **Search · Cost level** | **Fast:** LoRA-tuned SigLIP 2. **Standard:** + temporal re-ranker (z(student) + z(LoRA)). **Verified:** + Qwen3-VL-2B checks the top k (5–20) and re-orders them (γ = 2). Fast and Standard use 0 VLM calls; Verified uses k |
| **Search · Alert type** | Only alerts of the selected classes |
| **Search · MEVA filters** | Site, camera, day, hour range; applied after the whole corpus is scored, as in the Q2 evaluation |
| **Search · Show MEVA labels** | Each MEVA result's annotated activity. For an evaluation phrasing, matches get a ✓ and a green frame, and the top-10 count is shown |
| **Search · Compare with frozen SigLIP 2** | The same query on the frozen baseline (left) and the final system (right) |
| **Search · Grid / 3D clip map** | Results as cards, or the 3D map: every clip is a point (PCA of its LoRA embedding), the top results light up and the query sits at the centre of its top 10. Drag to rotate, scroll to zoom, click a number to open that result |
| **Result card** | Hover to play (alerts: the 5 s clip; MEVA: the stored crops). Click for the detail view: the clip, the 11 embedded frames (alerts) or tight crops (MEVA), metadata, and the score at each stage |

**What the cost line shows:**
- the seconds per stage;
- the number of VLM calls;
- what a VLM on every searched clip would cost, using the per-call time measured in that search when
  the VLM ran, and Result 04's 0.168 s otherwise.

**Confidence band:** where the clip's score falls among all 3,726 MEVA clips for the same query
(**High:** top 1 %; **Medium:** top 5 %; **Low:** the rest).
- It is relative, not a probability. Real probabilities would need labelled test videos, which were
  out of scope.
- At Verified, the VLM's yes-probability is shown as well.

**Linking straight to a search:** add the query to the URL, e.g.
`http://127.0.0.1:8000/?q=a%20person%20opening%20a%20car%20door&collection=meva&labels#search`.
- Other options: `&level=fast|standard|verified`, `&view=map`, `&compare`.
- This is handy when recording, since each shot can start from a link.

## What the numbers mean

- **MEVA:**
  - For the evaluation phrasings, the MEVA results reproduce the evaluated system: Success@1 and
    Success@10 are identical on every Q1 and Q2 query (`pytest -m slow tests/test_demo.py`).
  - Adding alerts never changes a MEVA score: MEVA is scored in full, and its statistics are the
    reference for the alerts (tested).
- **Your videos:**
  - They are a **demonstration, not a measured result.** There are no labels for them.
  - Footage unlike MEVA (indoor, close-up, edited YouTube video) usually ranks "Low" against MEVA's
    best clips, and the 3D map shows it sitting in its own region.
  - Pick videos where the action is clear and filmed like CCTV (fixed camera, people and cars in
    view).
- **What the models were trained for:** MEVA's activities, such as opening doors, getting in and out
  of cars, carrying, phone use and vehicles turning. Appearance wording like "blue jacket" was not
  measured (D-055).
- **Videos you use:** use videos you may show: your own recordings, or Creative Commons videos on
  YouTube.

## Recording walk-through (about 6 minutes)

The demo is presented as a **screen recording**, not live (D-058). Before recording:
- start the server;
- run one search at each level, so the GPU is warm;
- have the video to upload ready.

1. **Detect: an alert from a new video.** Drop a CCTV-style video with people and a car into the Detect
   tab. Show the three progress stages, then the alerts list: "person · near car" at 00:12, and so on.
   Click an alert: its 5-second clip loops in the player, with the boxes drawn.
2. **Search: find it with words.** Click *Search these alerts →* and type *a person getting into a car*
   at **Standard**. The alert from your video comes up, with its confidence band.
3. **The problem: cost.** Switch the collection to **Both**. Point at the cost line: about 3,700 clips
   searched in under a second with 0 VLM calls, against ≈ 10 minutes for a VLM on every clip.
4. **Paying for checking.** Choose **Verified, k = 10**: 10 VLM calls instead of ~3,700, and the VLM's
   yes-probability on the checked clips.
5. **Fine-tuning matters.** Collection **MEVA**, turn on *Show MEVA labels* and *Compare with frozen
   SigLIP 2*, then search *a person opening a car door*. The ✓ count shows the gain (here 6 of the top
   10 for the final system).
6. **The 3D map.** Switch to *3D clip map* and rotate it. Similar clips cluster, your alerts sit in
   their own region, and the results light up around the query.
7. **Honest limits.**
   - Turning left / right and very short actions are the weakest (Results 03–05).
   - Footage unlike MEVA is not measured.
   - The confidence band is relative.
