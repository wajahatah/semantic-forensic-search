# Spec — Alerts from your own videos + new dashboard

**Date:** 2026-11-07 · **Status:** decisions D1–D6 answered by the owner 2026-11-07 (D-060); **built and checked 2026-11-07 (D-061, D-062)**; §4.2 and §4.5 amended after the first run on real footage · **Log:** D-059 – D-062
**Scope (owner, 2026-11-07):**
- Upload any video (YouTube, another dataset). YOLO runs on it, and each detection becomes an alert
  named after its class. The alert keeps frames from before and after it, is embedded, and can then be
  found from the search.
- New dashboard with two tabs: **Detect** (upload → alerts) and **Search** (query → alert clips).
- **No sign-in** (owner dropped it).
- **Docker comes after these two** (separate spec).

**Time budget:** ingestion 1.5–2 days, dashboard 2–2.5 days, out of the ~11 left before submission.
**No training and no new experiments:** every model is used as it is (Results 03–05).

---

## 1. What this change does

1. A new package, `sfs.alerts`, turns a video file into searchable alerts:
   1. YOLO detection + tracking;
   2. the alert rule;
   3. an 11-frame clip around each alert;
   4. the same crops, embeddings and re-ranker features as MEVA;
   5. a local alert store.
2. The search engine (`sfs.demo.search`) searches **MEVA, your alerts, or both** with the same three
   cost levels. MEVA results stay exactly as they are now.
3. A new web dashboard replaces the Gradio page:
   - FastAPI back end;
   - a designed front end (the owner chooses one of the design options);
   - two tabs, **Detect** and **Search**.

## 2. State before

- The search runs only over the 3,726 MEVA eval clips (`data/demo/A`, D-056/D-057). It uses MEVA's
  human-drawn boxes, and only stored crops can be shown.
- The Gradio UI (`sfs.demo.app`) works, but the owner finds it plain.
- There is no detector in this project. `yolov8m.pt` (COCO, 80 classes, including person and car)
  is on the laptop in `C:\wajahat\nexa\drink_sprinkle_detection\poc\`.
- **Packages:**
  - `ultralytics` and `lap` are not installed (detection and tracking);
  - `opencv`, `fastapi` and `uvicorn` are installed;
  - Node 24 / npm 11 are installed.

## 3. State after

- **Detect tab:**
  1. You upload a video and see progress.
  2. You get a list of alerts. Each has:
     - its **type** (YOLO class, e.g. `person`, `car`) and any other objects next to it ("person · near car");
     - its time in the video and a thumbnail;
     - an "embedded ✓" mark when it is ready.
  3. Clicking an alert plays its clip with the boxes drawn.
- **Search tab:**
  1. You type a query and pick the collection (**Your videos** / MEVA / both) and the cost level.
  2. Each result shows:
     - a playable clip, in **full frame** for your videos (we have the video file) and as crops for MEVA;
     - a **confidence band** (§4.6);
     - its alert type and time.
  3. The cost line and the comparison with frozen SigLIP 2 stay.
- **Command line:** `python -m sfs.alerts.ingest <video>` does the same as the Detect tab, for batch
  use.
- **Docs:** `docs/demo.md` is rewritten for the new dashboard; README section; experiment log D-059+.

## 4. Design

### 4.1 Detection and tracking
- **Model:** `yolov8m.pt` (COCO), copied into `data/models/yolo/`. Its sha256 is recorded, so no
  download is needed.
- **Tracking:** Ultralytics `model.track(..., tracker="bytetrack.yaml", persist=True)` on every frame.
  For a 1080p video on the laptop GPU this is roughly real time. A track id follows each object
  across frames, and the alert rule (§4.2) needs it.
- **Classes that raise alerts (D3, owner 2026-11-07):** person, the vehicles (bicycle, car,
  motorcycle, bus, truck) and **one common object: backpack** (my pick; the most frequent
  carried object in street footage; handbag or suitcase are drop-in swaps). They can be changed
  in the Detect tab.

### 4.2 Alert rule (D1, owner 2026-11-07)
The owner's rule is "whatever the model detects becomes an alert, named by its class". YOLO detects
the same person in every frame, so the rule has to say **when** a detection is a new alert:
- **An alert fires when an object's track is confirmed:** seen for **0.5 s** with YOLO confidence
  **≥ 0.5**.
  - *Amended 2026-11-07 (D-061):* the draft said "≥ 3 frames". On a real 14 s crowded clip (at most 6
    people at once), 3 frames gave 38 alerts, mostly from low-confidence boxes.
- **Track stitching (added, D-061):** ByteTrack gives a briefly occluded person a new id. A new track
  that appears next to a just-lost track of the same class (within `gone_s`) is the same object and
  keeps its alert.
  - Measured on three clips: 20 → 6, 7 → 3 and 4 → 2 alerts, against 6 / 4 / 2 people at most.
  - A person who leaves, while another enters at the same spot within 2 s, would be merged: a known
    limit.
- **It is held while the object stays in view** (owner: "hold it till he is gone and detected
  again"). The same object raises a new alert only after it has been **gone for ≥ 2 s** and is
  detected again.
  - ByteTrack keeps a lost track for 2 s (`track_buffer` = 2 s of frames), so a short occlusion does
    not raise a second alert.
  - Limit (stated to the owner): an action late in a long stay (e.g. 40 s after the person appeared)
    is not in any clip. A repeat interval exists as a setting (`repeat_s`, off by default).
- **Alert type** = the YOLO class of the object.
- **"Near"** = other tracked objects whose box overlaps it or lies within one box height of it at
  the alert time, recorded as "person · near car".

### 4.3 The clip: exactly 5 seconds (D2, owner 2026-11-07)
- **5 *sampled* frames at 2 fps before and after**, not 5 raw video frames.
  - At 30 fps, 5 raw frames are 0.17 s, too short to show any action.
  - At 2 fps, the clip is 11 frames over 5 seconds (2.5 s before, 2.5 s after). This is the same
    sampling the models were trained and evaluated on (MEVA: 2 fps, ≤ 16 frames).
- **Always 5 s, "not more not less" (owner):**
  - At the start or end of the video, the window **shifts** to stay inside the video, so the
    alert frame is then not centred.
  - Only a video shorter than 5 s gives a shorter clip, flagged in the alert list.
  - Playback is that 5 s segment of the video's annotated WebM (§4.5).

### 4.4 Crops, embeddings, re-ranker features (same code as MEVA)
- **Regions:** the actor set = the alert's object plus the objects "near" it (§4.2), like MEVA's
  multi-actor activities (a person and the car they open).
  - Per frame, the box comes from that frame's tracks, using `lookup_union` with its ±5-frame window.
  - The crops use the unchanged `tight_region` / `context_region` from `sfs.data.regions`.
- **Embeddings:**
  - LoRA A-middle on the tight view (the search model), with the same preprocessing
    (`preprocess_eval`, `resize_bilinear`);
  - frozen SigLIP 2 on the tight view (comparison) and the context view (re-ranker input).
- **Re-ranker features:** `sfs.rerank.features.box_features` with the YOLO tracks. It is the same
  function the student was trained with: the motion of the boxes relative to their own size.
- **Verified level:** the 2B VLM reads the alert's tight crops, as for MEVA.

### 4.5 Alert store (`data/alerts/`, git-ignored)
- **`alerts.sqlite`:**
  - table `videos`: id, file name, sha256, fps, size, length, upload time;
  - table `alerts`: id, video, class, near-classes, track id, alert frame / time, YOLO confidence,
    clip frame numbers, status.
- **`vectors/`:** one `.npz` per video (frame embeddings + features), written atomically. A crash
  never leaves a half-written alert.
- **`media/<video>/`:**
  - the uploaded video;
  - **`annotated.webm`:** the whole video once, with every tracked box drawn (VP8, ≤ 960 px, 15 fps;
    browsers play it natively, while OpenCV here has no H.264 encoder);
  - per alert: 11 full frames (≤ 960 px JPEG) with the alert's boxes drawn, its 11 tight crops (PNG,
    for the VLM and the detail view), and a thumbnail.
  - An alert's clip is its 5 s segment of `annotated.webm`.
  - *Amended 2026-11-07 (D-061):* one WebM per alert was measured at 4.1 s of encoding per alert
    (156 of the 303 s for 38 alerts, 436 MB). One video per upload brought the same clip to 50 s and
    61 MB.
- **Deleting** a video removes its alerts (button in the Detect tab, with a confirm step).

### 4.6 Search over alerts and the confidence band
- **Scoring:** alerts are scored with the same functions as MEVA (Fast / Standard / Verified).
- **z-normalisation for Standard:** uses the mean and spread of the **MEVA corpus for that query**,
  and these are applied to the alerts too. As a result:
  1. MEVA results are bit-for-bit what they are now (the parity tests stay valid);
  2. a handful of alerts does not produce unstable z-scores;
  3. alert and MEVA scores are on one scale, so "both" can be merged into one ranking.
- **Confidence band** = where the clip's score falls among the 3,726 MEVA clips for the same query:
  - **High:** above MEVA's 99th percentile;
  - **Medium:** above the 95th;
  - **Low:** otherwise.

  It is labelled as *relative, not a probability*. Real probabilities would need labelled test
  videos, which is out of scope (owner: no new experiments). At Verified, the VLM's yes-probability
  is shown as well.

### 4.7 Dashboard
- **Back end:** FastAPI (`sfs.web`). Endpoints:
  - upload a video → background job;
  - job progress;
  - list / delete alerts;
  - search;
  - serve the clips.

  One job at a time, so the GPU is not shared between two detections. Listens on 127.0.0.1 only.
- **Front end:** React + Vite. It is built to static files that FastAPI serves, so no Node is needed
  at run time (this also helps the Docker step later). **Look: design A, "Control room"** (owner,
  D4): dark console, filters on the left, results grid.
- **3D clip map (D5, owner: yes):** a three.js view where similar clips sit close together. The query
  appears as a point, and its results light up.
  - Positions come from a fixed 3D PCA of the LoRA clip vectors (mean of each clip's frame
    embeddings), fitted on MEVA. New alerts are projected with the same matrix.
  - It is shown only as an illustration. Search never uses the 3D positions.
- **The Gradio UI stays** until the new one is verified. Removing it is the owner's call.

## 5. Tests
- **Alert rule:** confirmation after 3 frames; the 5.5 s repeat; clip edges at the start or end of a
  video; "near" objects. Run on synthetic tracks.
- **Clip sampling:** 2 fps, 5 + 1 + 5 frames, frame numbers checked.
- **Same embeddings as MEVA (slow, real models):** stored MEVA tight crops through `ClipEmbedder`
  equal the evaluated LoRA and frozen stores (measured cosine 1.00000 / 0.99999, bar 0.9999).
  - *Amended:* the draft fed a MEVA video through the whole path, but the raw MEVA videos are on
    Drive, not the laptop.
  - The region step is the unchanged `regions_for`, and a test checks an alert's tight crop against
    it.
- **Features:** shapes and the `has_box` / missing-box handling.
- **Store:** atomic writes; deleting a video removes its alerts.
- **Search:** MEVA-only results are identical to the current engine (exact). The alerts get bands.
  "Both" merges correctly.
- **API:** every endpoint with FastAPI's test client, with a fake detector and fake models.
- **Slow end-to-end test:** a short real video through the real YOLO and models → its alerts are
  found by a matching query.
- **Full suite** before handing over.

## 6. Risks
| Risk | Mitigation |
|---|---|
| YouTube footage differs from MEVA (angle, quality, editing) and is not measured | Presented as a demonstration, not a result; pick clips with clear actions; the confidence band is honest about being relative |
| Laptop GPU memory (8 GB): YOLO ~1 GB + SigLIP + student + 2B VLM (~5 GB) | Detection and Verified search never run at the same time (one job); the VLM can be switched off |
| YOLO misses or swaps a track id | Only changes which frames an alert covers; ByteTrack is the standard choice |
| Licences | Ultralytics is AGPL-3.0 (fine for a private course project; say so in the report). Use videos you may show: Creative Commons on YouTube, or your own recording |

## 7. Decisions for the owner
**Answered by the owner on 2026-11-07 (D-060):**
1. **D1 Alert rule:** alert when an object is first confirmed; held while it stays; a new alert only
   after it was gone and is detected again.
2. **D2 Clip:** 5 sampled frames at 2 fps before and after; always exactly 5 s.
3. **D3 Alert classes:** person, vehicles and one common object (backpack).
4. **D4 Design:** A, Control room.
5. **D5 3D clip map:** yes.
6. **D6 Installs:** approved, into the `dlproj` env. `ultralytics 8.4.204` + `lap 0.5.14` (the dry
   run showed no change to torch, numpy or OpenCV; the package list from before was saved).
   Front end: React, Vite and three.js from npm.
