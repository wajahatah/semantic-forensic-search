"""Dashboard back end (spec-video-alerts-dashboard §4.7): FastAPI, local only.

    python -m sfs.web.server            # http://127.0.0.1:8000

Detect tab: POST /api/videos (upload) -> a background job (sfs.alerts.ingest) -> GET /api/jobs/{id};
GET /api/videos, /api/videos/{id}/alerts, DELETE /api/videos/{id}.
Search tab: POST /api/search, GET /api/clips/{id}, GET /api/map; media under /media/...
The front end (web/, React + three.js) is built to web/dist and served at /.
"""
# no `from __future__ import annotations`: FastAPI must see the real types of the route parameters,
# which are defined inside create_app
import argparse
import math
import os
import re
import threading
import uuid

import numpy as np
from pydantic import BaseModel, Field

from sfs.alerts.rule import DEFAULT_CLASSES
from sfs.demo.search import COLLECTIONS, LEVELS, VLM_SECONDS_PER_CALL
from sfs.web.clipmap import ClipMap
from sfs.web.jobs import JobRunner

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD = 4 * 1024 ** 3
FREE_EXAMPLES = ("someone carrying a box out of a building", "two people chatting near a car",
                 "a person on the phone at the bus stop", "a person walking past a parked car")


class Filters(BaseModel):
    site: list[str] = Field(default_factory=list)
    camera: list[str] = Field(default_factory=list)
    date: list[str] = Field(default_factory=list)
    hour_from: int | None = None
    hour_to: int | None = None


class SearchIn(BaseModel):
    query: str
    collection: str = "both"
    level: str = "standard"
    k: int = Field(10, ge=1, le=50)
    classes: list[str] = Field(default_factory=list)
    videos: list[int] = Field(default_factory=list)
    filters: Filters = Field(default_factory=Filters)
    top: int = Field(24, ge=1, le=100)
    system: str = "final"


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def _alert_media(video_id: int, alert_id: int, start_frame: int, end_frame: int, fps: float) -> dict:
    """An alert's clip is its 5 s segment [start, end] of the video's annotated WebM."""
    base = f"/media/alerts/{video_id}"
    return {"media": f"{base}/annotated.webm", "start_s": round(start_frame / fps, 3), "end_s": round(end_frame / fps, 3),
            "thumb": f"{base}/alert_{alert_id}/thumb.jpg"}


def _fmt_t(s: float) -> str:
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


class LockedEmbedder:
    """The ingest worker and the search share the GPU models; one lock keeps their calls apart."""

    def __init__(self, embedder, lock):
        self.e, self.lock = embedder, lock

    def lora(self, crops):
        with self.lock:
            return self.e.lora(crops)

    def frozen(self, crops):
        with self.lock:
            return self.e.frozen(crops)


def create_app(searcher, store, ingest_job, previews_dir: str, static_dir: str | None = None,
               phrasings: dict[str, list[str]] | None = None, gpu_lock=None):
    """`ingest_job(job, progress) -> video id` runs in the background worker (sfs.alerts.ingest in main)."""
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, Response
    from fastapi.staticfiles import StaticFiles

    lock = gpu_lock or threading.RLock()
    phr = phrasings or {}
    text_activity = {t: a for a, ts in phr.items() for t in ts}
    runner = JobRunner(ingest_job)
    cmap = ClipMap(searcher.b.lora.emb, searcher.b.lora.starts)
    cache: dict = {}
    os.makedirs(os.path.join(store.root, "uploads"), exist_ok=True)

    app = FastAPI(title="Forensic Search", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.runner = runner

    # ------------------------------------------------------------------ helpers
    def clip_info(cid: int) -> dict:
        if cid >= 0:
            row = searcher.b.clips.iloc[searcher.pos[cid]]
            return {"id": cid, "source": "meva", "title": f"MEVA · {row.camera} {row.site}",
                    "meta": f"{row.date} · {int(row.hour):02d}h", "activity": row.activity,
                    "camera": row.camera, "site": row.site, "date": row.date, "hour": int(row.hour),
                    "media": f"/media/previews/{cid}_context.gif", "kind": "gif"}
        row = searcher.alert_row(cid)
        near = f" · near {row.near.replace(',', ', ')}" if row.near else ""
        info = {"id": cid, "source": "alert", "title": row.video, "meta": f"{_fmt_t(row.time_s)} · {row.cls}{near}",
                "cls": row.cls, "near": [n for n in row.near.split(",") if n], "video_id": int(row.video_id),
                "alert_id": int(row.id), "time_s": float(row.time_s), "conf": float(row.conf), "kind": "video"}
        return info | _alert_media(int(row.video_id), int(row.id), row.start_frame, row.end_frame, row.fps)

    def alert_points():
        ac = searcher.alert_corpus()
        key = store.version
        if cache.get("alert_points", (None,))[0] != key:
            pts = cmap.project(ac.lora_emb, ac.lora_starts) if ac is not None and len(ac) else np.zeros((0, 3))
            cache["alert_points"] = (key, ac.ids if ac is not None else np.zeros(0, np.int64), pts)
        return cache["alert_points"][1:]

    def point_of(ids) -> dict[int, list[float]]:
        a_ids, a_pts = alert_points()
        a_pos = {int(i): n for n, i in enumerate(a_ids)}
        out = {}
        for i in ids:
            i = int(i)
            p = cmap.meva[searcher.pos[i]] if i >= 0 else a_pts[a_pos[i]]
            out[i] = [round(float(v), 4) for v in p]
        return out

    # ------------------------------------------------------------------ status
    @app.get("/api/status")
    def status():
        import torch
        ac = searcher.alert_corpus()
        return {"meva_clips": int(len(searcher.b.ids)), "alerts": int(len(ac)) if ac is not None else 0,
                "videos": len(store.videos()), "device": searcher.device,
                "gpu": torch.cuda.get_device_name(0) if searcher.device == "cuda" else None,
                "vlm": searcher.vlm_available(), "levels": list(LEVELS), "collections": list(COLLECTIONS),
                "classes": list(DEFAULT_CLASSES), "examples": [ts[0] for _, ts in sorted(phr.items())] + list(FREE_EXAMPLES),
                "filters": {"site": sorted(searcher.b.clips.site.unique()), "camera": sorted(searcher.b.clips.camera.unique()),
                            "date": sorted(searcher.b.clips.date.unique()),
                            "hours": [int(searcher.b.clips.hour.min()), int(searcher.b.clips.hour.max())]}}

    # ------------------------------------------------------------------ detect tab
    @app.post("/api/videos")
    async def upload(file: UploadFile = File(...), classes: str = Form(",".join(DEFAULT_CLASSES))):
        name = os.path.basename(file.filename or "video")
        ext = os.path.splitext(name)[1].lower()
        if ext not in VIDEO_EXT:
            raise HTTPException(400, f"not a video file type: {ext or 'none'} (use {', '.join(sorted(VIDEO_EXT))})")
        chosen = tuple(c for c in classes.split(",") if c)
        bad = [c for c in chosen if c not in DEFAULT_CLASSES]
        if bad or not chosen:
            raise HTTPException(400, f"alert classes must be a non-empty subset of {list(DEFAULT_CLASSES)}")
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)[-80:]
        path = os.path.join(store.root, "uploads", f"{uuid.uuid4().hex[:8]}_{safe}")
        size = 0
        with open(path, "wb") as f:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    f.close()
                    os.remove(path)
                    raise HTTPException(413, "video larger than 4 GB")
                f.write(chunk)
        return runner.submit(name, path, chosen).public()

    @app.get("/api/jobs")
    def jobs():
        return sorted((j.public() for j in runner.jobs.values()), key=lambda j: -j["created"])

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        if job_id not in runner.jobs:
            raise HTTPException(404, "no such job")
        return runner.jobs[job_id].public()

    @app.get("/api/videos")
    def videos():
        return store.videos()

    @app.get("/api/videos/{video_id}/alerts")
    def video_alerts(video_id: int):
        al = store.alerts(video_id)
        out = []
        for r in al.itertuples():
            out.append({"id": -int(r.id), "alert_id": int(r.id), "cls": r.cls, "near": [n for n in r.near.split(",") if n],
                        "time_s": float(r.time_s), "time": _fmt_t(r.time_s), "conf": float(r.conf), "full": bool(r.full)}
                       | _alert_media(video_id, int(r.id), r.start_frame, r.end_frame, r.fps))
        return out

    @app.delete("/api/videos/{video_id}")
    def delete_video(video_id: int):
        v = next((v for v in store.videos() if v["id"] == video_id), None)
        if v is None:
            raise HTTPException(404, "no such video")
        if v["status"] == "processing":
            raise HTTPException(409, "the video is still being processed")
        store.delete_video(video_id)
        return {"deleted": video_id}

    # ------------------------------------------------------------------ search tab
    @app.post("/api/search")
    def search(body: SearchIn):
        text = body.query.strip()
        if not text:
            raise HTTPException(400, "type a description of what you are looking for")
        if body.level not in LEVELS or body.collection not in COLLECTIONS or body.system not in ("final", "frozen"):
            raise HTTPException(400, "unknown level, collection or system")
        if body.level == "verified" and body.system == "final" and not searcher.vlm_available():
            raise HTTPException(409, "the Verified level needs the VLM on a GPU; it is not available here")
        f = body.filters
        filters = {"site": f.site or None, "camera": f.camera or None, "date": f.date or None,
                   "hour_from": f.hour_from, "hour_to": f.hour_to}
        with lock:
            r = searcher.search(text, level=body.level, k=body.k, top=body.top, filters=filters,
                                collection=body.collection, classes=body.classes or None,
                                videos=body.videos or None, system=body.system)
        want = text_activity.get(text)
        pts = point_of(r.ids)
        results = []
        for n, cid in enumerate(r.ids):
            cid = int(cid)
            sc = {k: _num(v[n]) for k, v in r.scores.items()}
            vlm = sc.get("vlm")
            item = clip_info(cid) | {"rank": n + 1, "band": r.bands[n], "scores": sc, "point": pts[cid],
                                     "vlm_prob": None if vlm is None else 1 / (1 + math.exp(-vlm))}
            if want is not None and cid >= 0:
                item["match"] = item["activity"] == want
            results.append(item)
        top10 = [pts[int(i)] for i in r.ids[:10]]
        total = sum(r.seconds.values())
        return {"results": results, "level": r.level, "system": r.system, "collection": r.collection,
                "expected_activity": want,
                "query_point": [round(float(v), 4) for v in np.mean(top10, 0)] if top10 else None,
                "cost": {"seconds": r.seconds, "total": total, "vlm_calls": r.vlm_calls, "n_searched": r.n_searched,
                         "vlm_all_seconds": r.vlm_all_seconds, "vlm_per_call": r.vlm_seconds_per_call,
                         "vlm_per_call_source": "measured now" if r.vlm_calls else f"Result 04 ({VLM_SECONDS_PER_CALL} s)"}}

    @app.get("/api/clips/{clip_id}")
    def clip(clip_id: int):
        try:
            info = clip_info(clip_id)
        except (KeyError, IndexError):
            raise HTTPException(404, "no such clip")
        if clip_id >= 0:
            n = len(searcher.clip_frames(clip_id, k=16)[1])
            info["frames"] = [f"/api/clips/{clip_id}/frame/{j}.jpg" for j in range(n)]
        else:
            import json as _json
            row = searcher.alert_row(clip_id)
            info["frames"] = [f"/media/alerts/{int(row.video_id)}/alert_{int(row.id)}/frame_{j:02d}.jpg"
                              for j in range(len(_json.loads(row.frames)))]
        return info

    @app.get("/api/clips/{clip_id}/frame/{j}.jpg")
    def meva_frame(clip_id: int, j: int):
        import cv2
        if clip_id < 0:
            raise HTTPException(404, "alert frames are served under /media/alerts")
        frames, _, _ = searcher.clip_frames(clip_id, k=16)
        if not 0 <= j < len(frames):
            raise HTTPException(404, "no such frame")
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frames[j], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/map")
    def clip_map():
        a_ids, a_pts = alert_points()
        meta = searcher.alert_corpus().meta if len(a_ids) else None
        return {"meva": {"ids": [int(i) for i in searcher.b.ids], "points": np.round(cmap.meva, 4).tolist(),
                         "labels": searcher.b.clips.activity.tolist()},
                "alerts": {"ids": [int(i) for i in a_ids], "points": np.round(a_pts, 4).tolist(),
                           "labels": meta.cls.tolist() if meta is not None else []}}

    # ------------------------------------------------------------------ media + front end
    os.makedirs(os.path.join(store.root, "media"), exist_ok=True)
    app.mount("/media/alerts", StaticFiles(directory=os.path.join(store.root, "media")), name="alerts")
    if os.path.isdir(previews_dir):
        app.mount("/media/previews", StaticFiles(directory=previews_dir), name="previews")
    if static_dir and os.path.isdir(static_dir):
        index = os.path.join(static_dir, "index.html")
        app.mount("/assets", StaticFiles(directory=os.path.join(static_dir, "assets")), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def front(path: str):
            p = os.path.normpath(os.path.join(static_dir, path))
            if path and p.startswith(os.path.normpath(static_dir)) and os.path.isfile(p):
                return FileResponse(p)
            return FileResponse(index)
    return app


def build_parser() -> argparse.ArgumentParser:
    from sfs.alerts.detect import DEFAULT_WEIGHTS
    from sfs.alerts.embed import DEFAULT_ADAPTER
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", default="data/demo/A")
    ap.add_argument("--crops", default="data/crops_v2")
    ap.add_argument("--alerts", default="data/alerts")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--phrasings", default="queries/phrasings.csv")
    ap.add_argument("--static", default="web/dist")
    ap.add_argument("--no-vlm", action="store_true", help="hide the Verified level's VLM")
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to listen on: 127.0.0.1 on the laptop; 0.0.0.0 inside Docker, where Compose "
                         "publishes the port on the laptop's loopback only (spec-docker §1)")
    ap.add_argument("--port", type=int, default=8000)
    return ap


def main(argv=None) -> int:
    from sfs.alerts.detect import YoloTracker
    from sfs.alerts.embed import ClipEmbedder
    from sfs.alerts.ingest import ingest
    from sfs.alerts.store import AlertStore
    from sfs.demo.search import Bundle, Searcher
    from sfs.eval.queries import load_phrasings
    a = build_parser().parse_args(argv)
    store = AlertStore(a.alerts)
    s = Searcher(Bundle(a.bundle), a.crops, alerts=store)
    if a.no_vlm:
        s.vlm_path = ""
    lock = threading.RLock()
    print("loading models ...", flush=True)
    s.embed("warm up")
    _ = s.student
    if s.vlm_available():
        _ = s.vlm
    embedder = LockedEmbedder(ClipEmbedder(s.encoder, a.adapter), lock)
    trackers: dict = {}

    def run(job, progress):
        key = tuple(job.classes)
        if key not in trackers:
            trackers[key] = YoloTracker(a.weights, key)
        try:
            return ingest(job.path, store, trackers[key], embedder, key, job.name, progress)
        finally:
            if os.path.exists(job.path):
                os.remove(job.path)                 # the store keeps its own copy of the video

    app = create_app(s, store, run, os.path.join(a.bundle, "previews"), a.static, load_phrasings(a.phrasings), lock)
    import uvicorn
    shown = "127.0.0.1" if a.host == "0.0.0.0" else a.host     # what to open in the browser
    print(f"open http://{shown}:{a.port}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
