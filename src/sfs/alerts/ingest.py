"""Video file -> searchable alerts (spec-video-alerts-dashboard §4).

    python -m sfs.alerts.ingest path/to/video.mp4 [--name NAME]

1. Detection pass: YOLO + ByteTrack over every frame -> tracks; the alert rule (rule.py) fires alerts.
2. Each alert gets its 5 s clip (clip.py): 11 frames at 2 fps. Its actors = the alert's object plus
   the objects next to it, exactly like MEVA's multi-actor activities.
3. Crop pass (sequential decode, no seeking): the unchanged `regions_for` gives the tight / context
   regions per frame; the same embeddings and student features as the MEVA stores are computed
   (embed.py, `box_features`).
   The whole video is written once as a playable WebM with every tracked box drawn; an alert's clip is its
   5 s segment of that video. One WebM per alert was measured at 4.1 s of encoding per alert (D-061).
4. The vectors and alert rows are written to the store; the video becomes searchable.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from sfs.alerts.clip import ClipWindow, clip_window
from sfs.alerts.detect import VideoInfo, file_sha256, probe
from sfs.alerts.rule import DEFAULT_CLASSES, Alert, AlertRule
from sfs.data.kpf import Tracks
from sfs.data.regions import VIEWS, lookup_union
from sfs.embed.run import EmbedConfig, iter_frames, regions_for
from sfs.rerank.features import BOX_FEATURES, EMB, T_MAX, box_features

CFG = EmbedConfig(views=("tight", "context"))
PLAYBACK_FPS = 15
PLAYBACK_MAX_W = 960
FRAME_MAX_W = 960
THUMB_W = 320
COLOR_MAIN = (34, 211, 238)      # RGB: the alert's object
COLOR_NEAR = (245, 158, 11)      # RGB: objects next to it
CLASS_COLOR = {"person": (34, 211, 238), "backpack": (167, 139, 250)}   # vehicles: COLOR_NEAR
ANNOTATED = "annotated.webm"

Progress = Callable[[str, float, str], None]


def _quiet(stage: str, frac: float, msg: str) -> None:
    pass


@dataclass
class _Work:
    """One alert while its frames are being read."""
    n: int
    alert: Alert
    window: ClipWindow
    actors: tuple[int, ...]
    tight: list[np.ndarray] = field(default_factory=list)
    context: list[np.ndarray] = field(default_factory=list)
    sides: list[int] = field(default_factory=list)
    read: list[int] = field(default_factory=list)    # sampled frames actually decoded

    @property
    def last(self) -> int:
        """The last video frame this alert needs; once it is read the alert can be finished."""
        return max(self.window.frames)

    @property
    def thumb_frame(self) -> int:
        """The sampled frame closest to the alert (the alert frame itself unless the window shifted)."""
        return min(self.window.frames, key=lambda f: (abs(f - self.alert.frame), f))


def _scaled(w: int, h: int, max_w: int) -> tuple[int, int]:
    s = min(1.0, max_w / w)
    return max(2, int(w * s) // 2 * 2), max(2, int(h * s) // 2 * 2)


def draw(rgb: np.ndarray, tracks: Tracks, classes: dict[int, str], main: int, others: tuple[int, ...],
         frame: int, window: int = 5) -> np.ndarray:
    """A copy of the frame with the alert's boxes and labels drawn."""
    import cv2
    out = rgb.copy()
    thick = max(2, round(out.shape[1] / 640))
    for tid, color in [(t, COLOR_NEAR) for t in others] + [(main, COLOR_MAIN)]:
        box = lookup_union(tracks, (tid,), frame, window)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        label = classes.get(tid, "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thick, thick)
        y0 = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, y0), (x1 + tw + 6, y0 + th + 6), color, -1)
        cv2.putText(out, label, (x1 + 3, y0 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thick, (8, 16, 22), thick)
    return out


def draw_all(rgb: np.ndarray, tracks: Tracks, classes: dict[int, str], frame: int) -> np.ndarray:
    """A copy of the frame with every object tracked at this frame drawn, coloured by class."""
    import cv2
    out = rgb.copy()
    thick = max(2, round(out.shape[1] / 640))
    for tid, track in tracks.items():
        box = track.get(frame)
        if box is None:
            continue
        cls = classes.get(tid, "")
        color = CLASS_COLOR.get(cls, COLOR_NEAR)
        cv2.rectangle(out, box[:2], box[2:], color, thick)
        (tw, th), _ = cv2.getTextSize(cls, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thick, thick)
        y0 = max(0, box[1] - th - 6)
        cv2.rectangle(out, (box[0], y0), (box[0] + tw + 6, y0 + th + 6), color, -1)
        cv2.putText(out, cls, (box[0] + 3, y0 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thick, (8, 16, 22), thick)
    return out


def _save_jpg(path: str, rgb: np.ndarray, max_w: int, quality: int = 88) -> None:
    import cv2
    h, w = rgb.shape[:2]
    if w > max_w:
        rgb = cv2.resize(rgb, _scaled(w, h, max_w), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])


def detect_pass(video_path: str, info: VideoInfo, tracker, rule: AlertRule, progress: Progress):
    tracks: Tracks = defaultdict(dict)
    classes: dict[int, str] = {}
    alerts: list[Alert] = []
    n = 0
    for idx, dets in tracker.track(video_path, info.fps):
        for d in dets:
            tracks[d.track_id][idx] = d.box
            classes.setdefault(d.track_id, d.cls)
        alerts.extend(rule.feed(idx, dets))
        n = idx + 1
        if info.n_frames and idx % 25 == 0:
            progress("detect", min(1.0, idx / info.n_frames), f"frame {idx:,} / {info.n_frames:,}")
    return dict(tracks), classes, alerts, n


def ingest(video_path: str, store, tracker, embedder, classes: tuple[str, ...] = DEFAULT_CLASSES,
           name: str | None = None, progress: Progress = _quiet, gone_s: float = 2.0,
           repeat_s: float | None = None) -> int:
    """Returns the video id. A video already in the store (same sha256) is not ingested twice."""
    import cv2
    sha = file_sha256(video_path)
    known = store.find_video(sha)
    if known is not None and known["status"] == "ready":
        progress("done", 1.0, "already in the store")
        return int(known["id"])
    if known is not None:
        store.delete_video(known["id"])           # failed or interrupted earlier: start again
    info = probe(video_path)
    vid = store.add_video(name or os.path.basename(video_path), sha, info, tracker.weights_id, classes)
    try:
        media = store.media_dir(vid)
        os.makedirs(media, exist_ok=True)
        dst = os.path.join(media, "video" + os.path.splitext(video_path)[1].lower())
        shutil.copyfile(video_path, dst)
        store.set_video(vid, path=dst)

        progress("detect", 0.0, "detecting and tracking")
        rule = AlertRule(info.fps, classes, gone_s=gone_s, repeat_s=repeat_s)
        tracks, cls_of, alerts, n_frames = detect_pass(video_path, info, tracker, rule, progress)
        if n_frames == 0:
            raise IOError("no frames could be decoded")
        store.set_video(vid, n_frames=n_frames, duration_s=n_frames / info.fps)
        progress("detect", 1.0, f"{len(alerts)} alerts")

        stride = max(1, round(info.fps / PLAYBACK_FPS))
        playback = set(range(0, n_frames, stride))
        work = [_Work(k, a, clip_window(a.frame, info.fps, n_frames), (a.track_id, *a.near_ids))
                for k, a in enumerate(alerts)]
        needed: dict[int, list[_Work]] = {f: [] for f in playback}
        for wk in work:
            for f in wk.window.frames:
                needed.setdefault(f, []).append(wk)

        out_alerts, lora, frozen = [None] * len(work), [None] * len(work), [None] * len(work)
        feats = np.zeros((len(work), T_MAX, 2 * EMB + len(BOX_FEATURES)), np.float16)
        mask = np.zeros((len(work), T_MAX), bool)
        pending = {wk.n for wk in work}
        cap = cv2.VideoCapture(video_path)
        play_size = _scaled(info.width, info.height, PLAYBACK_MAX_W)
        writer = cv2.VideoWriter(os.path.join(media, ANNOTATED), cv2.VideoWriter_fourcc(*"VP80"),
                                 info.fps / stride, play_size)
        try:
            order = sorted(needed)
            for i, (idx, frame) in enumerate(iter_frames(cap, order)):
                if frame is None:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if idx in playback:
                    writer.write(cv2.cvtColor(cv2.resize(draw_all(rgb, tracks, cls_of, idx), play_size,
                                                         interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
                for wk in needed[idx]:
                    others = wk.actors[1:]
                    if idx in wk.window.frames:
                        regions = {VIEWS[vi]: (region, side) for vi, region, _fb, side in
                                   regions_for(wk.actors, idx, tracks, info.width, info.height, CFG)}
                        t_reg, side = regions["tight"]
                        c_reg, _ = regions["context"]
                        wk.tight.append(rgb[t_reg[1]:t_reg[3], t_reg[0]:t_reg[2]].copy())
                        wk.context.append(rgb[c_reg[1]:c_reg[3], c_reg[0]:c_reg[2]].copy())
                        wk.sides.append(int(side))
                        wk.read.append(idx)
                        j = len(wk.tight) - 1
                        d = store.alert_dir(vid, wk.n)
                        os.makedirs(d, exist_ok=True)
                        drawn = draw(rgb, tracks, cls_of, wk.alert.track_id, others, idx)
                        _save_jpg(os.path.join(d, f"frame_{j:02d}.jpg"), drawn, FRAME_MAX_W)
                        cv2.imwrite(os.path.join(d, f"tight_{j:02d}.png"), cv2.cvtColor(wk.tight[-1], cv2.COLOR_RGB2BGR))
                        if idx == wk.thumb_frame:
                            _save_jpg(os.path.join(d, "thumb.jpg"), drawn, THUMB_W, 82)
                    if wk.n in pending and idx == wk.last:
                        _finish(wk, info, tracks, embedder, out_alerts, lora, frozen, feats, mask)
                        pending.discard(wk.n)
                if i % 10 == 0:
                    progress("embed", i / max(1, len(order)), f"{len(work) - len(pending)} / {len(work)} alerts embedded")
            for wk in work:                          # the stream ended early: finish with what was read
                if wk.n in pending and wk.tight:
                    _finish(wk, info, tracks, embedder, out_alerts, lora, frozen, feats, mask)
                    pending.discard(wk.n)
        finally:
            cap.release()
            writer.release()
        keep = [k for k in range(len(work)) if out_alerts[k] is not None]
        ids = store.write_alerts(vid, [out_alerts[k] for k in keep],
                                 {"lora": [lora[k] for k in keep], "frozen": [frozen[k] for k in keep],
                                  "feats": feats[keep], "mask": mask[keep]})
        for k in range(len(work)):                    # an alert without a single decoded frame is dropped
            if out_alerts[k] is None:
                shutil.rmtree(store.alert_dir(vid, k), ignore_errors=True)
        _rename_media(store, vid, keep, ids)
        progress("done", 1.0, f"{len(ids)} alerts ready")
        return vid
    except Exception as e:
        store.set_video(vid, status="failed", error=f"{type(e).__name__}: {e}"[:500])
        raise


def _rename_media(store, vid: int, positions: list[int], ids: list[int]) -> None:
    """Media were written under alert_<position>; the store names them alert_<id>. Two steps avoid clashes."""
    staged = []
    for k, aid in zip(positions, ids):
        src = store.alert_dir(vid, k)
        if os.path.isdir(src):
            tmp = src + ".mv"
            os.replace(src, tmp)
            staged.append((tmp, store.alert_dir(vid, aid)))
    for tmp, dst in staged:
        os.replace(tmp, dst)


def _finish(wk: _Work, info: VideoInfo, tracks: Tracks, embedder, out_alerts, lora, frozen, feats, mask) -> None:
    k = len(wk.tight)
    frames = list(wk.read)
    lora[wk.n] = embedder.lora(wk.tight)
    ctx = embedder.frozen(wk.context)
    frozen[wk.n] = embedder.frozen(wk.tight)
    feats[wk.n, :k, :EMB] = lora[wk.n]
    feats[wk.n, :k, EMB:2 * EMB] = ctx
    feats[wk.n, :k, 2 * EMB:] = box_features(tracks, wk.actors, frames, wk.sides, info.fps)
    mask[wk.n, :k] = True
    out_alerts[wk.n] = {"cls": wk.alert.cls, "near": list(wk.alert.near_classes), "track_id": wk.alert.track_id,
                        "frame": wk.alert.frame, "time_s": wk.alert.frame / info.fps, "conf": wk.alert.conf,
                        "start_frame": wk.window.start, "end_frame": wk.window.end, "frames": frames,
                        "full": wk.window.full}
    wk.tight, wk.context = [], []            # free the crops


def main(argv=None) -> int:
    from sfs.alerts.detect import DEFAULT_WEIGHTS, YoloTracker
    from sfs.alerts.embed import DEFAULT_ADAPTER, ClipEmbedder
    from sfs.alerts.store import AlertStore
    from sfs.embed.encoder import Encoder
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video")
    ap.add_argument("--name")
    ap.add_argument("--store", default="data/alerts")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    a = ap.parse_args(argv)
    classes = tuple(c.strip() for c in a.classes.split(",") if c.strip())
    store = AlertStore(a.store)
    tracker = YoloTracker(a.weights, classes)
    embedder = ClipEmbedder(Encoder("siglip2"), a.adapter)

    def show(stage, frac, msg):
        print(f"\r{stage:7s} {frac:6.1%}  {msg:50s}", end="", flush=True)
    vid = ingest(a.video, store, tracker, embedder, classes, a.name, show)
    print()
    al = store.alerts(vid)
    print(f"video {vid}: {len(al)} alerts")
    for r in al.itertuples():
        print(f"  #{r.id:<5} {r.cls:<11} {('near ' + r.near) if r.near else '':<22} {r.time_s:7.1f} s  conf {r.conf:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
