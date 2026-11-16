"""YOLO detection + ByteTrack tracking over a whole video (spec-video-alerts-dashboard §4.1).

`YoloTracker.track(video)` yields (frame_index, [Det]) for every frame, in order. ByteTrack keeps a lost
track for `gone_s` seconds (its `track_buffer`), the same time the alert rule waits before an object
counts as gone. Ultralytics is imported only here, so the rest of the package and its tests run without it.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from typing import Iterator, Protocol

from sfs.alerts.rule import DEFAULT_CLASSES, Det

DEFAULT_WEIGHTS = "data/models/yolo/yolov8m.pt"


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    n_frames: int          # as reported by the container; the detection pass gives the real count


def probe(video_path: str) -> VideoInfo:
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        info = VideoInfo(fps, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                         int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()
    if not fps or fps <= 0 or info.width <= 0 or info.height <= 0:
        raise IOError(f"invalid video properties for {video_path}: {info}")
    return info


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Tracker(Protocol):
    weights_id: str

    def track(self, video_path: str, fps: float) -> Iterator[tuple[int, list[Det]]]: ...


class YoloTracker:
    def __init__(self, weights: str = DEFAULT_WEIGHTS, classes: tuple[str, ...] = DEFAULT_CLASSES,
                 conf: float = 0.25, gone_s: float = 2.0, device: str | None = None):
        from ultralytics import YOLO
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        self.model = YOLO(weights)
        names = self.model.names                                 # {id: name}
        unknown = [c for c in classes if c not in names.values()]
        if unknown:
            raise ValueError(f"classes not in the model: {unknown}")
        self.class_ids = [i for i, n in names.items() if n in classes]
        self.names, self.weights = names, weights
        self.conf, self.gone_s, self.device = conf, gone_s, device
        self.weights_id = f"{os.path.basename(weights)}:{file_sha256(weights)[:12]}"

    def _tracker_yaml(self, fps: float, out_dir: str) -> str:
        import yaml
        from ultralytics.utils import ROOT
        with open(ROOT / "cfg" / "trackers" / "bytetrack.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # ByteTracker drops a track after `track_buffer` frames (it is built with frame_rate=30 and
        # scales by frame_rate / 30, i.e. not at all), so the buffer is given in this video's frames
        cfg["track_buffer"] = max(1, round(self.gone_s * fps))
        path = os.path.join(out_dir, "bytetrack.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        return path

    def track(self, video_path: str, fps: float) -> Iterator[tuple[int, list[Det]]]:
        from ultralytics import YOLO
        model = YOLO(self.weights)          # a fresh tracker per video: persist=True keeps ids within one video only
        with tempfile.TemporaryDirectory() as tmp:
            results = model.track(source=video_path, stream=True, persist=True, verbose=False,
                                       tracker=self._tracker_yaml(fps, tmp), classes=self.class_ids,
                                       conf=self.conf, device=self.device)
            for idx, r in enumerate(results):
                b = r.boxes
                if b is None or b.id is None:
                    yield idx, []
                    continue
                ids = b.id.int().tolist()
                cls = b.cls.int().tolist()
                conf = b.conf.tolist()
                xyxy = b.xyxy.round().int().tolist()
                yield idx, [Det(int(t), self.names[c], float(p), tuple(int(v) for v in box))
                            for t, c, p, box in zip(ids, cls, conf, xyxy)]
