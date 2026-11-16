"""Alert rule (spec-video-alerts-dashboard §4.2, owner decision D1).

YOLO + ByteTrack report the same object in every frame; the rule decides when a detection is a NEW alert:
- an object raises an alert once its track is confirmed (seen in `confirm_frames` frames);
- the alert is held while the object stays in view;
- the same object raises a new alert only after it was gone for >= `gone_s` and is detected again
  (ByteTrack drops a track lost for longer than its buffer, so the object usually returns under a new
  id; both cases are handled here);
- `repeat_s` (off by default) re-raises an alert every `repeat_s` seconds while the object stays.
Only `classes` raise alerts; every tracked detection can be "near" one.

Measured on real footage (D-061): with a 3-frame confirmation, a 14 s crowded clip (at most 6 people at
once) raised 38 alerts, mostly from low-confidence boxes and from ByteTrack giving an occluded person a
new id. Hence: only detections with confidence >= `min_conf` count towards the confirmation, an object
must be seen for `confirm_s`, and a new track that appears where a track of the same class was lost
within `gone_s` is the same object (track stitching), so it keeps that object's alert.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sfs.data.kpf import Box

# person, vehicles and one common object (owner D3; COCO names as YOLO reports them)
DEFAULT_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck", "backpack")


@dataclass(frozen=True)
class Det:
    track_id: int
    cls: str
    conf: float
    box: Box


@dataclass(frozen=True)
class Alert:
    track_id: int
    cls: str
    frame: int                       # the frame the alert fired at
    conf: float                      # YOLO confidence at that frame
    near_ids: tuple[int, ...] = ()   # tracks next to it at that frame
    near_classes: tuple[str, ...] = ()


@dataclass
class _Track:
    cls: str = ""
    seen: int = 0
    last_seen: int = -1
    last_box: Box | None = None
    alert_frame: int | None = None   # frame of the alert currently held; None = none held
    stitched: bool = False           # its alert was handed on to a track that replaced it


def is_near(a: Box, b: Box, scale: float = 1.0) -> bool:
    """Boxes overlap, or the gap between them is at most `scale` x the height of `a`."""
    gap_x = max(0, max(a[0] - b[2], b[0] - a[2]))
    gap_y = max(0, max(a[1] - b[3], b[1] - a[3]))
    return max(gap_x, gap_y) <= scale * max(1, a[3] - a[1])


@dataclass
class AlertRule:
    fps: float
    classes: tuple[str, ...] = DEFAULT_CLASSES
    confirm_s: float = 0.5
    min_conf: float = 0.5
    gone_s: float = 2.0
    repeat_s: float | None = None
    near_scale: float = 1.0
    stitch: bool = True
    _tracks: dict[int, _Track] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.fps <= 0 or self.confirm_s < 0 or self.gone_s <= 0:
            raise ValueError("fps and gone_s must be positive, confirm_s >= 0")
        if self.repeat_s is not None and self.repeat_s <= 0:
            raise ValueError("repeat_s must be positive or None")

    @property
    def gone_frames(self) -> int:
        return max(1, round(self.gone_s * self.fps))

    @property
    def confirm_frames(self) -> int:
        return max(1, round(self.confirm_s * self.fps))

    def _replaced(self, frame: int, d: Det, present: set[int]) -> _Track | None:
        """A recently lost track of the same class, next to `d`, whose alert `d` should inherit."""
        best, best_gap = None, None
        for tid, t in self._tracks.items():
            if (tid in present or t.cls != d.cls or t.alert_frame is None or t.stitched or t.last_box is None
                    or frame - t.last_seen > self.gone_frames or not is_near(t.last_box, d.box, 0.5)):
                continue
            gap = frame - t.last_seen
            if best_gap is None or gap < best_gap:
                best, best_gap = t, gap
        return best

    def feed(self, frame: int, dets: list[Det]) -> list[Alert]:
        """Detections of one frame (frames fed in increasing order) -> alerts that fire at this frame."""
        out = []
        present = {d.track_id for d in dets}
        for d in dets:
            if d.cls not in self.classes:
                continue
            t = self._tracks.setdefault(d.track_id, _Track(cls=d.cls))
            if t.last_seen >= 0 and frame - t.last_seen > self.gone_frames:
                t.seen, t.alert_frame, t.stitched = 0, None, False      # it was gone: start again
            t.last_seen, t.last_box = frame, d.box
            if d.conf < self.min_conf:
                continue
            t.seen += 1
            fire = t.alert_frame is None and t.seen >= self.confirm_frames
            if fire and self.stitch:
                old = self._replaced(frame, d, present)
                if old is not None:                      # the same object under a new track id
                    t.alert_frame, old.stitched, fire = old.alert_frame, True, False
            if not fire and t.alert_frame is not None and self.repeat_s is not None:
                fire = frame - t.alert_frame >= round(self.repeat_s * self.fps)
            if fire:
                t.alert_frame = frame
                near = [o for o in dets if o.track_id != d.track_id and is_near(d.box, o.box, self.near_scale)]
                out.append(Alert(d.track_id, d.cls, frame, d.conf, tuple(o.track_id for o in near),
                                 tuple(sorted({o.cls for o in near}))))
        return out
