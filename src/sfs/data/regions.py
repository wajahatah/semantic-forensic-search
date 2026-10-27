"""Image regions per view (spec-offline-embedding D4, amended).

- tight:   union of actor boxes, padded, squared, min side      -> detail on the actors
- context: union expanded `scale`x, squared, larger min side   -> actors + surroundings
- full:    whole frame                                          -> ablation

All square regions are clipped *by shifting* so they stay square and inside the image.
The same functions must be used at alert-ingestion time, or offline results will not transfer.
"""
from __future__ import annotations

import numpy as np

from sfs.data.kpf import Box, Tracks

VIEWS = ("tight", "context", "full")


def union(boxes: list[Box]) -> Box:
    if not boxes:
        raise ValueError("union of no boxes")
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def lookup_union(tracks: Tracks, actor_ids, frame: int, window: int = 5) -> Box | None:
    """Union of the actors' boxes at `frame`, using the nearest box within ±window per actor."""
    found: list[Box] = []
    for actor in actor_ids:
        track = tracks.get(actor)
        if not track:
            continue
        if frame in track:
            found.append(track[frame])
            continue
        for d in range(1, window + 1):
            box = track.get(frame - d) or track.get(frame + d)
            if box:
                found.append(box)
                break
    return union(found) if found else None


def _square(cx: float, cy: float, side: float, img_w: int, img_h: int) -> Box:
    # Round the side ONCE and derive both edges from it; rounding each edge separately can
    # produce a non-square box (banker's rounding: round(1042.5)=1042, round(1405.5)=1406).
    side_i = int(round(min(side, img_w, img_h)))
    x1 = int(round(min(max(cx - side_i / 2, 0), img_w - side_i)))
    y1 = int(round(min(max(cy - side_i / 2, 0), img_h - side_i)))
    return (x1, y1, x1 + side_i, y1 + side_i)


def _wh(box: Box) -> tuple[int, int]:
    return max(1, box[2] - box[0]), max(1, box[3] - box[1])


def tight_region(box: Box, img_w: int, img_h: int, pad: float = 0.25, min_side: int = 224) -> Box:
    w, h = _wh(box)
    side = max(w * (1 + 2 * pad), h * (1 + 2 * pad), min_side)
    return _square((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, side, img_w, img_h)


def context_region(box: Box, img_w: int, img_h: int, scale: float = 3.0, min_side: int = 448) -> Box:
    w, h = _wh(box)
    side = max(w * scale, h * scale, min_side)
    return _square((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, side, img_w, img_h)


def full_region(img_w: int, img_h: int) -> Box:
    return (0, 0, img_w, img_h)


def crop(frame: np.ndarray, box: Box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return frame[y1:y2, x1:x2]


def box_side(box: Box | None) -> int:
    """Largest side of the actor box in pixels (0 if none) — recorded for accuracy-vs-size analysis."""
    return 0 if box is None else max(_wh(box))
