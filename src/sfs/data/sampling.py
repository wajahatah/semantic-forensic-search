"""Frame sampling for activity instances (spec-offline-embedding D3).

N = clamp(ceil(duration_s * rate), 1, n_max), spread uniformly over the inclusive window.
Rate-based sampling keeps evidence proportional to duration; the cap bounds length bias.
"""
from __future__ import annotations

import math


def n_samples(frame_start: int, frame_end: int, fps: float,
              rate: float = 2.0, n_max: int = 16) -> int:
    if frame_end < frame_start:
        raise ValueError(f"frame_end {frame_end} < frame_start {frame_start}")
    if fps <= 0 or rate <= 0 or n_max < 1:
        raise ValueError("fps and rate must be positive, n_max >= 1")
    duration_s = (frame_end - frame_start + 1) / fps
    return max(1, min(n_max, math.ceil(duration_s * rate)))


def sample_frames(frame_start: int, frame_end: int, fps: float,
                  rate: float = 2.0, n_max: int = 16) -> list[int]:
    """Sorted, unique frame indices inside [frame_start, frame_end]; endpoints included when N >= 2."""
    n = n_samples(frame_start, frame_end, fps, rate, n_max)
    if n == 1:
        return [(frame_start + frame_end) // 2]
    span = frame_end - frame_start
    return sorted({frame_start + round(i * span / (n - 1)) for i in range(n)})
