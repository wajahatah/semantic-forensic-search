"""The alert clip (spec-video-alerts-dashboard §4.3, owner decision D2): exactly 5 seconds.

5 frames before + the alert frame + 5 after, sampled at 2 fps (the rate the models were trained and
evaluated on), i.e. 11 frames over 5.0 s. At the start or end of the video the window shifts to stay
inside it, so the clip stays 5 s; only a video shorter than 5 s gives a shorter clip.
"""
from __future__ import annotations

from dataclasses import dataclass

BEFORE = AFTER = 5
STEP_S = 0.5                          # 2 fps
CLIP_S = (BEFORE + AFTER) * STEP_S    # 5.0 s


@dataclass(frozen=True)
class ClipWindow:
    frames: tuple[int, ...]   # the sampled frames (embedded, shown in the frame strip)
    start: int                # first video frame of the 5 s window (playback)
    end: int                  # last video frame of the window, inclusive
    full: bool                # False only when the video is shorter than the clip


def clip_window(alert_frame: int, fps: float, n_frames: int, before: int = BEFORE, after: int = AFTER,
                step_s: float = STEP_S) -> ClipWindow:
    if fps <= 0 or n_frames < 1:
        raise ValueError("fps must be positive and the video must have frames")
    if not 0 <= alert_frame < n_frames:
        raise ValueError(f"alert frame {alert_frame} outside the video (0..{n_frames - 1})")
    span = (before + after) * step_s
    last_t = (n_frames - 1) / fps
    start_t = alert_frame / fps - before * step_s
    start_t = 0.0 if last_t < span else min(max(start_t, 0.0), last_t - span)
    times = [start_t + i * step_s for i in range(before + after + 1)]
    frames = sorted({min(n_frames - 1, round(t * fps)) for t in times if t <= last_t + 1e-9})
    end = min(n_frames - 1, round((start_t + span) * fps))
    return ClipWindow(tuple(frames), round(start_t * fps), end, last_t >= span)
