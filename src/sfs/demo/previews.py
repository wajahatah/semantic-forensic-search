"""Clip previews (spec-search-demo §4.4): a clip's stored crops as a short looping animation.

The crop store keeps <= 16 frames per clip at 2 fps; they are played at 4 fps, scaled to fit
`side` px. Rendered once per clip into <bundle>/previews/<id>_<view>.gif and read from there.
"""
from __future__ import annotations

import io
import os

import numpy as np

PLAY_FPS = 4
SIDE = 256
MAX_FRAMES = 16


def to_gif(frames: list[np.ndarray], side: int = SIDE, fps: int = PLAY_FPS) -> bytes:
    """RGB uint8 frames (any sizes) -> looping GIF, each frame letterboxed onto one side x side canvas."""
    from PIL import Image
    if not frames:
        raise ValueError("no frames")
    out = []
    for f in frames:
        im = Image.fromarray(np.asarray(f, np.uint8))
        im.thumbnail((side, side), Image.BILINEAR)
        canvas = Image.new("RGB", (side, side))
        canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
        out.append(canvas)
    buf = io.BytesIO()
    out[0].save(buf, format="GIF", save_all=True, append_images=out[1:], duration=int(1000 / fps), loop=0)
    return buf.getvalue()


class Previews:
    def __init__(self, crops, cache_dir: str, view: str = "context"):
        self.crops, self.dir, self.view = crops, cache_dir, view
        os.makedirs(cache_dir, exist_ok=True)

    def path(self, instance_id: int) -> str:
        p = os.path.join(self.dir, f"{int(instance_id)}_{self.view}.gif")
        if not os.path.exists(p):
            frames, _ = self.crops.frames(int(instance_id), self.view, MAX_FRAMES)
            tmp = p + ".tmp"
            with open(tmp, "wb") as f:
                f.write(to_gif(frames))
            os.replace(tmp, p)
        return p

    def frames(self, instance_id: int, view: str = "tight") -> tuple[list[np.ndarray], list[int]]:
        return self.crops.frames(int(instance_id), view, MAX_FRAMES)

    def render_all(self, ids, log=print) -> int:
        """Render every missing preview, visiting clips grouped by crop shard (one shard read each)."""
        todo = [int(i) for i in ids if not os.path.exists(os.path.join(self.dir, f"{int(i)}_{self.view}.gif"))]
        todo.sort(key=lambda i: (self.crops.shard_of(i), i))
        for n, i in enumerate(todo, 1):
            self.path(i)
            if n % 500 == 0:
                log(f"previews: {n:,}/{len(todo):,}")
        return len(todo)
