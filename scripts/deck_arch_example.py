"""One worked example for the architecture slide of the 6-slide deck (scripts/deck6_build.js, slide 4).

Rule: query "a person picking up an object" (the first frozen phrasing of person_picks_up_object, a motion-defined
activity, the case the temporal student was built for) over the MEVA demo index (protocol A eval split, 3,726 clips);
the example is the top-ranked clip at Standard with that label, at least 2 frames and an actor box in every frame;
up to 4 of its frames are saved (pick-up clips are mostly 2–3 frames long at 2 fps).
An illustration, not a measurement: a first try with "a person opening a car door" gave crops dominated by the cars
(the box covers the vehicle too), so the activity was changed before the slide was built. For the clip the
script records the rank and score at each stage (frozen SigLIP 2, + LoRA, + temporal student), the first numbers of
one frame's 768-number vector before and after LoRA, and saves 4 of its tight crops as PNGs.

    python scripts/deck_arch_example.py      # -> report/figures/arch_example.json, arch_frame_0..3.png
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "src")
from sfs.demo.search import Bundle, Searcher  # noqa: E402
from sfs.teacher.frames import pick_even  # noqa: E402

QUERY = "a person picking up an object"
ACTIVITY = "person_picks_up_object"
MIN_FRAMES = 2
OUT = "report/figures"


def rank_of(res, clip_id: int) -> int:
    return int(np.flatnonzero(res.ids == clip_id)[0]) + 1


def main() -> int:
    b = Bundle("data/demo/A")
    s = Searcher(b)
    q = s.embed(QUERY)
    n = len(b.ids)
    frozen = s.search_vector(q, QUERY, level="fast", system="frozen", top=n)
    fast = s.search_vector(q, QUERY, level="fast", top=n)
    std = s.search_vector(q, QUERY, level="standard", top=n)
    act = b.clips.set_index("instance_id").activity
    def usable(i: int) -> bool:
        r = s.pos[i]
        m = b.mask[r]
        return act[i] == ACTIVITY and m.sum() >= MIN_FRAMES and bool((b.feats[r, :, -1][m] > 0).all())
    clip = next(int(i) for i in std.ids if usable(int(i)))
    row = s.pos[clip]
    lo, lf = b.lora.starts[row], b.frozen.starts[row]
    n_frames = int(b.lora.starts[row + 1] - lo) if row + 1 < len(b.lora.starts) else len(b.lora.emb) - lo
    first = lambda v: [round(float(x), 3) for x in v[:3]]            # noqa: E731
    out = {
        "query": QUERY, "activity": ACTIVITY, "instance_id": clip, "corpus": n,
        "rule": f"top-ranked clip at Standard whose label is the query's activity, >= {MIN_FRAMES} frames, a box in every frame",
        "frames_in_clip": n_frames,
        "frame0_frozen_first3": first(b.frozen.emb[lf]), "frame0_lora_first3": first(b.lora.emb[lo]),
        "score_frozen": float(frozen.scores["final"][rank_of(frozen, clip) - 1]),
        "score_lora": float(fast.scores["final"][rank_of(fast, clip) - 1]),
        "score_student": float(std.scores["student"][rank_of(std, clip) - 1]),
        "rank_frozen": rank_of(frozen, clip), "rank_lora": rank_of(fast, clip), "rank_standard": rank_of(std, clip),
        "positives_in_corpus": int((act == ACTIVITY).sum()),
    }
    frames, _, _ = s.clip_frames(clip, k=16)
    picked = pick_even(len(frames), 4)
    out["frames_shown"] = len(picked)
    for old in range(len(picked), 4):                                 # no stale frames from an earlier run
        if os.path.exists(os.path.join(OUT, f"arch_frame_{old}.png")):
            os.remove(os.path.join(OUT, f"arch_frame_{old}.png"))
    for k, i in enumerate(picked):
        cv2.imwrite(os.path.join(OUT, f"arch_frame_{k}.png"), cv2.cvtColor(cv2.resize(frames[i], (224, 224)), cv2.COLOR_RGB2BGR))
    with open(os.path.join(OUT, "arch_example.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
