"""Crop store: one packed ``.npz`` per video holding encoded crops — PNG or JPEG per view (spec-crops-queries-baseline §3, D-014).

~147k crops as individual files would be very slow on a Drive mount, so each video's crops are
concatenated into one byte buffer with offsets. No pickle. Atomic writes, validated reads.
"""
from __future__ import annotations

import json
import os

import numpy as np

ROW_ARRAYS = ("instance_id", "frame_idx", "view")
_DTYPES = {"instance_id": np.int32, "frame_idx": np.int32, "view": np.uint8}


def write_crop_shard(path: str, blobs: list[bytes], meta: dict, **rows) -> None:
    n = len(blobs)
    if set(rows) != set(ROW_ARRAYS):
        raise ValueError(f"rows must be exactly {ROW_ARRAYS}")
    if any(len(rows[k]) != n for k in ROW_ARRAYS):
        raise ValueError("row arrays and blobs differ in length")
    if "store_id" not in meta:
        raise ValueError("meta must contain store_id")
    offsets = np.zeros(n + 1, np.int64)
    offsets[1:] = np.cumsum([len(j) for j in blobs])
    data = np.frombuffer(b"".join(blobs), np.uint8) if n else np.zeros(0, np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, data=data, offsets=offsets, meta=np.array(json.dumps(meta, sort_keys=True)),
                 **{k: np.asarray(rows[k], _DTYPES[k]) for k in ROW_ARRAYS})
    os.replace(tmp, path)


def read_crop_rows(path: str) -> dict:
    """Row arrays + meta only, WITHOUT the image bytes (npz members load lazily).

    Building an index over ~250 shards must not load several GB of image data.
    """
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in ROW_ARRAYS}
        out["meta"] = json.loads(str(z["meta"]))
    return out


class CropShard:
    def __init__(self, path: str):
        with np.load(path, allow_pickle=False) as z:
            self.data, self.offsets = z["data"], z["offsets"]
            self.instance_id, self.frame_idx, self.view = z["instance_id"], z["frame_idx"], z["view"]
            self.meta = json.loads(str(z["meta"]))
        if len(self.offsets) != len(self.instance_id) + 1 or self.offsets[-1] != len(self.data):
            raise ValueError(f"corrupt crop shard: {path}")

    def __len__(self) -> int:
        return len(self.instance_id)

    def blob(self, i: int) -> bytes:
        return self.data[self.offsets[i]:self.offsets[i + 1]].tobytes()

    def image(self, i: int) -> np.ndarray:
        """Decoded RGB uint8 array."""
        import cv2
        bgr = cv2.imdecode(np.frombuffer(self.blob(i), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot decode crop {i}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def crop_shard_is_valid(path: str, store_id: str | None = None) -> bool:
    if not os.path.exists(path):
        return False
    try:
        shard = CropShard(path)
    except Exception:
        return False
    return store_id is None or shard.meta.get("store_id") == store_id
