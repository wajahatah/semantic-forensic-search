"""Training data: packed tight-crop file, instance sampler, augmentation (spec-lora-finetune D4, D5, D9).

Packed file: every tight crop's encoded bytes (PNG, from the verified crop store) concatenated into
``tight.bin`` with a row table ``tight_rows.npz`` (instance_id, frame_idx, offset, length). Opened
with ``numpy.memmap``, so any crop can be read at random with no RAM pressure. The bytes are copied,
never re-encoded, so the store's cos-1.000 integrity guarantee carries over.
"""
from __future__ import annotations

import json
import os

import numpy as np

TARGET = 224          # SigLIP 2 base input size
ROWS_FILE, BIN_FILE, META_FILE = "tight_rows.npz", "tight.bin", "tight_meta.json"
FRAME_MODES = ("uniform", "middle")
EDGE_WEIGHT = 1 / 3   # D4 (b): first/last frame weight relative to the middle frame


# ---------------------------------------------------------------- packed file
def build_packed(crops_dir: str, out_dir: str, view: str = "tight") -> dict:
    """Copy every `view` crop of the crop store into one packed file. Atomic: all or nothing."""
    from sfs.data.regions import VIEWS
    from sfs.store.crops import CropShard

    files = sorted(f for f in os.listdir(crops_dir) if f.endswith(".npz"))
    if not files:
        raise ValueError(f"no crop shards in {crops_dir}")
    os.makedirs(out_dir, exist_ok=True)
    vi = VIEWS.index(view)
    tmp_bin = os.path.join(out_dir, BIN_FILE + ".tmp")
    inst, frame, offset, length, store_ids = [], [], [], [], set()
    pos = 0
    with open(tmp_bin, "wb") as f:
        for name in files:
            shard = CropShard(os.path.join(crops_dir, name))
            store_ids.add(shard.meta.get("store_id"))
            for i in np.flatnonzero(shard.view == vi):
                blob = shard.blob(int(i))
                f.write(blob)
                inst.append(int(shard.instance_id[i]))
                frame.append(int(shard.frame_idx[i]))
                offset.append(pos)
                length.append(len(blob))
                pos += len(blob)
            del shard
    if len(store_ids) != 1:
        os.remove(tmp_bin)
        raise ValueError(f"crop shards come from different stores: {sorted(map(str, store_ids))}")
    inst, frame = np.asarray(inst, np.int64), np.asarray(frame, np.int64)
    order = np.lexsort((frame, inst))          # rows grouped by instance, frames in time order
    meta = {"store_id": store_ids.pop(), "view": view, "rows": len(inst), "bytes": pos}
    tmp_rows = os.path.join(out_dir, ROWS_FILE + ".tmp.npz")
    np.savez(tmp_rows, instance_id=inst[order], frame_idx=frame[order],
             offset=np.asarray(offset, np.int64)[order], length=np.asarray(length, np.int64)[order])
    os.replace(tmp_bin, os.path.join(out_dir, BIN_FILE))
    os.replace(tmp_rows, os.path.join(out_dir, ROWS_FILE))
    with open(os.path.join(out_dir, META_FILE), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta


class PackedCrops:
    """Random access to the packed crops. The memmap opens lazily, so the object pickles cheaply
    into DataLoader worker processes (Windows uses spawn)."""

    def __init__(self, packed_dir: str):
        self.dir = packed_dir
        with open(os.path.join(packed_dir, META_FILE), encoding="utf-8") as f:
            self.meta = json.load(f)
        with np.load(os.path.join(packed_dir, ROWS_FILE)) as z:
            self.instance_id, self.frame_idx = z["instance_id"], z["frame_idx"]
            self.offset, self.length = z["offset"], z["length"]
        if len(self.instance_id) != self.meta["rows"]:
            raise ValueError("packed row table does not match its meta")
        ids, first = np.unique(self.instance_id, return_index=True)
        self._first = dict(zip(ids.tolist(), first.tolist()))
        self._end = dict(zip(ids.tolist(), np.append(first[1:], len(self.instance_id)).tolist()))
        self._mm = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"] = None
        return state

    def __len__(self) -> int:
        return len(self.instance_id)

    def rows_of(self, instance_id: int) -> np.ndarray:
        """Row numbers of one instance, frames in time order (empty if it has no crops)."""
        iid = int(instance_id)
        return np.arange(self._first[iid], self._end[iid]) if iid in self._first else np.zeros(0, np.int64)

    def blob(self, row: int) -> bytes:
        if self._mm is None:
            self._mm = np.memmap(os.path.join(self.dir, BIN_FILE), np.uint8, mode="r")
        o = int(self.offset[row])
        return self._mm[o:o + int(self.length[row])].tobytes()

    def image(self, row: int) -> np.ndarray:
        import cv2
        bgr = cv2.imdecode(np.frombuffer(self.blob(row), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot decode packed crop {row}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- sampling (D4)
def frame_weights(n: int, mode: str) -> np.ndarray:
    """Probability of picking each of an instance's n frames (time order).

    uniform: equal. middle: triangular — the middle frame weighs 1, the first/last EDGE_WEIGHT,
    because MEVA instances include ~1 s of lead-in/lead-out (D-019).
    """
    if mode not in FRAME_MODES:
        raise ValueError(f"frame mode must be one of {FRAME_MODES}")
    if n < 1:
        raise ValueError("instance has no frames")
    if mode == "uniform" or n <= 2:
        w = np.ones(n)
    else:
        mid = (n - 1) / 2
        w = 1 - (1 - EDGE_WEIGHT) * np.abs(np.arange(n) - mid) / mid
    return w / w.sum()


class InstanceSampler:
    """One epoch = as many draws as there are training instances; each draw picks an instance with
    probability ∝ 1/sqrt(its class size) (class mass ∝ sqrt(class size)), then one of its frames.

    Leakage guard: every instance must belong to the protocol's training subset.
    """

    def __init__(self, manifest, split_col: str, packed: PackedCrops, frame_mode: str, seed: int):
        train = manifest[manifest[split_col] == "train"]
        ids = train.instance_id.astype(np.int64).to_numpy()
        rows = [packed.rows_of(i) for i in ids]
        keep = np.array([len(r) > 0 for r in rows])
        self.instance_ids = ids[keep]
        self.activities = train.activity.to_numpy()[keep]
        self.rows = [r for r, k in zip(rows, keep) if k]
        self.n_missing = int((~keep).sum())
        allowed = set(ids.tolist())
        if not set(self.instance_ids.tolist()) <= allowed:
            raise AssertionError("sampler holds a non-training instance")
        counts = dict(zip(*np.unique(self.activities, return_counts=True)))
        w = np.array([1 / np.sqrt(counts[a]) for a in self.activities])
        self.p = w / w.sum()
        self.frame_p = [frame_weights(len(r), frame_mode) for r in self.rows]
        self.frame_mode = frame_mode
        self.rng = np.random.default_rng(seed)
        self._split_col, self._allowed = split_col, allowed

    def __len__(self) -> int:
        return len(self.instance_ids)

    def epoch(self) -> list[tuple[int, int, int]]:
        """[(packed_row, instance_index, augmentation_seed)] for one epoch. Advances the RNG."""
        picks = self.rng.choice(len(self.instance_ids), size=len(self.instance_ids), p=self.p)
        out = []
        for k in picks:
            row = self.rng.choice(self.rows[k], p=self.frame_p[k])
            out.append((int(row), int(k), int(self.rng.integers(2**31))))
        for _, k, _ in out:   # leakage guard on what is actually yielded
            if int(self.instance_ids[k]) not in self._allowed:
                raise AssertionError("sampler yielded a non-training instance")
        return out


# ---------------------------------------------------------------- image processing (D5)
def resize_bilinear(img: np.ndarray, w: int = TARGET, h: int = TARGET) -> np.ndarray:
    """Bilinear, antialiased, on uint8 — exactly the resize the SigLIP processor applies
    (transformers 5 uses torchvision for it; a PIL resize differs by one level on ~0.3 % of pixels)."""
    import torch
    from torchvision.transforms.v2 import InterpolationMode
    from torchvision.transforms.v2 import functional as tvF
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
    out = tvF.resize(t, [h, w], interpolation=InterpolationMode.BILINEAR, antialias=True)
    return out.permute(1, 2, 0).contiguous().numpy()


def preprocess_eval(img: np.ndarray) -> np.ndarray:
    """No augmentation: exactly the processor's resize. uint8 (224, 224, 3)."""
    return resize_bilinear(img)


def to_pixels(batch_uint8):
    """uint8 (B, H, W, 3) array or tensor -> normalised float (B, 3, H, W), as the processor does."""
    import torch
    t = torch.as_tensor(batch_uint8)
    return (t.permute(0, 3, 1, 2).float() / 255.0 - 0.5) / 0.5


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Training augmentation (D5 revised). Never flips. Returns uint8 (224, 224, 3)."""
    h, w = img.shape[:2]
    # zoom in + stretch: a crop covering 60–100 % of the area, aspect 0.8–1.25, resized to square
    for _ in range(10):
        area = rng.uniform(0.6, 1.0) * h * w
        aspect = np.exp(rng.uniform(np.log(0.8), np.log(1.25)))
        cw, ch = int(round(np.sqrt(area * aspect))), int(round(np.sqrt(area / aspect)))
        if 0 < cw <= w and 0 < ch <= h:
            break
    else:
        cw, ch = w, h
    x0, y0 = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
    out = resize_bilinear(np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw]))
    # zoom out: shrink and reflection-pad the border
    if rng.random() < 0.25:
        s = int(round(TARGET * rng.uniform(0.75, 1.0)))
        if s < TARGET:
            import cv2
            small = resize_bilinear(out, s, s)
            top, left = int(rng.integers(0, TARGET - s + 1)), int(rng.integers(0, TARGET - s + 1))
            out = cv2.copyMakeBorder(small, top, TARGET - s - top, left, TARGET - s - left,
                                     cv2.BORDER_REFLECT_101)
    # resolution degradation: down, then back up
    if rng.random() < 0.3:
        s = max(8, int(round(TARGET * rng.uniform(0.5, 1.0))))
        out = resize_bilinear(resize_bilinear(out, s, s))
    # colour jitter: brightness and contrast ±0.15
    f = out.astype(np.float32)
    c, b = rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15)
    f = ((f - f.mean()) * c + f.mean()) * b
    return np.clip(np.rint(f), 0, 255).astype(np.uint8)


class TrainDataset:
    """Dataset indexed by a draw (packed_row, instance_index, augmentation_seed).

    Augmentation is seeded per draw, so results do not depend on the number of DataLoader workers.
    """

    def __init__(self, packed: PackedCrops, labels: np.ndarray):
        self.packed, self.labels = packed, labels

    def __getitem__(self, draw):
        row, k, seed = draw
        img = augment(self.packed.image(row), np.random.default_rng(seed))
        return img, int(self.labels[k])


class EpochDraws:
    """DataLoader sampler that yields the current epoch's draws. Swapping the list each epoch lets
    the DataLoader keep persistent workers (worker start-up is slow on Windows)."""

    def __init__(self):
        self.draws: list = []

    def set(self, draws) -> None:
        self.draws = list(draws)

    def __iter__(self):
        return iter(self.draws)

    def __len__(self) -> int:
        return len(self.draws)
