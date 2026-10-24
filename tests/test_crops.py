"""Crop store: format round trip, pixel-correct extraction, and the integrity gate end to end."""
import os

import cv2
import numpy as np
import pytest

from sfs.data.extract_crops import build_index, encode_crop, extract_video, make_store_id, verify
from sfs.embed.run import EmbedConfig, process_video
from sfs.store.crops import CropShard, crop_shard_is_valid, write_crop_shard
from sfs.store.shards import consolidate

FPS, W, H = 30, 640, 480
BOX = (300, 200, 340, 280)  # 40 x 80 white "person" on grey
STEM = "2018-03-13.17-05-00.17-10-00.school.G328"
INSTANCES = [{"instance_id": 0, "frame_start": 5, "frame_end": 40, "actor_ids": "0"},
             {"instance_id": 1, "frame_start": 10, "frame_end": 12, "actor_ids": "0"}]
TRACKS = {0: {f: BOX for f in range(60)}}


@pytest.fixture
def video(tmp_path):
    path = str(tmp_path / f"{STEM}.r13.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    frame = np.full((H, W, 3), 100, np.uint8)
    frame[BOX[1]:BOX[3], BOX[0]:BOX[2]] = 255
    for _ in range(60):
        w.write(frame)
    w.release()
    return path


def test_shard_round_trip(tmp_path):
    imgs = [np.full((20, 20, 3), v, np.uint8) for v in (10, 200)]
    jpegs = [encode_crop(i, 32, 95) for i in imgs]
    p = str(tmp_path / "s.npz")
    write_crop_shard(p, jpegs, {"store_id": "x"}, instance_id=[4, 5], frame_idx=[1, 2], view=[0, 1])
    s = CropShard(p)
    assert len(s) == 2 and s.jpeg(1) == jpegs[1]
    assert s.image(0).shape == (32, 32, 3) and abs(int(s.image(1).mean()) - 200) <= 2
    assert crop_shard_is_valid(p, "x") and not crop_shard_is_valid(p, "y")


def test_shard_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError):
        write_crop_shard(str(tmp_path / "a.npz"), [b"x"], {}, instance_id=[0], frame_idx=[0], view=[0])
    with pytest.raises(ValueError):
        write_crop_shard(str(tmp_path / "a.npz"), [b"x"], {"store_id": "s"},
                         instance_id=[0, 1], frame_idx=[0], view=[0])


def test_extracted_crops_are_centred_on_the_actor(video, tmp_path):
    out = str(tmp_path / "crops")
    stats = extract_video(video, INSTANCES, TRACKS, EmbedConfig(), out, "sid", size=336, quality=95)
    assert stats["crops"] == (3 + 1) * 2 and stats["frames_missing"] == 0  # 4 frames x tight+context

    s = CropShard(os.path.join(out, f"{STEM}.npz"))
    for i in range(len(s)):
        img = s.image(i)
        assert img.shape == (336, 336, 3)
        assert img[168, 168].mean() > 230, "crop centre should be the white actor"
        assert img[5, 5].mean() < 130, "crop corner should be grey background"
    tight = [s.image(i) for i in range(len(s)) if s.view[i] == 0][0]
    ctx = [s.image(i) for i in range(len(s)) if s.view[i] == 1][0]
    assert (tight > 230).mean() > (ctx > 230).mean()  # context zooms out: actor covers less


def test_resume_and_index(video, tmp_path):
    out = str(tmp_path / "crops")
    extract_video(video, INSTANCES, TRACKS, EmbedConfig(), out, "sid")
    assert extract_video(video, INSTANCES, TRACKS, EmbedConfig(), out, "sid")["skipped"]
    idx = build_index(out)
    assert len(idx) == 8 and set(idx.view) == {"tight", "context"}


def test_store_id_changes_with_settings():
    a = make_store_id(EmbedConfig(), ("tight",), 336, 95)
    assert a != make_store_id(EmbedConfig(), ("tight",), 224, 95)
    assert a != make_store_id(EmbedConfig(), ("tight",), 336, 80)


class ColourEncoder:
    """Deterministic stand-in for SigLIP: unit vector of the mean colour, robust to resize/JPEG."""
    def __init__(self, name="siglip2"):
        self.name, self.dim = name, 4

    def embed_images(self, images):
        f = np.array([[*img.reshape(-1, 3).mean(0) / 255.0, 0.3] for img in images], np.float32)
        return f / np.linalg.norm(f, axis=1, keepdims=True)


def test_integrity_gate_passes_when_crops_match_embeddings(video, tmp_path, monkeypatch):
    out_emb = str(tmp_path / "emb")
    process_video(video, INSTANCES, TRACKS, [ColourEncoder()], EmbedConfig(), out_emb,
                  {"siglip2": "e"}, batch_size=4)
    consolidate([os.path.join(out_emb, "siglip2", f"{STEM}.npz")], str(tmp_path / "all.npz"))
    crops = str(tmp_path / "crops")
    extract_video(video, INSTANCES, TRACKS, EmbedConfig(), crops, "sid")

    monkeypatch.setattr("sfs.embed.encoder.Encoder", ColourEncoder)
    r = verify(crops, str(tmp_path / "all.npz"), n=50)
    assert r["compared"] == 4 and r["passed"] and r["mean_cosine"] > 0.99
