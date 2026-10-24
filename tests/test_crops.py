"""Crop store: format round trip, pixel-correct extraction, and the integrity gate end to end."""
import os

import cv2
import numpy as np
import pytest

from sfs.data.extract_crops import (VIEW_FORMATS, build_index, encode_crop, extract_video,
                                    make_store_id, verify)
from sfs.data.regions import crop
from sfs.embed.run import EmbedConfig, iter_frames, process_video, regions_for
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
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (30, 20, 3), dtype=np.uint8)
    blobs = [encode_crop(img, "png"), encode_crop(img, "jpg", max_side=10)]
    p = str(tmp_path / "s.npz")
    write_crop_shard(p, blobs, {"store_id": "x"}, instance_id=[4, 5], frame_idx=[1, 2], view=[0, 1])
    s = CropShard(p)
    assert len(s) == 2 and s.blob(1) == blobs[1]
    assert np.array_equal(s.image(0), img), "PNG must round-trip pixel-exact"
    assert s.image(1).shape == (10, 7, 3)                # downscaled, aspect ratio kept
    assert crop_shard_is_valid(p, "x") and not crop_shard_is_valid(p, "y")


def test_encode_never_upscales():
    img = np.zeros((224, 224, 3), np.uint8)
    out = cv2.imdecode(np.frombuffer(encode_crop(img, "jpg", max_side=448), np.uint8), cv2.IMREAD_COLOR)
    assert out.shape == (224, 224, 3)
    with pytest.raises(ValueError):
        encode_crop(img, "gif")


def test_shard_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError):
        write_crop_shard(str(tmp_path / "a.npz"), [b"x"], {}, instance_id=[0], frame_idx=[0], view=[0])
    with pytest.raises(ValueError):
        write_crop_shard(str(tmp_path / "a.npz"), [b"x"], {"store_id": "s"},
                         instance_id=[0, 1], frame_idx=[0], view=[0])


def test_extracted_crops_are_centred_on_the_actor(video, tmp_path):
    out = str(tmp_path / "crops")
    stats = extract_video(video, INSTANCES, TRACKS, EmbedConfig(), out, "sid")
    assert stats["crops"] == (3 + 1) * 2 and stats["frames_missing"] == 0  # 4 frames x tight+context

    s = CropShard(os.path.join(out, f"{STEM}.npz"))
    for i in range(len(s)):
        img = s.image(i)
        assert img.shape == ((224, 224, 3) if s.view[i] == 0 else (448, 448, 3))  # native sizes
        cy, cx = img.shape[0] // 2, img.shape[1] // 2
        assert img[cy, cx].mean() > 230, "crop centre should be the white actor"
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


def test_tight_crops_are_pixel_identical_to_what_was_embedded(video, tmp_path):
    """Regression for D-014: the stored tight crop must equal the region the embedding job cropped."""
    out = str(tmp_path / "crops")
    extract_video(video, INSTANCES, TRACKS, EmbedConfig(), out, "sid")
    s = CropShard(os.path.join(out, f"{STEM}.npz"))
    tight = {int(s.frame_idx[i]): s.image(i) for i in range(len(s)) if s.view[i] == 0}

    cap = cv2.VideoCapture(video)
    for f, frame in iter_frames(cap, sorted(tight)):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        region = regions_for((0,), f, TRACKS, W, H, EmbedConfig())[0][1]
        assert np.array_equal(tight[f], crop(rgb, region)), f"frame {f}: stored crop differs"
    cap.release()


def test_store_id_changes_with_settings():
    a = make_store_id(EmbedConfig(), VIEW_FORMATS)
    jpeg_tight = {**VIEW_FORMATS, "tight": {"format": "jpg", "max_side": None, "quality": 95}}
    assert a != make_store_id(EmbedConfig(), jpeg_tight)
    assert a == make_store_id(EmbedConfig(), dict(VIEW_FORMATS))


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
