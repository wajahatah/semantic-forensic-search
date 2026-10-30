"""End-to-end embedding job on a synthetic video, with a fake encoder (no model download).

Frame i is filled with grey level 4*i, so the fake encoder's first feature (mean intensity / 4)
recovers the frame index — this verifies the sequential decoder reads the right frames.
"""
import cv2
import numpy as np
import pytest

from sfs.embed.encoder import make_encoder_id
from sfs.embed.run import EmbedConfig, plan, process_video
from sfs.store.shards import read_shard

FPS, W, H, N_FRAMES = 30, 320, 240, 60
ACTOR_BOX = (100, 80, 140, 160)  # 40 x 80


class FakeEncoder:
    def __init__(self, name):
        self.name, self.dim = name, 3

    def embed_images(self, images):
        return np.array([[img.mean() / 4.0, img.shape[0], img.shape[1]] for img in images], np.float32)


@pytest.fixture
def video(tmp_path):
    path = str(tmp_path / "2018-03-13.17-05-00.17-10-00.school.G328.r13.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(N_FRAMES):
        w.write(np.full((H, W, 3), 4 * i, np.uint8))
    w.release()
    return path


INSTANCES = [
    {"instance_id": 0, "frame_start": 5, "frame_end": 40, "actor_ids": "0"},  # 1.2 s -> 3 frames
    {"instance_id": 1, "frame_start": 10, "frame_end": 12, "actor_ids": ""},  # no actors -> 1 frame
]
TRACKS = {0: {f: ACTOR_BOX for f in range(N_FRAMES)}}


def _run(video, out, encoders):
    ids = {e.name: f"id-{e.name}" for e in encoders}
    return process_video(video, INSTANCES, TRACKS, encoders, EmbedConfig(), str(out), ids, batch_size=4)


def test_plan_samples_per_instance():
    p = plan(INSTANCES, FPS, EmbedConfig())
    assert sorted(p) == [5, 11, 23, 40]  # 5 + round(17.5) = 23 (banker's rounding)
    assert p[11] == [(1, ())] and p[5] == [(0, (0,))]


def test_process_video_writes_aligned_shards(video, tmp_path):
    stats = _run(video, tmp_path / "out", [FakeEncoder("a"), FakeEncoder("b")])
    assert stats["vectors"] == 12 and stats["frames_missing"] == 0  # (3 + 1) frames x 3 views

    d = read_shard(str(tmp_path / "out" / "a" / f"{stats['clip_stem']}.npz"))
    emb = d["emb"].astype(np.float32)
    assert np.allclose(emb[:, 0], d["frame_idx"], atol=1.0), "decoded the wrong frames"
    assert d["meta"]["encoder_id"] == "id-a" and d["meta"]["fps"] == FPS

    sizes = {(int(i), int(v)): (int(h), int(w)) for i, v, h, w in
             zip(d["instance_id"], d["view"], emb[:, 1], emb[:, 2])}
    assert sizes[(0, 0)] == (224, 224)   # tight: min side 224
    assert sizes[(0, 1)] == (240, 240)   # context: min side 448, capped by image height
    assert sizes[(0, 2)] == (240, 320)   # full frame

    no_actor = d["instance_id"] == 1
    assert d["fallback"][no_actor].sum() == 2  # tight + context fell back; full is not a fallback
    assert (d["box_side"][d["instance_id"] == 0] == 80).all()


def test_resumes_without_recomputing(video, tmp_path):
    enc = [FakeEncoder("a")]
    _run(video, tmp_path / "out", enc)
    assert _run(video, tmp_path / "out", enc)["skipped"] is True


def test_frames_past_end_reported_missing(video, tmp_path):
    late = INSTANCES + [{"instance_id": 2, "frame_start": 100, "frame_end": 100, "actor_ids": "0"}]
    stats = process_video(video, late, TRACKS, [FakeEncoder("a")], EmbedConfig(),
                          str(tmp_path / "o"), {"a": "x"}, batch_size=4)
    assert stats["frames_missing"] == 1 and stats["vectors"] == 12


def test_encoder_id_depends_on_config():
    a = make_encoder_id({"model": "m", "embed_config": EmbedConfig().as_dict()})
    b = make_encoder_id({"model": "m", "embed_config": EmbedConfig(rate=4.0).as_dict()})
    assert a != b and a == make_encoder_id({"model": "m", "embed_config": EmbedConfig().as_dict()})


def test_refuses_clip_whose_actors_have_no_tracks(video, tmp_path):
    """Regression for the 99.5%-fallback run: must fail loudly, not embed full frames."""
    with pytest.raises(ValueError, match="none of 1 actors have box tracks"):
        process_video(video, INSTANCES, {}, [FakeEncoder("a")], EmbedConfig(),
                      str(tmp_path / "o"), {"a": "x"})


def test_read_manifest_keeps_huge_actor_ids_exact(tmp_path):
    import pandas as pd
    from sfs.data.manifest import read_manifest
    p = tmp_path / "m.csv"
    pd.DataFrame({"instance_id": [0, 1], "actor_ids": ["6068717397083399654", "1;2"],
                  "clip_stem": ["a", "b"], "camera": ["G1", "G2"]}).to_csv(p, index=False)
    assert read_manifest(str(p)).actor_ids.tolist() == ["6068717397083399654", "1;2"]


def test_pipeline_version_changes_encoder_id():
    """A parser fix must invalidate shards from the broken run, or resume would skip them."""
    old = make_encoder_id({"m": "x", "embed_config": EmbedConfig(pipeline_version=1).as_dict()})
    new = make_encoder_id({"m": "x", "embed_config": EmbedConfig().as_dict()})
    assert old != new
