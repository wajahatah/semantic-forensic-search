import numpy as np
import pytest

from sfs.store.shards import consolidate, read_shard, shard_is_valid, write_shard


def _arrays(n=3, dim=4):
    return dict(emb=np.random.rand(n, dim), instance_id=np.arange(n), frame_idx=np.arange(n) * 10,
                view=np.zeros(n), fallback=np.zeros(n, bool), box_side=np.full(n, 100))


def test_roundtrip(tmp_path):
    p = str(tmp_path / "a" / "x.npz")
    write_shard(p, {"encoder_id": "e1", "clip_stem": "x"}, **_arrays())
    d = read_shard(p)
    assert d["emb"].dtype == np.float16 and d["emb"].shape == (3, 4)
    assert d["meta"]["clip_stem"] == "x"
    assert shard_is_valid(p, "e1") and not shard_is_valid(p, "other")


def test_requires_encoder_id_and_equal_lengths(tmp_path):
    with pytest.raises(ValueError):
        write_shard(str(tmp_path / "x.npz"), {}, **_arrays())
    bad = _arrays()
    bad["frame_idx"] = np.arange(2)
    with pytest.raises(ValueError):
        write_shard(str(tmp_path / "x.npz"), {"encoder_id": "e"}, **bad)


def test_invalid_when_missing_or_corrupt(tmp_path):
    p = tmp_path / "x.npz"
    assert not shard_is_valid(str(p))
    p.write_bytes(b"not a zip")
    assert not shard_is_valid(str(p))


def test_consolidate_same_encoder(tmp_path):
    a, b = str(tmp_path / "a.npz"), str(tmp_path / "b.npz")
    write_shard(a, {"encoder_id": "e1", "clip_stem": "a"}, **_arrays(2))
    write_shard(b, {"encoder_id": "e1", "clip_stem": "b"}, **_arrays(3))
    meta = consolidate([a, b], str(tmp_path / "all.npz"))
    assert meta["clip_stems"] == ["a", "b"]
    assert read_shard(str(tmp_path / "all.npz"))["emb"].shape[0] == 5


def test_consolidate_refuses_mixed_encoders(tmp_path):
    a, b = str(tmp_path / "a.npz"), str(tmp_path / "b.npz")
    write_shard(a, {"encoder_id": "e1"}, **_arrays())
    write_shard(b, {"encoder_id": "e2"}, **_arrays())
    with pytest.raises(ValueError):
        consolidate([a, b], str(tmp_path / "all.npz"))
