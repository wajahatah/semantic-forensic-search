"""Student re-ranker (spec-rerank-cascade §5): box-motion features, sequence building, masking,
pair pool, targets, and crash-safe resume. Tiny synthetic data — no downloads."""
import math
import os

import numpy as np
import pandas as pd
import pytest
import torch

from sfs.data.regions import VIEWS
from sfs.rerank.features import (BOX_FEATURES, build_features, box_features, local_geom_path,
                                 parse_actors)
from sfs.rerank.model import TemporalReranker
from sfs.rerank.train import (LABEL_CAPTIONS, StudentConfig, build_pool, teacher_requests,
                              train_student, zcols)

F = {name: i for i, name in enumerate(BOX_FEATURES)}


# ---------------------------------------------------------------- features
def test_box_features_hand_computed():
    tracks = {1: {0: (0, 0, 10, 20), 2: (10, 0, 20, 20)}}
    out = box_features(tracks, (1,), [0, 2], [224, 448], fps=1.0)
    assert out[0, F["dx_cum"]] == 0 and out[0, F["has_box"]] == 1
    assert out[1, F["dx_cum"]] == pytest.approx(0.5) and out[1, F["dx_step"]] == pytest.approx(0.5)
    assert out[1, F["dy_cum"]] == 0 and out[1, F["log_h"]] == pytest.approx(0.0)
    assert out[1, F["log_w"]] == pytest.approx(math.log(0.5)) and out[1, F["log_aspect"]] == pytest.approx(math.log(0.5))
    assert out[1, F["t_sec"]] == pytest.approx(2 / 8) and out[1, F["log_side"]] == pytest.approx(math.log(2))
    assert out[1, F["n_actors"]] == pytest.approx(0.25) and out[1, F["pair_dist"]] == 0


def test_box_features_two_actors_and_missing_boxes():
    tracks = {1: {0: (0, 0, 10, 20)}, 2: {0: (40, 0, 50, 20)}}
    out = box_features(tracks, (1, 2), [0, 100], [224, 224], fps=10.0)
    assert out[0, F["pair_dist"]] == pytest.approx(40 / 20) and out[0, F["n_actors"]] == pytest.approx(0.5)
    assert out[1, F["has_box"]] == 0 and out[1, F["dx_cum"]] == 0       # frame 100: no box within ±5
    none = box_features({}, (1,), [0, 1], [224, 224], fps=1.0)
    assert none[:, F["has_box"]].sum() == 0


def test_paths_and_actor_parsing():
    drive = "/content/drive/MyDrive/DL-Project/meva/annotations/meva-data-repo/annotation/x/c.activities.yml"
    assert local_geom_path(drive, "root") == os.path.join("root", "annotation", "x", "c.geom.yml")
    assert parse_actors("11;12") == (11, 12) and parse_actors("3") == (3,)


def _shard(inst, frames, view, dim=768, seed=0):
    rng = np.random.default_rng(seed)
    return {"emb": rng.standard_normal((len(inst), dim)).astype(np.float16), "instance_id": np.array(inst),
            "frame_idx": np.array(frames), "view": np.full(len(inst), VIEWS.index(view)),
            "box_side": np.full(len(inst), 224)}


def test_build_features_alignment(tmp_path):
    geom = tmp_path / "annotation" / "c.geom.yml"
    geom.parent.mkdir()
    geom.write_text("- { geom: {id1: 1, ts0: 0, g0: 0 0 10 20} }\n- { geom: {id1: 1, ts0: 30, g0: 5 0 15 20} }\n")
    m = pd.DataFrame({"instance_id": ["4"], "anno_file": ["annotation/c.activities.yml"], "actor_ids": ["1"]})
    lora = _shard([4, 4], [30, 0], "tight")            # stored out of order on purpose
    frozen = _shard([4, 4], [0, 30], "context", seed=1)
    out = build_features(m, lora, frozen, str(tmp_path))
    assert out["mask"][0].tolist()[:3] == [True, True, False]
    assert np.array_equal(out["feats"][0, 0, :768], lora["emb"][1])          # time order restored
    assert out["feats"][0, 1, 768 + 768 + F["dx_cum"]] == pytest.approx(0.25)
    bad = _shard([4, 4], [0, 15], "context")
    with pytest.raises(ValueError):
        build_features(m, lora, bad, str(tmp_path))


# ---------------------------------------------------------------- model
def test_padded_frames_do_not_change_the_score():
    torch.manual_seed(0)
    model = TemporalReranker(frame_dim=20, text_dim=8, d=16, layers=1, heads=2, t_max=6).eval()
    x = torch.randn(2, 6, 20)
    mask = torch.tensor([[True] * 3 + [False] * 3, [True] * 6])
    text = torch.randn(2, 8)
    a = model(x, mask, text)
    x2 = x.clone()
    x2[0, 3:] = 100 * torch.randn(3, 20)                 # garbage in the padded slots
    assert torch.allclose(a, model(x2, mask, text), atol=1e-5)
    assert a.shape == (2,)
    perm = x[:, [2, 1, 0, 3, 4, 5]]
    assert not torch.allclose(model(perm, mask, text)[0], a[0])      # frame order matters


def test_zcols():
    z = zcols(np.array([[1.0, 10.0], [3.0, 10.0]]))
    assert np.allclose(z[:, 0], [-1, 1]) and np.allclose(z[:, 1], 0)


# ---------------------------------------------------------------- pool + targets
def _pool_data():
    acts = ["a"] * 3 + ["b"] * 3 + ["c"] * 2 + ["a"]
    split = ["train"] * 8 + ["val"]
    m = pd.DataFrame({"instance_id": [str(i) for i in range(9)], "activity": acts, "split": split})
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((9, 4)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    shard = {"emb": emb, "instance_id": np.arange(9), "frame_idx": np.zeros(9, int),
             "view": np.zeros(9, np.uint8)}
    proto = rng.standard_normal((3, 4)).astype(np.float32)
    return m, shard, proto


def test_pool_positives_hard_and_random_negatives():
    m, shard, proto = _pool_data()
    pool = build_pool(m, "split", ["a", "b", "c"], proto, shard, n_hard=2, n_rand=1)
    assert 8 not in set(pool.instance_id)                     # validation never enters the pool
    a = pool[pool.cls == 0]
    assert set(a[a.y == 1].instance_id) == {0, 1, 2} and len(a[a.y == 0]) == 3
    scores = shard["emb"][3:8] @ proto[0]
    hard = set((np.arange(3, 8)[np.argsort(-scores)][:2]).tolist())
    assert hard <= set(a[a.y == 0].instance_id)
    assert not pool.duplicated(["instance_id", "cls"]).any()


def test_teacher_requests_use_only_labelled_captions():
    pool = pd.DataFrame({"instance_id": [1, 2], "cls": [0, 1], "y": [1, 0]})
    caps = {"a": [f"a{i}" for i in range(6)], "b": [f"b{i}" for i in range(6)]}
    req = teacher_requests(pool, ["a", "b"], caps)
    assert req == sorted([(1, f"a{j}") for j in LABEL_CAPTIONS] + [(2, f"b{j}") for j in LABEL_CAPTIONS])


def test_kd_variant_needs_a_teacher():
    with pytest.raises(ValueError):
        StudentConfig("A", "KD", "A-middle", "f", "", "c", "p", "pool")


# ---------------------------------------------------------------- crash-safe resume
class _StubValidator:
    """Deterministic 'validation mAP' from the weights; raises on call `crash_at` (a power cut
    during that epoch, after the previous epoch's checkpoint was written)."""
    lora_map = (0.1, {})

    def __init__(self, crash_at=None):
        self.calls, self.crash_at = 0, crash_at

    def __call__(self, model, feats, mask, device):
        self.calls += 1
        if self.calls == self.crash_at:
            raise RuntimeError("simulated power cut")
        w = sum(float(p.detach().double().abs().sum()) for p in model.parameters())
        return {"val_map": (w * 1e3) % 1.0, "beta": 0.5, "per_activity": {}}


def _tiny_run(out_dir, epochs=4, variant="GTKD", crash_at=None):
    rng = np.random.default_rng(0)
    feats = {"instance_id": np.arange(6), "feats": rng.standard_normal((6, 4, 10)).astype(np.float16),
             "mask": np.array([[True, True, False, False]] * 6)}
    pool = pd.DataFrame({"instance_id": [0, 1, 2, 3, 4, 5], "cls": [0, 0, 1, 1, 0, 1], "y": [1, 0, 1, 0, 1, 1]})
    caps = {"a": ["a0", "a1"], "b": ["b0", "b1"]}
    teacher = {(i, t): float(rng.standard_normal()) for i in range(6) for t in ["a0", "a1", "b0", "b1"]}
    cap_emb = rng.standard_normal((2, 2, 8)).astype(np.float32)
    cfg = StudentConfig("A", variant, "A-middle", "f", "t" if variant != "GT" else "", "c", "p", "pool",
                        d=8, layers=1, heads=2, batch=4, max_epochs=epochs, patience=99)
    return train_student(cfg, feats, pool, cap_emb, teacher, caps, ["a", "b"], _StubValidator(crash_at),
                         out_dir, "cpu", log=lambda *_: None)


def test_resume_after_crash_is_exact(tmp_path):
    import json
    full = _tiny_run(str(tmp_path / "full"))
    with pytest.raises(RuntimeError):
        _tiny_run(str(tmp_path / "crash"), crash_at=3)          # dies during epoch 3
    assert torch.load(tmp_path / "crash" / "last.pt", weights_only=False)["state"]["epoch"] == 2
    resumed = _tiny_run(str(tmp_path / "crash"))
    a = torch.load(tmp_path / "full" / "last.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "crash" / "last.pt", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert resumed["best_epoch"] == full["best_epoch"] and resumed["best_val_map"] == full["best_val_map"]
    assert len(open(tmp_path / "crash" / "train_log.csv").read().strip().splitlines()) == 5   # header + 4
    assert json.load(open(tmp_path / "crash" / "done.json"))["epochs_run"] == 4


def test_resume_refuses_a_different_config(tmp_path):
    with pytest.raises(RuntimeError):
        _tiny_run(str(tmp_path / "r"), crash_at=2)
    with pytest.raises(ValueError):
        _tiny_run(str(tmp_path / "r"), variant="GT")            # different config -> refused

def test_text_embeddings_from_an_exported_file(tmp_path):
    from sfs.rerank.train import _text_embeddings
    e0, e1 = np.arange(12, dtype=np.float32).reshape(3, 4), np.ones((2, 4), np.float32)
    np.savez(tmp_path / "t.npz", texts_0=np.array(["b", "a", "c"]), emb_0=e0, texts_1=np.array(["x", "y"]), emb_1=e1)
    assert np.array_equal(_text_embeddings(["x", "y"], str(tmp_path / "t.npz")), e1)
    assert np.array_equal(_text_embeddings(["b", "a", "c"], str(tmp_path / "t.npz")), e0)
    with pytest.raises(KeyError):                                     # only the exact list of one call
        _text_embeddings(["a", "b", "c"], str(tmp_path / "t.npz"))
