"""scripts/clip_baseline.py: zero-shot CLIP baseline (spec-submission §2). No download: a tiny random
CLIP is built from a config; the real model runs only in the measured run."""
import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "clip_baseline", os.path.join(ROOT, "scripts", "clip_baseline.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cb = _load()


def test_auc_perfect_inverted_and_ties():
    assert cb.auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert cb.auc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0
    assert cb.auc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5
    # one of four positive-negative pairs wrong
    assert cb.auc(np.array([2.0, 4.0]), np.array([1.0, 3.0])) == 0.75


def test_prompt_uses_activity_words():
    assert cb.prompt_for("person_texts_on_phone") == "a surveillance camera photo of a person texts on phone"


def test_val_activities_keep_types_with_five_validation_instances():
    import pandas as pd
    m = pd.DataFrame({"split": ["val"] * 9 + ["train"] * 5,
                      "activity": ["a"] * 5 + ["b"] * 4 + ["b"] * 5})
    assert cb.val_activities(m) == ["a"]


def test_zero_shot_val_per_activity_auc_and_p10():
    labels = np.array(["a"] * 10 + ["b"] * 10)
    ids = np.arange(20)
    s = np.zeros((20, 2))
    s[:10, 0] = 1.0                       # activity a: positives on top -> AUC 1, P@10 1
    s[:, 1] = np.linspace(1, 0, 20)       # activity b: positives at the bottom -> AUC 0, P@10 0
    r = cb.zero_shot_val(ids, labels, s, ["a", "b"])
    assert r["per_activity"]["a"] == {"n_pos": 10, "auc": 1.0, "p10": 1.0}
    assert r["per_activity"]["b"] == {"n_pos": 10, "auc": 0.0, "p10": 0.0}
    assert r["auc"] == 0.5 and r["p10"] == 0.5 and r["chance_p10"] == 0.5


@pytest.fixture(scope="module")
def tiny_clip():
    torch = pytest.importorskip("torch")
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel
    torch.manual_seed(0)
    cfg = CLIPConfig(
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                     "num_attention_heads": 2, "vocab_size": 100, "max_position_embeddings": 77,
                     "eos_token_id": 2},
        vision_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                       "num_attention_heads": 2, "image_size": 32, "patch_size": 16},
        projection_dim=16)
    proc = CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})

    def tokenizer(texts, **_):
        ids = torch.tensor([[1, 5 + len(t) % 50, 2] for t in texts])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    return cb.ClipEncoder(CLIPModel(cfg), proc, tokenizer, "cpu")


def test_clip_encoder_image_and_text_vectors_are_unit_length(tiny_clip):
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 256, (40, 40, 3), dtype=np.uint8),
            rng.integers(0, 256, (60, 30, 3), dtype=np.uint8)]   # non-square: resized + centre-cropped
    X = tiny_clip.embed_images(imgs)
    T = tiny_clip.embed_texts(["a person", "a car"])
    assert X.shape == (2, 16) and T.shape == (2, 16)
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(T, axis=1), 1, atol=1e-5)
    assert tiny_clip.embed_images([]).shape == (0, 16)
    # same image -> same vector regardless of batch company
    np.testing.assert_allclose(tiny_clip.embed_images(imgs[:1])[0], X[0], atol=1e-5)


class _FakePacked:
    def __init__(self):
        self.instance_id = np.array([7, 7, 9, 9, 9], np.int64)
        self.frame_idx = np.array([0, 4, 2, 6, 10], np.int64)
        self.meta = {"store_id": "s1"}

    def __len__(self):
        return len(self.instance_id)

    def image(self, row):
        return np.full((8, 8, 3), row, np.uint8)


class _FakeEncoder:
    def __init__(self):
        self.calls = 0

    def embed_images(self, images):
        self.calls += len(images)
        v = np.array([[float(im[0, 0, 0]), 1.0] for im in images], np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def _baseline():
    from sfs.data.regions import VIEWS
    # rows in a different order than the packed file, plus other views that must be ignored
    inst = np.array([9, 9, 9, 7, 7, 7], np.int64)
    frame = np.array([10, 6, 2, 4, 0, 0], np.int64)
    view = np.array([VIEWS.index("tight")] * 5 + [VIEWS.index("full")])
    return {"instance_id": inst, "frame_idx": frame, "view": view,
            "fallback": np.array([1, 0, 0, 1, 0, 1], bool),
            "box_side": np.array([50, 40, 30, 20, 10, 99], np.int32)}


def test_embed_packed_aligns_rows_with_the_baseline_and_resumes(tmp_path, monkeypatch):
    from sfs.store.shards import read_shard
    monkeypatch.setattr(cb, "BLOCK", 2)                 # 3 part files
    enc = _FakeEncoder()
    out = cb.embed_packed(_FakePacked(), _baseline(), enc, str(tmp_path), {"encoder": "clip"},
                          batch=1, log=lambda *_: None)
    s = read_shard(out)
    assert s["meta"]["encoder"] == "clip"
    assert s["instance_id"].tolist() == [7, 7, 9, 9, 9] and s["frame_idx"].tolist() == [0, 4, 2, 6, 10]
    assert s["box_side"].tolist() == [10, 20, 30, 40, 50]      # looked up by (instance, frame)
    assert s["fallback"].tolist() == [False, True, False, False, True]
    assert s["emb"].shape == (5, 2) and enc.calls == 5
    cb.embed_packed(_FakePacked(), _baseline(), enc, str(tmp_path), {"encoder": "clip"},
                    batch=1, log=lambda *_: None)
    assert enc.calls == 5                                       # finished parts are kept


def test_embed_packed_refuses_crops_without_a_baseline_row(tmp_path):
    b = _baseline()
    b["frame_idx"] = b["frame_idx"] + 1
    with pytest.raises(ValueError, match="no baseline row"):
        cb.embed_packed(_FakePacked(), b, _FakeEncoder(), str(tmp_path), {"encoder": "clip"})
