"""LoRA training (spec-lora-finetune §5): loss, sampler, augmentation, packed file, LoRA wrap,
LR rule, and crash-safe resume (D13). Uses a tiny random SigLIP vision tower — no downloads."""
import math
import os
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from sfs.data.regions import VIEWS
from sfs.store.crops import write_crop_shard
from sfs.train.captions import disjointness_violations, load_captions, similarity
from sfs.train.data import (EDGE_WEIGHT, InstanceSampler, PackedCrops, augment, build_packed,
                            frame_weights, preprocess_eval, to_pixels)
from sfs.train.loss import label_aware_sigmoid_loss
from sfs.train.lora import (adapter_hash, adapter_state, image_features, load_adapter_state,
                            trainable_report, wrap_lora)
from sfs.train.run import TrainConfig, after_epoch, lr_at, should_stop, train

ROOT = os.path.dirname(os.path.dirname(__file__))


# ---------------------------------------------------------------- loss (D3)
def test_loss_hand_computed():
    img = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    cap = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    labels = torch.tensor([0, 1])
    scale, bias = torch.tensor(math.log(2.0)), torch.tensor(-1.0)
    # logits = 2 * img·capᵀ - 1 = [[1, 0.2], [-1, 0.6]]; targets = [[1, 0], [0, 1]]
    sp = lambda x: math.log1p(math.exp(x))            # BCE(z, 1) = softplus(-z); BCE(z, 0) = softplus(z)
    expected = (sp(-1) + sp(0.2) + sp(-1) + sp(-0.6)) / 2
    got = label_aware_sigmoid_loss(img, cap, labels, scale, bias)
    assert float(got) == pytest.approx(expected, rel=1e-6)


def test_loss_rewards_the_correct_caption():
    cap = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    labels = torch.arange(4)
    s, b = torch.tensor(math.log(10.0)), torch.tensor(-5.0)
    good = label_aware_sigmoid_loss(cap, cap, labels, s, b)
    bad = label_aware_sigmoid_loss(cap, cap, labels.roll(1), s, b)
    assert good < bad


# ---------------------------------------------------------------- frame weights (D4)
def test_frame_weights():
    assert frame_weights(5, "uniform") == pytest.approx(np.full(5, 0.2))
    w = frame_weights(5, "middle")
    assert w.sum() == pytest.approx(1) and w == pytest.approx(w[::-1])
    assert w[0] / w[2] == pytest.approx(EDGE_WEIGHT) and w.argmax() == 2
    assert frame_weights(1, "middle") == pytest.approx([1.0])
    assert frame_weights(2, "middle") == pytest.approx([0.5, 0.5])
    with pytest.raises(ValueError):
        frame_weights(3, "edges")


# ---------------------------------------------------------------- synthetic crop store
def _png(rng, w, h):
    import cv2
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()


def _store(tmp_path, n_videos=3, inst_per_video=4, frames=3, store_id="s1", seed=0):
    rng = np.random.default_rng(seed)
    crops = tmp_path / "crops"
    iid = 0
    for v in range(n_videos):
        blobs, inst, fr, view = [], [], [], []
        for _ in range(inst_per_video):
            for f in range(frames):
                for vw in ("tight", "context"):
                    blobs.append(_png(rng, 40 + f, 36))
                    inst.append(iid), fr.append(100 + 15 * f), view.append(VIEWS.index(vw))
            iid += 1
        write_crop_shard(str(crops / f"video{v}.npz"), blobs, {"store_id": store_id},
                         instance_id=inst, frame_idx=fr, view=view)
    return crops, iid


def test_packed_file_is_byte_identical_and_tight_only(tmp_path):
    from sfs.store.crops import CropShard
    crops, n_inst = _store(tmp_path)
    meta = build_packed(str(crops), str(tmp_path / "packed"))
    p = PackedCrops(str(tmp_path / "packed"))
    assert meta["rows"] == len(p) == n_inst * 3 and meta["store_id"] == "s1"
    want = {}
    for f in sorted(os.listdir(crops)):
        s = CropShard(str(crops / f))
        for i in range(len(s)):
            if s.view[i] == VIEWS.index("tight"):
                want[(int(s.instance_id[i]), int(s.frame_idx[i]))] = s.blob(i)
    for r in range(len(p)):
        assert p.blob(r) == want[(int(p.instance_id[r]), int(p.frame_idx[r]))]
    rows = p.rows_of(2)
    assert list(p.frame_idx[rows]) == sorted(p.frame_idx[rows]) and len(p.rows_of(999)) == 0
    p.blob(0)
    assert pickle.loads(pickle.dumps(p))._mm is None          # memmap is reopened in each worker


def test_packed_refuses_mixed_stores(tmp_path):
    crops, _ = _store(tmp_path)
    write_crop_shard(str(crops / "zz.npz"), [_png(np.random.default_rng(1), 8, 8)], {"store_id": "other"},
                     instance_id=[99], frame_idx=[0], view=[0])
    with pytest.raises(ValueError, match="different stores"):
        build_packed(str(crops), str(tmp_path / "packed"))


# ---------------------------------------------------------------- sampler (D4)
def _manifest(n_inst, split_col="split"):
    acts = ["walk"] * (n_inst - n_inst // 4) + ["rare"] * (n_inst // 4)
    splits = ["train"] * n_inst
    for i in range(0, n_inst, 6):
        splits[i] = "val"
    return pd.DataFrame({"instance_id": np.arange(n_inst), "activity": acts, split_col: splits,
                         "split_b": splits})


def test_sampler_leakage_guard_balance_and_determinism(tmp_path):
    crops, n = _store(tmp_path, n_videos=6, inst_per_video=6)
    p = PackedCrops_(build_packed(str(crops), str(tmp_path / "pk")), tmp_path / "pk")
    m = _manifest(n)
    s = InstanceSampler(m, "split", p, "uniform", seed=3)
    train_ids = set(m.loc[m.split == "train", "instance_id"])
    draws = [d for _ in range(300) for d in s.epoch()]
    picked = [int(s.instance_ids[k]) for _, k, _ in draws]
    assert set(picked) <= train_ids
    assert all(int(p.instance_id[row]) == int(s.instance_ids[k]) for row, k, _ in draws[:500])
    acts = s.activities[[k for _, k, _ in draws]]
    counts = pd.Series(s.activities).value_counts()
    share = (acts == "rare").mean()
    want = math.sqrt(counts["rare"]) / (math.sqrt(counts["rare"]) + math.sqrt(counts["walk"]))
    assert share == pytest.approx(want, abs=0.02)             # class mass ∝ sqrt(class size)
    a, b = InstanceSampler(m, "split", p, "middle", 7), InstanceSampler(m, "split", p, "middle", 7)
    a.epoch()
    state = a.rng.bit_generator.state
    nxt = a.epoch()
    b.rng.bit_generator.state = state
    assert b.epoch() == nxt                                   # RNG state restores the next epoch exactly


def PackedCrops_(meta, path):
    return PackedCrops(str(path))


# ---------------------------------------------------------------- augmentation (D5) and preprocessing
def test_augment_never_flips_and_is_seeded():
    ramp = np.tile(np.linspace(0, 255, 120, dtype=np.float32), (90, 1))
    img = np.repeat(ramp[:, :, None], 3, axis=2).astype(np.uint8)
    for seed in range(200):
        out = augment(img, np.random.default_rng(seed))
        assert out.shape == (224, 224, 3) and out.dtype == np.uint8
        col = out.mean(axis=(0, 2))
        assert np.corrcoef(np.arange(224), col)[0, 1] > 0.5, seed   # left stays dark, right bright
    assert (augment(img, np.random.default_rng(5)) == augment(img, np.random.default_rng(5))).all()


def test_eval_preprocessing_matches_the_siglip_processor():
    from transformers import SiglipImageProcessor
    proc = SiglipImageProcessor(size={"height": 224, "width": 224}, resample=2,
                                image_mean=[0.5] * 3, image_std=[0.5] * 3)
    rng = np.random.default_rng(0)
    for h, w in [(224, 224), (300, 260), (150, 410)]:
        img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        ref = proc(images=[img], return_tensors="pt")["pixel_values"]
        ours = to_pixels(preprocess_eval(img)[None])
        assert torch.allclose(ours, ref, atol=1e-6)                  # pixel-identical


# ---------------------------------------------------------------- caption bank (D1)
def test_real_caption_bank_is_disjoint_from_eval_phrasings():
    from sfs.eval.queries import load_phrasings
    caps = load_captions(os.path.join(ROOT, "queries", "train_captions.csv"))
    phr = load_phrasings(os.path.join(ROOT, "queries", "phrasings.csv"))
    assert disjointness_violations(caps, phr) == []
    assert set(phr) <= set(caps) and all(len(c) == 6 for c in caps.values())


def test_disjointness_rule():
    phr = {"open": ["a person opening a door"]}
    assert disjointness_violations({"open": ["A person opening a door!"]}, phr)          # exact
    assert disjointness_violations({"open": ["a person opening the door"]}, phr)         # near
    assert not disjointness_violations({"close": ["a person closing a door"]}, phr)     # other activity
    assert similarity("x y", "y x") == 1.0


# ---------------------------------------------------------------- LoRA wrap (D2)
def _tiny_vision(seed=0):
    from transformers import SiglipVisionConfig, SiglipVisionModel
    torch.manual_seed(seed)
    cfg = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=2, image_size=224, patch_size=32)
    return SiglipVisionModel(cfg)        # transformers 5: same class as SiglipModel.vision_model


def test_lora_trains_only_adapters_and_starts_as_identity():
    base = _tiny_vision()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        before = image_features(base, x)
    frozen = {n: p.detach().clone() for n, p in base.named_parameters()}
    v = wrap_lora(base, rank=4, alpha=8, dropout=0.0)
    rep = trainable_report(v)
    assert rep["adapted_modules"] == 2 * 4 and 0 < rep["trainable"] < rep["total"]
    v.eval()
    with torch.no_grad():
        assert torch.allclose(image_features(v, x), before, atol=1e-6)    # LoRA B = 0 at start
    opt = torch.optim.AdamW([p for p in v.parameters() if p.requires_grad], lr=1e-2)
    v.train()
    image_features(v, x).sum().backward()
    opt.step()
    for n, p in v.named_parameters():
        if "lora_" not in n:
            key = n.replace("base_model.model.", "").replace(".base_layer", "")
            assert torch.equal(p, frozen[key]), n                         # base weights unchanged
    st = adapter_state(v)
    h = adapter_hash(st)
    w = wrap_lora(_tiny_vision(), rank=4, alpha=8, dropout=0.0)
    assert adapter_hash(adapter_state(w)) != h
    load_adapter_state(w, st)
    assert adapter_hash(adapter_state(w)) == h


# ---------------------------------------------------------------- LR rule (D6 option 1)
def test_lr_warmup_plateau_and_stop():
    cfg = TrainConfig("A", "uniform", warmup_epochs=2, lr=1.0)
    assert lr_at(cfg, 0, 10, 1.0) == pytest.approx(0.05) and lr_at(cfg, 19, 10, 1.0) == 1.0
    assert lr_at(cfg, 500, 10, 0.25) == 0.25
    st = {"best_map": 0.10, "best_epoch": 0, "since_best": 0, "factor": 1.0}
    assert after_epoch(cfg, st, 0.20, 1) and st["best_epoch"] == 1
    factors = []
    for e in range(2, 7):
        after_epoch(cfg, st, 0.19, e)
        factors.append(st["factor"])
    assert factors == [1.0, 0.5, 0.5, 0.25, 0.25]     # halved after 2 and 4 epochs without improvement
    assert should_stop(cfg, st, 6) and not should_stop(cfg, {**st, "since_best": 4}, 6)
    assert should_stop(cfg, {**st, "since_best": 0}, 100) == "max epochs"


# ---------------------------------------------------------------- end-to-end + crash-safe resume (D13)
def _setup(tmp_path):
    crops, n = _store(tmp_path, n_videos=4, inst_per_video=6)
    build_packed(str(crops), str(tmp_path / "pk"))
    packed = PackedCrops(str(tmp_path / "pk"))
    m = _manifest(n)
    rng = np.random.default_rng(0)
    unit = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
    caps = {a: unit(rng.standard_normal((6, 32))).astype(np.float32) for a in ("rare", "walk")}
    phr = {"walk": ["p1", "p2"], "rare": ["p3", "p4"]}
    q = {a: unit(rng.standard_normal((2, 32))).astype(np.float32) for a in phr}
    return packed, m, caps, phr, q


def _run(tmp_path, out, packed, m, caps, phr, q, crash_after=None):
    cfg = TrainConfig("A", "middle", batch=8, max_epochs=3, lora_rank=4, lora_alpha=8,
                      lora_dropout=0.1, val_min_instances=1, stop_patience=10, lr=5e-3)
    v = wrap_lora(_tiny_vision(), cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout)
    torch.manual_seed(123)

    def log(msg):
        if crash_after is not None and f"epoch {crash_after}:" in msg:
            raise KeyboardInterrupt("simulated power cut")

    return train(cfg, v, torch.tensor(2.0), torch.tensor(-3.0), packed, m, caps, phr, q,
                 str(tmp_path / out), "cpu", 0, log)


def test_resume_after_a_crash_reproduces_the_uninterrupted_run(tmp_path):
    packed, m, caps, phr, q = _setup(tmp_path)
    straight = _run(tmp_path, "straight", packed, m, caps, phr, q)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, "crashed", packed, m, caps, phr, q, crash_after=2)
    assert not os.path.exists(tmp_path / "crashed" / "done.json")
    resumed = _run(tmp_path, "crashed", packed, m, caps, phr, q)
    a = torch.load(tmp_path / "straight" / "last.pt", weights_only=False)
    b = torch.load(tmp_path / "crashed" / "last.pt", weights_only=False)
    assert adapter_hash(a["adapter"]) == adapter_hash(b["adapter"])
    assert a["state"] == b["state"] and straight["adapter_hash"] == resumed["adapter_hash"]
    log = pd.read_csv(tmp_path / "crashed" / "train_log.csv")
    assert log.epoch.tolist() == [0, 1, 2, 3]
    assert os.path.exists(tmp_path / "crashed" / "prev.pt")
    again = _run(tmp_path, "crashed", packed, m, caps, phr, q)         # finished run is skipped
    assert again["adapter_hash"] == resumed["adapter_hash"]


def test_resume_refuses_a_different_config(tmp_path):
    packed, m, caps, phr, q = _setup(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, "run", packed, m, caps, phr, q, crash_after=1)
    cfg = TrainConfig("A", "uniform", batch=8, max_epochs=3, lora_rank=4, lora_alpha=8,
                      val_min_instances=1)
    v = wrap_lora(_tiny_vision(), 4, 8, 0.1)
    with pytest.raises(ValueError, match="different training config"):
        train(cfg, v, torch.tensor(2.0), torch.tensor(-3.0), packed, m, caps, phr, q,
              str(tmp_path / "run"), "cpu", 0, lambda s: None)


def test_epoch_zero_is_the_frozen_model_and_best_is_kept(tmp_path):
    packed, m, caps, phr, q = _setup(tmp_path)
    done = _run(tmp_path, "r", packed, m, caps, phr, q)
    log = pd.read_csv(tmp_path / "r" / "train_log.csv")
    assert done["frozen_val_map"] == pytest.approx(log.val_map[0], abs=1e-4)
    best = torch.load(tmp_path / "r" / "best" / "adapter.pt", weights_only=False)
    assert best["epoch"] == done["best_epoch"] and best["val_map"] == pytest.approx(done["best_val_map"])
    assert done["best_val_map"] == pytest.approx(log.val_map.max(), abs=1e-4)
