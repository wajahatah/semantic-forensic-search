"""Qwen2.5-VL through VLMScorer (spec-vlm-followups E7, D-038), on a tiny random-weight Qwen2.5-VL built
from the real config and processor of Qwen2.5-VL-3B-Instruct (config files only, no weights; skipped
without them): the video-once cached path must equal full passes, as for Qwen3-VL."""
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
CFG = os.path.join(ROOT, "data", "models", "Qwen2.5-VL-3B-Instruct-config")
HAVE = all(os.path.exists(os.path.join(CFG, f)) for f in ("config.json", "preprocessor_config.json", "tokenizer.json"))


@pytest.fixture(scope="module")
def tiny_qwen25():
    if not HAVE:
        pytest.skip("Qwen2.5-VL-3B-Instruct config files not present")
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    from sfs.teacher.vlm import VLMScorer
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(CFG)
    t = cfg.text_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 64, 128, 2
    t.num_attention_heads, t.num_key_value_heads = 4, 2
    t.layer_types = t.layer_types[:2] if getattr(t, "layer_types", None) else None
    for attr in ("rope_scaling", "rope_parameters"):
        if getattr(t, attr, None):
            setattr(t, attr, {**getattr(t, attr), "mrope_section": [4, 2, 2]})
    v = cfg.vision_config
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads = 2, 32, 64, 2
    v.out_hidden_size, v.fullatt_block_indexes = 64, [1]
    model = AutoModelForImageTextToText.from_config(cfg).float()
    return VLMScorer.from_parts(model, AutoProcessor.from_pretrained(CFG), "instruct")


def test_qwen25_cached_scoring_equals_full_forward(tiny_qwen25):
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (60, 40, 3), np.uint8) for _ in range(4)]
    queries = ["a person opens a car door", "someone sits down", "a car turns left at the junction"]
    full = tiny_qwen25.score_full(frames, [0, 15, 30, 45], 30.0, queries)
    cached = tiny_qwen25.score_clip(frames, [0, 15, 30, 45], 30.0, queries, batch=2)
    assert np.isfinite(full).all() and len(set(np.round(full, 5))) > 1
    assert np.allclose(full, cached, atol=1e-4), (full, cached)


def test_scorer_refuses_unknown_model_types(tmp_path):
    import json
    from sfs.teacher.vlm import VLMScorer
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llava"}))
    with pytest.raises(ValueError, match="not one of"):
        VLMScorer(str(tmp_path), "instruct", device="cpu")
