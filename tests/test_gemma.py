"""Gemma 4 scorer (spec-vlm-followups E6, D-032), on a tiny random-weight Gemma 4 built from the real
config and processor of gemma-4-E4B-it (no weight download; skipped without those files): the video
comes before the question, scores are deterministic, `score_clip` equals `score_full`, the yes/no mass
is a probability, and the CPU/GPU split (per-layer table on the CPU, decoder layers offloaded) gives
the same scores as one device. Also: the loader picks the scorer by config, and G0 records placement
and yes/no mass."""
import copy
import json
import os

import numpy as np
import pandas as pd
import pytest


ROOT = os.path.dirname(os.path.dirname(__file__))
GEMMA = os.path.join(ROOT, "data", "models", "gemma-4-E4B-it")
HAVE_GEMMA = all(os.path.exists(os.path.join(GEMMA, f)) for f in ("config.json", "processor_config.json", "tokenizer.json"))
QUERIES = ["a person opens a car door", "someone sits down", "a car turns left at the junction"]


@pytest.fixture(scope="module")
def tiny_gemma():
    if not HAVE_GEMMA:
        pytest.skip("gemma-4-E4B-it config/processor files not present")
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(GEMMA)
    t = cfg.text_config
    t.num_hidden_layers, t.layer_types = 4, ["sliding_attention", "full_attention"] * 2
    t.hidden_size, t.intermediate_size, t.num_attention_heads, t.num_key_value_heads = 64, 128, 2, 1
    t.head_dim, t.global_head_dim, t.hidden_size_per_layer_input = 32, 32, 8
    t.num_kv_shared_layers, t.sliding_window = 2, 16   # like E4B: the last layers reuse KV of the same type
    v = cfg.vision_config
    v.num_hidden_layers, v.hidden_size, v.intermediate_size = 1, 32, 64
    v.num_attention_heads, v.num_key_value_heads, v.head_dim, v.global_head_dim = 2, 2, 16, 16
    cfg.audio_config = None
    model = AutoModelForImageTextToText.from_config(cfg).float().eval()
    return model, AutoProcessor.from_pretrained(GEMMA)


def _frames(n=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (60, 40, 3), np.uint8) for _ in range(n)]


def test_prompt_puts_the_video_before_the_question(tiny_gemma):
    from sfs.teacher.gemma import GemmaScorer
    s = GemmaScorer.from_parts(copy.deepcopy(tiny_gemma[0]), tiny_gemma[1])
    p = s.prompt("a person opens a car door")
    assert p.index("<|video|>") < p.index("Does this video show the following: a person opens a car door?")
    assert p.rstrip().endswith("model")


def test_scores_are_deterministic_and_clip_equals_full(tiny_gemma):
    from sfs.teacher.gemma import GemmaScorer
    s = GemmaScorer.from_parts(copy.deepcopy(tiny_gemma[0]), tiny_gemma[1])
    frames, nums = _frames(), [0, 15, 30, 45]
    a = s.score_full(frames, nums, 30.0, QUERIES)
    assert np.isfinite(a).all() and len(set(np.round(a, 5))) > 1            # the question matters
    assert np.array_equal(a, s.score_full(frames, nums, 30.0, QUERIES))
    assert np.array_equal(a, s.score_clip(frames, nums, 30.0, QUERIES))
    assert ((s.last_mass > 0) & (s.last_mass <= 1)).all() and not s.shortcut


def test_cpu_gpu_split_gives_the_same_scores(tiny_gemma):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from sfs.teacher.gemma import GemmaScorer, _nbytes
    model, proc = tiny_gemma
    ref = GemmaScorer.from_parts(copy.deepcopy(model), proc).score_full(_frames(), [0, 15, 30, 45], 30.0, QUERIES)
    m = copy.deepcopy(model)
    lm = m.model.language_model
    fixed = _nbytes(m, skip=[lm.embed_tokens_per_layer] + list(lm.layers))
    budget = (fixed + _nbytes(lm.layers[0]) + 1) / 1e9                    # room for exactly one layer
    split = GemmaScorer.from_parts(m, proc, device="cuda", gpu_budget_gb=budget)
    assert split.placement["layers_on_gpu"] == 1 and split.placement["layers_offloaded"] == 3
    assert split.placement["per_layer_table_on_cpu"]
    assert lm.embed_tokens_per_layer.emb.weight.device.type == "cpu"
    got = split.score_full(_frames(), [0, 15, 30, 45], 30.0, QUERIES)
    assert np.allclose(got, ref, atol=1e-3), (got, ref)



def test_real_config_builds_with_the_scorer_attention_and_switches_kernels():
    """The full E4B architecture (audio tower included) on the meta device: no memory, no weights."""
    if not HAVE_GEMMA:
        pytest.skip("gemma-4-E4B-it config files not present")
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText
    from sfs.teacher.gemma import GemmaScorer, attention
    with torch.device("meta"):
        m = AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(GEMMA), attn_implementation=attention())
    s = GemmaScorer.__new__(GemmaScorer)
    s.model = m
    for name in ("eager", "sdpa"):
        s.set_attn(name)
        assert m.model.language_model.config._attn_implementation == name
        assert m.model.vision_tower.config._attn_implementation == name
    assert m.model.audio_tower.config._attn_implementation == "eager"


def test_size_count_leaves_out_whole_subtrees():
    import torch.nn as nn
    from sfs.teacher.gemma import _nbytes
    inner = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))   # 2 x 16 floats
    outer = nn.Sequential(nn.Linear(4, 2, bias=False), inner)                          # + 8 floats
    assert _nbytes(outer) == (8 + 32) * 4
    assert _nbytes(outer, skip=[inner]) == 8 * 4                    # children of a skipped module too
    outer[0].weight = inner[0].weight                              # shared tensor counted once
    assert _nbytes(outer) == 32 * 4

def test_loader_picks_the_scorer_from_the_config(tmp_path, monkeypatch):
    import sfs.teacher.gemma as gm
    import sfs.teacher.vlm as vlm
    monkeypatch.setattr(gm, "GemmaScorer", lambda path, kind, **kw: ("gemma", kw))
    monkeypatch.setattr(vlm, "VLMScorer", lambda path, kind, **kw: ("qwen", kw))
    for mt, want in [("gemma4", "gemma"), ("gemma4_unified", "gemma"), ("qwen3_vl", "qwen"), ("qwen2_5_vl", "qwen")]:
        d = tmp_path / mt
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"model_type": mt}))
        assert vlm.load_scorer(str(d), "instruct", gpu_budget_gb=5.0)[0] == want


class _FakeGemma:
    kind, model_id, revision, shortcut = "instruct", "fake-gemma@0", "0", False
    placement = {"layers_on_gpu": 20, "layers_offloaded": 22}

    def __init__(self):
        self.last_mass = np.empty(0, np.float32)

    def score_full(self, frames, nums, fps, queries):
        self.last_mass = np.full(len(queries), 0.9, np.float32)
        return np.array([len(q) / 10 for q in queries], np.float32)

    score_clip = score_full

    def sync(self):
        pass

    def reset_peak_memory(self):
        pass

    def peak_memory_gb(self):
        return 6.0

    def set_attn(self, name):
        pass


class _Crops:
    def frames(self, iid, view, k=8):
        return _frames(4, iid), [0, 1, 2, 3]

    def fps_of(self, iid):
        return 2.0


def test_g0_records_placement_shortcut_and_yes_no_mass():
    from sfs.teacher.gate import g0
    manifest = pd.DataFrame({"instance_id": [str(i) for i in range(40)], "split": ["val"] * 40})
    captions = {f"c{i:02d}": [f"caption {i} a", f"caption {i} b"] for i in range(32)}
    rep = g0(_FakeGemma(), _Crops(), manifest, captions)
    assert rep["shortcut"] is False and rep["placement"]["layers_offloaded"] == 22
    assert rep["mean_yes_no_mass"] == pytest.approx(0.9)

