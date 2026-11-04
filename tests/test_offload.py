"""CPU/GPU split for Qwen-VL (D-041, E4 32B): the tiny random-weight Qwen3-VL split across CPU and GPU
(token embeddings looked up on the CPU, decoder layers streamed) scores exactly like one device, on the
cached and the full path; the question batch size is configurable."""
import copy

import numpy as np
import pytest

QUERIES = ["a person opens a car door", "someone sits down", "a car turns left at the junction"]


def _frames():
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (60, 40, 3), np.uint8) for _ in range(4)], [0, 15, 30, 45]


@pytest.mark.parametrize("tied", [True, False])
def test_split_qwen_scores_like_one_device(tiny_instruct_scorer, tied):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from sfs.teacher.offload import _nbytes
    from sfs.teacher.vlm import VLMScorer
    model = copy.deepcopy(tiny_instruct_scorer.model)
    if not tied:                                     # like the 32B: lm_head has its own weight
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    assert (model.lm_head.weight is model.model.language_model.embed_tokens.weight) == tied
    ref = VLMScorer.from_parts(copy.deepcopy(model).to("cuda"), tiny_instruct_scorer.processor, "instruct", device="cuda")
    frames, nums = _frames()
    want_full, want_clip = ref.score_full(frames, nums, 30.0, QUERIES), ref.score_clip(frames, nums, 30.0, QUERIES)
    s = VLMScorer.from_parts(model, tiny_instruct_scorer.processor, "instruct", device="cuda")
    lm = model.model.language_model
    emb_skip = [] if tied else [lm.embed_tokens]
    budget = (_nbytes(model, skip=emb_skip + list(lm.layers)) + _nbytes(lm.layers[0]) + 1) / 1e9
    p = s.split(budget, pin=False)
    assert p["layers_on_gpu"] == 1 and p["layers_offloaded"] == 1 and p["token_embeddings_on_cpu"] == (not tied)
    if not tied:
        assert lm.embed_tokens.emb.weight.device.type == "cpu"
    assert lm.layers[1].mlp.down_proj.weight.device.type == "cpu"
    assert np.allclose(s.score_full(frames, nums, 30.0, QUERIES), want_full, atol=1e-4)
    s.clip_batch = 1
    assert np.allclose(s.score_clip(frames, nums, 30.0, QUERIES), want_clip, atol=1e-4)
    assert lm.layers[1].mlp.down_proj.weight.device.type == "cpu"            # back on the CPU after the pass


def test_pinning_is_skipped_when_ram_is_short(monkeypatch, tiny_instruct_scorer):
    import sfs.teacher.offload as off
    model = copy.deepcopy(tiny_instruct_scorer.model)
    lm = model.model.language_model
    monkeypatch.setattr(off, "mem_available_gb", lambda: 1.0)                 # 1 GB free < margin
    p = off.place_layers(model, lm.layers, "cpu", 0.0, [lm.embed_tokens])
    assert p["layers_offloaded"] == len(lm.layers) and p["pinned"] is False
