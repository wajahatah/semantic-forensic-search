import os

import pytest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
STEM = "2018-03-13.17-05-00.17-10-00.school.G328"


@pytest.fixture
def act_path():
    return os.path.join(FIXTURES, f"{STEM}.activities.yml")


@pytest.fixture
def geom_path():
    return os.path.join(FIXTURES, f"{STEM}.geom.yml")


# --- tiny random-weight Qwen3-VL instruct scorer (VLM teacher and office-server tests). No weight
# download; needs the real processor files of the 2B model in data/models, skipped without them.
PROCESSOR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models", "Qwen3-VL-2B-Instruct")


@pytest.fixture(scope="session")
def tiny_instruct_scorer():
    if not os.path.exists(os.path.join(PROCESSOR, "preprocessor_config.json")):
        pytest.skip("Qwen3-VL-2B-Instruct processor files not present")
    import torch
    from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from sfs.teacher.vlm import VLMScorer

    torch.manual_seed(0)
    cfg = Qwen3VLConfig.from_pretrained(PROCESSOR)
    t = cfg.text_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 64, 128, 2
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 16
    t.rope_scaling = {**t.rope_scaling, "mrope_section": [4, 2, 2]}
    if getattr(t, "rope_parameters", None):
        t.rope_parameters = {**t.rope_parameters, "mrope_section": [4, 2, 2]}
    v = cfg.vision_config
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads = 2, 32, 64, 2
    v.out_hidden_size, v.deepstack_visual_indexes = 64, [0, 1]
    model = Qwen3VLForConditionalGeneration(cfg).float()
    return VLMScorer.from_parts(model, AutoProcessor.from_pretrained(PROCESSOR), "instruct")
