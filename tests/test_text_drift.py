import importlib.util
import os

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
_spec = importlib.util.spec_from_file_location("text_drift_check", os.path.join(ROOT, "scripts", "text_drift_check.py"))
td = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(td)


def _tiny_text():
    from transformers import SiglipTextConfig, SiglipTextModel
    torch.manual_seed(0)
    return SiglipTextModel(SiglipTextConfig(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                            num_attention_heads=2, max_position_embeddings=64, projection_size=32))


def _tokens(n, seed):
    g = torch.Generator().manual_seed(seed)
    return {"input_ids": torch.randint(1, 50, (n, 64), generator=g)}


def test_spearman_matches_scipy_definition_with_ties():
    a = np.array([1.0, 2.0, 2.0, 3.0, 5.0])
    b = np.array([2.0, 1.0, 4.0, 4.0, 9.0])
    assert td.spearman(a, a) == pytest.approx(1.0)
    assert td.spearman(a, -a) == pytest.approx(-1.0)
    # ranks a = [1, 2.5, 2.5, 4, 5], b = [2, 1, 3.5, 3.5, 5]  -> Pearson of the ranks
    ra, rb = np.array([1, 2.5, 2.5, 4, 5]), np.array([2, 1, 3.5, 3.5, 5])
    assert td.spearman(a, b) == pytest.approx(np.corrcoef(ra, rb)[0, 1])


def test_drift_is_perfect_for_identical_embeddings_and_detects_a_shuffle():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(12, 8))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    assert td.drift(E, E) == {"rho": 1.0, "self_cos": 1.0, "nn_agree": 1.0}
    d = td.drift(E, E[rng.permutation(12)])
    assert d["rho"] < 0.9 and d["self_cos"] < 0.9


def test_zero_adapter_equals_the_frozen_text_side_and_a_trained_one_moves_it():
    from sfs.train.lora import adapter_state, wrap_lora
    text = _tiny_text()
    sets = {"sentences": _tokens(10, 1), "phrasings": _tokens(6, 2)}
    t = wrap_lora(_tiny_text(), 4, 8, 0.0)                 # fresh LoRA: B = 0, so the output is unchanged
    zero = adapter_state(t)
    moved = {k: (v + 0.5 * torch.randn_like(v)) for k, v in zero.items()}
    payload = lambda st: {"text_adapter": st, "config": {"lora_rank": 4, "lora_alpha": 8}}
    res = td.run(text, {"zero": payload(zero), "moved": payload(moved)}, sets)
    assert res["zero"]["sentences"] == {"rho": 1.0, "self_cos": 1.0, "nn_agree": 1.0} and res["zero"]["pass"]
    assert res["moved"]["sentences"]["self_cos"] < 1.0


def test_adapters_with_different_rank_are_refused():
    p = lambda r: {"text_adapter": {}, "config": {"lora_rank": r, "lora_alpha": 8}}
    with pytest.raises(ValueError, match="rank"):
        td.run(_tiny_text(), {"a": p(4), "b": p(8)}, {"sentences": _tokens(3, 0)})


def test_sentence_file_is_frozen_and_disjoint_from_training_captions():
    from sfs.train.captions import load_captions, normalise, similarity
    s = pd.read_csv(os.path.join(ROOT, "queries", "appearance_sentences.csv"), comment="#")
    assert len(s) == 40 and s.sentence.is_unique
    caps = [c for cs in load_captions(os.path.join(ROOT, "queries", "train_captions.csv")).values() for c in cs]
    for t in s.sentence:
        assert normalise(t) not in {normalise(c) for c in caps}
        assert max(similarity(t, c) for c in caps) < 0.80
