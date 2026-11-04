"""Gates (spec-rerank-cascade D3/D4, spec-vlm-followups E2/E3): the shortcut check against the bf16
noise floor, and G1 at full depth (every validation clip)."""
import numpy as np

from sfs.eval.queries import Query
from sfs.teacher.gate import ValQ1, g1_evaluate, shortcut_check


def test_shortcut_check_passes_rounding_noise_and_fails_a_real_mismatch():
    rng = np.random.default_rng(0)
    full = rng.normal(0, 5, 48)
    other = full + rng.uniform(-0.9, 0.9, 48)              # two attention kernels: bf16 noise
    ok = shortcut_check(full + rng.uniform(-0.4, 0.4, 48), full, other)
    assert ok["shortcut_pass"] and ok["cached_vs_full_spearman"] > 0.99
    shifted = full.copy()
    shifted[:5] += 3.0                                       # a real bug: some scores move by 3
    assert not shortcut_check(shifted, full, other)["shortcut_pass"]
    shuffled = rng.permutation(full)                         # same range, wrong order
    assert not shortcut_check(shuffled, full, full + 100)["shortcut_pass"]


def _val():
    v = ValQ1.__new__(ValQ1)
    v.ids = np.array([1, 2, 3, 4, 5])
    v.texts = [("a", 0, "text a")]
    v.text_col = {("a", 0): 0}
    v.lora = np.array([[5.0], [4.0], [3.0], [2.0], [1.0]])  # relevant clip 5 ranked last by LoRA
    v.queries = [Query("q1|split:val|a|0", "q1", "a", 0, "text a", frozenset(range(1, 6)), frozenset({5}))]
    v.activities = ["a"]
    return v


def test_g1_whole_validation_depth_reaches_every_clip():
    v = _val()
    assert len(v.top(2)) == 2 and {i for i, _ in v.top(None)} == {1, 2, 3, 4, 5}
    vlm = {(i, "text a"): (9.0 if i == 5 else 0.0) for i in range(1, 6)}
    top3 = g1_evaluate(v, vlm, k=3)                          # clip 5 is outside the top 3: no gain
    every = g1_evaluate(v, vlm, k=None)                      # whole depth: the VLM lifts clip 5 to rank 1
    assert top3["gain"] == 0.0 and every["best_val_map"] == 1.0 and every["best_weight"] == 0.0
    small = g1_evaluate(v, vlm, k=5)                         # k >= corpus size equals whole depth
    assert small["best_val_map"] == every["best_val_map"]


def _diag():
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "g1_frames_diagnostic", os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "g1_frames_diagnostic.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_frames_diagnostic_splits_long_and_short_clips():
    d = _diag()
    assert d.auc([2, 2], [1, 2]) == 0.75 and d.auc([], [1]) is None
    q = Query("q", "q1", "a", 0, "t", frozenset(range(1, 5)), frozenset({1, 3}))
    nf = {1: 3, 2: 4, 3: 16, 4: 12}                          # 1, 2 short (<= 8 frames); 3, 4 long
    s8 = {(1, "t"): 1.0, (2, "t"): 0.0, (3, "t"): 0.0, (4, "t"): 1.0}
    s16 = {**s8, (3, "t"): 2.0}                              # 16 frames fix the long pair only
    out = d.long_short_auc([q], {"t": [1, 2, 3, 4]}, nf, {"8": s8, "16": s16})
    assert out["short"]["8"]["mean_auc"] == out["short"]["16"]["mean_auc"] == 1.0
    assert out["long"]["8"]["mean_auc"] == 0.0 and out["long"]["16"]["mean_auc"] == 1.0
    assert out["long"]["16"]["n_queries"] == 1
