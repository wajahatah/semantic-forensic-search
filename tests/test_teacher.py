"""VLM teacher (spec-rerank-cascade §5): frame picking, crop index, frame preparation, score store,
request grouping, and cached == full scoring on a tiny random-weight Qwen3-VL (no weight download;
the real processor files are needed for that one test and it is skipped without them)."""
import os

import numpy as np
import pytest

from sfs.data.regions import VIEWS
from sfs.store.crops import write_crop_shard
from sfs.teacher.frames import CropIndex, pick_even
from sfs.teacher.run import group_requests
from sfs.teacher.store import load_scores, read_requests, save_scores, write_requests
from sfs.teacher.vlm import _metadata, prepare_frames

ROOT = os.path.dirname(os.path.dirname(__file__))
PROCESSOR = os.path.join(ROOT, "data", "models", "Qwen3-VL-2B-Instruct")


def test_pick_even():
    assert pick_even(16, 8) == [0, 2, 4, 6, 9, 11, 13, 15]
    assert pick_even(5, 8) == [0, 1, 2, 3, 4]
    assert pick_even(1, 8) == [0]
    with pytest.raises(ValueError):
        pick_even(0)


def _png(value: int, side: int = 40) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".png", np.full((side, side, 3), value, np.uint8))
    return buf.tobytes()


def _store(tmp_path, store_id="s1"):
    # clip 0: instance 7 with frames 30, 10, 20 (stored out of order), tight + context; clip 1: instance 9
    rows = [(7, 30, "tight", 3), (7, 10, "tight", 1), (7, 20, "tight", 2), (7, 10, "context", 101)]
    write_crop_shard(str(tmp_path / "c0.npz"), [_png(v) for *_, v in rows],
                     {"store_id": store_id, "fps": 30.0},
                     instance_id=[r[0] for r in rows], frame_idx=[r[1] for r in rows],
                     view=[VIEWS.index(r[2]) for r in rows])
    write_crop_shard(str(tmp_path / "c1.npz"), [_png(50)], {"store_id": store_id, "fps": 25.0},
                     instance_id=[9], frame_idx=[5], view=[0])


def test_crop_index_time_order_views_and_fps(tmp_path):
    _store(tmp_path)
    idx = CropIndex(str(tmp_path))
    frames, nums = idx.frames(7, "tight")
    assert nums == [10, 20, 30] and [int(f[0, 0, 0]) for f in frames] == [1, 2, 3]
    assert idx.frames(7, "context")[1] == [10]
    assert idx.fps_of(7) == 30.0 and idx.fps_of(9) == 25.0
    with pytest.raises(KeyError):
        idx.frames(9, "context")


def test_crop_index_refuses_mixed_stores(tmp_path):
    _store(tmp_path)
    write_crop_shard(str(tmp_path / "c2.npz"), [_png(1)], {"store_id": "other", "fps": 30.0},
                     instance_id=[11], frame_idx=[0], view=[0])
    with pytest.raises(ValueError):
        CropIndex(str(tmp_path))


def test_prepare_frames_side_and_even_count():
    out = prepare_frames([np.zeros((224, 112, 3), np.uint8)] * 3)
    assert out.shape == (4, 448, 224, 3)              # longest side 448, padded to an even count
    assert prepare_frames([np.zeros((900, 600, 3), np.uint8)] * 2).shape == (2, 448, 299, 3)


def test_prepare_frames_letterboxes_a_full_frame_fallback():
    crop = np.full((224, 224, 3), 200, np.uint8)
    full = np.full((1080, 1920, 3), 100, np.uint8)          # no actor box at this frame -> full frame
    out = prepare_frames([crop, full])
    assert out.shape == (2, 448, 448, 3)
    assert (out[1, :98] == 0).all() and (out[1, 350:] == 0).all()      # 448x252 centred, black bars
    assert (out[1, 98:350] == 100).all() and (out[0] == 200).all()
    import cv2                                               # same-shape clips: plain resize, no padding
    same = [np.random.default_rng(v).integers(0, 255, (300, 300, 3), np.uint8) for v in (1, 2)]
    plain = np.stack([cv2.resize(f, (448, 448), interpolation=cv2.INTER_CUBIC) for f in same])
    assert np.array_equal(prepare_frames(same), plain)


def test_metadata_relative_frames():
    meta = _metadata([300, 315, 330], 30.0, 4)
    assert meta["frames_indices"] == [0, 15, 30, 30] and meta["fps"] == 30.0


def test_score_store_round_trip_and_eval_refusal(tmp_path):
    p = str(tmp_path / "s.npz")
    save_scores(p, [1, 1, 2], ["a b", "c", "a b"], [0.5, -1.0, 2.0], {"model_id": "m", "subset": "trainval"})
    scores, sid = load_scores(p)
    assert scores == {(1, "a b"): 0.5, (1, "c"): -1.0, (2, "a b"): 2.0} and len(sid) == 12
    save_scores(p, [1], ["x"], [0.0], {"model_id": "m", "subset": "eval"})
    with pytest.raises(PermissionError):
        load_scores(p)
    assert load_scores(p, allow_eval=True)[0] == {(1, "x"): 0.0}


def test_requests_round_trip_and_grouping(tmp_path):
    _store(tmp_path)
    p = str(tmp_path / "r.csv")
    write_requests(p, [(9, "q2"), (7, "q1"), (9, "q1"), (7, "q1")])
    pairs = read_requests(p)
    assert pairs == [(7, "q1"), (9, "q1"), (9, "q2")]       # de-duplicated, sorted
    groups = group_requests(pairs, CropIndex(str(tmp_path)))
    assert groups == [(7, ["q1"]), (9, ["q1", "q2"])]       # grouped per instance, clip order


@pytest.mark.skipif(not os.path.exists(os.path.join(PROCESSOR, "tokenizer.json")),
                    reason="needs the Qwen3-VL processor files")
def test_cached_scoring_equals_full_forward(tiny_instruct_scorer):
    """The KV-cache path (video once, many questions) must give the same margins as full passes."""
    scorer = tiny_instruct_scorer
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (60, 40, 3), np.uint8) for _ in range(4)]
    queries = ["a person opens a car door", "someone sits down", "a car turns left at the junction"]
    full = scorer.score_full(frames, [0, 15, 30, 45], 30.0, queries)
    cached = scorer.score_clip(frames, [0, 15, 30, 45], 30.0, queries, batch=2)
    assert np.allclose(full, cached, atol=1e-4), (full, cached)


def test_feedback_picks_most_confident_mistakes():
    from sfs.teacher.feedback import mistakes
    scores = {(1, "open door"): 5.0, (2, "open door"): 1.0, (3, "open door"): -4.0, (4, "open door"): -1.0}
    act_text = {"open door": "opens"}
    act_inst = {1: "sits", 2: "sits", 3: "opens", 4: "opens"}
    out = mistakes(scores, act_text, act_inst)["opens"]
    assert [(k, i) for k, i, _, _ in out] == [("said YES, wrong", 1), ("said YES, wrong", 2),
                                              ("said NO, wrong", 3), ("said NO, wrong", 4)]


def test_feedback_page_plain_questions_options_and_order():
    from sfs.teacher.feedback import _sure, render

    class Crops:
        def frames(self, iid, view):
            return [np.zeros((20, 20, 3), np.uint8)] * 2, [0, 1]
    page = {"intro": ["hello"], "reply_hint": "Reply Q1 A",
            "questions": [{"title": "Pick one?", "importance": "Most important", "what": "W", "did": "D",
                           "options": [{"key": "A", "text": "first", "recommended": True}, {"key": "B", "text": "second"}],
                           "details": "raw <numbers>"}]}
    g0 = [{"model_id": "m", "peak_vram_gb": 6.0, "seconds_per_pair_uncached": 0.2, "pass": True}]
    g1 = {"teacher": "t", "configs": {"t": {"pass": False, **{p: {"lora_val_map": .4, "best_val_map": .41,
                                                                 "vlm_only_val_map": .3} for p in "AB"}}}}
    ex = {"opens": [("said YES, wrong", 1, "open door", 6.0)], "turns": [("said NO, wrong", 2, "turn", -1.0)]}
    out = render(page, g0, g1, ex, Crops(), {1: "sits", 2: "turns"}, "tight", {"A": {"turns": -0.5, "opens": 0.1}})
    assert "Q1 · Most important" in out and "<b>A</b> — first <span class='rec'>my recommendation</span>" in out
    assert "<b>B</b> — second</li>" in out and "raw &lt;numbers&gt;" in out      # escaped, inside details
    assert out.index("<h3 class='act'>turns") < out.index("<h3 class='act'>opens")  # most hurt first
    assert "VLM very sure YES (score +6.0)" in out and "really <b>sits</b>" in out
    assert _sure(-1.0) == "leaning NO" and _sure(3.0) == "fairly sure YES"
