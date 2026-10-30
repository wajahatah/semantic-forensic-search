import math

import numpy as np
import pandas as pd
import pytest

from sfs.data.regions import VIEWS
from sfs.eval.baseline import evaluate, summarise
from sfs.eval.metrics import (average_precision, precision_at_k, r_precision, rank, recall_at_k,
                              reciprocal_rank, success_at_k)
from sfs.eval.queries import (build_q1, build_q2, check_phrasings, eligible_activities,
                              load_phrasings)
from sfs.eval.retrieval import InstanceIndex


# ---------------- metrics: hand-computed ----------------
def test_metrics_hand_computed():
    rel = np.array([1, 0, 1, 0, 0], bool)            # relevant at ranks 1 and 3, R = 2
    assert average_precision(rel) == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert precision_at_k(rel, 2) == 0.5
    assert recall_at_k(rel, 1) == 0.5 and recall_at_k(rel, 5) == 1.0
    assert r_precision(rel) == 0.5                    # top-2 contains 1 of 2 relevant
    assert reciprocal_rank(rel) == 1.0
    late = np.array([0, 0, 1], bool)
    assert reciprocal_rank(late) == pytest.approx(1 / 3) and average_precision(late) == pytest.approx(1 / 3)


def test_metrics_no_relevant():
    rel = np.zeros(4, bool)
    assert math.isnan(average_precision(rel)) and math.isnan(recall_at_k(rel, 2))
    assert reciprocal_rank(rel) == 0.0 and precision_at_k(rel, 2) == 0.0


def test_success_at_k_not_capped_by_many_relevant():
    many = np.array([1, 1, 1, 1, 0, 0], bool)         # 4 relevant, k = 1: recall is capped at 1/4
    assert success_at_k(many, 1) == 1.0 and recall_at_k(many, 1) == 0.25
    late = np.array([0, 0, 0, 1], bool)
    assert success_at_k(late, 3) == 0.0 and success_at_k(late, 4) == 1.0
    assert math.isnan(success_at_k(np.zeros(3, bool), 2))


def test_rank_is_deterministic_on_ties():
    assert rank(np.array([3, 1, 2]), np.array([0.5, 0.5, 0.9])).tolist() == [2, 1, 3]


# ---------------- queries ----------------
def _manifest():
    rows = []
    iid = 0
    for cam, hour, act, n, split, split_b in [
        ("G1", "10", "open_door", 25, "eval", "eval"), ("G1", "10", "walk", 3, "eval", "eval"),
        ("G2", "11", "open_door", 2, "eval", "train"), ("G2", "11", "rare", 4, "eval", "train"),
        ("G3", "12", "walk", 30, "eval", "train"), ("G3", "12", "open_door", 5, "train", "val"),
    ]:
        for _ in range(n):
            rows.append({"instance_id": iid, "activity": act, "camera": cam, "date": "2018-03-07",
                         "hour": hour, "split": split, "split_b": split_b})
            iid += 1
    return pd.DataFrame(rows)


PHR = {"open_door": ["a person opening a door", "someone opens a door"],
       "walk": ["a person walking", "someone walks"]}


def test_eligible_activities_threshold():
    assert eligible_activities(_manifest()) == ["open_door", "walk"]  # 'rare' has 4 eval


def test_q1_relevant_sets_and_corpus():
    m = _manifest()
    q = build_q1(m, "split", ["open_door", "walk"], PHR)
    assert len(q) == 4                                  # 2 activities x 2 phrasings
    door = [x for x in q if x.activity == "open_door"][0]
    assert len(door.relevant) == 27 and len(door.corpus) == 64  # train rows excluded from both
    assert door.relevant <= door.corpus


def test_q2_windows_filter_corpus():
    m = _manifest()
    q = build_q2(m, "split", ["open_door", "walk"], PHR)
    g2 = [x for x in q if x.filter["camera"] == "G2"]
    assert {x.activity for x in g2} == {"open_door"}    # no 'walk' in that window
    assert len(g2[0].corpus) == 6 and len(g2[0].relevant) == 2  # rare instances are distractors
    assert all(x.relevant <= x.corpus for x in q)


def test_q1_validation_subset():
    m = _manifest()
    q = build_q1(m, "split_b", ["open_door", "walk"], PHR, subset="val")
    assert {x.activity for x in q} == {"open_door"}    # only G3 open_door rows are val under B
    assert len(q[0].corpus) == 5 and q[0].relevant == q[0].corpus
    assert q[0].qid == "q1|split_b:val|open_door|0"    # never collides with an eval query id
    assert build_q1(m, "split_b", ["open_door"], PHR)[0].qid == "q1|split_b|open_door|0"


def test_protocol_b_uses_its_own_eval_rows():
    q = build_q1(_manifest(), "split_b", ["open_door", "walk"], PHR)
    assert len(q[0].corpus) == 28                       # only G1 rows are eval under protocol B


def test_phrasings_file_validation(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text("# reviewed\nactivity,phrasing\nopen_door,a person opening a door\nwalk,someone walks\n")
    phr = load_phrasings(str(p))
    check_phrasings(phr, ["open_door", "walk"])
    with pytest.raises(ValueError, match="missing"):
        check_phrasings(phr, ["open_door", "walk", "sit"])
    p.write_text("activity,phrasing\nwalk,x\nwalk,x\n")
    with pytest.raises(ValueError):
        load_phrasings(str(p))


# ---------------- retrieval + evaluate ----------------
def _shard():
    # instance 0: frames close to e0; instance 1: close to e1; instance 2 (other view) ignored
    e0, e1 = np.eye(4)[0], np.eye(4)[1]
    emb = np.stack([e0, e0 * 0.9 + e1 * 0.1, e1, e1, e1, e0])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return {"emb": emb.astype(np.float16), "instance_id": np.array([0, 0, 1, 1, 1, 2]),
            "frame_idx": np.arange(6), "view": np.array([0, 0, 0, 0, 0, VIEWS.index("full")])}


def test_instance_index_mean_of_top_m():
    idx = InstanceIndex.from_shard(_shard(), "tight")
    assert idx.instance_ids.tolist() == [0, 1]
    s = idx.score(np.eye(4)[:2], m=3)
    assert s[0, 0] > s[1, 0] and s[1, 1] > s[0, 1]
    assert s[1, 1] == pytest.approx(1.0, abs=1e-3)       # all three frames of instance 1 are e1


def test_instance_index_subset():
    assert InstanceIndex.from_shard(_shard(), "tight", {1}).instance_ids.tolist() == [1]


def test_evaluate_perfect_and_summary():
    from sfs.eval.queries import Query
    ids = np.array([10, 11, 12])
    scores = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]])
    qs = [Query("a", "q1", "door", 0, "t", frozenset({10, 11, 12}), frozenset({10, 11})),
          Query("b", "q1", "walk", 0, "t", frozenset({10, 11, 12}), frozenset({12}))]
    df = evaluate(qs, ids, scores, {("door", 0): 0, ("walk", 0): 1})
    assert df.ap.tolist() == [1.0, 1.0] and df.rr.tolist() == [1.0, 1.0]
    s = summarise(df)["q1"]
    assert s["activities"] == 2 and s["macro"]["ap"] == 1.0
