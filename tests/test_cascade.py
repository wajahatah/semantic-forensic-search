"""Cascade evaluation (spec-rerank-cascade D8): top-k re-ranking, exact cost counting, and refusal
to evaluate a system whose VLM scores were never computed."""
import numpy as np
import pytest

from sfs.eval.cascade import System, evaluate_systems, top_needed, topk_rerank
from sfs.eval.metrics import rank
from sfs.eval.queries import Query


def test_topk_rerank_only_reorders_the_top_k():
    ids = np.array([1, 2, 3, 4, 5])
    base = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    new = topk_rerank(ids, base, np.array([0.0, 1.0, 2.0]), k=3, gamma=0.0)   # VLM prefers 3 > 2 > 1
    assert rank(ids, new).tolist() == [3, 2, 1, 4, 5]
    keep = topk_rerank(ids, base, np.array([0.0, 1.0, 2.0]), k=3, gamma=100.0)
    assert rank(ids, keep).tolist() == [1, 2, 3, 4, 5]                        # first stage dominates
    assert np.array_equal(topk_rerank(ids, base, np.array([]), k=0, gamma=0.0), base)
    with pytest.raises(ValueError):
        topk_rerank(ids, base, np.array([1.0]), k=2, gamma=0.0)


def _setup():
    ids = np.arange(1, 7)
    q = Query("q1|x|a|0", "q2", "a", 0, "text a", frozenset({1, 2, 3, 4}), frozenset({4}))
    first = np.array([[6.0], [5.0], [4.0], [3.0], [2.0], [1.0]])           # instance 4 ranked 4th
    vlm = {(i, "text a"): (10.0 if i == 4 else 0.0) for i in range(1, 7)}
    return ids, q, {"lora": first}, {("a", 0): 0}, vlm


def test_costs_and_success():
    ids, q, firsts, col, vlm = _setup()
    systems = [System("lora", "lora"), System("v@2", "lora", 2), System("v@5", "lora", 5),
               System("all", "lora", None)]
    df = evaluate_systems([q], ids, firsts, col, vlm, systems).set_index("system")
    assert df.cost.tolist() == [0, 2, 4, 4]                  # k capped at the corpus size (4)
    assert df.loc["lora", "s1"] == 0 and df.loc["v@2", "s1"] == 0  # relevant item outside the top 2
    assert df.loc["v@5", "s1"] == 1 and df.loc["all", "rr"] == 1


def test_missing_vlm_score_is_an_error():
    ids, q, firsts, col, vlm = _setup()
    del vlm[(1, "text a")]
    with pytest.raises(KeyError, match="score it first"):
        evaluate_systems([q], ids, firsts, col, vlm, [System("v@2", "lora", 2)])


def test_top_needed():
    ids, q, firsts, col, _ = _setup()
    assert top_needed([q], ids, firsts["lora"], col, 2) == {(1, "text a"), (2, "text a")}
    assert len(top_needed([q], ids, firsts["lora"], col, None)) == 4


def test_direction_test_uses_only_vlm_scored_pairs():
    from sfs.eval.cascade_run import direction_test
    L, R = "vehicle_turns_left", "vehicle_turns_right"
    ids = np.arange(1, 6)
    act = {1: L, 2: L, 3: R, 4: R, 5: "person_sits_down"}
    q = Query("q2|w|L|0", "q2", L, 0, "turn left", frozenset({1, 2, 3, 4, 5}), frozenset({1, 2}))
    perfect = np.array([[3.0], [2.0], [1.0], [0.0], [9.0]])            # left above right
    flipped = -perfect
    vlm = {(1, "turn left"): 0.0, (3, "turn left"): 1.0, (5, "turn left"): 5.0}   # 2 and 4 never scored
    out = direction_test([q], ids, {"good": perfect, "bad": flipped}, {(L, 0): 0}, vlm, act)
    assert out == {"n_queries": 1, "vlm_auc": 0.0, "good_auc": 1.0, "bad_auc": 0.0}


def test_cascade_report_figure_and_tables(tmp_path):
    import importlib.util
    import json
    import os
    import pandas as pd
    spec = importlib.util.spec_from_file_location(
        "cascade_report", os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "cascade_report.py"))
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)
    row = lambda c, v: {"cost": c, "s1": v, "s5": v, "s10": v, "rr": v, "ap": v, "p10": v}   # noqa: E731
    tab = {"lora": row(0, .3), "student": row(0, .5), "lora>instruct@5": row(5, .4), "lora>instruct@10": row(10, .45),
           "student>instruct@1": row(1, .5), "instruct-all": row(56, .35)}
    rep = {"summary": {f"{p}/{k}": tab for p in "AB" for k in ("q1", "q2")},
           "students": {p: {"seconds_per_query_whole_eval_corpus": 0.01} for p in "AB"},
           "vlm_seconds_per_call_uncached": {"instruct": 0.2},
           "h5_direction": {"A/instruct": {"n_queries": 3, "vlm_auc": 0.5, "lora_auc": 0.6,
                                           "student_only_auc": None, "student_auc": 0.9}}}
    base = str(tmp_path / "r")
    json.dump(rep, open(base + ".json", "w"))
    pd.DataFrame([{"kind": "q2", "protocol": "A", "activity": "a", "system": "lora", "s1": 1, "ap": 1}]).to_csv(
        base + "_queries.csv", index=False)
    assert cr.main(["--report", base, "--fig", str(tmp_path / "f.svg"), "--tables", str(tmp_path / "t.md"),
                    "--per-activity", str(tmp_path / "p.csv")]) == 0
    svg = open(tmp_path / "f.svg", encoding="utf-8").read()
    assert svg.startswith("<svg") and svg.count("<polyline") == 4          # two series x two panels
    md = open(tmp_path / "t.md", encoding="utf-8").read()
    assert "| A | lora>instruct@10 | 10.0 | 2.00 |" in md and "| A | student | 0.0 | 0.01 |" in md
    assert "| A/instruct | 3 | 0.500 | 0.600 | — | 0.900 |" in md


def test_cascade_report_hypotheses_groups_and_bootstrap(tmp_path):
    import importlib.util
    import os
    import pandas as pd
    spec = importlib.util.spec_from_file_location(
        "cascade_report", os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "cascade_report.py"))
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)
    assert cr.boot([1.0, 1.0, 1.0]) == (1.0, 1.0, 1.0)
    systems = {"lora": 0.2, "student": 0.5, "student[GTKD]": 0.4, "student[KD]": 0.1, "lora>instruct@10": 0.3,
               "lora>reranker@10": 0.3, "student>instruct@10": 0.6, "student>reranker@10": 0.5,
               "instruct-all": 0.25, "student>instruct@5": 0.55}
    rows = []
    for p in "AB":
        for act, bump in (("move", 0.3), ("look", 0.0)):
            for s_, v in systems.items():
                cost = {"instruct-all": 50, "student>instruct@5": 5}.get(s_, 0)
                for kind in ("q1", "q2"):
                    if kind == "q1" and s_ == "instruct-all":
                        continue
                    val = v + (bump if s_ == "student" else 0)
                    rows.append({"protocol": p, "kind": kind, "activity": act, "system": s_, "ap": val, "rr": val,
                                 "s1": val, "cost": cost})
    q = str(tmp_path / "q.csv")
    pd.DataFrame(rows).to_csv(q, index=False)
    b = str(tmp_path / "b.csv")
    pd.DataFrame([{"protocol": p, "activity": a, "lift_base": l} for p in "AB" for a, l in (("move", 1.5), ("look", 9.0))]
                 ).to_csv(b, index=False)
    md = cr.hypotheses(q, b)
    assert "| A motion (lift ≤ 2): move | 1 | 0.200 → 0.800 | +0.600 |" in md       # motion group gets the bump
    assert "| A rest:  | 1 | 0.200 → 0.500 | +0.300 |" in md
    assert "| A | 0.550 | 0.250 | 2.20 | 5.0 | 50.0 | 10.0% |" in md


def test_everything_q1_requests_every_candidate_and_adds_all_points():
    from sfs.eval.cascade_run import build_requests, systems_for
    ids, q2, firsts, col, _ = _setup()
    q1 = Query("q1|a|0", "q1", "a", 0, "text a", frozenset(range(1, 7)), frozenset({4}))
    data = {"queries": [q1, q2], "ids": ids, "firsts": {**firsts, "student": firsts["lora"]}, "text_col": col}
    top = build_requests(data, everything_q2=False, k_q2=2, k_q1=3)
    assert {i for i, _ in top} == {1, 2, 3}                       # Q1 top 3, Q2 top 2 (inside 1..3)
    every = build_requests(data, everything_q2=False, k_q2=2, k_q1=3, everything_q1=True)
    assert {i for i, _ in every} == set(range(1, 7))              # every Q1 candidate
    names = [s.name for s in systems_for("instruct", 2.0, 1.0, None)]
    assert "instruct-all" in names and "instruct-all+lora" in names
    assert not any(n.endswith("-all") for n in [s.name for s in systems_for("instruct", 2.0, 1.0, 50)])


def test_merge_turns_relabels_before_queries_and_parses_stores():
    import pandas as pd
    from sfs.eval.cascade_run import parse_merge, parse_stores
    from sfs.eval.queries import build_q1
    mp = parse_merge("vehicle_turns_left, vehicle_turns_right=vehicle_turns; a,b=c")
    assert mp == {"vehicle_turns_left": "vehicle_turns", "vehicle_turns_right": "vehicle_turns", "a": "c", "b": "c"}
    assert parse_stores("instruct=x,instruct=y,reranker=z") == {"instruct": ["x", "y"], "reranker": ["z"]}
    m = pd.DataFrame({"instance_id": [1, 2, 3, 4], "split": ["eval"] * 4,
                      "activity": ["vehicle_turns_left", "vehicle_turns_right", "vehicle_stops", "vehicle_turns_left"]})
    m["activity"] = m.activity.replace(mp)
    qs = build_q1(m, "split", ["vehicle_stops", "vehicle_turns"],
                  {"vehicle_stops": ["a car stopping"], "vehicle_turns": ["a car making a turn"]})
    turn = [q for q in qs if q.activity == "vehicle_turns"][0]
    assert turn.relevant == frozenset({1, 2, 4}) and turn.corpus == frozenset({1, 2, 3, 4})


def test_g0_file_per_model(tmp_path):
    from sfs.eval.cascade_run import g0_files
    g1 = str(tmp_path / "g1_32b.json")
    big = tmp_path / "g0_32b.json"
    big.write_text("{}")
    # default: Result 04 layout next to the G1 file
    assert g0_files("", g1, ["instruct", "reranker"]) == {
        "instruct": str(tmp_path / "g0_instruct.json"), "reranker": str(tmp_path / "g0_reranker.json")}
    # named: the model's own file, never the 2B's
    assert g0_files(f"instruct={big}", g1, ["instruct"]) == {"instruct": str(big)}
    with pytest.raises(FileNotFoundError):
        g0_files(f"instruct={tmp_path / 'missing.json'}", g1, ["instruct"])
    with pytest.raises(ValueError):
        g0_files(f"instruct={big},instruct={big}", g1, ["instruct"])
