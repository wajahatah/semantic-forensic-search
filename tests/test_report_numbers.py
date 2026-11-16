"""The report's numbers file reproduces the headline cells of the Result files (spec-report §8.1)."""
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("report_numbers", os.path.join(ROOT, "scripts", "report_numbers.py"))
rn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rn)

RAW = os.path.join(ROOT, "docs", "results", "raw")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(RAW, "04_cascade.json")), reason="raw results not here")


@pytest.fixture(scope="module")
def n():
    return rn.collect(RAW, manifest=os.path.join(ROOT, "manifest_v2_drive", "manifest_v2.csv"))


class _R3(float):
    """A value rounded to 3 decimals, compared with a half-unit tolerance (0.5965 == 0.596 and 0.597)."""

    def __eq__(self, other):
        return abs(float(self) - float(other)) <= 0.0005 + 1e-12


def r3(x):
    return _R3(x)


def test_baseline_and_lora_match_results_02_and_03(n):
    assert n["baseline"]["A"]["corpus"] == 3726 and n["baseline"]["B"]["corpus"] == 3938
    assert r3(n["baseline"]["A"]["systems"]["siglip2-tight"]["q1"]["ap"]) == 0.179
    assert r3(n["baseline"]["B"]["systems"]["siglip2-tight"]["q1"]["ap"]) == 0.174
    assert r3(n["baseline"]["A"]["systems"]["siglip2-full"]["q1"]["ap"]) == 0.096
    assert r3(n["baseline"]["A"]["systems"]["siglip2-tight"]["q2"]["r10"]) == 0.511 or r3(n["baseline"]["A"]["systems"]["siglip2-tight"]["q2"]["r10"]) == 0.510
    assert n["lora"]["selection"]["A"]["selected"] == "A-middle" and n["lora"]["selection"]["B"]["selected"] == "B-uniform"
    assert r3(n["lora"]["runs"]["A-middle"]["q1"]["ap"]) == 0.313 and r3(n["lora"]["runs"]["B-uniform"]["q1"]["ap"]) == 0.280
    assert r3(n["lora"]["runs"]["A-middle"]["q2"]["s1"]) == 0.389 and r3(n["lora"]["runs"]["B-uniform"]["q2"]["s1"]) == 0.369
    assert r3(n["lora"]["runs"]["A-middle"]["q2"]["s10"]) == 0.902
    assert len(n["lora"]["per_activity"]) == 44                           # 22 activities x 2 protocols


def test_cascade_cells_match_results_04_and_05(n):
    a = n["cascade"]["2b"]["summary"]["A/q2"]
    b = n["cascade"]["2b"]["summary"]["B/q2"]
    assert r3(a["student"]["s1"]) == 0.531 and r3(b["student"]["s1"]) == 0.519
    assert r3(a["student>instruct@10"]["s1"]) == 0.560 and r3(a["student>instruct@5"]["s1"]) == 0.546
    assert r3(n["cascade"]["2b"]["summary"]["A/q1"]["student"]["ap"]) == 0.476
    assert r3(n["cascade"]["32b"]["summary"]["A/q2"]["student>instruct@10"]["s1"]) == 0.596
    assert r3(n["cascade"]["q1all"]["summary"]["A/q1"]["instruct-all"]["ap"]) == 0.207
    assert n["students"]["A"]["selected"] == "A-GT" and r3(n["students"]["A"]["val_map"]["GT"]) == 0.600
    assert r3(n["students"]["A"]["val_map"]["KD"]) == 0.424
    assert r3(n["cascade"]["2b"]["vlm_seconds_per_call"]["instruct"]) == 0.168
    assert r3(n["cascade"]["32b"]["vlm_seconds_per_call"]["instruct"]) == 1.742
    bs = n["bootstrap"]["32b"]["A: 32B@10 - student"]
    assert bs["diff"] == 0.065 and bs["ci95"] == [0.04, 0.095]
    assert n["bootstrap"]["8b"]["A: 8B@10 - student"]["diff"] == 0.058


def test_gates_e8_and_drift_match_results_04_to_06(n):
    g = n["g1"]
    assert not g["2b_top50"]["pass"] and g["2b_top50"]["teacher"] == "reranker-tight"
    assert not g["8b_top50"]["pass"] and r3(g["8b_top50"]["configs"]["instruct-tight"]["B"]["gain"]) == 0.019
    assert g["32b_top50"]["pass"] and g["32b_wholeval"]["pass"]
    assert r3(g["32b_wholeval"]["configs"]["instruct-tight"]["A"]["gain"]) == 0.018
    assert r3(g["32b_wholeval"]["configs"]["instruct-tight"]["B"]["gain"]) == 0.047
    assert r3(g["gemma_top50"]["configs"]["gemma-e4b-tight"]["mean_gain"]) == -0.002
    assert not g["qwen25_3b_top50"]["pass"] and not g["qwen25_7b_top50"]["pass"] and not g["2b_16frames"]["pass"]
    assert n["g0"]["qwen3vl_8b_fp8"]["pass"] and not n["g0"]["qwen3vl_32b_fp8"]["pass"]
    assert r3(n["g1_auc"]["8b"]["A"]["Qwen8B"]) == 0.661
    v = n["e8"]["verdict"]
    assert not v["adopt"] and r3(v["mean_diff"]["A-middle"]) == -0.019 and r3(v["mean_diff"]["B-uniform"]) == 0.047
    assert not n["drift"]["all_pass"] and r3(n["drift"]["results"]["A0"]["sentences"]["rho"]) == 0.813
    assert n["drift"]["revision"].startswith("75de2d55")
