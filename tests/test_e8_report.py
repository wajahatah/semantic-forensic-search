import importlib.util
import json
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "e8_report", os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "e8_report.py"))
e8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e8)


def test_seed0_numbers_fail_the_rule_on_the_drop():
    v = e8.verdict({"A-middle": [-0.0284], "B-uniform": [0.0507]})
    assert v["gain_ok"] and not v["no_drop_ok"] and not v["adopt"]
    assert v["repeats"] == {"A-middle": False, "B-uniform": False}      # one seed never repeats


def test_rule_uses_the_seed_means_and_repeats_need_same_sign_above_the_bar():
    v = e8.verdict({"A-middle": [-0.020, 0.018], "B-uniform": [0.050, 0.030]})
    assert v["mean_diff"]["A-middle"] == pytest.approx(-0.001) and v["adopt"]
    assert v["repeats"] == {"A-middle": False, "B-uniform": True}
    assert not e8.verdict({"A-middle": [0.02, 0.005], "B-uniform": [0.03, 0.03]})["repeats"]["A-middle"]


def test_main_reads_the_run_folders_and_checks_the_seed(tmp_path, capsys):
    vals = {("io", 0): (0.40, 0.30), ("it", 0): (0.38, 0.36), ("io", 1): (0.41, 0.31), ("it", 1): (0.39, 0.35)}
    per_act = {}
    for (arm, s), (a, b) in vals.items():
        for r, m in zip(e8.RUNS, (a, b)):
            d = tmp_path / e8.run_dir(arm, s) / r
            d.mkdir(parents=True)
            (d / "done.json").write_text(json.dumps({"best_val_map": m, "best_epoch": 3, "epochs_run": 8,
                                                     "frozen_val_map": 0.27, "config_hash": "x",
                                                     "config": {"seed": s}}))
            per_act[f"{e8.run_dir(arm, s)}/{r}"] = {"x": m, "y": 1 - m if arm == "io" else 1 - m + (0.1 if s else -0.1)}
    (tmp_path / "per_activity.json").write_text(json.dumps(per_act))
    out = tmp_path / "report.json"
    assert e8.main(["--root", str(tmp_path), "--seeds", "0,1", "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["verdict"]["mean_diff"]["A-middle"] == pytest.approx(-0.02)
    assert rep["verdict"]["repeats"] == {"A-middle": True, "B-uniform": True} and not rep["verdict"]["adopt"]
    rows = {x["activity"]: x for x in rep["per_activity"]["A-middle"]}
    assert rows["x"]["agree"] and not rows["y"]["agree"]
    assert "mean" in capsys.readouterr().out

    bad = tmp_path / "lora_it_s1" / "A-middle" / "done.json"
    bad.write_text(bad.read_text().replace('"seed": 1', '"seed": 0'))
    with pytest.raises(ValueError, match="seed"):
        e8.main(["--root", str(tmp_path), "--seeds", "0,1"])
