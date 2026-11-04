"""One-process job queue (D-041): jobs run in order, each model is released before the next loads,
the GPU memory is reserved once, and a failing job is recorded without stopping the queue."""
import weakref

import pytest

from sfs.teacher.queue import run_queue, validate


class _Scorer:
    pass


def test_queue_runs_in_order_releases_models_and_continues_after_a_failure():
    events, refs = [], []
    jobs = [{"type": "g0", "model": "m/a", "kind": "instruct", "out": "o1"},
            {"type": "score", "model": "m/b", "kind": "instruct", "view": "tight", "requests": "r", "subset": "val", "out": "o2"},
            {"type": "score", "model": "m/c", "kind": "instruct", "view": "tight", "requests": "r", "subset": "val", "out": "o3"}]

    def load(job):
        assert all(r() is None for r in refs), "previous model still alive when the next one loads"
        s = _Scorer()
        refs.append(weakref.ref(s))
        events.append(("load", job["model"]))
        return s

    def do_score(scorer, job):
        events.append(("score", job["out"]))
        if job["out"] == "o2":
            raise RuntimeError("CUDA out of memory")

    status = run_queue(jobs, load, lambda s, j: events.append(("g0", j["out"])), do_score,
                       reserve=lambda: events.append(("reserve",)) or 14.5, log=lambda m: None)
    assert events[0] == ("reserve",) and events.count(("reserve",)) == 1
    assert [e for e in events if e[0] != "reserve"] == [("load", "m/a"), ("g0", "o1"), ("load", "m/b"), ("score", "o2"),
                                                       ("load", "m/c"), ("score", "o3")]
    assert [s["ok"] for s in status] == [True, False, True] and "out of memory" in status[1]["error"]


def test_bad_jobs_are_refused_before_anything_runs():
    with pytest.raises(ValueError, match="type must be"):
        validate([{"type": "train"}])
    with pytest.raises(ValueError, match="missing"):
        validate([{"type": "score", "model": "m", "kind": "instruct", "out": "o"}])


def test_watch_runs_dropped_files_in_order_and_stops_on_stop(tmp_path):
    import json
    from sfs.teacher.queue import watch
    (tmp_path / "02_b.json").write_text(json.dumps([{"job": "b"}]))
    (tmp_path / "01_a.json").write_text(json.dumps([{"job": "a"}]))
    (tmp_path / "03_bad.json").write_text("not json")
    seen, polls = [], []

    def run_jobs(jobs):
        seen.append(jobs[0]["job"])
        return [{"job": jobs[0]["job"], "ok": jobs[0]["job"] != "b"}]

    def sleep(s):                                    # idle: nothing left, so ask the holder to stop
        polls.append(s)
        (tmp_path / "STOP").write_text("")

    assert watch(str(tmp_path), run_jobs, sleep=sleep, log=lambda m: None) == 3
    assert seen == ["a", "b"] and polls == [30.0]
    assert (tmp_path / "01_a.done").exists() and (tmp_path / "02_b.failed").exists()
    assert (tmp_path / "03_bad.failed").exists() and "JSONDecodeError" in (tmp_path / "03_bad.status.json").read_text()
