"""Docker support (spec-docker): the server's --host option, the SigLIP 2 cache copy, and the parity rule."""
import hashlib
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


prep = _load("prepare_docker_models")
parity = _load("docker_parity")
diag = _load("docker_platform_diag")


def test_the_server_parser_has_host_and_keeps_its_defaults():
    from sfs.web.server import build_parser
    a = build_parser().parse_args([])
    assert (a.host, a.port, a.alerts, a.bundle) == ("127.0.0.1", 8000, "data/alerts", "data/demo/A")
    b = build_parser().parse_args(["--host", "0.0.0.0", "--alerts", "/alerts", "--no-vlm"])
    assert b.host == "0.0.0.0" and b.alerts == "/alerts" and b.no_vlm


# --------------------------------------------------------------------------- prepare_docker_models
def _git_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _fake_cache(root, model="org/m", commit="abc123", use_symlinks=False):
    """A Hugging Face hub cache with one model: a small file (git sha1 blob) and an LFS file (sha256 blob)."""
    repo = os.path.join(root, prep.repo_dirname(model))
    blobs, snap = os.path.join(repo, "blobs"), os.path.join(repo, "snapshots", commit)
    for d in (blobs, snap, os.path.join(repo, "refs")):
        os.makedirs(d)
    files = {"config.json": b'{"a": 1}', "model.safetensors": bytes(range(256)) * 20}
    names = {"config.json": _git_sha1(files["config.json"]),
             "model.safetensors": hashlib.sha256(files["model.safetensors"]).hexdigest()}
    for n, data in files.items():
        with open(os.path.join(blobs, names[n]), "wb") as f:
            f.write(data)
        target = os.path.join(snap, n)
        if use_symlinks:
            os.symlink(os.path.join("..", "..", "blobs", names[n]), target)
        else:
            with open(target, "wb") as f:
                f.write(data)
    with open(os.path.join(repo, "refs", "main"), "w") as f:
        f.write(commit)
    return files, names


def test_prepare_copies_real_files_checks_hashes_and_pins_the_commit(tmp_path):
    cache, out = str(tmp_path / "hub"), str(tmp_path / "out")
    files, names = _fake_cache(cache)
    r = prep.prepare(cache, "org/m", out)
    assert r["ok"] and r["commit"] == "abc123"
    assert [(f["name"], f["blob"]) for f in r["files"]] == [("config.json", names["config.json"]),
                                                            ("model.safetensors", names["model.safetensors"])]
    snap = os.path.join(out, "hub", "models--org--m", "snapshots", "abc123")
    for n, data in files.items():
        p = os.path.join(snap, n)
        assert not os.path.islink(p) and open(p, "rb").read() == data
    assert open(os.path.join(out, "hub", "models--org--m", "refs", "main")).read() == "abc123"
    assert prep.prepare(cache, "org/m", out, check_only=True)["ok"]
    # a corrupted copy is found by --check and repaired by a plain run
    with open(os.path.join(snap, "model.safetensors"), "r+b") as f:
        f.seek(10)
        f.write(b"\0\0\0\0")
    bad = prep.prepare(cache, "org/m", out, check_only=True)
    assert not bad["ok"] and [f["ok"] for f in bad["files"]] == [True, False]
    assert prep.prepare(cache, "org/m", out)["ok"]
    assert open(os.path.join(snap, "model.safetensors"), "rb").read() == files["model.safetensors"]


def test_prepare_follows_symlinked_snapshots_like_the_real_cache(tmp_path):
    cache, out = str(tmp_path / "hub"), str(tmp_path / "out")
    try:
        files, names = _fake_cache(cache, use_symlinks=True)
    except OSError:
        pytest.skip("symbolic links are not permitted here")
    r = prep.prepare(cache, "org/m", out)
    assert r["ok"] and [f["blob"] for f in r["files"]] == [names["config.json"], names["model.safetensors"]]
    snap = os.path.join(out, "hub", "models--org--m", "snapshots", "abc123")
    assert not os.path.islink(os.path.join(snap, "model.safetensors"))


def test_prepare_refuses_a_model_that_is_not_cached(tmp_path):
    with pytest.raises(FileNotFoundError):
        prep.prepare(str(tmp_path), "org/missing", str(tmp_path / "out"))


# --------------------------------------------------------------------------- docker_parity
def _run(entries):
    return {"queries": [dict(e) for e in entries]}


def _q(query, level, ids, scores, match, k=10, seconds=0.5):
    return {"query": query, "level": level, "k": k, "ids": list(ids), "scores": list(scores), "match": list(match),
            "vlm_prob": [None] * len(ids), "seconds": seconds, "vlm_calls": k if level == "verified" else 0}


def test_record_asks_every_phrasing_at_fast_and_standard_and_a_few_at_verified():
    seen = []

    def fetch(body):
        seen.append((body["query"], body["level"], body["collection"], body["top"]))
        return {"results": [{"id": i, "scores": {"final": 1.0 - i / 10}, "match": i == 0} for i in range(3)],
                "cost": {"total": 0.25, "vlm_calls": body["k"] if body["level"] == "verified" else 0}}

    phr = {"b": ["b1", "b2"], "a": ["a1"]}
    out = parity.record(fetch, phr, vlm=True, verified_n=2)
    assert [(q["query"], q["level"]) for q in out] == [
        ("a1", "fast"), ("b1", "fast"), ("b2", "fast"), ("a1", "standard"), ("b1", "standard"), ("b2", "standard"),
        ("a1", "verified"), ("b1", "verified")]
    assert out[0]["ids"] == [0, 1, 2] and out[0]["scores"] == [1.0, 0.9, 0.8] and out[0]["match"] == [True, False, False]
    assert all(s[2] == "meva" and s[3] == parity.TOP for s in seen) and out[-1]["vlm_calls"] == 10
    assert len(parity.record(fetch, phr, vlm=False)) == 6          # no Verified without the VLM


def test_compare_accepts_identical_runs_and_near_tie_swaps_only():
    ref = _run([_q("x", "fast", [1, 2, 3, 4], [0.9, 0.5, 0.4995, 0.1], [True, False, False, False]),
                _q("x", "verified", [1, 2, 3], [0.9, 0.5, 0.4], [True, False, False], k=3)])
    same = parity.compare(ref, ref)
    assert same["ok"] and same["near_tie_swaps"] == 0 and same["verified_checked"] == 1
    swap = _run([_q("x", "fast", [1, 3, 2, 4], [0.9, 0.4999, 0.5002, 0.1], [True, False, False, False], seconds=0.7),
                 _q("x", "verified", [1, 3, 2], [0.9, 0.4, 0.5], [True, False, False], k=3)])
    r = parity.compare(ref, swap)
    assert r["ok"] and r["near_tie_swaps"] == 2 and r["max_score_diff"] < 1e-3
    assert r["median_seconds"]["fast"] == {"ref": 0.5, "other": 0.7}


def test_compare_rejects_real_differences():
    ref = _run([_q("x", "standard", [1, 2, 3], [0.9, 0.5, 0.1], [True, False, False])])
    r = parity.compare(ref, _run([_q("x", "standard", [2, 1, 3], [0.9, 0.5, 0.1], [False, True, False])]))
    assert not r["ok"] and any("Success@1" in p for p in r["problems"]) and any("not a near-tie" in p for p in r["problems"])
    r = parity.compare(ref, _run([_q("x", "standard", [1, 2, 3], [0.9, 0.503, 0.1], [True, False, False])]))
    assert not r["ok"] and r["problems"] == [p for p in r["problems"] if "differs by" in p]
    ref_v = _run([_q("x", "verified", [1, 2, 3], [0.9, 0.5, 0.1], [True, False, False], k=3)])
    r = parity.compare(ref_v, _run([_q("x", "verified", [1, 2, 9], [0.9, 0.5, 0.1], [True, False, False], k=3)]))
    assert not r["ok"] and any("different set" in p for p in r["problems"])
    assert not parity.compare(ref, _run([]))["ok"]


# --------------------------------------------------------------------------- docker_platform_diag
def _diag_dir(path, frames, q, student_own, student_refq=None):
    import json

    import numpy as np
    os.makedirs(path)
    np.savez(os.path.join(path, "frames.npz"), **{str(i): f for i, f in frames.items()})
    np.save(os.path.join(path, "q.npy"), q)
    np.save(os.path.join(path, "student_own.npy"), student_own)
    if student_refq is not None:
        np.save(os.path.join(path, "student_refq.npy"), student_refq)
    with open(os.path.join(path, "env.json"), "w") as f:
        json.dump({"platform": os.path.basename(path)}, f)


def test_diag_compare_splits_the_difference_into_frames_text_tower_and_student_kernels(tmp_path):
    import numpy as np
    rng = np.random.default_rng(0)
    f0 = rng.integers(0, 256, (4, 6, 3), dtype=np.uint8)
    f1 = f0.copy()
    f1[0, 0] = np.clip(f1[0, 0].astype(int) + 3, 0, 255)          # one pixel, +3 on each channel
    q = rng.standard_normal((5, 8)).astype(np.float32)
    s = rng.standard_normal((20, 5)).astype(np.float32)
    # b: same frames except one pixel, q moved by 1e-4, student logits moved only through q (kernels exact)
    qb = q + 1e-4
    sb = s + 0.02
    _diag_dir(str(tmp_path / "a"), {0: f0, 1: f0}, q, s)
    _diag_dir(str(tmp_path / "b"), {0: f0, 1: f1}, qb, sb, student_refq=s.copy())
    r = diag.compare(str(tmp_path / "a"), str(tmp_path / "b"))
    assert r["frames"]["0"]["identical"] and not r["frames"]["1"]["identical"]
    assert r["frames"]["1"]["max"] == 3 and r["frames"]["1"]["pixels_differing"] == pytest.approx(3 / 72)
    assert r["q"]["max"] == pytest.approx(1e-4, rel=1e-2) and r["q"]["min_cosine"] > 0.999
    assert r["student_kernels_only"]["max"] == 0.0
    assert r["student_from_q_only"]["max"] == pytest.approx(0.02, rel=1e-3)
    assert r["student_total"]["max"] == pytest.approx(0.02, rel=1e-3)
    assert r["env"]["a"]["platform"] == "a" and r["env"]["b"]["platform"] == "b"
    # kernels that differ show up in their own line and nowhere else
    np.save(str(tmp_path / "b" / "student_refq.npy"), s + 0.005)
    r = diag.compare(str(tmp_path / "a"), str(tmp_path / "b"))
    assert r["student_kernels_only"]["max"] == pytest.approx(0.005, rel=1e-3)
    assert r["student_from_q_only"]["max"] == pytest.approx(0.015, rel=1e-3)


def test_diag_record_frames_reads_the_requested_frames_in_order(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    path = str(tmp_path / "v.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 24))
    for i in range(6):
        w.write(np.full((24, 32, 3), i * 40, np.uint8))
    w.release()
    r = diag.record_frames(path, [0, 3, 5])
    assert sorted(r["frames"]) == [0, 3, 5] and r["info"]["read"] == 6 and r["info"]["shape"] == [24, 32, 3]
    assert r["frames"][0].mean() < r["frames"][3].mean() < r["frames"][5].mean()


# --------------------------------------------------------------------------- Dockerfile
def _copied_into_image(dockerfile_text):
    """Source paths the Dockerfile's COPY lines take from the build context (--from lines copy build output)."""
    out = set()
    for line in dockerfile_text.splitlines():
        parts = line.split()
        if parts and parts[0] == "COPY" and not any(p.startswith("--from") for p in parts):
            out.update(p.rstrip("/") for p in parts[1:-1] if not p.startswith("--"))
    return out


def _requirement_includes(path):
    """Files a pip requirements file pulls in with -c/--constraint or -r/--requirement, resolved next to it."""
    here = os.path.relpath(os.path.dirname(path), ROOT)
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) > 1 and parts[0] in ("-c", "--constraint", "-r", "--requirement"):
                out.append(os.path.normpath(os.path.join(here, parts[1])).replace(os.sep, "/"))
    return out


def _missing_from_image(dockerfile_text):
    """Requirement includes under docker/ that no COPY line puts into the image (the first build failed on one)."""
    copied = _copied_into_image(dockerfile_text)

    def in_image(rel):
        return any(rel == c or rel.startswith(c + "/") for c in copied)

    problems = []
    for name in sorted(os.listdir(os.path.join(ROOT, "docker"))):
        if name.endswith(".txt"):
            for inc in _requirement_includes(os.path.join(ROOT, "docker", name)):
                if not os.path.exists(os.path.join(ROOT, inc)):
                    problems.append(f"docker/{name} includes {inc}, which does not exist")
                elif not in_image(inc):
                    problems.append(f"docker/{name} includes {inc}, which no COPY line puts into the image")
    for rel in ("docker/requirements.txt", "docker/requirements-test.txt"):
        if not in_image(rel):
            problems.append(f"{rel} is never copied")
    return problems


def test_dockerfile_copies_every_file_the_requirement_files_include():
    with open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8") as f:
        text = f.read()
    assert _missing_from_image(text) == []
    # the Dockerfile as first written: requirements.txt copied alone, so the `-c constraints.txt` of both
    # requirement files pointed at nothing (the first build failed there, 2026-11-08)
    old = text.replace("COPY docker/requirements.txt docker/constraints.txt docker/",
                       "COPY docker/requirements.txt docker/requirements.txt")
    assert old != text
    assert _missing_from_image(old) == [
        "docker/requirements-test.txt includes docker/constraints.txt, which no COPY line puts into the image",
        "docker/requirements.txt includes docker/constraints.txt, which no COPY line puts into the image"]
