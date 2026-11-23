"""Hand-in folder builder (spec-submission §1): moves leave working links, nothing is lost or duplicated."""
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("build_submission", os.path.join(ROOT, "scripts", "build_submission.py"))
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)


def _write(path, text="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


@pytest.fixture
def root(tmp_path):
    """A tiny project with every folder the builder moves or copies."""
    r = str(tmp_path / "project")
    for old, _ in bs.MOVES:
        _write(os.path.join(r, old, "file.bin"), old)
    for named in ("data/hf-cache/hub/model.bin", "runs/lora/A-middle/best/adapter.pt", "data/models/yolo/yolov8m.pt"):
        _write(os.path.join(r, named))
    for name in bs.INDEX_FILES:
        _write(os.path.join(r, bs.INDEX_OLD, name), name)
    _write(os.path.join(r, "src/sfs/web/server.py"), "# server")
    _write(os.path.join(r, "src/sfs/__pycache__/junk.pyc"), "junk")
    _write(os.path.join(r, "web/dist/index.html"), "<html>")
    _write(os.path.join(r, "queries/phrasings.csv"), "activity,text\n")
    _write(os.path.join(r, "docker/requirements.txt"),
           "--extra-index-url https://download.pytorch.org/whl/cu126\n-c constraints.txt\n\ntorch==2.14.1+cu126\nnumpy==2.5.2\n")
    _write(os.path.join(r, "docker/constraints.txt"), "numpy==2.5.2\ntorch==2.14.1+cu126\n")
    _write(os.path.join(r, bs.TEST_VIDEO[0]), "video")
    return r


def test_move_leaves_a_link_that_reads_the_moved_data(root, tmp_path):
    out = os.path.join(root, "submission")
    msgs = []
    bs.move(root, out, msgs.append, with_meva=True)
    for old, new in bs.MOVES:
        assert bs.is_junction(os.path.join(root, old)) and not bs.is_junction(os.path.join(out, new))
        with open(os.path.join(root, old, "file.bin"), encoding="utf-8") as f:      # the old path still works
            assert f.read() == old
        assert os.path.samefile(os.path.join(root, old, "file.bin"), os.path.join(out, new, "file.bin"))
    for name in bs.INDEX_FILES:                                                     # one file, two names
        assert os.path.samefile(os.path.join(root, bs.INDEX_OLD, name), os.path.join(out, bs.INDEX_NEW, name))
    assert not os.path.exists(os.path.join(out, bs.INDEX_NEW, "previews"))          # previews live under meva/
    assert os.path.isfile(os.path.join(root, bs.INDEX_OLD, "previews", "file.bin"))

    again = []
    bs.move(root, out, again.append, with_meva=True)                                                # repeatable: nothing moves twice
    assert all("already moved" in m for m in again) and len(again) == len(bs.MOVES) + len(bs.INDEX_FILES)


def test_move_never_overwrites_and_reports_missing_data(tmp_path):
    old, new = str(tmp_path / "old"), str(tmp_path / "sub" / "new")
    _write(os.path.join(old, "a.txt"), "mine")
    _write(os.path.join(new, "b.txt"), "theirs")
    with pytest.raises(FileExistsError):
        bs.move_with_link(old, new, lambda m: None)
    assert os.path.isfile(os.path.join(old, "a.txt")) and not bs.is_junction(old)   # untouched
    with pytest.raises(FileNotFoundError):
        bs.move_with_link(str(tmp_path / "nothing"), str(tmp_path / "x"), lambda m: None)


def test_a_failed_link_puts_the_folder_back(tmp_path, monkeypatch):
    old, new = str(tmp_path / "old"), str(tmp_path / "sub" / "new")
    _write(os.path.join(old, "a.txt"), "mine")

    def boom(link, target):
        raise OSError("no links here")
    monkeypatch.setattr(bs, "make_junction", boom)
    with pytest.raises(OSError):
        bs.move_with_link(old, new, lambda m: None)
    assert os.path.isfile(os.path.join(old, "a.txt")) and not os.path.exists(new)


def test_assemble_copies_code_and_derives_the_cpu_requirements(root):
    out = os.path.join(root, "submission")
    bs.assemble(root, out, lambda m: None)
    bs.assemble(root, out, lambda m: None)                                          # repeatable
    code = os.path.join(out, "code")
    assert os.path.isfile(os.path.join(code, "sfs", "web", "server.py"))
    assert not os.path.exists(os.path.join(code, "sfs", "__pycache__"))
    with open(os.path.join(code, "run.bat"), "rb") as f:
        assert b"\r\n" in f.read()                                                  # cmd.exe needs CRLF
    with open(os.path.join(code, "run.sh"), "rb") as f:
        assert b"\r" not in f.read()
    with open(os.path.join(code, "requirements-cpu.txt"), encoding="utf-8") as f:
        cpu = f.read()
    assert "torch==2.14.1+cpu" in cpu and "cu126" not in cpu and "numpy==2.5.2" in cpu and "whl/cpu" in cpu
    with open(os.path.join(code, "docker", "constraints-cpu.txt"), encoding="utf-8") as f:
        assert f.read() == "numpy==2.5.2\ntorch==2.14.1+cpu\n"
    assert os.path.isfile(os.path.join(out, bs.TEST_VIDEO[1]))
    assert all(os.path.isdir(os.path.join(out, d)) for d in bs.EMPTY_DIRS)

    bs.move(root, out, lambda m: None)                                              # default: no MEVA media
    for old, _ in bs.MEVA_MOVES:
        assert not bs.is_junction(os.path.join(root, old))                          # left where they were
    assert not os.path.exists(os.path.join(out, "meva"))
    r = bs.check(out, lambda m: None)
    assert [m for m in r["missing"] if m != "README.md"] == []          # the guide is written by hand, checked below
    assert set(r["parts"]) == {new for _, new in bs.MODEL_MOVES} | {bs.INDEX_NEW}
    assert set(bs.check(out, lambda m: None, with_meva=True)["missing"]) >= set(bs.EXPECTED_MEVA)


def test_run_scripts_and_docker_files_name_the_folders_the_builder_fills():
    """The paths in run.bat / run.sh / Dockerfile must be the ones `move` creates."""
    targets = {new for _, new in bs.MOVES} | {bs.INDEX_NEW}
    for name, sep, prefix in (("run.bat", "\\", "%ROOT%\\"), ("run.sh", "/", "$ROOT/"), ("Dockerfile", "/", "/")):
        with open(os.path.join(bs.TEMPLATES, name), encoding="utf-8") as f:
            text = f.read()
        for t in targets - {"meva/example_videos"}:
            path = prefix + t.replace("/", sep)
            assert path in text, f"{name} does not use {path}"
        for flag in ("--bundle", "--previews", "--crops", "--weights", "--adapter", "--vlm", "--alerts"):
            assert flag in text, f"{name} lacks {flag}"
    with open(os.path.join(bs.TEMPLATES, "compose.yaml"), encoding="utf-8") as f:
        compose = f.read()
    assert "../models:/models:ro" in compose and "../meva:/meva:ro" in compose and "127.0.0.1:8000:8000" in compose


def test_cpu_requirements_refuse_an_unexpected_file():
    with pytest.raises(ValueError):
        bs.cpu_requirements("torch==2.14.1+cu126\n", "torch==2.14.1+cu126\n")
    with pytest.raises(ValueError):
        bs.cpu_requirements("-c constraints.txt\nnumpy==1\n", "numpy==1\n")
