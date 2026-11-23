"""Hand-in folder for the professor (docs/planning/spec-submission.md §1).

    python scripts/build_submission.py assemble     # code, run scripts, Docker files, guide (copies; repeatable)
    python scripts/build_submission.py move         # models moved in, links left at the old paths
    python scripts/build_submission.py move --with-meva   # also the MEVA media (not handed in: the professor
                                                          # downloads MEVA from its site; 2026-11-23)
    python scripts/build_submission.py check        # every expected file is there; sizes for the guide

`move` renames folders on the same drive (nothing is copied or stored twice) and leaves a directory
junction at each old path, so the laptop's dashboard, tests and Docker setup keep reading `data/...` and
`runs/...`. The MEVA index is split: its four core files go to models/index (always needed: the temporal
re-ranker, and the reference scores alerts are standardised with), its previews to meva/previews; the old
folder data/demo/A keeps hard links to the four files and a junction to the previews.
Nothing is ever deleted or overwritten: a step whose target already exists is reported and skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = os.path.join(ROOT, "packaging", "submission")

# old path (kept as a junction) -> new path inside submission/
MODEL_MOVES = (
    ("data/hf-cache", "models/siglip2"),
    ("runs/lora/A-middle/best", "models/lora_adapter"),
    ("data/models/yolo", "models/yolo"),
    ("data/models/Qwen3-VL-2B-Instruct", "models/qwen3-vl-2b"),
)
MEVA_MOVES = (                      # MEVA media: only with --with-meva (the dataset itself is not handed in)
    ("data/crops_v2", "meva/crops"),
    ("data/meva/examples/videos", "meva/example_videos"),
    ("data/demo/A/previews", "meva/previews"),
)
MOVES = MODEL_MOVES + MEVA_MOVES
MEVA_URL = "https://mevadata.org"
INDEX_OLD, INDEX_NEW = "data/demo/A", "models/index"
INDEX_FILES = ("bundle.json", "meta.sqlite", "student.pt", "vectors.npz")
TEST_VIDEO = ("data/alerts/media/1/video.mp4", "test_videos/garage_cctv.mp4")
EMPTY_DIRS = ("test_videos", "report", "presentation", "demo_video")
CODE_COPIES = (                     # source in the repo -> path under submission/code
    ("src/sfs", "sfs"),
    ("web/dist", "web/dist"),
    ("queries/phrasings.csv", "queries/phrasings.csv"),
    ("docker/requirements.txt", "docker/requirements.txt"),
    ("docker/constraints.txt", "docker/constraints.txt"),
)
TEMPLATE_FILES = ("run.bat", "run.sh", "requirements.txt", "Dockerfile", "compose.yaml")
EXPECTED = (
    "README.md", "code/run.bat", "code/run.sh", "code/requirements.txt", "code/requirements-cpu.txt",
    "code/docker/requirements.txt", "code/docker/constraints.txt", "code/docker/constraints-cpu.txt",
    "code/Dockerfile", "code/compose.yaml", "code/sfs/web/server.py", "code/web/dist/index.html",
    "code/queries/phrasings.csv", "models/siglip2/hub", "models/lora_adapter/adapter.pt",
    "models/yolo/yolov8m.pt", "models/qwen3-vl-2b", "models/index/vectors.npz", "models/index/student.pt",
    "models/index/meta.sqlite", "models/index/bundle.json", "test_videos/garage_cctv.mp4",
)
EXPECTED_MEVA = ("meva/previews", "meva/crops", "meva/example_videos")
CPU_HEAD = (
    "# Python 3.12, no GPU (Windows or Linux). Same package versions as requirements.txt, with the CPU build of torch.\n"
    "#   pip install -r requirements-cpu.txt\n"
    "# Fast and Standard search and the Detect tab work; the Verified level is hidden (its VLM needs a GPU).\n"
    "--extra-index-url https://download.pytorch.org/whl/cpu\n"
    "-c docker/constraints-cpu.txt\n"
)


def cpu_requirements(gpu_requirements: str, gpu_constraints: str) -> tuple[str, str]:
    """The CPU requirement and constraint files, derived from the tested GPU ones (same versions, torch +cpu)."""
    marker = "-c constraints.txt"
    if marker not in gpu_requirements:
        raise ValueError("docker/requirements.txt no longer names constraints.txt")
    swap = lambda s: s.replace("+cu126", "+cpu")          # noqa: E731
    body = swap(gpu_requirements.split(marker, 1)[1])
    cons = swap(gpu_constraints)
    for text in (body, cons):
        if "cu126" in text or "torch==" not in text:
            raise ValueError("torch pins not found or not converted")
    return CPU_HEAD + body, cons


def is_junction(path: str) -> bool:
    return os.path.isdir(path) and (os.path.islink(path) or getattr(os.path, "isjunction", lambda p: False)(path))


def make_junction(link: str, target: str) -> None:
    """Directory junction (Windows, no admin rights); a relative symlink elsewhere."""
    if os.name == "nt":
        r = subprocess.run(["cmd", "/c", "mklink", "/J", os.path.normpath(link), os.path.normpath(target)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise OSError(f"mklink /J {link} failed: {r.stdout.strip()} {r.stderr.strip()}")
    else:
        os.symlink(os.path.relpath(target, os.path.dirname(link)), link, target_is_directory=True)


def move_with_link(old: str, new: str, log=print) -> str:
    """Rename `old` to `new` and leave a junction at `old`. Returns what happened."""
    if is_junction(old):
        if not os.path.isdir(new):
            raise FileNotFoundError(f"{old} is a link but {new} is missing")
        log(f"  already moved   {old}")
        return "already"
    if not os.path.isdir(old):
        raise FileNotFoundError(f"{old} is not a folder")
    if os.path.exists(new):
        raise FileExistsError(f"{new} already exists; nothing moved")
    os.makedirs(os.path.dirname(new), exist_ok=True)
    os.rename(old, new)                                   # same drive: a rename, not a copy
    try:
        make_junction(old, new)
    except OSError:
        os.rename(new, old)                               # put it back; never leave the laptop without its data
        raise
    log(f"  moved           {old} -> {new}")
    return "moved"


def move_file_with_hardlink(old: str, new: str, log=print) -> str:
    """Move one file and leave a hard link at the old path (the same bytes on disk, not a copy)."""
    if os.path.exists(new):
        if not os.path.exists(old) or not os.path.samefile(old, new):
            raise FileExistsError(f"{new} exists and is not the same file as {old}")
        log(f"  already moved   {old}")
        return "already"
    os.makedirs(os.path.dirname(new), exist_ok=True)
    os.rename(old, new)
    try:
        os.link(new, old)
    except OSError:
        os.rename(new, old)
        raise
    log(f"  moved           {old} -> {new} (hard link left)")
    return "moved"


def assemble(root: str, out: str, log=print) -> None:
    code = os.path.join(out, "code")
    for src, dst in CODE_COPIES:
        s, d = os.path.join(root, src), os.path.join(code, dst)
        if not os.path.exists(s):
            raise FileNotFoundError(f"{src} is missing" + (" (run `npm run build` in web/)" if "dist" in src else ""))
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if os.path.isdir(s):
            if os.path.isdir(d):
                shutil.rmtree(d)                          # a copy this script made; rebuilt from the repo
            shutil.copytree(s, d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copyfile(s, d)
    for name in TEMPLATE_FILES:
        with open(os.path.join(TEMPLATES, name), encoding="utf-8") as f:
            text = f.read()
        with open(os.path.join(code, name), "w", encoding="utf-8", newline="\r\n" if name.endswith(".bat") else "\n") as f:
            f.write(text)
    with open(os.path.join(root, "docker", "requirements.txt"), encoding="utf-8") as f:
        req = f.read()
    with open(os.path.join(root, "docker", "constraints.txt"), encoding="utf-8") as f:
        con = f.read()
    cpu_req, cpu_con = cpu_requirements(req, con)
    for path, text in ((os.path.join(code, "requirements-cpu.txt"), cpu_req),
                       (os.path.join(code, "docker", "constraints-cpu.txt"), cpu_con)):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    guide = os.path.join(TEMPLATES, "README.md")
    if os.path.exists(guide):
        shutil.copyfile(guide, os.path.join(out, "README.md"))
    for d in EMPTY_DIRS:
        os.makedirs(os.path.join(out, d), exist_ok=True)
    s, d = os.path.join(root, TEST_VIDEO[0]), os.path.join(out, TEST_VIDEO[1])
    if os.path.exists(s) and not os.path.exists(d):
        shutil.copyfile(s, d)
    log(f"assembled {code}")


def move(root: str, out: str, log=print, with_meva: bool = False) -> None:
    # the previews first: afterwards data/demo/A holds only its four files and the previews junction
    moves = MOVES if with_meva else MODEL_MOVES
    for old, new in sorted(moves, key=lambda m: m[0] != "data/demo/A/previews"):
        move_with_link(os.path.join(root, old), os.path.join(out, new), log)
    for name in INDEX_FILES:
        move_file_with_hardlink(os.path.join(root, INDEX_OLD, name), os.path.join(out, INDEX_NEW, name), log)


def tree_size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for base, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(base, f)) for f in files)
    return total


def check(out: str, log=print, with_meva: bool = False) -> dict:
    expected = EXPECTED + (EXPECTED_MEVA if with_meva else ())
    missing = [p for p in expected if not os.path.exists(os.path.join(out, p))]
    sizes = {d: tree_size(os.path.join(out, d)) for d in sorted(os.listdir(out)) if os.path.isdir(os.path.join(out, d))}
    sub = {f"{top}/{d}": tree_size(os.path.join(out, top, d)) for top in ("models", "meva")
           if os.path.isdir(os.path.join(out, top)) for d in sorted(os.listdir(os.path.join(out, top)))}
    for k, v in {**sizes, **sub}.items():
        log(f"  {k:28s} {v / 1024 ** 3:7.2f} GB")
    log(f"  {'total':28s} {sum(sizes.values()) / 1024 ** 3:7.2f} GB")
    for p in missing:
        log(f"  MISSING {p}")
    return {"missing": missing, "sizes": sizes, "parts": sub}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("assemble", "move", "check"))
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=None, help="default <root>/submission")
    ap.add_argument("--with-meva", action="store_true", help="also move / expect the MEVA media")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.root, "submission")
    os.makedirs(out, exist_ok=True)
    if a.step == "assemble":
        assemble(a.root, out)
    elif a.step == "move":
        move(a.root, out, with_meva=a.with_meva)
    else:
        r = check(out, with_meva=a.with_meva)
        print(json.dumps({"missing": r["missing"]}))
        return 1 if r["missing"] else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
