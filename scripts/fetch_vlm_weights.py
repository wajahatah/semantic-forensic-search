"""Download VLM weights from the official ModelScope mirror, resumably, with sha256 checks
(spec-rerank-cascade D1, O1).

Why ModelScope: Hugging Face rate-limits this connection (HTTP 429, 2026-10-31); ModelScope is
Qwen's own mirror with the same files and publishes a sha256 per file. Each model is pinned to
the repository revision listed at download time, recorded in <out>/<name>/_source.json.

Usage (repo root):  python scripts/fetch_vlm_weights.py Qwen/Qwen3-VL-2B-Instruct Qwen/Qwen3-VL-Reranker-2B
Re-running resumes partial files and skips verified ones. Nothing downloaded is ever executed.
"""
import hashlib
import json
import os
import subprocess
import sys
import urllib.request

API = "https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Recursive=true"
FILE = "https://www.modelscope.cn/models/{repo}/resolve/{rev}/{path}"
OUT = os.path.join("data", "models")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def listing(repo: str) -> list[dict]:
    with urllib.request.urlopen(API.format(repo=repo), timeout=60) as r:
        files = json.load(r)["Data"]["Files"]
    return [f for f in files if f["Type"] == "blob"]


def fetch(repo: str) -> str:
    name = repo.split("/")[-1]
    root = os.path.join(OUT, name)
    os.makedirs(root, exist_ok=True)
    src_path = os.path.join(root, "_source.json")
    files = listing(repo)
    head = max(files, key=lambda f: f["CommittedDate"])["Revision"]
    if os.path.exists(src_path):  # the revision is pinned at first download
        src = json.load(open(src_path))
        if src["revision"] != head:
            raise RuntimeError(f"{repo} changed upstream ({src['revision']} -> {head}); not mixing revisions")
    else:
        src = {"repo": repo, "mirror": "modelscope", "revision": head, "files": {}}
    for f in files:
        dest = os.path.join(root, *f["Path"].split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if src["files"].get(f["Path"]) == f["Sha256"] and os.path.exists(dest):
            continue
        if not (os.path.exists(dest) and os.path.getsize(dest) == f["Size"]):
            url = FILE.format(repo=repo, rev="master", path=f["Path"])
            print(f"{name}: {f['Path']} ({f['Size'] / 1e6:.1f} MB)", flush=True)
            subprocess.run(["curl", "-sSL", "--fail", "--retry", "20", "--retry-all-errors",
                            "--retry-delay", "10", "-C", "-", "-o", dest, url], check=True)
        got = sha256(dest)
        if got != f["Sha256"]:
            os.remove(dest)
            raise RuntimeError(f"{dest}: sha256 mismatch (removed; re-run to fetch again)")
        src["files"][f["Path"]] = got
        json.dump(src, open(src_path, "w"), indent=1)
    print(f"{name}: complete, revision {src['revision']}", flush=True)
    return root


if __name__ == "__main__":
    for repo in sys.argv[1:]:
        fetch(repo)
