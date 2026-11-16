"""Copy the SigLIP 2 snapshot out of the Hugging Face cache into a plain-file cache for Docker.

The laptop's cache stores the weights as symbolic links, two levels deep (Xet layout), which a Linux
container cannot follow through a bind mount. This script writes real files in the layout Hugging Face
reads offline (`hub/models--<org>--<name>/refs/main` + `snapshots/<commit>/`), under `data/hf-cache`
(spec-docker §4.4). Nothing is downloaded; every file is checked against its blob name (sha256 for
LFS files, git blob sha1 for the small ones), so a corrupt or truncated copy cannot pass.

    python scripts/prepare_docker_models.py            # google/siglip2-base-patch16-224 -> data/hf-cache
    python scripts/prepare_docker_models.py --check    # only verify an existing copy

The commit hash it copies is the revision the container runs, i.e. the model is pinned there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

DEFAULT_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_OUT = "data/hf-cache"


def default_cache() -> str:
    """Where this machine's Hugging Face hub cache is (same rule as huggingface_hub)."""
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(home, "hub")


def repo_dirname(model: str) -> str:
    return "models--" + model.replace("/", "--")


def blob_hash(path: str, name: str) -> str:
    """The hash the cache names this blob by: sha256 (64 hex) for LFS files, git blob sha1 (40 hex) otherwise."""
    if len(name) == 64:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    h = hashlib.sha1()
    h.update(f"blob {os.path.getsize(path)}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def blob_name(entry: str, blobs_dir: str) -> str:
    """The cache's blob name for a snapshot entry: the basename of its link target or, for a plain file (a
    cache written without symlink permission copies the files), the blob in `blobs_dir` with its hash."""
    if os.path.islink(entry):
        return os.path.basename(os.readlink(entry))
    for kind in (64, 40):
        h = blob_hash(entry, "x" * kind)
        if os.path.isfile(os.path.join(blobs_dir, h)):
            return h
    raise ValueError(f"{os.path.basename(entry)}: not a link, and no blob with its hash in {blobs_dir}")


def snapshot_commit(repo: str) -> str:
    ref = os.path.join(repo, "refs", "main")
    if not os.path.isfile(ref):
        raise FileNotFoundError(f"no refs/main in {repo}: the model is not in this cache")
    with open(ref, encoding="utf-8") as f:
        return f.read().strip()


def prepare(cache: str, model: str, out: str, check_only: bool = False) -> dict:
    """Copy (or verify) one model. Returns {model, commit, files: [{name, blob, bytes, ok}], ok}."""
    src_repo = os.path.join(cache, repo_dirname(model))
    commit = snapshot_commit(src_repo)
    src_snap = os.path.join(src_repo, "snapshots", commit)
    dst_repo = os.path.join(out, "hub", repo_dirname(model))
    dst_snap = os.path.join(dst_repo, "snapshots", commit)
    files, ok = [], True
    for name in sorted(os.listdir(src_snap)):
        src = os.path.join(src_snap, name)
        if os.path.isdir(src):
            raise NotImplementedError(f"nested snapshot folder {name!r} (not expected for {model})")
        blob = blob_name(src, os.path.join(src_repo, "blobs"))
        real = os.path.realpath(src)
        if not (len(blob) in (40, 64) and all(c in "0123456789abcdef" for c in blob)):
            raise ValueError(f"{name}: cache entry is not named by a hash ({blob})")
        dst = os.path.join(dst_snap, name)
        good = os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(real) and blob_hash(dst, blob) == blob
        if not good and not check_only:
            os.makedirs(dst_snap, exist_ok=True)
            tmp = dst + ".part"
            shutil.copyfile(real, tmp)
            os.replace(tmp, dst)
            good = blob_hash(dst, blob) == blob
            if not good:
                os.remove(dst)
        ok &= good
        files.append({"name": name, "blob": blob, "bytes": os.path.getsize(real), "ok": good})
    if not check_only and ok:
        os.makedirs(os.path.join(dst_repo, "refs"), exist_ok=True)
        with open(os.path.join(dst_repo, "refs", "main"), "w", encoding="utf-8") as f:
            f.write(commit)
    if check_only:
        ref = os.path.join(dst_repo, "refs", "main")
        ok &= os.path.isfile(ref) and open(ref, encoding="utf-8").read().strip() == commit
    return {"model": model, "commit": commit, "source": src_snap, "target": dst_snap, "files": files, "ok": bool(ok)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cache", default=default_cache(), help="this machine's Hugging Face hub cache")
    ap.add_argument("--out", default=DEFAULT_OUT, help="plain-file cache to mount as /hf-cache")
    ap.add_argument("--check", action="store_true", help="verify an existing copy; copy nothing")
    a = ap.parse_args(argv)
    r = prepare(a.cache, a.model, a.out, check_only=a.check)
    total = sum(f["bytes"] for f in r["files"])
    for f in r["files"]:
        print(f"{'ok ' if f['ok'] else 'BAD'} {f['name']:28s} {f['bytes']:>13,d}  {f['blob'][:12]}")
    print(f"{r['model']} @ {r['commit'][:12]}: {len(r['files'])} files, {total / 1e9:.2f} GB -> {r['target']}: "
          f"{'ok' if r['ok'] else 'FAILED'}")
    if r["ok"] and not a.check:
        with open(os.path.join(a.out, "prepared.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1)
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
