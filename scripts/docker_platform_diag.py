"""Where do the laptop (Windows) and the container (Linux) disagree? (docs/docker.md, parity checks of 2026-11-08)

The parity run gave identical rankings but Standard scores apart by up to 5e-3, and the same video gave
21 alerts in the container against 22 on the laptop. This records, on one platform, the three things
that could differ and lets `compare` split the difference into its parts:

  frames   the decoded pixels of a few frames of the test video (OpenCV uses another video back-end per OS)
  q        the SigLIP 2 text embeddings of the frozen phrasings (float16 text tower on the GPU)
  student  the temporal student's logits for every MEVA clip, with this platform's own q and, when
           --ref points at the other platform's recording, with that platform's q (same input, so any
           difference there is the student's GPU kernels alone)

  python scripts/docker_platform_diag.py record --out runs/docker/diag/laptop
  python scripts/docker_platform_diag.py record --out /diag/docker --video /alerts/media/1/video.mp4 --ref /diag/laptop
  python scripts/docker_platform_diag.py compare runs/docker/diag/laptop runs/docker/diag/docker
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys

import numpy as np

FRAMES = (0, 1200, 1668, 2824)      # 0 s, 48.0 s, 66.7 s and 113.0 s of the garage clip: alerts that differed


def load_phrasings(path: str) -> list[str]:
    import pandas as pd
    df = pd.read_csv(path, comment="#")
    for c in df.columns:
        if c.lower() in ("phrasing", "text", "query", "phrase"):
            return [str(t) for t in df[c]]
    raise SystemExit(f"{path}: no phrasing column among {list(df.columns)}")


def record_frames(video: str, indices) -> dict:
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    want, out, n = set(int(i) for i in indices), {}, 0
    while n <= max(want):
        ok, frame = cap.read()
        if not ok:
            break
        if n in want:
            out[n] = frame.copy()
        n += 1
    info = {"backend": cap.getBackendName(), "fps": cap.get(cv2.CAP_PROP_FPS),
            "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "read": n,
            "shape": list(next(iter(out.values())).shape) if out else None,
            "means": {str(i): float(f.mean()) for i, f in sorted(out.items())}}
    cap.release()
    return {"frames": out, "info": info}


def record(args) -> None:
    import torch
    from sfs.demo.search import Bundle, Searcher
    from sfs.rerank.train import score_matrix
    os.makedirs(args.out, exist_ok=True)
    env = {"platform": platform.platform(), "python": sys.version.split()[0], "torch": torch.__version__,
           "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32}
    if args.video:
        import cv2
        env["cv2"] = cv2.__version__
        fr = record_frames(args.video, args.frames)
        np.savez(os.path.join(args.out, "frames.npz"), **{str(i): f for i, f in fr["frames"].items()})
        env["video"] = fr["info"]
        print("frames:", fr["info"])
    texts = load_phrasings(args.phrasings)
    s = Searcher(Bundle(args.bundle), device=args.device)
    q = np.stack([s.embed(t) for t in texts]).astype(np.float32)
    np.save(os.path.join(args.out, "q.npy"), q)
    with open(os.path.join(args.out, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=0)
    env["encoder"] = s.b.info["encoder"]
    print(f"q: {q.shape} from {env['encoder']} on {s.device}")
    own = score_matrix(s.student, s.b.feats, s.b.mask, s.rows, q, s.device)
    np.save(os.path.join(args.out, "student_own.npy"), own)
    print(f"student with own q: {own.shape}, mean {own.mean():.4f}, std {own.std():.4f}")
    if args.ref:
        ref_q = np.load(os.path.join(args.ref, "q.npy"))
        with open(os.path.join(args.ref, "texts.json"), encoding="utf-8") as f:
            if json.load(f) != texts:
                raise SystemExit("the reference recording used other phrasings")
        with_ref = score_matrix(s.student, s.b.feats, s.b.mask, s.rows, ref_q, s.device)
        np.save(os.path.join(args.out, "student_refq.npy"), with_ref)
        env["ref"] = args.ref
        print(f"student with the reference q: max |diff to own| {np.abs(with_ref - own).max():.2e}")
    with open(os.path.join(args.out, "env.json"), "w", encoding="utf-8") as f:
        json.dump(env, f, indent=1)
    print("env:", json.dumps(env))


def _stats(d: np.ndarray) -> dict:
    d = np.abs(np.asarray(d, np.float64))
    return {"max": float(d.max()), "median": float(np.median(d)), "mean": float(d.mean())}


def compare(a_dir: str, b_dir: str) -> dict:
    """a = reference (laptop), b = the other platform. Returns the decomposition; prints it too."""
    env = {}
    for k, d in (("a", a_dir), ("b", b_dir)):
        with open(os.path.join(d, "env.json"), encoding="utf-8") as f:
            env[k] = json.load(f)
    out = {"env": env}
    fa, fb = os.path.join(a_dir, "frames.npz"), os.path.join(b_dir, "frames.npz")
    if os.path.exists(fa) and os.path.exists(fb):
        A, B = np.load(fa), np.load(fb)
        frames = {}
        for k in sorted(A.files, key=int):
            if k in B.files and A[k].shape == B[k].shape:
                d = np.abs(A[k].astype(np.int16) - B[k].astype(np.int16))
                frames[k] = {"identical": bool(d.max() == 0), "max": int(d.max()), "mean": float(d.mean()),
                             "pixels_differing": float((d > 0).mean())}
            else:
                frames[k] = {"identical": False, "note": "missing or different shape"}
        out["frames"] = frames
    qa, qb = np.load(os.path.join(a_dir, "q.npy")), np.load(os.path.join(b_dir, "q.npy"))
    cos = (qa * qb).sum(1) / (np.linalg.norm(qa, axis=1) * np.linalg.norm(qb, axis=1))
    out["q"] = {**_stats(qa - qb), "min_cosine": float(cos.min()), "n": int(len(qa))}
    sa, sb = np.load(os.path.join(a_dir, "student_own.npy")), np.load(os.path.join(b_dir, "student_own.npy"))
    out["student_total"] = _stats(sa - sb)                        # each platform end to end
    out["student_std_over_clips"] = float(sa.std(axis=0).mean())  # the z-scoring divides by about this
    rq = os.path.join(b_dir, "student_refq.npy")
    if os.path.exists(rq):
        sr = np.load(rq)
        out["student_kernels_only"] = _stats(sa - sr)            # same q as a: the student's own arithmetic
        out["student_from_q_only"] = _stats(sb - sr)             # same platform: the text tower's part
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    r.add_argument("--bundle", default="data/demo/A")
    r.add_argument("--phrasings", default="queries/phrasings.csv")
    r.add_argument("--video", default="data/alerts/media/1/video.mp4", help="'' to skip the frames")
    r.add_argument("--frames", type=lambda s: [int(x) for x in s.split(",")], default=list(FRAMES))
    r.add_argument("--ref", help="the other platform's recording: also score the student with its q")
    r.add_argument("--device", default=None)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out")
    a = p.parse_args(argv)
    if a.cmd == "record":
        record(a)
        return 0
    res = compare(a.a, a.b)
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
