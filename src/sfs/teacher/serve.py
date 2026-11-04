"""Scoring service for the office server (spec-vlm-followups E4, D-029): our own VLMScorer behind a
small HTTP API, so a bigger VLM on the office GPU scores exactly as the 2B models did on the laptop
(same frame preparation, prompt, margin and video-once shortcut).

- Binds to 127.0.0.1 by default; the laptop reaches it through an SSH tunnel
  (`ssh -N -L 8765:127.0.0.1:8765 lt_internal_server`). No port is opened on the server.
- Every request carries the shared secret in the `X-SFS-Secret` header. The secret is read from the
  environment variable SFS_SERVE_SECRET, or from the file named by SFS_SERVE_SECRET_FILE (default
  ~/.config/sfs/secret). The owner creates it; it is never logged or returned.
- Requests are size-limited (MAX_BODY_BYTES, MAX_FRAMES, MAX_QUERIES, MAX_QUERY_CHARS). The service
  only scores: after loading the model it reads and writes no files.
- Shared GPU: before loading, the service waits until at least `--min-free-gb` of GPU memory is free
  (the team's Ollama models unload when idle). A request that runs out of GPU memory returns 503 and
  the client retries; nothing is cached on the server, so a retry is a clean re-run.
- One request at a time (plain HTTPServer): GPU work is never interleaved.

Frames travel as PNG (lossless): the laptop decodes its crop store as usual, re-encodes each frame as
PNG, and the server decodes it to the identical uint8 array before `prepare_frames`.

API (JSON):
  GET  /info     -> model id, revision, kind, versions, GPU name, limits
  POST /score    {"mode": "clip"|"full", "frames": [base64 PNG], "frame_numbers": [int], "fps": float,
                  "queries": [str]} -> {"scores": [float], "seconds": float}
  POST /control  {"op": "reset_peak"|"peak"|"set_attn", "name": "sdpa"|"eager"} -> {"peak_memory_gb": float}

CLI (on the server, in ~/envs/sfs):
  python -m sfs.teacher.serve --model ~/sfs/models/Qwen3-VL-8B-Instruct-FP8 --kind instruct \
      --port 8765 --min-free-gb 14
"""
from __future__ import annotations

import argparse
import base64
import hmac
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

PROTOCOL = 1
SECRET_HEADER = "X-SFS-Secret"
DEFAULT_SECRET_FILE = os.path.join("~", ".config", "sfs", "secret")
MIN_SECRET_CHARS = 16
MAX_BODY_BYTES = 64 * 1024 * 1024
MAX_FRAMES = 32
MAX_QUERIES = 512
MAX_QUERY_CHARS = 500
ATTN = ("sdpa", "eager")


def load_secret(environ=None) -> str:
    """The shared secret from SFS_SERVE_SECRET or the file named by SFS_SERVE_SECRET_FILE."""
    env = os.environ if environ is None else environ
    secret = env.get("SFS_SERVE_SECRET", "")
    if not secret:
        path = os.path.expanduser(env.get("SFS_SERVE_SECRET_FILE", DEFAULT_SECRET_FILE))
        if os.path.exists(path):
            secret = open(path, encoding="utf-8").read().strip()
    if len(secret) < MIN_SECRET_CHARS:
        raise RuntimeError("no shared secret: set SFS_SERVE_SECRET or create ~/.config/sfs/secret "
                           f"(at least {MIN_SECRET_CHARS} characters)")
    return secret


def encode_frames(frames: list[np.ndarray]) -> list[str]:
    """uint8 frames -> base64 PNG (lossless; channel order is kept as given)."""
    import cv2
    out = []
    for f in frames:
        ok, buf = cv2.imencode(".png", np.ascontiguousarray(f))
        if not ok:
            raise ValueError("cannot encode frame")
        out.append(base64.b64encode(buf.tobytes()).decode("ascii"))
    return out


def decode_frames(items: list[str]) -> list[np.ndarray]:
    import cv2
    out = []
    for s in items:
        f = cv2.imdecode(np.frombuffer(base64.b64decode(s, validate=True), np.uint8), cv2.IMREAD_UNCHANGED)
        if f is None or f.ndim != 3 or f.shape[2] != 3:
            raise ValueError("frame is not a 3-channel image")
        out.append(f)
    return out


def validate_score_request(req: dict, kind: str) -> str | None:
    """None if the request is well-formed for a scorer of this kind, else the reason."""
    if not isinstance(req, dict):
        return "body must be a JSON object"
    if req.get("mode") not in ("clip", "full"):
        return "mode must be 'clip' or 'full'"
    if req["mode"] == "clip" and kind != "instruct":
        return "mode 'clip' needs an instruct model"
    frames, nums, queries = req.get("frames"), req.get("frame_numbers"), req.get("queries")
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES:
        return f"frames: 1..{MAX_FRAMES} items"
    if not isinstance(nums, list) or len(nums) != len(frames) or not all(isinstance(x, int) for x in nums):
        return "frame_numbers: one int per frame"
    if not isinstance(req.get("fps"), (int, float)) or not req["fps"] > 0:
        return "fps must be > 0"
    if not isinstance(queries, list) or not 1 <= len(queries) <= MAX_QUERIES:
        return f"queries: 1..{MAX_QUERIES} items"
    if not all(isinstance(q, str) and 0 < len(q) <= MAX_QUERY_CHARS for q in queries):
        return f"each query is a non-empty string of at most {MAX_QUERY_CHARS} characters"
    return None


def server_info(scorer) -> dict:
    import torch
    import transformers
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"protocol": PROTOCOL, "model_id": scorer.model_id, "revision": scorer.revision, "kind": scorer.kind,
            "torch": torch.__version__, "transformers": transformers.__version__, "gpu": gpu,
            "limits": {"max_body_bytes": MAX_BODY_BYTES, "max_frames": MAX_FRAMES, "max_queries": MAX_QUERIES,
                       "max_query_chars": MAX_QUERY_CHARS}}


def wait_for_gpu(min_free_gb: float, poll_s: float = 30.0, mem_info=None, sleep=time.sleep, log=print) -> float:
    """Block until the GPU has at least min_free_gb free; returns the free GB seen."""
    if mem_info is None:
        import torch
        mem_info = torch.cuda.mem_get_info
    while True:
        free, total = mem_info()
        free_gb = free / 1e9
        if free_gb >= min_free_gb:
            log(f"GPU free {free_gb:.1f} of {total / 1e9:.1f} GB; loading")
            return free_gb
        log(f"GPU free {free_gb:.1f} GB < {min_free_gb:.1f} GB needed; waiting {poll_s:.0f} s")
        sleep(poll_s)


class Handler(BaseHTTPRequestHandler):
    server_version = f"sfs-score/{PROTOCOL}"

    def log_message(self, fmt, *args):          # the default logger is replaced by one line per request
        pass

    def _reply(self, code: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.server.log(f"{time.strftime('%H:%M:%S')} {self.command} {self.path} -> {code}"
                        + (f" {obj['seconds']:.2f}s" if "seconds" in obj else ""))

    def _authorised(self) -> bool:
        given = self.headers.get(SECRET_HEADER, "")
        if hmac.compare_digest(given.encode("utf-8"), self.server.secret.encode("utf-8")):
            return True
        self._reply(401, {"error": "unauthorised"})
        return False

    def _read_body(self) -> bytes | None:
        """The whole request body, read before any reply (closing a socket with unread data resets the
        connection, and the client would never see the reply); None after a 413."""
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = -1
        if n < 0 or n > MAX_BODY_BYTES:
            self._reply(413, {"error": f"body must be at most {MAX_BODY_BYTES} bytes"})
            return None
        return self.rfile.read(n)

    def do_GET(self):
        if not self._authorised():
            return
        if self.path == "/info":
            return self._reply(200, self.server.info)
        self._reply(404, {"error": "not found"})

    def do_POST(self):
        raw = self._read_body()
        if raw is None or not self._authorised():
            return
        if self.path not in ("/score", "/control"):
            return self._reply(404, {"error": "not found"})
        try:
            req = json.loads(raw or b"null")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._reply(400, {"error": "body is not JSON"})
        scorer = self.server.scorer
        if self.path == "/control":
            return self._control(scorer, req)
        reason = validate_score_request(req, scorer.kind)
        if reason:
            return self._reply(400, {"error": reason})
        try:
            frames = decode_frames(req["frames"])
        except (ValueError, TypeError):
            return self._reply(400, {"error": "frames must be base64 PNG images"})
        import torch
        t0 = time.perf_counter()
        try:
            fn = scorer.score_clip if req["mode"] == "clip" else scorer.score_full
            scores = fn(frames, req["frame_numbers"], float(req["fps"]), req["queries"])
            scorer.sync()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return self._reply(503, {"error": "GPU out of memory", "retry": True})
        except Exception as e:                    # noqa: BLE001 - report the type, never a traceback
            return self._reply(500, {"error": type(e).__name__})
        self._reply(200, {"scores": [float(x) for x in scores], "seconds": time.perf_counter() - t0})

    def _control(self, scorer, req: dict):
        op = req.get("op") if isinstance(req, dict) else None
        if op == "reset_peak":
            scorer.reset_peak_memory()
        elif op == "set_attn":
            if req.get("name") not in ATTN:
                return self._reply(400, {"error": f"name must be one of {ATTN}"})
            scorer.set_attn(req["name"])
        elif op != "peak":
            return self._reply(400, {"error": "op must be reset_peak, peak or set_attn"})
        self._reply(200, {"peak_memory_gb": scorer.peak_memory_gb()})


def make_server(scorer, secret: str, host: str = "127.0.0.1", port: int = 8765, log=print) -> HTTPServer:
    srv = HTTPServer((host, port), Handler)
    srv.scorer, srv.secret, srv.log = scorer, secret, log
    srv.info = server_info(scorer)
    return srv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", required=True, choices=["instruct", "reranker"])
    ap.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 and use an SSH tunnel")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--min-free-gb", type=float, default=14.0, help="wait for this much free GPU memory")
    ap.add_argument("--gpu-budget-gb", type=float, help="Gemma 4: GPU memory for weights (rest kept in CPU memory)")
    a = ap.parse_args(argv)
    log = lambda m: print(m, flush=True)                                   # noqa: E731
    secret = load_secret()
    wait_for_gpu(a.min_free_gb, log=log)
    from sfs.teacher.run import make_scorer
    scorer = make_scorer(os.path.expanduser(a.model), None, a.kind, a.gpu_budget_gb)
    srv = make_server(scorer, secret, a.host, a.port, log)
    log(f"serving {scorer.model_id} ({a.kind}) on {a.host}:{a.port}; "
        f"loaded with {scorer.peak_memory_gb():.1f} GB peak GPU memory")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
