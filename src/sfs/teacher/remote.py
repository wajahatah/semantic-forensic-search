"""Laptop side of the office-server scoring service (spec-vlm-followups E4): `RemoteScorer` has the
same interface as `sfs.teacher.vlm.VLMScorer`, so `sfs.teacher.run` and the G0 gate work unchanged
(`--remote http://127.0.0.1:8765` through the SSH tunnel).

- Model id, revision and kind come from the server's /info and go into every score file's identity.
- Connection failures and 502/503/504 (e.g. GPU out of memory while the team's Ollama reloads) are
  retried with exponential back-off; a refused secret (401) or a bad request (400/413) stops at once.
- The secret is read like the server's (SFS_SERVE_SECRET or the secret file) and only ever sent in the
  request header.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import numpy as np

from sfs.teacher.serve import MAX_BODY_BYTES, PROTOCOL, SECRET_HEADER, encode_frames, load_secret

RETRY_STATUS = (502, 503, 504)


class RemoteError(RuntimeError):
    pass


class RemoteScorer:
    def __init__(self, url: str, secret: str | None = None, timeout: float = 600.0, retries: int = 8,
                 backoff_s: float = 5.0, max_backoff_s: float = 300.0, sleep=time.sleep, log=print):
        self.url = url.rstrip("/")
        self._secret = secret if secret is not None else load_secret()
        self.timeout, self.retries, self.backoff_s, self.max_backoff_s = timeout, retries, backoff_s, max_backoff_s
        self._sleep, self._log = sleep, log
        self.server_seconds = 0.0                  # GPU-side time of all score calls (excludes transfer)
        self.info = self._call("GET", "/info")
        if self.info.get("protocol") != PROTOCOL:
            raise RemoteError(f"server protocol {self.info.get('protocol')} != client {PROTOCOL}")
        self.kind, self.model_id, self.revision = self.info["kind"], self.info["model_id"], self.info["revision"]

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        if data is not None and len(data) > MAX_BODY_BYTES:
            raise RemoteError(f"{method} {path}: request of {len(data):,} bytes exceeds the server limit")
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(self.url + path, data=data, method=method,
                                         headers={SECRET_HEADER: self._secret, "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                try:
                    reason = json.loads(e.read()).get("error", "")
                except (json.JSONDecodeError, AttributeError):
                    reason = ""
                if e.code == 401:
                    raise RemoteError("the server refused the shared secret") from None
                if e.code not in RETRY_STATUS:
                    raise RemoteError(f"{method} {path}: HTTP {e.code} {reason}") from None
                why = f"HTTP {e.code} {reason}"
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                why = type(e).__name__
            if attempt == self.retries:
                raise RemoteError(f"{method} {path}: gave up after {self.retries + 1} attempts ({why})")
            wait = min(self.backoff_s * 2 ** attempt, self.max_backoff_s)
            self._log(f"{method} {path}: {why}; retry {attempt + 1}/{self.retries} in {wait:.0f} s")
            self._sleep(wait)
        raise AssertionError("unreachable")

    def _score(self, mode: str, frames, frame_numbers, fps, queries) -> np.ndarray:
        out = self._call("POST", "/score", {"mode": mode, "frames": encode_frames(frames),
                                             "frame_numbers": [int(x) for x in frame_numbers],
                                             "fps": float(fps), "queries": list(queries)})
        self.server_seconds += out["seconds"]
        return np.asarray(out["scores"], np.float32)

    def score_clip(self, frames, frame_numbers, fps, queries) -> np.ndarray:
        return self._score("clip", frames, frame_numbers, fps, queries)

    def score_full(self, frames, frame_numbers, fps, queries) -> np.ndarray:
        return self._score("full", frames, frame_numbers, fps, queries)

    # ------------------------------------------------------------ device hooks (as VLMScorer)
    def sync(self):
        pass                                       # every call returns after the server finished

    def reset_peak_memory(self):
        self._call("POST", "/control", {"op": "reset_peak"})

    def peak_memory_gb(self) -> float:
        return float(self._call("POST", "/control", {"op": "peak"})["peak_memory_gb"])

    def set_attn(self, name: str):
        self._call("POST", "/control", {"op": "set_attn", "name": name})
