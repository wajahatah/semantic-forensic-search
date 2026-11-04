"""Office-server scoring service and client (spec-vlm-followups E4, D-029): end-to-end on a local port
with the tiny random-weight Qwen3-VL, secret refusal, size limits, retry on GPU out-of-memory, the
secret never logged, lossless frames, waiting for free GPU memory, and G0 / remote-check through the
scorer hooks."""
import http.client
import json
import threading

import numpy as np
import pandas as pd
import pytest

from sfs.teacher.remote import RemoteError, RemoteScorer
from sfs.teacher.serve import (MAX_BODY_BYTES, MAX_QUERIES, SECRET_HEADER, decode_frames, encode_frames,
                               load_secret, make_server, validate_score_request, wait_for_gpu)

SECRET = "test-secret-0123456789abcdef"
NO_SLEEP = lambda s: None                                       # noqa: E731


class Serving:
    """A scoring server on a free local port, in a thread; collects its log lines."""

    def __init__(self, scorer, secret=SECRET):
        self.lines = []
        self.srv = make_server(scorer, secret, "127.0.0.1", 0, log=self.lines.append)
        self.port = self.srv.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.srv.shutdown()
        self.srv.server_close()


def _frames(n=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (60, 40, 3), np.uint8) for _ in range(n)]


QUERIES = ["a person opens a car door", "someone sits down", "a car turns left at the junction"]


def test_remote_scores_equal_local_and_hooks_work(tiny_instruct_scorer):
    local = tiny_instruct_scorer
    frames, nums = _frames(), [0, 15, 30, 45]
    with Serving(local) as s:
        client_log = []
        remote = RemoteScorer(s.url, secret=SECRET, sleep=NO_SLEEP, log=client_log.append)
        assert remote.kind == "instruct" and remote.model_id == local.model_id
        assert remote.info["limits"]["max_queries"] == MAX_QUERIES
        assert np.array_equal(remote.score_full(frames, nums, 30.0, QUERIES), local.score_full(frames, nums, 30.0, QUERIES))
        assert np.array_equal(remote.score_clip(frames, nums, 30.0, QUERIES), local.score_clip(frames, nums, 30.0, QUERIES))
        assert remote.server_seconds > 0
        remote.reset_peak_memory()
        assert remote.peak_memory_gb() >= 0.0
        remote.set_attn("eager")
        remote.set_attn("sdpa")
        assert local.model.config._attn_implementation == "sdpa"
    logged = "\n".join(s.lines + client_log)
    assert SECRET not in logged and "/score -> 200" in logged


def test_wrong_secret_is_refused_without_retries(tiny_instruct_scorer):
    sleeps = []
    with Serving(tiny_instruct_scorer) as s:
        with pytest.raises(RemoteError, match="refused the shared secret"):
            RemoteScorer(s.url, secret="wrong-secret-0123456789", sleep=sleeps.append, log=NO_SLEEP)
    assert sleeps == [] and SECRET not in "\n".join(s.lines)


def _post(port, path, body: bytes, secret=SECRET, length=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.putrequest("POST", path)
    c.putheader(SECRET_HEADER, secret)
    c.putheader("Content-Length", str(len(body) if length is None else length))
    c.endheaders()
    if length is None:
        c.send(body)
    r = c.getresponse()
    out = r.status, json.loads(r.read())
    c.close()
    return out


def test_size_limits_and_bad_requests_are_rejected(tiny_instruct_scorer):
    ok = {"mode": "full", "frames": encode_frames(_frames(2)), "frame_numbers": [0, 1], "fps": 2.0, "queries": ["q"]}
    with Serving(tiny_instruct_scorer) as s:
        assert _post(s.port, "/score", b"", length=MAX_BODY_BYTES + 1)[0] == 413
        assert _post(s.port, "/score", b"not json")[0] == 400
        too_many = {**ok, "queries": ["q"] * (MAX_QUERIES + 1)}
        assert _post(s.port, "/score", json.dumps(too_many).encode())[0] == 400
        bad_frames = {**ok, "frames": ["aGVsbG8="] * 2}
        assert _post(s.port, "/score", json.dumps(bad_frames).encode()) == (400, {"error": "frames must be base64 PNG images"})
        assert _post(s.port, "/control", json.dumps({"op": "set_attn", "name": "flash"}).encode())[0] == 400
        assert _post(s.port, "/nowhere", b"{}")[0] == 404
        assert _post(s.port, "/score", json.dumps(ok).encode(), secret="x" * 20)[0] == 401
        assert _post(s.port, "/score", json.dumps(ok).encode())[0] == 200
    assert validate_score_request({**ok, "mode": "clip"}, "reranker") == "mode 'clip' needs an instruct model"
    assert validate_score_request({**ok, "frame_numbers": [0]}, "instruct") == "frame_numbers: one int per frame"


class FlakyScorer:
    """Runs out of GPU memory on the first `fails` score calls, then delegates."""

    def __init__(self, inner, fails):
        self.inner, self.fails = inner, fails
        self.kind, self.model_id, self.revision = inner.kind, inner.model_id, inner.revision

    def score_full(self, *args):
        import torch
        if self.fails:
            self.fails -= 1
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return self.inner.score_full(*args)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_out_of_memory_is_retried_with_backoff(tiny_instruct_scorer):
    frames, nums = _frames(2), [0, 1]
    sleeps, client_log = [], []
    with Serving(FlakyScorer(tiny_instruct_scorer, fails=2)) as s:
        remote = RemoteScorer(s.url, secret=SECRET, backoff_s=1.0, sleep=sleeps.append, log=client_log.append)
        got = remote.score_full(frames, nums, 2.0, QUERIES)
    assert sleeps == [1.0, 2.0] and "/score -> 503" in "\n".join(s.lines)
    assert np.array_equal(got, tiny_instruct_scorer.score_full(frames, nums, 2.0, QUERIES))
    with Serving(FlakyScorer(tiny_instruct_scorer, fails=5)) as s:
        remote = RemoteScorer(s.url, secret=SECRET, retries=2, sleep=NO_SLEEP, log=NO_SLEEP)
        with pytest.raises(RemoteError, match="gave up after 3 attempts"):
            remote.score_full(frames, nums, 2.0, QUERIES)


def test_unreachable_server_is_retried_then_reported():
    sleeps = []
    with pytest.raises(RemoteError, match="gave up after 3 attempts"):
        RemoteScorer("http://127.0.0.1:9", secret=SECRET, retries=2, timeout=2, sleep=sleeps.append, log=NO_SLEEP)
    assert len(sleeps) == 2


def test_frames_round_trip_losslessly_in_any_shape():
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 255, (7, 13, 3), np.uint8), rng.integers(0, 255, (448, 252, 3), np.uint8)]
    back = decode_frames(encode_frames(frames))
    assert all(np.array_equal(a, b) for a, b in zip(frames, back))


def test_secret_from_env_or_file_and_too_short_refused(tmp_path):
    f = tmp_path / "secret"
    f.write_text(SECRET + "\n", encoding="utf-8")
    assert load_secret({"SFS_SERVE_SECRET_FILE": str(f)}) == SECRET
    assert load_secret({"SFS_SERVE_SECRET": "e" * 20, "SFS_SERVE_SECRET_FILE": str(f)}) == "e" * 20
    with pytest.raises(RuntimeError, match="no shared secret"):
        load_secret({"SFS_SERVE_SECRET": "short"})
    with pytest.raises(RuntimeError):
        load_secret({"SFS_SERVE_SECRET_FILE": str(tmp_path / "missing")})


def test_wait_for_gpu_polls_until_enough_is_free():
    seen = iter([(6e9, 16e9), (9e9, 16e9), (15e9, 16e9)])
    sleeps, lines = [], []
    free = wait_for_gpu(14.0, poll_s=30, mem_info=lambda: next(seen), sleep=sleeps.append, log=lines.append)
    assert free == 15.0 and sleeps == [30, 30] and "waiting" in lines[0] and "loading" in lines[-1]


class FakeScorer:
    """Deterministic scores from frame content; records hook calls (stands in for a local or remote model)."""
    kind, model_id, revision = "instruct", "fake@0", "0"

    def __init__(self, offset=0.0):
        self.offset, self.calls, self.attn = offset, [], "sdpa"
        self.info = {"model_id": self.model_id}

    def _s(self, frames, queries):
        base = float(np.mean([f.mean() for f in frames]))
        return np.array([base / 50 + len(q) / 10 + self.offset + (0.05 if self.attn == "eager" else 0.0)
                         for q in queries], np.float32)

    def score_full(self, frames, nums, fps, queries):
        return self._s(frames, queries)

    def score_clip(self, frames, nums, fps, queries):
        return self._s(frames, queries) + 0.01

    def sync(self):
        self.calls.append("sync")

    def reset_peak_memory(self):
        self.calls.append("reset")

    def peak_memory_gb(self):
        return 12.5

    def set_attn(self, name):
        self.calls.append(name)
        self.attn = name


class FakeCrops:
    def frames(self, iid, view, k=8):
        rng = np.random.default_rng(iid)
        return [rng.integers(0, 255, (20, 20, 3), np.uint8) for _ in range(4)], [0, 1, 2, 3]

    def fps_of(self, iid):
        return 2.0


def _g0_inputs():
    manifest = pd.DataFrame({"instance_id": [str(i) for i in range(40)], "split": ["val"] * 40})
    captions = {f"c{i:02d}": [f"caption {i} a" * (1 + i % 3), f"caption {i} b"] for i in range(32)}
    return manifest, captions


def test_g0_uses_scorer_hooks_and_the_memory_limit():
    from sfs.teacher.gate import g0
    manifest, captions = _g0_inputs()
    s = FakeScorer()
    rep = g0(s, FakeCrops(), manifest, captions, max_vram_gb=15.0)
    assert rep["peak_vram_gb"] == 12.5 and rep["max_vram_gb"] == 15.0 and rep["pass"]
    assert s.calls[0] == "reset" and "eager" in s.calls and s.attn == "sdpa"
    assert not g0(FakeScorer(), FakeCrops(), manifest, captions)["pass"]          # 12.5 GB > laptop's 7.5


def test_remote_check_passes_a_faithful_server_and_fails_a_wrong_one():
    from sfs.teacher.gate import remote_check
    manifest, captions = _g0_inputs()
    ok = remote_check(FakeScorer(), FakeScorer(offset=0.001), FakeCrops(), manifest, captions)
    assert ok["pass"] and ok["full"]["shortcut_pass"] and "cached" in ok
    bad = remote_check(FakeScorer(), FakeScorer(offset=1.0), FakeCrops(), manifest, captions)
    assert not bad["pass"]


def test_client_refuses_a_request_over_the_server_limit(tiny_instruct_scorer, monkeypatch):
    import sfs.teacher.remote as rm
    with Serving(tiny_instruct_scorer) as s:
        remote = RemoteScorer(s.url, secret=SECRET, sleep=NO_SLEEP, log=NO_SLEEP)
        monkeypatch.setattr(rm, "MAX_BODY_BYTES", 100)
        with pytest.raises(RemoteError, match="exceeds the server limit"):
            remote.score_full(_frames(2), [0, 1], 2.0, QUERIES)


def test_reserve_gpu_holds_memory_in_the_cache_without_allocating():
    import torch
    from sfs.teacher.run import reserve_gpu
    assert reserve_gpu(0) == 0.0
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    got = reserve_gpu(torch.cuda.memory_reserved() / 1e9 + 0.2)
    assert got >= 0.2 and torch.cuda.memory_allocated() == before         # held, not in use
    x = torch.empty(100 * 2 ** 20, dtype=torch.uint8, device="cuda")       # served from the held block
    assert torch.cuda.memory_reserved() / 1e9 == pytest.approx(got)
    del x
    torch.cuda.empty_cache()
