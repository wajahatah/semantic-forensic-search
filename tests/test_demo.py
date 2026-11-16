"""Search demo (spec-search-demo): bundle, engine, filters, VLM level, previews; parity with the evaluation."""
import json
import os

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from sfs.demo.index import build
from sfs.demo.previews import Previews, to_gif
from sfs.demo.search import Bundle, Searcher, VLM_SECONDS_PER_CALL
from sfs.eval.cascade import topk_rerank
from sfs.eval.metrics import rank
from sfs.eval.retrieval import InstanceIndex
from sfs.rerank.model import TemporalReranker
from sfs.rerank.train import score_matrix, zcols
from sfs.store.crops import write_crop_shard
from sfs.store.shards import write_shard

DIM, T, F = 8, 4, 12


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@pytest.fixture
def sources(tmp_path):
    rng = np.random.default_rng(0)
    n = 14
    m = pd.DataFrame({
        "instance_id": np.arange(100, 100 + n),
        "split": ["eval"] * 10 + ["train"] * 4,
        "split_b": ["eval"] * n,
        "activity": [f"act{i % 3}" for i in range(n)],
        "site": ["school"] * 6 + ["bus"] * 8,
        "camera": ["G1", "G2"] * 7,
        "date": ["2018-03-07"] * 7 + ["2018-03-09"] * 7,
        "hour": [10, 11, 12, 13] * 3 + [14, 15],
        "frame_start": np.arange(n) * 30, "frame_end": np.arange(n) * 30 + 60,
        "clip_stem": [f"vid{i % 2}" for i in range(n)]})
    rows = []
    for i in m.instance_id:
        for v in range(3):
            for f in range(rng.integers(1, T + 1)):
                rows.append((i, f * 15, v))
    r = np.array(rows)
    emb = _unit(rng.normal(size=(len(r), DIM))).astype(np.float16)
    common = dict(instance_id=r[:, 0], frame_idx=r[:, 1], view=r[:, 2], fallback=np.zeros(len(r), bool),
                  box_side=np.full(len(r), 50))
    write_shard(str(tmp_path / "frozen.npz"), {"encoder_id": "frozen"}, emb=emb, **common)
    emb2 = _unit(rng.normal(size=(len(r), DIM))).astype(np.float16)
    write_shard(str(tmp_path / "lora.npz"), {"encoder_id": "lora1"}, emb=emb2, **common)

    ids = m.instance_id.to_numpy()[::-1]             # features in another order than the corpus
    feats = rng.normal(size=(n, T, F)).astype(np.float16)
    mask = np.zeros((n, T), bool)
    mask[:, :2] = True
    np.savez(tmp_path / "feats.npz", meta=np.array("{}"), instance_id=ids, feats=feats, mask=mask)

    sd = tmp_path / "students"
    (sd / "A-GT").mkdir(parents=True)
    torch.manual_seed(0)
    cfg = {"d": 16, "layers": 1, "heads": 2, "dropout": 0.0}
    model = TemporalReranker(F, DIM, cfg["d"], cfg["layers"], cfg["heads"], T, cfg["dropout"])
    torch.save({"config": cfg, "model": model.state_dict(), "beta": 0.5}, sd / "A-GT" / "best.pt")
    (sd / "selection.json").write_text(json.dumps({"A": {"selected": "A-GT"}}))
    ch = tmp_path / "choices.json"
    ch.write_text(json.dumps({"choices": {"A/instruct": {"config": "instruct-tight", "gamma_student": 2.0}}}))

    crops = tmp_path / "crops"
    for stem in ("vid0", "vid1"):
        sub = m[m.clip_stem == stem]
        blobs, ii, ff, vv = [], [], [], []
        for i in sub.instance_id:
            for v in (0, 1):
                for f in range(3):
                    img = np.full((20 + 4 * v, 20, 3), (i * 7 + f * 40) % 255, np.uint8)
                    blobs.append(cv2.imencode(".png", img)[1].tobytes())
                    ii.append(i), ff.append(f * 15), vv.append(v)
        write_crop_shard(str(crops / f"{stem}.npz"), blobs, {"store_id": "s1", "fps": 30.0},
                         instance_id=ii, frame_idx=ff, view=vv)
    return {"m": m, "dir": tmp_path, "crops": str(crops), "students": str(sd), "choices": str(ch)}


@pytest.fixture
def bundle(sources):
    d = sources["dir"]
    out = str(d / "bundle")
    build("A", sources["m"], str(d / "frozen.npz"), str(d / "lora.npz"), str(d / "feats.npz"), sources["students"],
          sources["crops"], sources["choices"], "instruct", out, log=lambda *_: None)
    return out


class FakeVLM:
    kind = "instruct"

    def __init__(self):
        self.calls = []

    def score_clip(self, frames, nums, fps, queries):
        self.calls.append((len(frames), tuple(nums), fps, tuple(queries)))
        return np.array([float(np.mean(frames[0])) / 10.0], np.float32)


def test_bundle_holds_the_eval_corpus_and_refuses_tampering(sources, bundle):
    b = Bundle(bundle)
    assert list(b.ids) == list(range(100, 110))               # protocol A eval only, sorted
    assert b.info["beta"] == 0.5 and b.info["vlm"] == {"kind": "instruct", "config": "instruct-tight",
                                                       "gamma": 2.0, "view": "tight"}
    assert list(b.clips.instance_id) == list(b.ids)
    assert set(b.clips.crop_shard) == {"vid0.npz", "vid1.npz"}
    assert b.clips.set_index("instance_id").loc[103, "hour"] == 13
    with open(os.path.join(bundle, "vectors.npz"), "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="vectors.npz"):
        Bundle(bundle)


def test_build_refuses_a_corpus_instance_without_crops(sources, tmp_path):
    os.remove(os.path.join(sources["crops"], "vid1.npz"))
    d = sources["dir"]
    with pytest.raises(ValueError, match="no crops"):
        build("A", sources["m"], str(d / "frozen.npz"), str(d / "lora.npz"), str(d / "feats.npz"),
              sources["students"], sources["crops"], sources["choices"], "instruct", str(tmp_path / "b2"),
              log=lambda *_: None)


def test_fast_standard_and_frozen_equal_the_evaluation_code_paths(sources, bundle):
    from sfs.store.shards import read_shard
    d = sources["dir"]
    s = Searcher(Bundle(bundle), sources["crops"], device="cpu")
    q = _unit(np.random.default_rng(5).normal(size=DIM)).astype(np.float32)
    corpus = set(range(100, 110))
    lora = InstanceIndex.from_shard(read_shard(str(d / "lora.npz")), "tight", corpus).score(q[None], 3)[:, 0]
    frozen = InstanceIndex.from_shard(read_shard(str(d / "frozen.npz")), "tight", corpus).score(q[None], 3)[:, 0]
    ids = np.arange(100, 110)

    r = s.search_vector(q, level="fast", top=10)
    assert np.array_equal(r.ids, rank(ids, lora)) and r.vlm_calls == 0 and set(r.seconds) == {"lora"}

    z = np.load(d / "feats.npz")
    rows = np.array([list(z["instance_id"]).index(i) for i in ids])
    best = torch.load(os.path.join(sources["students"], "A-GT", "best.pt"), weights_only=False)
    model = TemporalReranker(F, DIM, 16, 1, 2, T, 0.0)
    model.load_state_dict(best["model"])
    st = score_matrix(model, z["feats"], z["mask"], rows, q[None], "cpu")[:, 0]
    fused = zcols(st[:, None])[:, 0] + 0.5 * zcols(lora[:, None])[:, 0]
    r = s.search_vector(q, level="standard", top=10)
    assert np.array_equal(r.ids, rank(ids, fused))
    np.testing.assert_allclose(r.scores["final"], np.sort(fused)[::-1], rtol=0, atol=1e-12)

    r = s.search_vector(q, level="verified", system="frozen", top=10)     # the baseline ignores the level
    assert np.array_equal(r.ids, rank(ids, frozen)) and r.level == "fast" and r.vlm_calls == 0


def test_filters_select_the_window_after_scoring_the_whole_corpus(sources, bundle):
    s = Searcher(Bundle(bundle), sources["crops"], device="cpu")
    q = _unit(np.ones(DIM)).astype(np.float32)
    full = s.search_vector(q, level="standard", top=10)
    m = sources["m"].set_index("instance_id")
    for flt in ({"camera": "G1"}, {"site": ["bus"], "date": "2018-03-09"}, {"camera": "G2", "date": "2018-03-07",
                                                                             "hour": "11"},
                {"hour_from": 11, "hour_to": 12}):
        r = s.search_vector(q, level="standard", top=10, filters=flt)
        want = [i for i in full.ids if all(
            (str(m.loc[i, c]) in [str(x) for x in np.atleast_1d(v)]) if c in ("camera", "site", "date") else
            (int(m.loc[i, "hour"]) == int(v)) if c == "hour" else
            (int(m.loc[i, "hour"]) >= v) if c == "hour_from" else (int(m.loc[i, "hour"]) <= v)
            for c, v in flt.items())]
        assert list(r.ids) == want and r.n_searched == len(want) and len(want) > 0
    empty = s.search_vector(q, level="verified", filters={"camera": "nope"}, top=10)
    assert len(empty.ids) == 0 and empty.vlm_calls == 0


def test_verified_reorders_the_top_k_with_the_vlm_and_counts_calls(sources, bundle):
    vlm = FakeVLM()
    s = Searcher(Bundle(bundle), sources["crops"], device="cpu", vlm=vlm)
    q = _unit(np.arange(DIM, dtype=np.float32) - 3)
    std = s.search_vector(q, level="standard", top=10)
    r = s.search_vector(q, "a person", level="verified", k=4, top=10)
    assert r.vlm_calls == 4 and len(vlm.calls) == 4 and r.vlm_checked == set(int(i) for i in std.ids[:4])
    assert all(c[3] == ("a person",) and c[0] == 3 for c in vlm.calls)      # 3 tight frames each, query asked
    margins = np.array([s.vlm.score_clip(*s.crops.frames(int(i), "tight"), 30.0, ["a person"])[0]
                        for i in std.ids[:4]], np.float64)
    ids = np.arange(100, 110)
    base = np.empty(10)
    base[[list(ids).index(i) for i in std.ids]] = std.scores["final"]
    assert np.array_equal(r.ids, rank(ids, topk_rerank(ids, base, margins, 4, 2.0)))
    assert np.isnan(r.scores["vlm"]).sum() == 6 and {"frames", "vlm", "student", "lora"} <= set(r.seconds)
    small = s.search_vector(q, "a person", level="verified", k=20, filters={"camera": "G1"})
    assert small.vlm_calls == small.n_searched == 5                          # k capped at the clips left
    assert r.vlm_all_seconds == pytest.approx(10 * r.seconds["vlm"] / 4)       # measured per-call time
    assert std.vlm_all_seconds == pytest.approx(10 * VLM_SECONDS_PER_CALL)    # no VLM ran: Result 04


def test_verified_needs_a_gpu_without_a_loaded_vlm(sources, bundle):
    s = Searcher(Bundle(bundle), sources["crops"], device="cpu")
    assert not s.vlm_available()
    with pytest.raises(RuntimeError, match="GPU"):
        s.search_vector(_unit(np.ones(DIM)).astype(np.float32), "x", level="verified")
    with pytest.raises(ValueError, match="level"):
        s.search_vector(np.ones(DIM, np.float32), level="slow")


def test_previews_are_time_ordered_cached_animations(sources, bundle, tmp_path):
    from PIL import Image
    s = Searcher(Bundle(bundle), sources["crops"], device="cpu")
    p = Previews(s.crops, str(tmp_path / "prev"))
    path = p.path(104)
    im = Image.open(path)
    assert im.n_frames == 3 and im.size == (256, 256)
    shades = []
    for f in range(im.n_frames):
        im.seek(f)
        shades.append(np.asarray(im.convert("RGB"))[128, 128, 0])
    assert shades == [(104 * 7 + f * 40) % 255 for f in range(3)]           # frames in time order
    mtime = os.path.getmtime(path)
    assert p.path(104) == path and os.path.getmtime(path) == mtime           # read from the cache
    assert p.render_all(range(100, 110), log=lambda *_: None) == 9           # 104 already there
    assert p.render_all(range(100, 110), log=lambda *_: None) == 0
    with pytest.raises(ValueError):
        to_gif([])


@pytest.mark.slow
def test_demo_reproduces_the_evaluated_metrics():
    """Real data: for every Q1 and Q2 eval query, Fast and Standard give the evaluated Success@1/@10 exactly
    and AP within float rounding (scores differ by <= 1e-5: one text vs 66 in one BLAS call)."""
    if not os.path.exists("data/demo/A/bundle.json"):
        pytest.skip("demo bundle not built")
    from sfs.data.manifest import read_manifest
    from sfs.eval.baseline import score_query
    from sfs.eval.cascade_run import load_protocol
    from sfs.eval.queries import load_phrasings
    from sfs.teacher.gate import _phrasing_embeddings
    m = read_manifest("manifest_v2_drive/manifest_v2.csv")
    phr = load_phrasings("queries/phrasings.csv")
    pe = _phrasing_embeddings(phr)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = load_protocol("A", m, phr, pe, "runs/student", dev)
    s = Searcher(Bundle("data/demo/A"), device=dev)
    assert np.array_equal(d["ids"], s.b.ids)
    pos = {int(i): n for n, i in enumerate(d["ids"])}
    T_ = np.stack([pe[a][j] for a, j, _ in d["texts"]])
    ap_diff = {}
    for q in d["queries"]:
        c = d["text_col"][(q.activity, q.phrasing_idx)]
        ids = np.array(sorted(q.corpus), np.int64)
        rel = np.fromiter(q.relevant, np.int64)
        for lvl, key in (("fast", "lora"), ("standard", "student")):
            ref_scores = d["firsts"][key][[pos[int(i)] for i in ids], c]
            r = s.search_vector(T_[c], q.text, level=lvl, filters=q.filter or None, top=len(ids))
            assert sorted(r.ids) == list(ids)
            col = "lora" if lvl == "fast" else "standard"
            np.testing.assert_allclose(r.scores[col], d["firsts"][key][[pos[int(i)] for i in r.ids], c], atol=1e-5)
            a, b = score_query(np.isin(rank(ids, ref_scores), rel)), score_query(np.isin(r.ids, rel))
            assert a["s1"] == b["s1"] and a["s10"] == b["s10"], q.qid
            ap_diff.setdefault((q.kind, lvl), []).append(b["ap"] - a["ap"])
    for key, v in ap_diff.items():
        assert abs(np.mean(v)) < 1e-4, key


class FakeEncoder:
    def embed_texts(self, texts):
        rng = np.random.default_rng(len(texts[0]))
        return _unit(rng.normal(size=(len(texts), DIM))).astype(np.float32)


def _app(sources, bundle, tmp_path, vlm=None):
    from sfs.demo.app import DemoApp
    s = Searcher(Bundle(bundle), sources["crops"], device="cpu", encoder=FakeEncoder(), vlm=vlm)
    return DemoApp(s, Previews(s.crops, str(tmp_path / "prev")), {"act1": ["a person waving", "someone waving"],
                                                                   "act0": ["a car parking"]})


def test_app_search_handler_returns_previews_cost_and_labels(sources, bundle, tmp_path):
    from sfs.demo.app import LEVEL_LABELS
    app = _app(sources, bundle, tmp_path)
    assert app.examples()[:2] == ["a car parking", "a person waving"]
    items, md, state = app.run("a person waving", LEVEL_LABELS["standard"], 10, [], [], [], 10, 15, True)
    assert len(items) == 10 and all(os.path.exists(p) for p, _ in items) and len(state["ids"]) == 10
    assert "VLM calls: **0**" in md and "10 clips searched" in md and "of the top 10 match" in md
    acts = app.s.b.clips.set_index("instance_id").activity
    for (_, cap), i in zip(items, state["ids"]):
        assert cap.endswith("✓" if acts[i] == "act1" else "✗")
    items, md, _ = app.run("a person waving", LEVEL_LABELS["fast"], 10, [], ["G1"], [], 10, 15, False)
    assert len(items) == 5 and "[" not in items[0][1]
    assert app.run("  ", LEVEL_LABELS["fast"], 10, [], [], [], 10, 15, False)[0] == []
    items, md, state = app.run("x", LEVEL_LABELS["verified"], 5, [], [], [], 10, 15, False)
    assert items == [] and "not available" in md and state is None
    frames, info = app.detail(*_state_and_index(app))
    assert len(frames) == 3 and "Annotated MEVA activity" in info
    assert app.detail(None, 0) == ([], "")


def _state_and_index(app):
    from sfs.demo.app import LEVEL_LABELS
    _, _, state = app.run("a car parking", LEVEL_LABELS["standard"], 10, [], [], [], 10, 15, False)
    return state, 2


def test_app_verified_shows_vlm_confidence_and_compare_gives_two_lists(sources, bundle, tmp_path):
    from sfs.demo.app import LEVEL_LABELS
    app = _app(sources, bundle, tmp_path, vlm=FakeVLM())
    items, md, state = app.run("a car parking", LEVEL_LABELS["verified"], 5, [], [], [], 10, 15, False)
    assert "VLM calls: **5**" in md and sum("VLM yes" in c for _, c in items) == 5
    left, right, md = app.compare("a car parking", [], [], [], 10, 15, True)
    assert len(left) == len(right) == 10 and "Left:" in md and "Right:" in md
    assert app.filters([], [], [], 10, 15) == {"site": None, "camera": None, "date": None}
    assert app.hours == (10, 13)                     # eval clips only; full range = no hour filter
    assert app.filters(["bus"], [], [], 11, 12) == {"site": ["bus"], "camera": None, "date": None,
                                                    "hour_from": 11, "hour_to": 12}


def test_ui_builds_without_launching(sources, bundle, tmp_path):
    from sfs.demo.app import build_ui
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                # e.g. Gradio not seeing the SelectData parameter
        ui = build_ui(_app(sources, bundle, tmp_path))
    assert ui is not None


def test_durations_read_naturally():
    from sfs.demo.app import _fmt_s
    assert (_fmt_s(0.0421), _fmt_s(3.64), _fmt_s(626.0)) == ("42 ms", "3.6 s", "10.4 min")


# ---------------------------------------------------------------- your videos (spec-video-alerts-dashboard)
@pytest.fixture
def alert_store(tmp_path):
    from sfs.alerts.detect import VideoInfo
    from sfs.alerts.store import AlertStore
    rng = np.random.default_rng(9)
    st = AlertStore(str(tmp_path / "alerts"))
    vid = st.add_video("street.mp4", "sha-1", VideoInfo(25.0, 640, 360, 250), "fake", ("person", "car"))
    alerts = [{"cls": c, "near": [], "track_id": t, "frame": 50 * t, "time_s": 2.0 * t, "conf": 0.9,
               "start_frame": 0, "end_frame": 125, "frames": [0, 25, 50], "full": True}
              for t, c in [(1, "person"), (2, "car"), (3, "person")]]
    vec = lambda: _unit(rng.normal(size=(3, DIM))).astype(np.float32)
    ids = st.write_alerts(vid, alerts, {"lora": [vec() for _ in alerts], "frozen": [vec() for _ in alerts],
                                        "feats": rng.normal(size=(3, T, F)), "mask": np.ones((3, T), bool)})
    for aid in ids:
        d = st.alert_dir(vid, aid)
        os.makedirs(d, exist_ok=True)
        for j in range(3):
            cv2.imwrite(os.path.join(d, f"tight_{j:02d}.png"), np.full((30, 30, 3), 20 * aid + j, np.uint8))
    return st, ids


def test_alerts_never_change_meva_results_and_use_meva_statistics(sources, bundle, alert_store):
    st, ids = alert_store
    q = _unit(np.random.default_rng(5).normal(size=DIM)).astype(np.float32)
    plain = Searcher(Bundle(bundle), sources["crops"], device="cpu")
    with_alerts = Searcher(Bundle(bundle), sources["crops"], device="cpu", alerts=st)
    for lvl in ("fast", "standard"):
        a = plain.search_vector(q, level=lvl, top=10)
        b = with_alerts.search_vector(q, level=lvl, top=10, collection="meva")
        both = with_alerts.search_vector(q, level=lvl, top=20, collection="both")
        assert list(a.ids) == list(b.ids) and np.array_equal(a.scores["final"], b.scores["final"])
        meva_in_both = [int(i) for i in both.ids if i > 0]
        assert meva_in_both == [int(i) for i in a.ids]                  # same order, alerts interleaved
        assert {-i for i in both.ids if i < 0} == set(ids) and both.n_searched == len(a.ids) + 3

    # an alert's Standard score: z with the MEVA corpus's mean / spread, then fused as for MEVA
    s = with_alerts
    r = s.search_vector(q, level="standard", top=3, collection="alerts")
    ac = st.corpus()
    lm, la = s.b.lora.score(q[None], 3)[:, 0], InstanceIndex(ac.ids, ac.lora_emb, ac.lora_starts).score(q[None], 3)[:, 0]
    sm = score_matrix(s.student, s.b.feats, s.b.mask, s.rows, q[None], "cpu")[:, 0]
    sa = score_matrix(s.student, ac.feats, ac.mask, np.arange(3), q[None], "cpu")[:, 0]
    z = lambda x, ref: (x.astype(np.float64) - ref.astype(np.float64).mean()) / ref.astype(np.float64).std()
    want = z(sa, sm) + s.b.info["beta"] * z(la, lm)
    got = dict(zip(r.ids, r.scores["final"]))
    assert np.allclose([got[i] for i in ac.ids], want, atol=1e-12)
    assert all(i < 0 for i in r.ids) and set(r.bands) <= {"high", "medium", "low"}
    fused_m = zcols(sm[:, None])[:, 0] + s.b.info["beta"] * zcols(lm[:, None])[:, 0]
    p95, p99 = np.percentile(fused_m, [95, 99])
    assert r.bands == ["high" if v >= p99 else "medium" if v >= p95 else "low" for v in r.scores["final"]]


def test_alert_filters_and_the_verified_level_on_alerts(sources, bundle, alert_store):
    st, ids = alert_store
    vlm = FakeVLM()
    s = Searcher(Bundle(bundle), sources["crops"], device="cpu", vlm=vlm, alerts=st)
    q = _unit(np.random.default_rng(6).normal(size=DIM)).astype(np.float32)
    cars = s.search_vector(q, level="fast", collection="alerts", classes=["car"])
    assert [-int(i) for i in cars.ids] == [ids[1]]
    r = s.search_vector(q, "a person", level="verified", k=3, collection="alerts", classes=["person"])
    assert r.vlm_calls == 2 and r.n_searched == 2 and {-i for i in r.vlm_checked} == {ids[0], ids[2]}
    assert all(c[1] == (0, 25, 50) and c[2] == 25.0 for c in vlm.calls)   # alert frames, the video's fps
    st.delete_video(st.videos()[0]["id"])                                  # the engine follows the store
    assert s.search_vector(q, level="fast", collection="alerts").n_searched == 0
    with pytest.raises(ValueError):
        s.search_vector(q, collection="everything")
