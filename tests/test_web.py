"""Dashboard back end (spec-video-alerts-dashboard §4.7): every endpoint against fixture data, no GPU."""
import io
import os

import numpy as np
import pytest

from sfs.alerts.detect import VideoInfo
from sfs.demo.search import Bundle, Searcher
from sfs.web.clipmap import ClipMap, clip_means
from sfs.web.server import build_parser, create_app, meva_media_present
from tests.test_demo import DIM, F, T, FakeVLM, _unit, alert_store, bundle, sources  # noqa: F401 (fixtures)


@pytest.fixture
def client(sources, bundle, alert_store, tmp_path):
    from fastapi.testclient import TestClient
    st, _ = alert_store
    rng = np.random.default_rng(3)
    calls = []

    def fake_ingest(job, progress):                  # what sfs.alerts.ingest does, on fixture-sized vectors
        calls.append((job.name, job.classes, os.path.getsize(job.path)))
        progress("detect", 0.5, "half way")
        vid = st.add_video(job.name, f"upload-{len(calls)}", VideoInfo(30.0, 320, 240, 300), "fake", job.classes)
        st.write_alerts(vid, [{"cls": "person", "near": ["car"], "track_id": 1, "frame": 60, "time_s": 2.0,
                               "conf": 0.8, "start_frame": 0, "end_frame": 150, "frames": [0, 15], "full": True}],
                        {"lora": [_unit(rng.normal(size=(2, DIM)))], "frozen": [_unit(rng.normal(size=(2, DIM)))],
                         "feats": rng.normal(size=(1, T, F)), "mask": np.ones((1, T), bool)})
        return vid

    static = tmp_path / "dist"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<html>dashboard</html>")
    (static / "assets" / "app.js").write_text("console.log(1)")
    s = Searcher(Bundle(bundle), sources["crops"], device="cpu", vlm=FakeVLM(), alerts=st)
    s._encoder = type("E", (), {"embed_texts": lambda self, t: _unit(np.ones((len(t), DIM)))})()
    phr = {"act1": ["a person opening a car door"]}
    app = create_app(s, st, fake_ingest, os.path.join(bundle, "previews"), str(static), phr)
    c = TestClient(app)
    c.calls, c.store, c.app = calls, st, app
    # the same searcher and store with the MEVA collection hidden (spec-submission §1.5)
    c.without_meva = lambda: TestClient(create_app(s, st, fake_ingest, os.path.join(bundle, "previews"), str(static),
                                                   phr, meva=False))
    return c


def test_status_lists_the_corpus_filters_and_examples(client):
    j = client.get("/api/status").json()
    assert j["meva_clips"] == 10 and j["alerts"] == 3 and j["videos"] == 1
    assert j["classes"][0] == "person" and "backpack" in j["classes"]
    assert j["examples"][0] == "a person opening a car door" and j["filters"]["hours"] == [10, 13]


def test_search_returns_media_bands_points_and_cost(client):
    r = client.post("/api/search", json={"query": "a person opening a car door", "collection": "both",
                                         "level": "standard", "top": 13})
    assert r.status_code == 200
    j = r.json()
    res = j["results"]
    assert len(res) == 13 and [x["rank"] for x in res] == list(range(1, 14))
    assert {x["source"] for x in res} == {"meva", "alert"} and j["expected_activity"] == "act1"
    meva = next(x for x in res if x["source"] == "meva")
    alert = next(x for x in res if x["source"] == "alert")
    assert meva["media"].endswith("_context.gif") and "match" in meva and meva["band"] in ("high", "medium", "low")
    assert alert["media"].endswith("/annotated.webm") and alert["thumb"].endswith("/thumb.jpg") and "match" not in alert
    assert alert["end_s"] - alert["start_s"] == pytest.approx(5.0) and "match" not in alert
    assert len(meva["point"]) == 3 and len(j["query_point"]) == 3
    assert set(meva["scores"]) >= {"lora", "student", "standard", "final"}
    assert j["cost"]["n_searched"] == 13 and j["cost"]["vlm_calls"] == 0

    v = client.post("/api/search", json={"query": "a person", "collection": "alerts", "level": "verified",
                                         "k": 2, "classes": ["person"]}).json()
    assert v["cost"]["vlm_calls"] == 2 and all(x["source"] == "alert" for x in v["results"])
    assert sum(x["vlm_prob"] is not None for x in v["results"]) == 2
    assert client.post("/api/search", json={"query": "  "}).status_code == 400
    assert client.post("/api/search", json={"query": "x", "level": "slow"}).status_code == 400


def test_upload_runs_in_the_background_and_the_video_can_be_deleted(client):
    assert client.post("/api/videos", files={"file": ("notes.txt", b"hi")}).status_code == 400
    assert client.post("/api/videos", files={"file": ("a.mp4", b"x")}, data={"classes": "dragon"}).status_code == 400
    r = client.post("/api/videos", files={"file": ("my clip.mp4", io.BytesIO(b"\0" * 1000))},
                    data={"classes": "person,car"})
    assert r.status_code == 200 and r.json()["state"] in ("queued", "running", "done")
    job = client.app.state.runner.wait(r.json()["id"])
    assert job.state == "done" and client.calls == [("my clip.mp4", ("person", "car"), 1000)]
    assert client.get(f"/api/jobs/{job.id}").json()["video_id"] == job.video_id
    al = client.get(f"/api/videos/{job.video_id}/alerts").json()
    assert [(a["cls"], a["near"], a["time"]) for a in al] == [("person", ["car"], "00:02.0")]
    assert client.get("/api/status").json()["alerts"] == 4

    detail = client.get(f"/api/clips/{al[0]['id']}").json()
    assert detail["source"] == "alert" and len(detail["frames"]) == 2 and detail["near"] == ["car"]
    assert client.delete(f"/api/videos/{job.video_id}").status_code == 200
    assert client.get("/api/status").json()["alerts"] == 3
    assert client.delete(f"/api/videos/{job.video_id}").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404


def test_meva_clip_detail_frames_map_and_front_end(client):
    d = client.get("/api/clips/104").json()
    assert d["source"] == "meva" and d["activity"] == "act1" and len(d["frames"]) == 3
    img = client.get(d["frames"][0])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert client.get("/api/clips/999999").status_code == 404
    m = client.get("/api/map").json()
    assert len(m["meva"]["points"]) == 10 and len(m["alerts"]["points"]) == 3 and len(m["meva"]["points"][0]) == 3
    assert client.get("/").text == "<html>dashboard</html>" and client.get("/search/anything").text.startswith("<html>")
    assert client.get("/assets/app.js").text == "console.log(1)"


def test_clip_map_is_a_stable_pca_of_clip_means():
    rng = np.random.default_rng(0)
    emb = _unit(rng.normal(size=(40, 16))).astype(np.float32)
    starts = np.arange(0, 41, 4)
    m1, m2 = ClipMap(emb, starts), ClipMap(emb.copy(), starts)
    assert m1.meva.shape == (10, 3) and np.abs(m1.meva).max() == pytest.approx(1.0)
    assert np.allclose(m1.meva, m2.meva)
    assert np.allclose(m1.project(emb, starts), m1.meva)          # a MEVA clip projects onto its own point
    assert np.allclose(np.linalg.norm(clip_means(emb, starts), axis=1), 1)


def test_without_meva_media_only_alerts_are_searched_with_the_same_scores(client):
    """spec-submission §1.5: hiding MEVA changes what is offered, not one alert score."""
    q = {"query": "a person opening a car door", "collection": "alerts", "level": "standard", "top": 50}
    full = client.post("/api/search", json=q).json()
    bare = client.without_meva()
    alone = bare.post("/api/search", json=q).json()
    assert [x["id"] for x in alone["results"]] == [x["id"] for x in full["results"]] and len(alone["results"]) == 3
    assert [x["scores"] for x in alone["results"]] == [x["scores"] for x in full["results"]]      # bit-identical
    assert [x["band"] for x in alone["results"]] == [x["band"] for x in full["results"]]

    st = bare.get("/api/status").json()
    assert st["collections"] == ["alerts"] and st["meva_clips"] == 0 and st["alerts"] == 3
    assert client.get("/api/status").json()["collections"] == ["meva", "alerts", "both"]
    for collection in ("meva", "both"):
        r = bare.post("/api/search", json=q | {"collection": collection})
        assert r.status_code == 409 and "not installed" in r.json()["detail"]
    assert bare.get("/api/clips/104").status_code == 404 and bare.get("/api/clips/104/frame/0.jpg").status_code == 404
    m = bare.get("/api/map").json()
    assert m["meva"] == {"ids": [], "points": [], "labels": []} and len(m["alerts"]["points"]) == 3
    alert_id = alone["results"][0]["id"]
    assert bare.get(f"/api/clips/{alert_id}").json()["source"] == "alert"


def test_meva_media_need_both_previews_and_crops(tmp_path):
    previews, crops = tmp_path / "previews", tmp_path / "crops"
    assert not meva_media_present(str(previews), str(crops))
    previews.mkdir()
    assert not meva_media_present(str(previews), str(crops))
    crops.mkdir()
    assert meva_media_present(str(previews), str(crops))


def test_server_paths_are_options_with_the_laptop_defaults():
    a = build_parser().parse_args([])
    assert (a.bundle, a.previews, a.crops, a.vlm) == ("data/demo/A", None, "data/crops_v2",
                                                      "data/models/Qwen3-VL-2B-Instruct")
    b = build_parser().parse_args(["--vlm", "m/vlm", "--previews", "x/previews", "--bundle", "m/index"])
    assert (b.vlm, b.previews, b.bundle) == ("m/vlm", "x/previews", "m/index")
