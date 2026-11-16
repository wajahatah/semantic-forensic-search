"""Alerts from your own videos (spec-video-alerts-dashboard): rule, 5 s clip, store, ingestion end to end."""
import json
import os

import cv2
import numpy as np
import pytest

from sfs.alerts.clip import CLIP_S, clip_window
from sfs.alerts.ingest import CFG, ingest
from sfs.alerts.rule import AlertRule, Det, is_near
from sfs.alerts.store import AlertStore
from sfs.embed.run import regions_for
from sfs.rerank.features import BOX_FEATURES, EMB, T_MAX

FPS = 30.0


def P(tid, box, cls="person", conf=0.9):
    return Det(tid, cls, conf, box)


# ---------------------------------------------------------------- rule
def test_alert_fires_once_confirmed_and_is_held_while_the_object_stays():
    rule = AlertRule(FPS)                            # confirmed after 0.5 s = 15 frames
    fired = [(f, a.track_id) for f in range(300) for a in rule.feed(f, [P(1, (10, 10, 50, 120))])]
    assert fired == [(14, 1)]                        # then held for the rest of the 10 s


def test_low_confidence_detections_do_not_count_towards_confirmation():
    rule = AlertRule(FPS)
    weak = [f for f in range(100) for _ in rule.feed(f, [P(1, (10, 10, 50, 120), conf=0.3)])]
    mixed = [f for f in range(100, 200) for _ in rule.feed(f, [P(1, (10, 10, 50, 120), conf=0.3 + 0.4 * (f % 2))])]
    assert weak == [] and mixed == [129]             # 15 confident frames, every other frame from 100


def test_a_new_alert_only_after_the_object_was_gone_and_is_seen_again():
    rule = AlertRule(FPS, gone_s=2.0)                # gone = more than 60 frames unseen
    seen = list(range(0, 100)) + list(range(150, 200)) + list(range(300, 320))
    fired = [f for f in seen for _ in rule.feed(f, [P(1, (10, 10, 50, 120))])]
    assert fired == [14, 314]                        # a 50-frame gap is an occlusion; 100 frames is gone


def test_a_track_that_replaces_a_just_lost_one_keeps_its_alert():
    rule = AlertRule(FPS)
    fired = [(f, a.track_id) for f in range(60) for a in rule.feed(f, [P(1, (100, 10, 140, 120))])]
    # ByteTrack loses id 1 and re-finds the person 20 frames later as id 2, a few pixels away
    fired += [(f, a.track_id) for f in range(80, 200) for a in rule.feed(f, [P(2, (104, 12, 144, 122))])]
    # another person elsewhere is a new object
    fired += [(f, a.track_id) for f in range(200, 260) for a in rule.feed(f, [P(3, (400, 10, 440, 120))])]
    assert fired == [(14, 1), (214, 3)]
    plain = AlertRule(FPS, stitch=False)
    fired = [f for f in range(60) for _ in plain.feed(f, [P(1, (100, 10, 140, 120))])]
    fired += [f for f in range(80, 200) for _ in plain.feed(f, [P(2, (104, 12, 144, 122))])]
    assert fired == [14, 94]


def test_repeat_interval_is_off_by_default_and_works_when_set():
    rule = AlertRule(FPS, repeat_s=5.5)
    fired = [f for f in range(400) for _ in rule.feed(f, [P(1, (10, 10, 50, 120))])]
    assert fired == [14, 179, 344]


def test_only_the_chosen_classes_raise_alerts_but_any_tracked_object_can_be_near():
    rule = AlertRule(FPS, classes=("person",), confirm_s=0)
    dets = [P(1, (100, 100, 140, 200)), P(2, (150, 120, 300, 200), "car"), P(3, (600, 0, 640, 40), "car")]
    (a,) = rule.feed(0, dets)
    assert (a.cls, a.near_ids, a.near_classes) == ("person", (2,), ("car",))
    assert is_near((0, 0, 10, 100), (105, 0, 120, 10)) and not is_near((0, 0, 10, 100), (111, 0, 120, 10))


# ---------------------------------------------------------------- clip
@pytest.mark.parametrize("alert,n,start", [(300, 900, 225), (10, 900, 0), (890, 900, 749)])
def test_the_clip_is_always_exactly_5_seconds_of_11_frames_at_2_fps(alert, n, start):
    w = clip_window(alert, FPS, n)
    assert w.full and len(w.frames) == 11
    assert (w.frames[-1] - w.frames[0]) / FPS == CLIP_S == (w.end - w.start) / FPS
    assert np.allclose(np.diff(w.frames), 15) and w.start == start and w.frames[-1] <= n - 1
    if start not in (0, 749):
        assert w.frames[5] == alert                  # centred unless it had to shift


def test_a_video_shorter_than_the_clip_gives_a_flagged_short_clip():
    w = clip_window(20, FPS, 90)                     # 3 s video
    assert not w.full and w.frames[0] == 0 and w.frames[-1] == 89 - 89 % 15 and w.end == 89


# ---------------------------------------------------------------- ingestion end to end
W, H, N = 320, 240, 360      # 12 s at 30 fps


def person_box(f):
    x = 40 + f // 3
    return (x, 60, x + 30, 160)


CAR = (130, 110, 250, 180)
PERSON_VISIBLE = lambda f: 30 <= f < 150 or f >= 230      # gone 80 frames (> 2 s): two person alerts


@pytest.fixture
def video(tmp_path):
    path = str(tmp_path / "street.mp4")
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in range(N):
        img = np.full((H, W, 3), 40, np.uint8)
        cv2.rectangle(img, CAR[:2], CAR[2:], (30, 30, 200), -1)
        if PERSON_VISIBLE(f):
            b = person_box(f)
            cv2.rectangle(img, b[:2], b[2:], (200, 180, 20), -1)
        wr.write(img)
    wr.release()
    return path


class FakeTracker:
    weights_id = "fake:0"

    def track(self, video_path, fps):
        for f in range(N):
            dets = [Det(7, "car", 0.8, CAR)]
            if PERSON_VISIBLE(f):
                dets.append(Det(1, "person", 0.9, person_box(f)))
            yield f, dets


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def _vec(self, crops):
        v = np.zeros((len(crops), EMB), np.float32)
        for i, c in enumerate(crops):
            v[i, :3] = c.reshape(-1, 3).mean(0) + 1.0
            v[i, 3:5] = c.shape[:2]
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def lora(self, crops):
        self.calls.append(("lora", [c.shape for c in crops]))
        return self._vec(crops)

    def frozen(self, crops):
        self.calls.append(("frozen", [c.shape for c in crops]))
        return self._vec(crops) * -1


def test_ingest_turns_a_video_into_searchable_5s_alerts(video, tmp_path):
    store = AlertStore(str(tmp_path / "alerts"))
    emb = FakeEmbedder()
    stages = []
    vid = ingest(video, store, FakeTracker(), emb, progress=lambda s, f, m: stages.append(s))
    al = store.alerts(vid)
    assert list(zip(al.cls, al.frame)) == [("car", 14), ("person", 44), ("person", 244)]
    assert list(al.near) == ["", "car", "car"]      # the car fires before the person appears
    assert stages[0] == "detect" and stages[-1] == "done"
    v = store.videos()[0]
    assert (v["status"], v["n_frames"], v["fps"]) == ("ready", N, FPS) and v["n_alerts"] == 3

    for r in al.itertuples():
        frames = json.loads(r.frames)
        assert len(frames) == 11 and (frames[-1] - frames[0]) / FPS == CLIP_S and r.full
        d = store.alert_dir(vid, r.id)
        assert sorted(os.listdir(d)) == sorted(["thumb.jpg"] + [f"frame_{j:02d}.jpg" for j in range(11)]
                                               + [f"tight_{j:02d}.png" for j in range(11)])
    cap = cv2.VideoCapture(os.path.join(store.media_dir(vid), "annotated.webm"))     # one video, all boxes drawn
    n = 0
    while cap.read()[0]:
        n += 1
    assert n == N // 2 and cap.get(cv2.CAP_PROP_FPS) == pytest.approx(15)          # 12 s at 15 fps

    ac = store.corpus()
    assert len(ac) == 3 and list(ac.ids) == [-int(i) for i in al.id]
    assert ac.feats.shape == (3, T_MAX, 2 * EMB + len(BOX_FEATURES)) and ac.mask.sum(1).tolist() == [11, 11, 11]
    assert list(np.diff(ac.lora_starts)) == [11, 11, 11]
    has_box = ac.feats[:, :11, 2 * EMB + BOX_FEATURES.index("has_box")]
    assert np.all(has_box == 1)

    # the person alert's tight crops are the MEVA regions of the person + the car next to it
    person = al[al.cls == "person"].iloc[0]
    f0 = json.loads(person.frames)[0]
    tracks = {1: {f: person_box(f) for f in range(N) if PERSON_VISIBLE(f)}, 7: {f: CAR for f in range(N)}}
    (_, t_reg, _, _), _ctx = regions_for((1, 7), f0, tracks, W, H, CFG)
    png = cv2.imread(os.path.join(store.alert_dir(vid, person.id), "tight_00.png"))
    assert png.shape[:2] == (t_reg[3] - t_reg[1], t_reg[2] - t_reg[0])


def test_the_same_video_is_not_ingested_twice_and_delete_removes_everything(video, tmp_path):
    store = AlertStore(str(tmp_path / "alerts"))
    vid = ingest(video, store, FakeTracker(), FakeEmbedder())
    v0 = store.version
    assert ingest(video, store, FakeTracker(), FakeEmbedder()) == vid and store.version == v0
    store.delete_video(vid)
    assert store.videos() == [] and len(store.alerts()) == 0 and len(store.corpus()) == 0
    assert not os.path.exists(store.media_dir(vid)) and not os.path.exists(store.vectors_path(vid))


def test_a_failed_or_interrupted_ingestion_is_never_searchable(video, tmp_path):
    store = AlertStore(str(tmp_path / "alerts"))

    class Boom(FakeEmbedder):
        def lora(self, crops):
            raise RuntimeError("GPU fell over")
    with pytest.raises(RuntimeError):
        ingest(video, store, FakeTracker(), Boom())
    (v,) = store.videos()
    assert v["status"] == "failed" and "GPU fell over" in v["error"] and len(store.corpus()) == 0
    vid = ingest(video, store, FakeTracker(), FakeEmbedder())       # retried: the failed one is replaced
    assert [x["status"] for x in store.videos()] == ["ready"] and len(store.alerts(vid)) == 3

    store.set_video(vid, status="processing")                       # as if the app died mid-way
    assert [x["status"] for x in AlertStore(store.root).videos()] == ["failed"]


@pytest.mark.slow
def test_the_alert_embedding_path_reproduces_the_evaluated_meva_embeddings():
    """Stored MEVA tight crops through ClipEmbedder == the LoRA and frozen stores the results were measured on."""
    from sfs.alerts.embed import ClipEmbedder
    from sfs.demo.search import Bundle, Searcher
    from sfs.store.shards import read_shard
    if not os.path.isdir("data/demo/A") or not os.path.isdir("data/crops_v2"):
        pytest.skip("needs the local demo bundle and crop store")
    b = Bundle("data/demo/A")
    s = Searcher(b, "data/crops_v2")
    e = ClipEmbedder(s.encoder)
    stores = {"lora": read_shard("data/embeddings_lora/A-middle/embeddings.npz"),
              "frozen": read_shard("manifest_v2_drive/embeddings_v2_siglip2.npz")}
    cos = {"lora": [], "frozen": []}
    for i in [int(x) for x in b.ids[::400]]:
        sh, per = s.crops.where[i]
        shard = s.crops._shard(sh)
        imgs = [shard.image(int(r)) for r in per["tight"]]
        frames = [int(shard.frame_idx[r]) for r in per["tight"]]
        for name, got in (("lora", e.lora(imgs)), ("frozen", e.frozen(imgs))):
            st = stores[name]
            sel = np.flatnonzero((st["instance_id"] == i) & (st["view"] == 0))
            ref = {int(st["frame_idx"][r]): st["emb"][r] for r in sel}
            want = np.stack([ref[f] for f in frames]).astype(np.float32)
            cos[name] += ((got * want).sum(1) / np.linalg.norm(want, axis=1)).tolist()
    assert min(cos["lora"]) > 0.9999 and min(cos["frozen"]) > 0.9999      # measured 1.00000 / 0.99999
