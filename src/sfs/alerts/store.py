"""Local alert store (spec-video-alerts-dashboard §4.5), under data/alerts/ (git-ignored).

    alerts.sqlite        videos + alerts (metadata)
    vectors/<video>.npz  per video: LoRA tight + frozen tight frame embeddings, student features
    media/<video>/       the uploaded video and, per alert, clip.webm, thumb.jpg, frame_XX.jpg, tight_XX.png

A video's vectors are written atomically (tmp + replace) before its alerts are marked ready, so a crash
never leaves a half-ingested video searchable. Alert ids are positive in the store; the search engine
uses -id so they never collide with MEVA instance ids.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE, path TEXT NOT NULL,
    fps REAL, width INTEGER, height INTEGER, n_frames INTEGER, duration_s REAL,
    added TEXT NOT NULL, status TEXT NOT NULL, detector TEXT, classes TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY, video_id INTEGER NOT NULL REFERENCES videos(id), cls TEXT NOT NULL,
    near TEXT NOT NULL, track_id INTEGER NOT NULL, frame INTEGER NOT NULL, time_s REAL NOT NULL,
    conf REAL NOT NULL, start_frame INTEGER NOT NULL, end_frame INTEGER NOT NULL, frames TEXT NOT NULL,
    full INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS alerts_video ON alerts(video_id);
"""


@dataclass
class AlertCorpus:
    """Everything the search engine needs for the ready alerts, aligned row by row with `meta`."""
    ids: np.ndarray              # -alert_id, int64
    lora_emb: np.ndarray         # (frames, 768) float32
    lora_starts: np.ndarray      # (n + 1,)
    frozen_emb: np.ndarray
    frozen_starts: np.ndarray
    feats: np.ndarray            # (n, T, F) float16
    mask: np.ndarray             # (n, T) bool
    meta: pd.DataFrame           # alerts joined with their video's name and fps

    def __len__(self) -> int:
        return len(self.ids)


def _atomic_savez(path: str, **arrays) -> None:
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _starts(lengths) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(np.asarray(lengths, np.int64))]).astype(np.int64)


class AlertStore:
    def __init__(self, root: str = "data/alerts"):
        self.root = root
        os.makedirs(os.path.join(root, "vectors"), exist_ok=True)
        os.makedirs(os.path.join(root, "media"), exist_ok=True)
        self.db_path = os.path.join(root, "alerts.sqlite")
        self._lock = threading.Lock()
        self.version = 0                     # bumped on every change; the search engine reloads on change
        with self._con() as con:
            con.executescript(SCHEMA)
            # a video left 'processing' by a crash or restart is not searchable: mark it failed
            con.execute("UPDATE videos SET status='failed', error='interrupted' WHERE status='processing'")

    @contextmanager
    def _con(self):
        """A connection that commits on success, rolls back on error, and is always closed."""
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            with con:
                yield con
        finally:
            con.close()

    # ------------------------------------------------------------------ paths
    def media_dir(self, video_id: int) -> str:
        return os.path.join(self.root, "media", str(int(video_id)))

    def alert_dir(self, video_id: int, alert_id: int) -> str:
        return os.path.join(self.media_dir(video_id), f"alert_{int(alert_id)}")

    def vectors_path(self, video_id: int) -> str:
        return os.path.join(self.root, "vectors", f"{int(video_id)}.npz")

    # ------------------------------------------------------------------ videos
    def find_video(self, sha256: str) -> dict | None:
        with self._con() as con:
            r = con.execute("SELECT * FROM videos WHERE sha256=?", (sha256,)).fetchone()
        return dict(r) if r else None

    def add_video(self, name: str, sha256: str, info, detector: str, classes) -> int:
        with self._lock, self._con() as con:
            cur = con.execute(
                "INSERT INTO videos(name, sha256, path, fps, width, height, n_frames, duration_s, added, status,"
                " detector, classes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (name, sha256, "", info.fps, info.width, info.height, info.n_frames,
                 info.n_frames / info.fps if info.fps else None, time.strftime("%Y-%m-%d %H:%M:%S"),
                 "processing", detector, json.dumps(list(classes))))
            return int(cur.lastrowid)

    def set_video(self, video_id: int, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock, self._con() as con:
            con.execute(f"UPDATE videos SET {cols} WHERE id=?", (*fields.values(), int(video_id)))
        self.version += 1

    def videos(self) -> list[dict]:
        with self._con() as con:
            rows = con.execute(
                "SELECT v.*, (SELECT COUNT(*) FROM alerts a WHERE a.video_id=v.id) AS n_alerts "
                "FROM videos v ORDER BY v.id DESC").fetchall()
        return [dict(r) for r in rows]

    def delete_video(self, video_id: int) -> None:
        with self._lock, self._con() as con:
            con.execute("DELETE FROM alerts WHERE video_id=?", (int(video_id),))
            con.execute("DELETE FROM videos WHERE id=?", (int(video_id),))
        for p in (self.vectors_path(video_id),):
            if os.path.exists(p):
                os.remove(p)
        shutil.rmtree(self.media_dir(video_id), ignore_errors=True)
        self.version += 1

    # ------------------------------------------------------------------ alerts
    def write_alerts(self, video_id: int, alerts: list[dict], vectors: dict) -> list[int]:
        """Insert the video's alerts and their vectors; the video becomes 'ready'. Returns the alert ids.

        `alerts`: dicts with cls, near, track_id, frame, time_s, conf, start_frame, end_frame, frames, full.
        `vectors`: per alert (same order) lists `lora` (k, 768), `frozen` (k, 768), arrays `feats` (n, T, F),
        `mask` (n, T)."""
        if len(vectors["lora"]) != len(alerts) or len(vectors["frozen"]) != len(alerts):
            raise ValueError("one vector set per alert")
        with self._lock, self._con() as con:
            ids = []
            for a in alerts:
                cur = con.execute(
                    "INSERT INTO alerts(video_id, cls, near, track_id, frame, time_s, conf, start_frame, end_frame,"
                    " frames, full) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (int(video_id), a["cls"], ",".join(a["near"]), int(a["track_id"]), int(a["frame"]),
                     float(a["time_s"]), float(a["conf"]), int(a["start_frame"]), int(a["end_frame"]),
                     json.dumps([int(f) for f in a["frames"]]), int(bool(a["full"]))))
                ids.append(int(cur.lastrowid))
            empty = np.zeros((0, 768), np.float16)
            _atomic_savez(
                self.vectors_path(video_id), alert_id=np.array(ids, np.int64),
                lora_emb=np.concatenate(vectors["lora"]).astype(np.float16) if ids else empty,
                lora_starts=_starts([len(v) for v in vectors["lora"]]),
                frozen_emb=np.concatenate(vectors["frozen"]).astype(np.float16) if ids else empty,
                frozen_starts=_starts([len(v) for v in vectors["frozen"]]),
                feats=np.asarray(vectors["feats"], np.float16), mask=np.asarray(vectors["mask"], bool))
            con.execute("UPDATE videos SET status='ready', error=NULL WHERE id=?", (int(video_id),))
        self.version += 1
        return ids

    def alerts(self, video_id: int | None = None) -> pd.DataFrame:
        q = ("SELECT a.*, v.name AS video, v.fps AS fps, v.status AS video_status FROM alerts a "
             "JOIN videos v ON v.id=a.video_id")
        args: tuple = ()
        if video_id is not None:
            q, args = q + " WHERE a.video_id=?", (int(video_id),)
        with self._con() as con:
            return pd.read_sql_query(q + " ORDER BY a.video_id, a.frame, a.id", con, params=args)

    def corpus(self) -> AlertCorpus:
        """All alerts of ready videos, in (video, alert id) order."""
        meta = self.alerts()
        meta = meta[meta.video_status == "ready"].reset_index(drop=True)
        parts = {k: [] for k in ("ids", "lora", "lora_len", "frozen", "frozen_len", "feats", "mask")}
        rows = []
        for vid, grp in meta.groupby("video_id", sort=True):
            with np.load(self.vectors_path(vid), allow_pickle=False) as z:
                v = {k: z[k] for k in z.files}
            where = {int(a): n for n, a in enumerate(v["alert_id"])}
            for aid in grp.id:
                n = where[int(aid)]
                ls, fs = v["lora_starts"], v["frozen_starts"]
                parts["lora"].append(v["lora_emb"][ls[n]:ls[n + 1]])
                parts["lora_len"].append(ls[n + 1] - ls[n])
                parts["frozen"].append(v["frozen_emb"][fs[n]:fs[n + 1]])
                parts["frozen_len"].append(fs[n + 1] - fs[n])
                parts["feats"].append(v["feats"][n])
                parts["mask"].append(v["mask"][n])
                rows.append(int(aid))
        meta = meta.set_index("id").loc[rows].reset_index() if rows else meta
        f32 = lambda xs: np.concatenate(xs).astype(np.float32) if xs else np.zeros((0, 768), np.float32)
        t, f = (parts["feats"][0].shape if parts["feats"] else (16, 2 * 768 + 12))
        return AlertCorpus(
            ids=-np.array(rows, np.int64), lora_emb=f32(parts["lora"]), lora_starts=_starts(parts["lora_len"]),
            frozen_emb=f32(parts["frozen"]), frozen_starts=_starts(parts["frozen_len"]),
            feats=np.stack(parts["feats"]) if rows else np.zeros((0, t, f), np.float16),
            mask=np.stack(parts["mask"]) if rows else np.zeros((0, t), bool), meta=meta)
