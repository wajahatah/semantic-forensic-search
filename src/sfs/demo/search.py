"""Search engine of the demo (spec-search-demo §4.3): free text in, ranked clips + cost out. No UI code.

Levels (the evaluated systems of Result 04, same code paths):
  fast      LoRA: mean of the top-3 frame similarities per clip
  standard  LoRA + student: z(student) + beta * z(LoRA), z over the whole corpus
  verified  standard, then the VLM scores the top k and they are re-ordered by
            z(VLM) + gamma * z(standard) (sfs.eval.cascade.topk_rerank); k VLM calls
System "frozen" = frozen SigLIP 2 (the baseline of Result 02), for the comparison view.
Filters restrict the result to a site / camera / day / hour window AFTER the whole corpus is scored,
exactly as the Q2 evaluation does, so scores equal the evaluated ones.
"""
from __future__ import annotations

import glob
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sfs.demo.index import file_sha
from sfs.eval.cascade import topk_rerank
from sfs.eval.metrics import rank
from sfs.eval.retrieval import InstanceIndex
from sfs.rerank.train import zcols
from sfs.teacher.frames import N_FRAMES, pick_even

LEVELS = ("fast", "standard", "verified")
SYSTEMS = ("final", "frozen")
COLLECTIONS = ("meva", "alerts", "both")
VLM_SECONDS_PER_CALL = 0.168          # Result 04, G0 uncached 2B Instruct on the laptop
DEFAULT_VLM = "data/models/Qwen3-VL-2B-Instruct"
MEAN_TOP_M = 3


@dataclass
class SearchResult:
    ids: np.ndarray                     # ranked clip ids (top `top` of the filtered corpus)
    scores: dict[str, np.ndarray]       # per stage, aligned with ids (missing stages absent)
    seconds: dict[str, float]           # per stage
    vlm_calls: int
    n_searched: int                     # clips left after the filters
    level: str
    system: str
    vlm_checked: set = field(default_factory=set)
    collection: str = "meva"
    bands: list = field(default_factory=list)   # high / medium / low: the score's place among all MEVA clips

    @property
    def vlm_seconds_per_call(self) -> float:
        """Measured in this search when the VLM ran; Result 04's laptop figure otherwise."""
        return self.seconds["vlm"] / self.vlm_calls if self.vlm_calls else VLM_SECONDS_PER_CALL

    @property
    def vlm_all_seconds(self) -> float:
        """What a VLM on every clip left after the filters would cost."""
        return self.n_searched * self.vlm_seconds_per_call


def _z_like(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """z-scores of `x` with the mean / spread of `ref`, computed exactly as zcols computes them for `ref`."""
    r = np.asarray(ref, np.float64)[:, None]
    return (np.asarray(x, np.float64) - r.mean(0)[0]) / np.maximum(r.std(0), 1e-9)[0]


class Bundle:
    def __init__(self, path: str, verify: bool = True):
        with open(os.path.join(path, "bundle.json"), encoding="utf-8") as f:
            self.info = json.load(f)
        if verify:
            for name, sha in self.info["files"].items():
                if file_sha(os.path.join(path, name)) != sha:
                    raise ValueError(f"{name} does not match bundle.json (rebuild with python -m sfs.demo.index)")
        with np.load(os.path.join(path, "vectors.npz"), allow_pickle=False) as z:
            v = {k: z[k] for k in z.files}
        self.ids = v["instance_ids"]
        self.lora = InstanceIndex(self.ids, v["lora_emb"].astype(np.float32), v["lora_starts"])
        self.frozen = InstanceIndex(self.ids, v["frozen_emb"].astype(np.float32), v["frozen_starts"])
        self.feats, self.mask = v["feats"], v["mask"]
        with sqlite3.connect(os.path.join(path, "meta.sqlite")) as con:
            clips = pd.read_sql_query("SELECT * FROM clips", con)
        clips = clips.set_index("instance_id").loc[self.ids].reset_index()
        if len(clips) != len(self.ids):
            raise ValueError("meta.sqlite and vectors.npz disagree")
        self.clips = clips
        self.path = path


class Searcher:
    def __init__(self, bundle: Bundle, crops_dir: str = "data/crops_v2", device: str | None = None,
                 encoder=None, vlm=None, vlm_path: str = DEFAULT_VLM, alerts=None):
        import torch
        self.b = bundle
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.crops_dir = crops_dir
        self._encoder, self._vlm, self.vlm_path = encoder, vlm, vlm_path
        self._student = None
        self._crops = None
        self.alerts = alerts                    # sfs.alerts.store.AlertStore or None
        self._alerts_cache = None
        self.rows = np.arange(len(bundle.ids))
        self.pos = {int(i): n for n, i in enumerate(bundle.ids)}

    # ------------------------------------------------------------------ models (lazy)
    @property
    def encoder(self):
        if self._encoder is None:
            from sfs.embed.encoder import Encoder
            self._encoder = Encoder(self.b.info["encoder"], device=self.device)
        return self._encoder

    @property
    def student(self):
        if self._student is None:
            import torch

            from sfs.rerank.model import TemporalReranker
            best = torch.load(os.path.join(self.b.path, "student.pt"), map_location="cpu", weights_only=False)
            cfg = best["config"]
            model = TemporalReranker(self.b.feats.shape[-1], self.b.lora.emb.shape[1], cfg["d"], cfg["layers"],
                                     cfg["heads"], self.b.feats.shape[1], cfg["dropout"])
            model.load_state_dict(best["model"])
            self._student = model.to(self.device)
        return self._student

    @property
    def crops(self):
        if self._crops is None:
            from sfs.teacher.frames import CropIndex
            self._crops = CropIndex(self.crops_dir, cache_shards=8, names=list(self.b.clips.crop_shard.unique()))
        return self._crops

    def vlm_available(self) -> bool:
        return self._vlm is not None or (self.device == "cuda" and os.path.isdir(self.vlm_path))

    @property
    def vlm(self):
        if self._vlm is None:
            if self.device != "cuda":
                raise RuntimeError("the Verified level needs a GPU (the VLM is too slow on the CPU)")
            from sfs.teacher.vlm import load_scorer
            self._vlm = load_scorer(self.vlm_path, self.b.info["vlm"]["kind"])
        return self._vlm

    # ------------------------------------------------------------------ alerts (your videos)
    def alert_corpus(self):
        """The ready alerts of the store, reloaded when the store changed; None without a store."""
        if self.alerts is None:
            return None
        if self._alerts_cache is None or self._alerts_cache[0] != self.alerts.version:
            version = self.alerts.version
            ac = self.alerts.corpus()
            idx = {"lora": InstanceIndex(ac.ids, ac.lora_emb, ac.lora_starts),
                   "frozen": InstanceIndex(ac.ids, ac.frozen_emb, ac.frozen_starts)}
            self._alerts_cache = (version, ac, idx, {int(i): n for n, i in enumerate(ac.ids)})
        return self._alerts_cache[1]

    def _alert_index(self, kind: str):
        return self._alerts_cache[2][kind]

    def alert_row(self, clip_id: int):
        """The alert's metadata row (clip ids of alerts are negative)."""
        self.alert_corpus()
        return self._alerts_cache[1].meta.iloc[self._alerts_cache[3][int(clip_id)]]

    @staticmethod
    def alert_mask(ac, classes=None, videos=None) -> np.ndarray:
        keep = np.ones(len(ac), bool)
        if classes:
            keep &= ac.meta.cls.isin(list(np.atleast_1d(classes))).to_numpy()
        if videos:
            keep &= ac.meta.video_id.astype(int).isin([int(v) for v in np.atleast_1d(videos)]).to_numpy()
        return keep

    def clip_frames(self, clip_id: int, k: int = N_FRAMES) -> tuple[list[np.ndarray], list[int], float]:
        """(tight RGB frames, their video frame numbers, video fps) for the VLM and the detail view."""
        if clip_id >= 0:
            f, n = self.crops.frames(int(clip_id), self.b.info["vlm"]["view"], k)
            return f, n, self.crops.fps_of(int(clip_id))
        import cv2
        row = self.alert_row(clip_id)
        d = self.alerts.alert_dir(row.video_id, row.id)
        paths = sorted(glob.glob(os.path.join(d, "tight_*.png")))
        nums = json.loads(row.frames)
        if not paths or len(paths) != len(nums):
            raise KeyError(f"alert {row.id} has no tight crops on disk")
        picked = pick_even(len(paths), k)
        frames = [cv2.cvtColor(cv2.imread(paths[i]), cv2.COLOR_BGR2RGB) for i in picked]
        return frames, [int(nums[i]) for i in picked], float(row.fps)

    def _positions(self, all_ids: np.ndarray) -> dict[int, int]:
        if len(all_ids) == len(self.b.ids):
            return self.pos
        out = dict(self.pos)
        out.update({int(i): len(self.b.ids) + n for n, i in enumerate(all_ids[len(self.b.ids):])})
        return out

    # ------------------------------------------------------------------ search
    def embed(self, text: str) -> np.ndarray:
        return np.asarray(self.encoder.embed_texts([text]), np.float32)[0]

    def filter_mask(self, filters: dict | None) -> np.ndarray:
        c, keep = self.b.clips, np.ones(len(self.b.ids), bool)
        for col in ("site", "camera", "date"):
            vals = (filters or {}).get(col)
            if vals:
                keep &= c[col].astype(str).isin([str(v) for v in np.atleast_1d(vals)]).to_numpy()
        hours = (filters or {}).get("hour")                         # exact hour(s), as in a Q2 window
        if hours is not None and str(hours) != "":
            keep &= c.hour.astype(int).isin([int(h) for h in np.atleast_1d(hours)]).to_numpy()
        lo, hi = (filters or {}).get("hour_from"), (filters or {}).get("hour_to")
        if lo is not None:
            keep &= c.hour.astype(int).to_numpy() >= int(lo)
        if hi is not None:
            keep &= c.hour.astype(int).to_numpy() <= int(hi)
        return keep

    def _sync(self):
        if self.device == "cuda":
            import torch
            torch.cuda.synchronize()

    def search_vector(self, q: np.ndarray, text: str = "", level: str = "standard", k: int = 10,
                      filters: dict | None = None, top: int = 50, system: str = "final",
                      collection: str = "meva", classes=None, videos=None) -> SearchResult:
        """`q`: the L2-normalised query vector; `text`: the query as typed (the VLM's question).

        `collection`: "meva", "alerts" (your videos) or "both". MEVA is always scored in full and its
        z-statistics are the reference for the alerts, so MEVA scores never depend on the alerts.
        `filters` (site / camera / date / hour) apply to MEVA clips, `classes` / `videos` to alerts."""
        from sfs.rerank.train import score_matrix
        if level not in LEVELS or system not in SYSTEMS or collection not in COLLECTIONS:
            raise ValueError(f"level must be one of {LEVELS}, system one of {SYSTEMS}, "
                             f"collection one of {COLLECTIONS}")
        q = np.asarray(q, np.float32).reshape(1, -1)
        seconds, scores = {}, {}
        ac = self.alert_corpus() if collection != "meva" else None
        n_m, n_a = len(self.b.ids), (len(ac) if ac is not None else 0)
        all_ids = np.concatenate([self.b.ids, ac.ids]) if n_a else self.b.ids
        keep_m = self.filter_mask(filters) if collection != "alerts" else np.zeros(n_m, bool)
        keep_a = self.alert_mask(ac, classes, videos) if n_a else np.zeros(0, bool)
        keep = np.concatenate([keep_m, keep_a])
        ids = all_ids[keep]
        if system == "frozen":
            t0 = time.perf_counter()
            fm = self.b.frozen.score(q, MEAN_TOP_M)[:, 0].astype(np.float64)
            fa = self._alert_index("frozen").score(q, MEAN_TOP_M)[:, 0].astype(np.float64) if n_a else fm[:0]
            seconds["frozen"] = time.perf_counter() - t0
            scores["frozen"] = base = np.concatenate([fm, fa])
            ref = fm
            level = "fast"
        else:
            t0 = time.perf_counter()
            lm = self.b.lora.score(q, MEAN_TOP_M)[:, 0]
            la = self._alert_index("lora").score(q, MEAN_TOP_M)[:, 0] if n_a else lm[:0]
            seconds["lora"] = time.perf_counter() - t0
            scores["lora"] = base = np.concatenate([lm, la]).astype(np.float64)
            ref = base[:n_m]
            if level in ("standard", "verified"):
                self._sync()
                t0 = time.perf_counter()
                sm = score_matrix(self.student, self.b.feats, self.b.mask, self.rows, q, self.device)[:, 0]
                sa = (score_matrix(self.student, ac.feats, ac.mask, np.arange(n_a), q, self.device)[:, 0]
                      if n_a else sm[:0])
                self._sync()
                seconds["student"] = time.perf_counter() - t0
                beta = self.b.info["beta"]
                fused_m = zcols(sm[:, None])[:, 0] + beta * zcols(lm[:, None])[:, 0]
                fused_a = _z_like(sa, sm) + beta * _z_like(la, lm)
                scores["student"] = np.concatenate([sm, sa]).astype(np.float64)
                scores["standard"] = base = np.concatenate([fused_m, fused_a])
                ref = fused_m
        p95, p99 = np.percentile(ref, [95, 99]) if len(ref) else (np.inf, np.inf)
        base_f = base[keep]
        final = base_f
        checked: set = set()
        calls = 0
        pos = self._positions(all_ids)
        if level == "verified" and len(ids):
            k = min(int(k), len(ids))
            cand = rank(ids, base_f)[:k]
            t0 = time.perf_counter()
            frames = [self.clip_frames(int(i)) for i in cand]
            seconds["frames"] = time.perf_counter() - t0
            vlm = self.vlm                                      # loading (first use) is not search time
            self._sync()
            t0 = time.perf_counter()
            margins = np.array([vlm.score_clip(f, n, fps, [text])[0] for f, n, fps in frames], np.float64)
            self._sync()
            seconds["vlm"] = time.perf_counter() - t0
            final = topk_rerank(ids, base_f, margins, k, self.b.info["vlm"]["gamma"])
            calls, checked = k, {int(i) for i in cand}
            scores["vlm"] = np.full(len(all_ids), np.nan)            # NaN = not checked by the VLM
            scores["vlm"][[pos[int(i)] for i in cand]] = margins
        order = rank(ids, final)[:top]
        at = np.array([pos[int(i)] for i in order], np.int64)      # rows in the whole corpus
        out_scores = {name: col[at] for name, col in scores.items()}
        full_final = np.full(len(all_ids), np.nan)
        full_final[keep] = final
        out_scores["final"] = full_final[at]
        bands = ["high" if v >= p99 else "medium" if v >= p95 else "low" for v in base[at]]
        return SearchResult(order, out_scores, seconds, calls, int(len(ids)), level, system, checked,
                            collection, bands)

    def search(self, text: str, **kw) -> SearchResult:
        t0 = time.perf_counter()
        q = self.embed(text)
        dt = time.perf_counter() - t0
        res = self.search_vector(q, text, **kw)
        res.seconds = {"text": dt, **res.seconds}
        return res
