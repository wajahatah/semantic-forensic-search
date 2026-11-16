"""3D clip map for the dashboard (spec-video-alerts-dashboard §4.7, owner D5).

Each clip becomes one point: the mean of its LoRA frame embeddings, L2-normalised, projected onto the
first three principal components of the MEVA corpus. Alerts are projected with the same matrix, so they
land among the MEVA clips they resemble. The map is an illustration only: search never uses it.
"""
from __future__ import annotations

import numpy as np


def clip_means(emb: np.ndarray, starts: np.ndarray) -> np.ndarray:
    out = np.stack([emb[starts[i]:starts[i + 1]].mean(0) for i in range(len(starts) - 1)]) if len(starts) > 1 \
        else np.zeros((0, emb.shape[1]), np.float32)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


class ClipMap:
    def __init__(self, meva_emb: np.ndarray, meva_starts: np.ndarray):
        x = clip_means(np.asarray(meva_emb, np.float32), meva_starts).astype(np.float64)
        self.mean = x.mean(0)
        _, _, vt = np.linalg.svd(x - self.mean, full_matrices=False)
        self.components = vt[:3]
        for c in range(3):                  # fix each axis's sign, so the map looks the same every start
            if self.components[c, np.argmax(np.abs(self.components[c]))] < 0:
                self.components[c] *= -1
        raw = (x - self.mean) @ self.components.T
        self.scale = float(np.abs(raw).max()) or 1.0
        self.meva = raw / self.scale

    def project(self, emb: np.ndarray, starts: np.ndarray) -> np.ndarray:
        x = clip_means(np.asarray(emb, np.float32), starts).astype(np.float64)
        return (x - self.mean) @ self.components.T / self.scale if len(x) else np.zeros((0, 3))
