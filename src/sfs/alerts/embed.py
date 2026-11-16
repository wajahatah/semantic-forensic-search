"""Embeddings for alert crops — the same models and preprocessing as the MEVA stores (spec §4.4).

- LoRA tight: the selected adapter (Result 03, A-middle) on `preprocess_eval` crops, fp16 autocast, as
  `sfs.embed.reembed.embed_rows` computed the LoRA store.
- frozen tight / context: frozen SigLIP 2 through its own processor (`Encoder.embed_images`), as
  `sfs.embed.run` computed the baseline store. The search engine's encoder is shared, so the model
  is loaded once.
"""
from __future__ import annotations

import numpy as np

DEFAULT_ADAPTER = "runs/lora/A-middle/best/adapter.pt"


class ClipEmbedder:
    def __init__(self, encoder, adapter_path: str = DEFAULT_ADAPTER, device: str | None = None):
        from sfs.embed.reembed import load_encoder
        self.encoder = encoder
        self.device = device or encoder.device
        self.vision, self.info = load_encoder(adapter_path, self.device)

    def lora(self, crops: list[np.ndarray], batch: int = 64) -> np.ndarray:
        import torch

        from sfs.train.data import preprocess_eval, to_pixels
        from sfs.train.lora import image_features
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
            for s in range(0, len(crops), batch):
                x = np.stack([preprocess_eval(c) for c in crops[s:s + batch]])
                out.append(image_features(self.vision, to_pixels(x).to(self.device)).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

    def frozen(self, crops: list[np.ndarray], batch: int = 32) -> np.ndarray:
        out = [self.encoder.embed_images(crops[s:s + batch]) for s in range(0, len(crops), batch)]
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 768), np.float32)
