"""Frozen image/text encoders (spec-offline-embedding D5, D8; amended to compare SigLIP and SigLIP 2).

`encoder_id` hashes the model, its exact revision and the embedding configuration (sampling + crop
policy). Vectors with different ids live in different spaces and must never be mixed.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

ENCODERS = {
    "siglip": "google/siglip-base-patch16-224",
    "siglip2": "google/siglip2-base-patch16-224",
}
TEXT_MAX_LEN = 64  # SigLIP models are trained with max_length padding of 64 tokens


def make_encoder_id(identity: dict) -> str:
    blob = json.dumps(identity, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def _as_tensor(out):
    import torch
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("pooler_output", "image_embeds", "text_embeds"):
        val = getattr(out, attr, None)
        if isinstance(val, torch.Tensor):
            return val
    raise TypeError(f"cannot extract features from {type(out).__name__}")


def _normalise(t) -> np.ndarray:
    t = t.float()
    t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return t.cpu().numpy()


class Encoder:
    def __init__(self, name: str, device: str | None = None, revision: str | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor

        if name not in ENCODERS:
            raise ValueError(f"unknown encoder {name!r}; choose from {sorted(ENCODERS)}")
        self.name = name
        self.model_id = ENCODERS[name]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(self.model_id, revision=revision, dtype=self.dtype)
        self.model = self.model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id, revision=revision)
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision or "unpinned"

    @property
    def dim(self) -> int:
        return int(self.model.config.vision_config.hidden_size)

    def identity(self, embed_config: dict) -> dict:
        return {"encoder": self.name, "model_id": self.model_id, "revision": self.revision,
                "embed_config": embed_config}

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        """RGB uint8 HxWxC arrays -> (n, dim) float32, L2-normalised."""
        import torch
        if not images:
            return np.zeros((0, self.dim), np.float32)
        inputs = self.processor(images=images, return_tensors="pt")
        pixels = inputs["pixel_values"].to(self.device, self.dtype)
        with torch.inference_mode():
            return _normalise(_as_tensor(self.model.get_image_features(pixel_values=pixels)))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        import torch
        inputs = self.processor(text=texts, padding="max_length", max_length=TEXT_MAX_LEN,
                                truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()
                  if k in ("input_ids", "attention_mask")}
        with torch.inference_mode():
            return _normalise(_as_tensor(self.model.get_text_features(**inputs)))
