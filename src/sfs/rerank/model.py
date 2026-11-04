"""Student re-ranker: a small transformer over a candidate's frame sequence, conditioned on the
query (spec-rerank-cascade D6). ~1.5-2 M parameters; output = one relevance logit."""
from __future__ import annotations

import torch
from torch import nn


class TemporalReranker(nn.Module):
    def __init__(self, frame_dim: int, text_dim: int = 768, d: int = 256, layers: int = 2,
                 heads: int = 4, t_max: int = 16, dropout: float = 0.1):
        super().__init__()
        self.frame_in = nn.Sequential(nn.LayerNorm(frame_dim), nn.Linear(frame_dim, d))
        self.query_in = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d))
        self.pos = nn.Parameter(torch.zeros(t_max, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, frames: torch.Tensor, mask: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """frames (B, T, F), mask (B, T) True = real frame, text (B, text_dim) -> logits (B,)."""
        t = frames.shape[1]
        x = self.frame_in(frames) + self.pos[:t]
        q = self.query_in(text).unsqueeze(1)
        seq = torch.cat([q, x], dim=1)
        pad = torch.cat([torch.zeros_like(mask[:, :1]), ~mask], dim=1)   # the query token is never padding
        out = self.encoder(seq, src_key_padding_mask=pad)
        return self.head(out[:, 0]).squeeze(-1)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
