"""LoRA on the SigLIP 2 vision tower (spec-lora-finetune D2).

Only the attention projections (q, k, v, out) of the vision encoder's transformer layers get
adapters; the pooling head, MLPs, embeddings and the whole text tower stay frozen.
"""
from __future__ import annotations

import hashlib

LORA_TARGETS = r".*encoder\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)"


def load_vision_tower(model_id: str, revision: str | None = None):
    """The pretrained vision tower (SiglipVisionModel incl. its pooling head) and logit scale/bias."""
    from transformers import AutoModel
    # "unpinned" is the placeholder sfs.embed.encoder records when no commit hash is known
    full = AutoModel.from_pretrained(model_id, revision=None if revision in (None, "unpinned") else revision)
    vision = full.vision_model
    scale, bias = full.logit_scale.detach().clone(), full.logit_bias.detach().clone()
    revision = getattr(full.config, "_commit_hash", None) or revision or "unpinned"
    del full
    return vision, scale, bias, revision


def wrap_lora(vision, rank: int = 16, alpha: int = 32, dropout: float = 0.1):
    """Freeze every base weight and add LoRA adapters to the attention projections."""
    from peft import LoraConfig, get_peft_model
    for p in vision.parameters():
        p.requires_grad_(False)
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS,
                     bias="none")
    return get_peft_model(vision, cfg)


def trainable_report(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if any("lora_" not in n for n in names):
        raise AssertionError("a non-LoRA parameter is trainable")
    return {"total": total, "trainable": train, "percent": round(100 * train / total, 3),
            "adapted_modules": len(names) // 2}


def adapter_state(model) -> dict:
    """Only the LoRA tensors, on CPU (what is checkpointed and hashed)."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if "lora_" in n}


def load_adapter_state(model, state: dict) -> None:
    params = dict(model.named_parameters())
    if set(state) != {n for n in params if "lora_" in n}:
        raise ValueError("adapter state does not match the model's LoRA parameters")
    import torch
    with torch.no_grad():
        for n, t in state.items():
            params[n].copy_(t.to(params[n].device, params[n].dtype))


def adapter_hash(state: dict) -> str:
    h = hashlib.sha256()
    for n in sorted(state):
        h.update(n.encode())
        h.update(state[n].float().numpy().tobytes())
    return h.hexdigest()[:12]


def image_features(vision, pixel_values):
    """L2-normalised image embeddings — what SiglipModel.get_image_features returns, normalised."""
    out = vision(pixel_values=pixel_values).pooler_output
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
