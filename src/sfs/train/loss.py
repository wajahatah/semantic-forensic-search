"""Label-aware sigmoid loss against caption prototypes (spec-lora-finetune D3).

Plain SigLIP training treats every other pair in the batch as a negative. Here many images share an
activity, so the target is defined by labels instead: an image is positive for its own activity's
caption and negative for every other activity's caption.
"""
from __future__ import annotations


def label_aware_sigmoid_loss(img_emb, cap_emb, labels, logit_scale, logit_bias):
    """img_emb (B, D) and cap_emb (C, D) L2-normalised; labels (B,) in [0, C).

    logits = exp(logit_scale) * img·capᵀ + logit_bias (SigLIP's parameterisation; logit_scale is
    stored as a log). Loss = binary cross-entropy summed over the C captions, averaged over the
    batch — the same normalisation as SigLIP's own loss.
    """
    import torch
    import torch.nn.functional as F

    logits = logit_scale.exp() * img_emb.float() @ cap_emb.float().T + logit_bias
    target = F.one_hot(labels.long(), num_classes=cap_emb.shape[0]).float()
    return F.binary_cross_entropy_with_logits(logits, target, reduction="sum") / img_emb.shape[0]
