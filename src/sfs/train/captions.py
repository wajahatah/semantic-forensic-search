"""Training caption bank and its disjointness from the evaluation phrasings (spec-lora-finetune D1).

Rule: no training caption may equal an evaluation phrasing, and within the SAME activity no caption
may be a near-duplicate of a phrasing (similarity >= 0.80). Similar wording across DIFFERENT
activities is allowed — it provides hard negatives (e.g. opening vs closing a door).
"""
from __future__ import annotations

import difflib
import re

import pandas as pd

MAX_SAME_ACTIVITY_SIMILARITY = 0.80


def load_captions(path: str) -> dict[str, list[str]]:
    df = pd.read_csv(path, comment="#")
    if set(df.columns) != {"activity", "caption"}:
        raise ValueError("caption file needs exactly columns: activity, caption")
    df["caption"] = df.caption.str.strip()
    if df.caption.eq("").any() or df.duplicated().any():
        raise ValueError("empty or duplicate captions")
    return {a: g.caption.tolist() for a, g in df.groupby("activity", sort=True)}


def normalise(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def similarity(a: str, b: str) -> float:
    """max(character-sequence ratio, word-set Jaccard) on normalised text."""
    a, b = normalise(a), normalise(b)
    wa, wb = set(a.split()), set(b.split())
    jac = len(wa & wb) / len(wa | wb) if wa | wb else 1.0
    return max(difflib.SequenceMatcher(None, a, b).ratio(), jac)


def disjointness_violations(captions: dict[str, list[str]], phrasings: dict[str, list[str]],
                            limit: float = MAX_SAME_ACTIVITY_SIMILARITY) -> list[tuple]:
    """[(activity, caption, phrasing, similarity)] that break the rule; empty when disjoint."""
    all_phrasings = {normalise(p) for ps in phrasings.values() for p in ps}
    out = []
    for act, caps in captions.items():
        for c in caps:
            if normalise(c) in all_phrasings:
                out.append((act, c, c, 1.0))
            for p in phrasings.get(act, []):
                s = similarity(c, p)
                if s >= limit and normalise(c) != normalise(p):
                    out.append((act, c, p, round(s, 3)))
    return out
