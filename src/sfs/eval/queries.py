"""Query sets (spec-crops-queries-baseline §4). Ground truth comes only from MEVA human labels.

Q1 category: "find every instance of <activity>" over a split's whole corpus.
Q2 forensic: "find <activity> on camera C in hour H" — a metadata filter restricts the corpus to
             that camera-date-hour window; the relevant set is usually 1–5 instances.
Every phrasing is a separate query, so results also measure robustness to wording.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

MIN_EVAL_INSTANCES = 20  # D-012 option A


@dataclass(frozen=True)
class Query:
    qid: str
    kind: str                     # "q1" | "q2"
    activity: str
    phrasing_idx: int
    text: str
    corpus: frozenset             # instance ids searched
    relevant: frozenset           # instance ids that count as correct
    filter: dict = field(default_factory=dict, hash=False, compare=False)


def eligible_activities(manifest: pd.DataFrame, split_col: str = "split",
                        min_eval: int = MIN_EVAL_INSTANCES) -> list[str]:
    """Activity types with >= min_eval instances in Protocol A eval (fixed, then reused for B)."""
    counts = manifest.loc[manifest[split_col] == "eval", "activity"].value_counts()
    return sorted(counts[counts >= min_eval].index)


def load_phrasings(path: str) -> dict[str, list[str]]:
    df = pd.read_csv(path, comment="#")
    if set(df.columns) != {"activity", "phrasing"}:
        raise ValueError("phrasings file needs exactly columns: activity, phrasing")
    df["phrasing"] = df.phrasing.str.strip()
    if df.phrasing.eq("").any() or df.duplicated().any():
        raise ValueError("empty or duplicate phrasings")
    return {a: g.phrasing.tolist() for a, g in df.groupby("activity", sort=True)}


def check_phrasings(phrasings: dict[str, list[str]], activities: list[str]) -> None:
    missing = sorted(set(activities) - set(phrasings))
    extra = sorted(set(phrasings) - set(activities))
    if missing or extra:
        raise ValueError(f"phrasings mismatch — missing: {missing}, unexpected: {extra}")


def build_q1(manifest: pd.DataFrame, split_col: str, activities: list[str],
             phrasings: dict[str, list[str]]) -> list[Query]:
    ev = manifest[manifest[split_col] == "eval"]
    corpus = frozenset(ev.instance_id.tolist())
    out = []
    for a in activities:
        rel = frozenset(ev.loc[ev.activity == a, "instance_id"].tolist())
        if not rel:
            continue
        for j, text in enumerate(phrasings[a]):
            out.append(Query(f"q1|{split_col}|{a}|{j}", "q1", a, j, text, corpus, rel))
    return out


def build_q2(manifest: pd.DataFrame, split_col: str, activities: list[str],
             phrasings: dict[str, list[str]]) -> list[Query]:
    ev = manifest[manifest[split_col] == "eval"]
    out = []
    for (camera, date, hour), win in ev.groupby(["camera", "date", "hour"], sort=True):
        corpus = frozenset(win.instance_id.tolist())
        for a in activities:
            rel = frozenset(win.loc[win.activity == a, "instance_id"].tolist())
            if not rel:
                continue
            flt = {"camera": camera, "date": date, "hour": str(hour)}
            for j, text in enumerate(phrasings[a]):
                qid = f"q2|{split_col}|{a}|{camera}|{date}|{hour}|{j}"
                out.append(Query(qid, "q2", a, j, text, corpus, rel, flt))
    return out
