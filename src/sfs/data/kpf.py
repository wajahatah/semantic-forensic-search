"""Parsers for MEVA KPF annotation files (``*.activities.yml``, ``*.geom.yml``).

Frame numbers in KPF are 0-based and match OpenCV frame indices (``ts1 == ts0 / fps``).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import yaml

Box = tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
Tracks = dict[int, dict[int, Box]]  # actor id -> frame -> box

_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
_ACT_KEY = re.compile(r"^act\d+$")
_STEM_SUFFIXES = (".activities.yml", ".geom.yml", ".types.yml", ".regions.yml",
                  ".r13.avi", ".avi", ".mp4")


@dataclass(frozen=True)
class Activity:
    activity: str
    act_id: int
    frame_start: int
    frame_end: int
    actor_ids: tuple[int, ...]


@dataclass(frozen=True)
class ClipKey:
    date: str
    start: str
    end: str
    site: str
    camera: str

    @property
    def hour(self) -> str:
        return self.start[:2]


def clip_stem(path: str) -> str:
    """``.../2018-03-07.16-50-00.16-55-00.admin.G329.r13.avi`` -> ``2018-03-07.16-50-00.16-55-00.admin.G329``."""
    name = os.path.basename(path)
    for suffix in _STEM_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def parse_stem(stem: str) -> ClipKey:
    parts = stem.split(".")
    if len(parts) < 5:
        raise ValueError(f"unrecognised MEVA clip stem: {stem!r}")
    return ClipKey(*parts[:5])


def parse_activities(path: str) -> list[Activity]:
    with open(path, encoding="utf-8") as f:
        doc = yaml.load(f, Loader=_LOADER) or []
    out: list[Activity] = []
    for item in doc:
        act = item.get("act") if isinstance(item, dict) else None
        if not isinstance(act, dict):
            continue
        labels = next((v for k, v in act.items()
                       if _ACT_KEY.match(str(k)) and isinstance(v, dict) and v), None)
        if not labels:
            continue
        name = max(labels, key=labels.get)
        timespan = act.get("timespan") or []
        span = timespan[0].get("tsr0") if timespan and isinstance(timespan[0], dict) else None
        if not span or len(span) != 2:
            continue
        actors = tuple(sorted(int(a["id1"]) for a in (act.get("actors") or [])
                              if isinstance(a, dict) and "id1" in a))
        out.append(Activity(str(name), int(act.get("id2", -1)),
                            int(span[0]), int(span[1]), actors))
    return out


def parse_geom(path: str) -> Tracks:
    """Parse with a real YAML parser: the MEVA repo mixes styles for the same schema, e.g.
    ``{ geom: {id1: 6, ts0: 2478, g0: 487 902 967 1080} }`` and
    ``{'geom': {'g0': '1182 267 1326 540', 'id1': 1, 'ts0': 2677}}``. A text regex silently
    matched only the first style, so ~99% of clips lost their boxes."""
    with open(path, encoding="utf-8") as f:
        doc = yaml.load(f, Loader=_LOADER) or []
    tracks: Tracks = {}
    for item in doc:
        geom = item.get("geom") if isinstance(item, dict) else None
        if not isinstance(geom, dict) or not {"id1", "ts0", "g0"} <= geom.keys():
            continue
        coords = str(geom["g0"]).split()
        if len(coords) != 4:
            continue
        box = tuple(int(round(float(v))) for v in coords)
        tracks.setdefault(int(geom["id1"]), {})[int(geom["ts0"])] = box
    return tracks


def geom_path_for(activities_path: str) -> str:
    if not activities_path.endswith(".activities.yml"):
        raise ValueError(f"not an activities file: {activities_path}")
    return activities_path[: -len(".activities.yml")] + ".geom.yml"
