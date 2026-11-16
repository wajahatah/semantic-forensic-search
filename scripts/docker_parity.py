"""Search parity between two runs of the dashboard server (spec-docker §5.2): the conda run on the
laptop against the Docker container, on the same data.

    # 1. laptop run (python -m sfs.web.server), then:
    python scripts/docker_parity.py record --out runs/docker/laptop.json
    # 2. stop it; docker compose up; then:
    python scripts/docker_parity.py record --out runs/docker/docker.json
    # 3.
    python scripts/docker_parity.py compare runs/docker/laptop.json runs/docker/docker.json

`record` asks the server every frozen phrasing (queries/phrasings.csv) at Fast and Standard over MEVA
(top 50), and the first 5 phrasings at Verified with k = 10 when the VLM is loaded. It keeps, per query,
the returned ids, the final scores, the label matches and the measured seconds.

`compare` applies the rule fixed in the spec before any measurement, with the first file as the reference:
- every query: Success@1 and Success@10 (from the label matches) identical;
- Fast and Standard: every score of a clip both runs returned within `--tol` (1e-3) of the reference, and
  the top-10 order may differ only where the reference's score gap is below `--tol` (a near-tie);
- Verified: the set of the k clips the VLM checked is identical (the stage before it must agree); their
  order and the bf16 VLM probabilities are reported, not gated (Result 04 measured that noise at 0.37
  in the raw logit, rank correlation 0.995).
It also reports the median seconds per search of both runs (spec §4.5 risk 2).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

LEVELS = ("fast", "standard")
VERIFIED_N = 5
TOP = 50
K = 10


def http_json(url: str, body: dict | None = None, timeout: float = 600) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if body else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def record(fetch, phrasings: dict[str, list[str]], vlm: bool, verified_n: int = VERIFIED_N,
           k: int = K, top: int = TOP) -> list[dict]:
    """`fetch(body) -> /api/search answer`. Returns one entry per (phrasing, level)."""
    texts = [t for a in sorted(phrasings) for t in phrasings[a]]
    plan = [(t, lv) for lv in LEVELS for t in texts]
    if vlm:
        plan += [(t, "verified") for t in texts[:verified_n]]
    out = []
    for text, level in plan:
        j = fetch({"query": text, "collection": "meva", "level": level, "k": k, "top": top})
        res = j["results"]
        out.append({"query": text, "level": level, "k": k,
                    "ids": [int(r["id"]) for r in res],
                    "scores": [None if r["scores"].get("final") is None else float(r["scores"]["final"]) for r in res],
                    "match": [bool(r.get("match")) for r in res],
                    "vlm_prob": [r.get("vlm_prob") for r in res],
                    "seconds": float(j["cost"]["total"]), "vlm_calls": int(j["cost"]["vlm_calls"])})
    return out


def success(match: list[bool], k: int) -> bool:
    return any(match[:k])


def compare(ref: dict, other: dict, tol: float = 1e-3) -> dict:
    """The rule above; `ref` is the reference run (the laptop)."""
    rq = {(q["query"], q["level"]): q for q in ref["queries"]}
    oq = {(q["query"], q["level"]): q for q in other["queries"]}
    problems: list[str] = []
    if set(rq) != set(oq):
        problems.append(f"different query sets: {len(rq)} vs {len(oq)} entries "
                        f"(only in reference: {len(set(rq) - set(oq))}, only in other: {len(set(oq) - set(rq))})")
    max_diff, swaps, checked = 0.0, 0, 0
    for key in sorted(rq):
        if key not in oq:
            continue
        a, b = rq[key], oq[key]
        name = f"{key[1]} · {key[0]!r}"
        for kk in (1, 10):
            if success(a["match"], kk) != success(b["match"], kk):
                problems.append(f"{name}: Success@{kk} {success(a['match'], kk)} -> {success(b['match'], kk)}")
        if a["level"] == "verified":
            checked += 1
            if set(a["ids"][:a["k"]]) != set(b["ids"][:b["k"]]):
                problems.append(f"{name}: the VLM checked a different set of clips")
            continue
        sa = {i: s for i, s in zip(a["ids"], a["scores"]) if s is not None}
        sb = {i: s for i, s in zip(b["ids"], b["scores"]) if s is not None}
        common = sa.keys() & sb.keys()
        if not common:
            problems.append(f"{name}: no clip in common")
            continue
        worst = max(abs(sa[i] - sb[i]) for i in common)
        max_diff = max(max_diff, worst)
        if worst > tol:
            problems.append(f"{name}: a score differs by {worst:.2e} (> {tol:g})")
        for r in range(min(10, len(a["ids"]), len(b["ids"]))):
            ia, ib = a["ids"][r], b["ids"][r]
            if ia == ib:
                continue
            if ib not in sa or abs(sa[ia] - sa[ib]) >= tol:
                problems.append(f"{name}: rank {r + 1} is clip {ib} instead of {ia}, not a near-tie")
            else:
                swaps += 1
    sec = lambda run, lv: [q["seconds"] for q in run["queries"] if q["level"] == lv]  # noqa: E731
    timing = {lv: {"ref": statistics.median(sec(ref, lv)) if sec(ref, lv) else None,
                   "other": statistics.median(sec(other, lv)) if sec(other, lv) else None}
              for lv in LEVELS + ("verified",)}
    return {"ok": not problems, "problems": problems, "queries": len(rq), "verified_checked": checked,
            "max_score_diff": max_diff, "near_tie_swaps": swaps, "tol": tol, "median_seconds": timing}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="query a running server and save its answers")
    r.add_argument("--url", default="http://127.0.0.1:8000")
    r.add_argument("--out", required=True)
    r.add_argument("--phrasings", default="queries/phrasings.csv")
    r.add_argument("--verified", type=int, default=VERIFIED_N, help="phrasings to run at Verified (0 = none)")
    c = sub.add_parser("compare", help="apply the parity rule to two recordings")
    c.add_argument("reference")
    c.add_argument("other")
    c.add_argument("--tol", type=float, default=1e-3)
    c.add_argument("--out", help="write the comparison as JSON")
    a = ap.parse_args(argv)

    if a.cmd == "record":
        sys.path.insert(0, "src")
        from sfs.eval.queries import load_phrasings
        status = http_json(a.url + "/api/status")
        phr = load_phrasings(a.phrasings)
        t0 = time.time()
        queries = record(lambda body: http_json(a.url + "/api/search", body), phr, bool(status["vlm"]), a.verified)
        out = {"url": a.url, "recorded": time.strftime("%Y-%m-%d %H:%M:%S"), "status": status,
               "wall_seconds": round(time.time() - t0, 1), "queries": queries}
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f)
        print(f"{len(queries)} searches recorded in {out['wall_seconds']} s -> {a.out} "
              f"(meva_clips {status['meva_clips']}, vlm {status['vlm']}, device {status['device']})")
        return 0

    with open(a.reference, encoding="utf-8") as f:
        ref = json.load(f)
    with open(a.other, encoding="utf-8") as f:
        other = json.load(f)
    res = compare(ref, other, a.tol)
    for p in res["problems"]:
        print("  ", p)
    print(f"{res['queries']} queries, max score diff {res['max_score_diff']:.2e}, near-tie swaps {res['near_tie_swaps']}, "
          f"verified sets checked {res['verified_checked']}")
    for lv, t in res["median_seconds"].items():
        if t["ref"] is not None and t["other"] is not None:
            print(f"   median {lv}: {t['ref']:.3f} s -> {t['other']:.3f} s ({t['other'] / max(t['ref'], 1e-9):.2f}x)")
    print("PARITY OK" if res["ok"] else "PARITY FAILED")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
