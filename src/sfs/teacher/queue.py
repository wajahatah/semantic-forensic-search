"""Run several VLM jobs in ONE process that holds its GPU memory from the first job to the last
(D-041). Separate processes give the memory back between jobs, and on the shared office GPU the
team's Ollama reloads into that gap (seen 2026-11-03: the 7B then had to wait). Here each model is
released with `del` but its memory stays in PyTorch's cache (no `empty_cache`), so the next model
loads into memory this process already holds.

Jobs file (JSON list), relative paths from the working directory:
  [{"type": "g0", "model": "...", "kind": "instruct", "view": "context", "frames": 8, "max_vram_gb": 15,
    "out": "runs/cascade/g0_x.json", "fp8_at_load": false},
   {"type": "score", "model": "...", "kind": "instruct", "view": "tight", "requests": "...", "subset": "val",
    "out": "data/teacher/...", "frames": 8, "fp8_at_load": false}]
Optional per job: "gpu_budget_gb" (CPU/GPU split), "clip_batch" (questions per cached batch), "pin"
(true / false / omitted = decide from free RAM).
A failing job is logged with its error and the queue continues (gates never stop the work).

CLI:
  python -m sfs.teacher.queue jobs.json --reserve-gb 14.5 [--manifest ... --captions ... --crops ...]
  python -m sfs.teacher.queue --watch runs/queue --reserve-gb 14.5      # hold the GPU until told to stop
Watch mode (owner, 2026-11-03: "do not lose the memory till our whole experimentation is done"): the
process reserves the memory once and keeps it while idle; every `*.json` dropped into the watched
folder is run in name order (renamed `.running`, then `.done` / `.failed`, with a `.status.json`); a
file named `STOP` in the folder ends the process and frees the memory.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback

JOB_TYPES = ("g0", "score")


def validate(jobs: list) -> None:
    need = {"g0": ("model", "kind", "out"), "score": ("model", "kind", "view", "requests", "subset", "out")}
    for n, j in enumerate(jobs):
        if j.get("type") not in JOB_TYPES:
            raise ValueError(f"job {n}: type must be one of {JOB_TYPES}")
        missing = [k for k in need[j["type"]] if k not in j]
        if missing:
            raise ValueError(f"job {n} ({j['type']}): missing {missing}")


def run_queue(jobs: list, load, do_g0, do_score, reserve=None, log=print) -> list[dict]:
    """Run `jobs` in order. `load(job)` builds a scorer; `do_g0(scorer, job)` / `do_score(scorer, job)`
    run one job; `reserve()` is called once before the first load. Returns one status per job."""
    validate(jobs)
    if reserve is not None:
        log(f"holding {reserve():.1f} GB of GPU memory for the whole queue")
    status = []
    for n, job in enumerate(jobs):
        name = f"{n + 1}/{len(jobs)} {job['type']} {os.path.basename(os.path.normpath(job['model']))} -> {job['out']}"
        log(f"=== {name} start {time.strftime('%F %T')}")
        t0, scorer = time.time(), None
        try:
            scorer = load(job)
            (do_g0 if job["type"] == "g0" else do_score)(scorer, job)
            status.append({"job": name, "ok": True, "seconds": round(time.time() - t0, 1)})
            log(f"=== {name} end {time.strftime('%F %T')} ok")
        except Exception as e:                      # noqa: BLE001 - record, continue with the next job
            log(traceback.format_exc())
            status.append({"job": name, "ok": False, "error": f"{type(e).__name__}: {e}"[:500]})
            log(f"=== {name} end {time.strftime('%F %T')} FAILED {type(e).__name__}")
        finally:
            del scorer                              # release the model; memory stays cached in this process
            gc.collect()
    return status


def watch(folder: str, run_jobs, sleep=time.sleep, poll_s: float = 30.0, log=print) -> int:
    """Run job files dropped into `folder` until a STOP file appears; returns the number of files run."""
    os.makedirs(folder, exist_ok=True)
    done = 0
    while not os.path.exists(os.path.join(folder, "STOP")):
        todo = sorted(f for f in os.listdir(folder) if f.endswith(".json") and not f.endswith(".status.json"))
        if not todo:
            sleep(poll_s)
            continue
        path = os.path.join(folder, todo[0])
        running = path[:-5] + ".running"
        os.replace(path, running)
        try:
            status = run_jobs(json.load(open(running)))
            ok = all(s["ok"] for s in status)
        except Exception as e:                      # noqa: BLE001 - a bad file must not end the holder
            status, ok = [{"job": todo[0], "ok": False, "error": f"{type(e).__name__}: {e}"[:500]}], False
        json.dump(status, open(path[:-5] + ".status.json", "w"), indent=1)
        os.replace(running, path[:-5] + (".done" if ok else ".failed"))
        log(f"=== file {todo[0]} {'done' if ok else 'FAILED'}")
        done += 1
    log("STOP found; releasing the GPU")
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jobs", nargs="?", help="jobs file (or use --watch)")
    ap.add_argument("--watch", help="folder to watch for job files; the GPU is held until a STOP file appears")
    ap.add_argument("--reserve-gb", type=float, default=0.0)
    ap.add_argument("--manifest", default="manifest_v2_drive/manifest_v2.csv")
    ap.add_argument("--captions", default="queries/train_captions.csv")
    ap.add_argument("--crops", default="data/crops_v2")
    a = ap.parse_args(argv)
    if bool(a.jobs) == bool(a.watch):
        ap.error("give a jobs file or --watch, not both")
    log = lambda m: print(m, flush=True)            # noqa: E731

    from sfs.teacher.frames import N_FRAMES, CropIndex
    from sfs.teacher.run import make_scorer, reserve_gpu, run
    crops = CropIndex(a.crops)
    cache = {}

    def load(job):
        return make_scorer(job["model"], None, job["kind"], job.get("gpu_budget_gb"), job.get("fp8_at_load", False),
                           job.get("clip_batch"), job.get("pin"))

    def do_g0(scorer, job):
        from sfs.data.manifest import read_manifest
        from sfs.teacher.gate import g0
        from sfs.train.captions import load_captions
        if "manifest" not in cache:
            cache["manifest"], cache["captions"] = read_manifest(a.manifest), load_captions(a.captions)
        rep = g0(scorer, crops, cache["manifest"], cache["captions"], view=job.get("view", "context"),
                 n_frames=job.get("frames"), max_vram_gb=job.get("max_vram_gb", 7.5))
        os.makedirs(os.path.dirname(os.path.abspath(job["out"])), exist_ok=True)
        json.dump(rep, open(job["out"], "w"), indent=1)
        log(json.dumps({k: v for k, v in rep.items() if k != "clips"}))

    def do_score(scorer, job):
        run(scorer, crops, job["requests"], job["view"], job["out"], job["subset"], job.get("frames", N_FRAMES), log=log)

    if a.reserve_gb:
        log(f"holding {reserve_gpu(a.reserve_gb):.1f} GB of GPU memory")
    if a.watch:
        watch(a.watch, lambda jobs: run_queue(jobs, load, do_g0, do_score, log=log), log=log)
        return 0
    status = run_queue(json.load(open(a.jobs)), load, do_g0, do_score, log=log)
    log("QUEUE DONE " + json.dumps(status))
    return 0 if all(s["ok"] for s in status) else 1


if __name__ == "__main__":
    sys.exit(main())
