"""One background worker that ingests uploaded videos, one at a time (the laptop has one GPU)."""
from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field


@dataclass
class Job:
    id: str
    name: str
    path: str
    classes: tuple[str, ...]
    state: str = "queued"          # queued | running | done | failed
    stage: str = "queued"          # detect | embed | done
    progress: float = 0.0
    message: str = ""
    video_id: int | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("path")
        d["classes"] = list(self.classes)
        return d


class JobRunner:
    def __init__(self, run):
        """`run(job, progress)` ingests job.path and returns the video id."""
        self._run = run
        self.jobs: dict[str, Job] = {}
        self._q: queue.Queue = queue.Queue()
        self._t = threading.Thread(target=self._loop, daemon=True, name="ingest")
        self._t.start()

    def submit(self, name: str, path: str, classes) -> Job:
        job = Job(uuid.uuid4().hex[:12], name, path, tuple(classes))
        self.jobs[job.id] = job
        self._q.put(job.id)
        return job

    def wait(self, job_id: str, timeout: float = 60.0) -> Job:
        end = time.time() + timeout
        while time.time() < end and self.jobs[job_id].state in ("queued", "running"):
            time.sleep(0.02)
        return self.jobs[job_id]

    def _loop(self):
        while True:
            job = self.jobs[self._q.get()]
            job.state = "running"

            def progress(stage, frac, msg, job=job):
                job.stage, job.progress, job.message = stage, float(frac), msg
            try:
                job.video_id = int(self._run(job, progress))
                job.state, job.stage, job.progress = "done", "done", 1.0
            except Exception as e:                          # reported to the page, never swallowed
                job.state, job.error = "failed", f"{type(e).__name__}: {e}"
                traceback.print_exc()
