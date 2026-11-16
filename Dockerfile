# syntax=docker/dockerfile:1
# Search dashboard image (docs/planning/spec-docker.md §4.1; how-to: docs/docker.md).
#
#   docker compose up                               # the `runtime` stage, with the GPU and the data mounts
#   docker build --target test -t sfs-test .        # the unit suite inside Linux (fails the build on a failure)
#
# Three stages: the front end is built with Node (same version as the laptop), the runtime holds the
# pinned Python packages + src/ + the built front end, and `test` adds pytest and the tests.
# Data and model weights are never copied in; see compose.yaml for the read-only mounts.

# ---------------------------------------------------------------- 1. front end (web/ -> web/dist)
FROM node:24.12.0-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---------------------------------------------------------------- 2. runtime
FROM python:3.12.15-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
# OpenCV's wheel needs these two shared libraries; nothing else from apt.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 sfs
WORKDIR /app
# Exact versions from the tested `dlproj` env (spec §4.2): requirements.txt names the direct dependencies and
# pulls in constraints.txt (`-c`, resolved next to it) for everything else; torch comes from the cu126 index
# named in the file.
COPY docker/requirements.txt docker/constraints.txt docker/
RUN pip install -r docker/requirements.txt && pip check
COPY pyproject.toml ./
COPY src/ src/
COPY queries/phrasings.csv queries/phrasings.csv
COPY --from=web /web/dist web/dist
# Offline at run time: the container never downloads anything after the build (spec §4.1).
ENV PYTHONPATH=/app/src \
    HF_HOME=/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    YOLO_OFFLINE=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics
RUN mkdir -p /alerts /hf-cache /tmp/ultralytics && chown -R sfs:sfs /alerts /tmp/ultralytics
USER sfs
EXPOSE 8000
CMD ["python", "-m", "sfs.web.server", "--host", "0.0.0.0", "--alerts", "/alerts"]

# ---------------------------------------------------------------- 3. tests (not the image that runs)
FROM runtime AS test
USER root
COPY docker/requirements-test.txt docker/requirements-test.txt
RUN pip install -r docker/requirements-test.txt && pip check
COPY tests/ tests/
COPY scripts/ scripts/
COPY queries/ queries/
COPY Dockerfile ./
RUN chown -R sfs:sfs /app
USER sfs
# `python -m pytest` puts /app on sys.path (tests import `tests.test_demo`); pyproject deselects the slow tests;
# -rs lists the tests that skip here (no GPU, no data folders) in the build log.
RUN python -m pytest -q -rs
