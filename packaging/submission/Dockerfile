# syntax=docker/dockerfile:1
# Search dashboard image for the hand-in folder (guide: ../README.md, "Run with Docker").
# Holds the pinned Python packages, the `sfs` package and the built front end. Models and data are
# never copied in: compose.yaml mounts ../models and ../meva read-only.
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
COPY docker/requirements.txt docker/constraints.txt docker/
RUN pip install -r docker/requirements.txt && pip check
COPY sfs/ src/sfs/
COPY queries/phrasings.csv queries/phrasings.csv
COPY web/dist web/dist
# Offline at run time: the container never downloads anything after the build.
ENV PYTHONPATH=/app/src \
    HF_HOME=/models/siglip2 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    YOLO_OFFLINE=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics
RUN mkdir -p /alerts /tmp/ultralytics && chown -R sfs:sfs /alerts /tmp/ultralytics
USER sfs
EXPOSE 8000
CMD ["python", "-m", "sfs.web.server", "--host", "0.0.0.0", "--alerts", "/alerts", \
     "--bundle", "/models/index", "--previews", "/meva/previews", "--crops", "/meva/crops", \
     "--weights", "/models/yolo/yolov8m.pt", "--adapter", "/models/lora_adapter/adapter.pt", \
     "--vlm", "/models/qwen3-vl-2b"]
