#!/usr/bin/env sh
# Start the search dashboard from this folder (guide: ../README.md). Extra options go after the name,
# e.g.  ./run.sh --port 8010   or   ./run.sh --no-vlm
set -e
CODE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$CODE")"
cd "$CODE"
export PYTHONPATH="$CODE"
export HF_HOME="$ROOT/models/siglip2"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1
exec python -m sfs.web.server \
  --bundle "$ROOT/models/index" --previews "$ROOT/meva/previews" --crops "$ROOT/meva/crops" \
  --weights "$ROOT/models/yolo/yolov8m.pt" --adapter "$ROOT/models/lora_adapter/adapter.pt" \
  --vlm "$ROOT/models/qwen3-vl-2b" --alerts "$ROOT/my_alerts" \
  --phrasings queries/phrasings.csv --static web/dist "$@"
