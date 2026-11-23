@echo off
rem Start the search dashboard from this folder (guide: ..\README.md). Extra options go after the name,
rem e.g.  run.bat --port 8010   or   run.bat --no-vlm
setlocal
set "CODE=%~dp0"
set "ROOT=%~dp0.."
cd /d "%CODE%"
set "PYTHONPATH=%CODE%"
set "HF_HOME=%ROOT%\models\siglip2"
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set YOLO_OFFLINE=1
python -m sfs.web.server ^
  --bundle "%ROOT%\models\index" --previews "%ROOT%\meva\previews" --crops "%ROOT%\meva\crops" ^
  --weights "%ROOT%\models\yolo\yolov8m.pt" --adapter "%ROOT%\models\lora_adapter\adapter.pt" ^
  --vlm "%ROOT%\models\qwen3-vl-2b" --alerts "%ROOT%\my_alerts" ^
  --phrasings queries\phrasings.csv --static web\dist %*
