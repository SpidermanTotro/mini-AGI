#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
WEIGHTS_DIR="${WEIGHTS_DIR:-greenlight-16g-r1}"
SMOKE_MINUTES="${SMOKE_MINUTES:-1}"
PASSES="${PASSES:-1}"
VENV="$ROOT/.venv"

[[ -x "$VENV/bin/python" ]] || "$PYTHON" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip
# Install the PyTorch wheel matching your NVIDIA driver/CUDA from pytorch.org first if needed.
python -m pip install -r requirements.txt -r requirements-optional.txt
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; install a CUDA-enabled PyTorch build for this NVIDIA GPU.")
print(torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GiB")
PY
[[ -d data/train && -d data/val ]] || { echo "Build data/train and data/val first; see README.md." >&2; exit 1; }
free -h
df -h "$ROOT"
export GREENLIGHT_CONFIG="$ROOT/config-16gb.yaml"
mkdir -p runs
LOG="$ROOT/runs/train-greenlight-16g-r1.log"
echo "Weights: $WEIGHTS_DIR (existing manifest resumes; use a new directory for fresh weights)"
python -u train.py read data/train --weights-dir "$WEIGHTS_DIR" --held-out data/val \
  --save --passes "$PASSES" --minutes "$SMOKE_MINUTES" >"$LOG" 2>&1 &
TRAIN_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  date '+%H:%M:%S'
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
  tail -n 4 "$LOG"
  sleep 2
done
wait "$TRAIN_PID"
echo "Training finished. Log: $LOG"
python -m minagi.store "$WEIGHTS_DIR"
echo "Verifying complete Greenlight model artifact..."
python -m minagi.verify_model "$WEIGHTS_DIR"
EVAL_OUTPUT="$ROOT/runs/$(basename "$WEIGHTS_DIR")-eval-$(date +%Y%m%d-%H%M%S).json"
python -m minagi.evaluate --weights "$WEIGHTS_DIR" --output "$EVAL_OUTPUT"
