#!/usr/bin/env bash
set -e

echo "[SETUP] $(date) start"
VENV="$HOME/flagos-env"

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

echo "[SETUP] pip upgrage"
pip install --upgrade pip -q

echo "[SETUP] install vllm==0.24.0 (big download)"
pip install "vllm==0.24.0" -q

echo "[SETUP] install extra"
pip install "huggingface_hub>=0.26" -q

echo "[SETUP] verify torch cuda"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
print("device_count", torch.cuda.device_count())
PY

echo "[SETUP] verify vllm"
python -c "import vllm; print('vllm', vllm.__version__)"

echo "[SETUP] done $(date)"