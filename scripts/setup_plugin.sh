#!/usr/bin/env bash
set -e

# Activate the expected venv if it exists, otherwise continue with the
# current python environment (e.g. a Colab or fresh WSL runtime).
if [ -f "$HOME/flagos-env/bin/activate" ]; then
  source "$HOME/flagos-env/bin/activate"
else
  echo "[PLUGIN] warning: ~/flagos-env not found; using current python"
fi

# Repo root derived from this script's location (portable across Colab/WSL/Windows).
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[PLUGIN] repo root: $BASE"
echo "[PLUGIN] install vllm-plugin-FL (python-only)"
cd "$BASE/vllm-plugin-FL"
pip install --no-build-isolation -e . 2>&1 | tail -5

echo "[PLUGIN] verify vllm_fl"
python -c "import vllm_fl; print('vllm_fl OK', getattr(vllm_fl, '__version__', '?'))"

echo "[GEMS] install flaggems"
cd "$BASE/FlagGems"
pip install --no-build-isolation -e . 2>&1 | tail -5

echo "[GEMS] verify flag_gems"
python -c "import flag_gems; print('flag_gems OK')"

echo "[DONE] plugin+gems ready $(date)"