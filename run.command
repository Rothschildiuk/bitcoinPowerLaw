#!/bin/zsh
set -e
cd "$(dirname "$0")"

if [ ! -x "venv/bin/streamlit" ]; then
  echo "[setup] Installing dependencies into venv..."
  make install
fi

echo "[run] Starting Bitcoin PowerLaw..."
make run
