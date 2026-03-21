#!/bin/zsh
set -e
cd "$(dirname "$0")"

if [ ! -x "venv/bin/python" ]; then
  echo "[setup] Creating virtual environment..."
  python3 -m venv venv
fi

if ! venv/bin/python -m streamlit version >/dev/null 2>&1; then
  echo "[setup] Installing dependencies into venv..."
  make install
fi

echo "[run] Starting Bitcoin PowerLaw..."
make run
