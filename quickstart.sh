#!/usr/bin/env bash
set -euo pipefail

SESSION_DIR="${1:-.}"
CONFIG_PATH="${2:-$SESSION_DIR/config.json}"

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt
python3 dnd_pipeline.py run "$SESSION_DIR" --config "$CONFIG_PATH"
