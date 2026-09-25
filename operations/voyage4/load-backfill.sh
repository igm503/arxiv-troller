#!/bin/bash
set -euo pipefail
cd /home/arxiv/arxiv_troller
PY=/home/arxiv/arxiv_troller/venv/bin/python
while true; do
  if [ -f data/voyage4/progress.json ]; then
    "$PY" -u operations/voyage4/storage.py import
    "$PY" -u operations/voyage4/storage.py rolling
    "$PY" -u operations/voyage4/storage.py rolling-index
    if "$PY" -c 'import json;from pathlib import Path;p=json.loads(Path("data/voyage4/progress.json").read_text());raise SystemExit(0 if p["status"]=="complete" and p["missing"]==0 else 1)'; then
      "$PY" -u operations/voyage4/storage.py import
      "$PY" -u operations/voyage4/storage.py rolling
      exec "$PY" -u operations/voyage4/storage.py indexes
    fi
  fi
  sleep 60
done
