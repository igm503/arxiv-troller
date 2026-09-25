#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${VOYAGE4_PYTHON:-$PWD/venv/bin/python}"
export VOYAGE4_ROOT="${VOYAGE4_ROOT:-$PWD/data/voyage4}"
mkdir -p "$VOYAGE4_ROOT"
exec 9>"$VOYAGE4_ROOT/maintenance.lock"
flock -n 9
"$PY" -u operations/voyage4/archive.py all
"$PY" -u operations/voyage4/storage.py import
"$PY" -u operations/voyage4/storage.py rolling
"$PY" -u operations/voyage4/storage.py acknowledge
"$PY" -u operations/voyage4/storage.py status
