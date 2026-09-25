#!/bin/bash
set -euo pipefail
cd /home/arxiv/arxiv_troller
PY=/home/arxiv/arxiv_troller/venv/bin/python
"$PY" -u operations/voyage4/archive.py all --tpm 14000000 --rpm 3000 --workers 12
"$PY" -u operations/voyage4/storage.py import
"$PY" -u operations/voyage4/storage.py rolling
"$PY" -u operations/voyage4/storage.py acknowledge
