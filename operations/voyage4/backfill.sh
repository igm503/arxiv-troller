#!/bin/bash
set -euo pipefail
ROOT=/home/arxiv/arxiv_troller
cd "$ROOT"
mkdir -p data/voyage4/backups data/voyage4/shards
BACKUP=data/voyage4/backups/pre-voyage4-embeddings.dump
if [ ! -f "$BACKUP" ]; then
  pg_dump -d arxiv --format=custom --compress=1 \
    --table=public.papers_embeddingvoyagebit2048 \
    --table=public.papers_embeddingvoyagehalf2048 \
    --table=public.papers_embeddingvoyagehalf256 \
    --table=public.papers_embeddinggeminihalf3072 \
    --table=public.papers_embeddinggeminihalf512 \
    --file="$BACKUP.partial"
  pg_restore --list "$BACKUP.partial" > data/voyage4/backups/contents.txt
  mv "$BACKUP.partial" "$BACKUP"
  chmod 444 "$BACKUP"
  sha256sum "$BACKUP" > data/voyage4/backups/pre-voyage4-embeddings.sha256
fi
exec "$ROOT/venv/bin/python" -u operations/voyage4/archive.py all --tpm 14000000 --rpm 3000 --workers 12
