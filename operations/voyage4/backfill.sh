#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export VOYAGE4_ROOT="${VOYAGE4_ROOT:-$PWD/data/voyage4}"
mkdir -p "$VOYAGE4_ROOT/backups" "$VOYAGE4_ROOT/shards"
BACKUP="$VOYAGE4_ROOT/backups/pre-voyage4-embeddings.dump"
if [ ! -f "$BACKUP" ]; then
  pg_dump -d "${PGDATABASE:-arxiv}" --format=custom --compress=1 \
    --table=public.papers_embeddingvoyagebit2048 \
    --table=public.papers_embeddingvoyagehalf2048 \
    --table=public.papers_embeddingvoyagehalf256 \
    --table=public.papers_embeddinggeminihalf3072 \
    --table=public.papers_embeddinggeminihalf512 \
    --file="$BACKUP.partial"
  pg_restore --list "$BACKUP.partial" > "$VOYAGE4_ROOT/backups/contents.txt"
  mv "$BACKUP.partial" "$BACKUP"
  chmod 444 "$BACKUP"
  sha256sum "$BACKUP" > "$VOYAGE4_ROOT/backups/pre-voyage4-embeddings.sha256"
fi
exec "${VOYAGE4_PYTHON:-$PWD/venv/bin/python}" -u operations/voyage4/archive.py all --tpm 14000000 --rpm 3000 --workers 12
