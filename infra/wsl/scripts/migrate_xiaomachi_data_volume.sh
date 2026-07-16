#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <compose-file>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
repo_root="$(cd "${WSL_DIR}/../.." && pwd)"
compose_file="$1"
volume_name="xiaomachi-bot-data"
marker=".migration-complete"

if ! docker volume inspect "${volume_name}" >/dev/null 2>&1; then
  docker volume create "${volume_name}" >/dev/null
fi

if docker run --rm -v "${volume_name}:/target" xiaomachi-bot:local \
  sh -ec "test -f /target/${marker}"; then
  exit 0
fi

echo "Migrating Xiaomachi runtime data into Docker volume ${volume_name}..."
mkdir -p "${repo_root}/data"

# SQLite must not be copied while the old Windows-mounted writer is active.
docker compose -f "${compose_file}" stop xiaomachi || true

docker run --rm \
  -v "${volume_name}:/target" \
  -v "${repo_root}/data:/source:ro" \
  xiaomachi-bot:local sh -ec '
    cp -a /source/. /target/
    python - <<"PY"
import sqlite3
from pathlib import Path

database = Path("/target/bot.db")
if database.exists():
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        result = connection.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        connection.close()
    if result != "ok":
        raise SystemExit(f"SQLite integrity check failed: {result}")
PY
    touch /target/.migration-complete
  '

echo "Xiaomachi runtime data migration completed. The Windows data directory remains untouched as a backup."
