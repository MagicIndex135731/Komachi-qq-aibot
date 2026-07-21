#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKUP_ROOT="${XIAOMACHI_BACKUP_ROOT:-/var/backups/xiaomachi}"
timestamp="$(date +%Y%m%d-%H%M%S)"
backup_dir="${BACKUP_ROOT}/${timestamp}"

if [[ "${WSL_DIR}" != /mnt/* ]]; then
  # Installed releases have no Windows-backed source data to migrate.
  exit 0
fi

mkdir -p "${backup_dir}"
chmod 700 "${BACKUP_ROOT}" "${backup_dir}"

backup_tree() {
  local source_dir="$1"
  local archive_name="$2"
  [[ -d "${source_dir}" ]] || return 0
  tar -C "${source_dir}" -czf "${backup_dir}/${archive_name}.tar.gz" .
  (
    cd "${source_dir}"
    find . -type f -print0 | sort -z | xargs -0 -r sha256sum
  ) >"${backup_dir}/${archive_name}.sha256"
}

copy_tree_once() {
  local source_dir="$1"
  local volume_name="$2"
  [[ -d "${source_dir}" ]] || return 0
  docker volume inspect "${volume_name}" >/dev/null 2>&1 || docker volume create "${volume_name}" >/dev/null
  docker run --rm \
    -v "${source_dir}:/source:ro" \
    -v "${volume_name}:/target" \
    python:3.12-slim sh -eu -c '
      if [ ! -e /target/.migration-complete ]; then
        find /target -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
        cp -a /source/. /target/
      fi
    '
}

mark_volume_complete() {
  docker run --rm -v "$1:/target" python:3.12-slim touch /target/.migration-complete
}

# External volumes must exist before Compose can even resolve the new files for
# the stop operation below.  Creating an empty volume is harmless and prevents
# a validation failure from leaving an old writer running during backup.
docker volume inspect xiaomachi-bot-data >/dev/null 2>&1 || docker volume create xiaomachi-bot-data >/dev/null
docker volume inspect xiaomachi-llbot-data >/dev/null 2>&1 || docker volume create xiaomachi-llbot-data >/dev/null

# Stop every writer before taking a point-in-time copy.  Container names are
# used deliberately: Compose-file validation must not be able to bypass the
# stop.  Any migration error restarts exactly the containers that were running.
stopped_containers=()
migration_succeeded=false
restore_on_failure() {
  if [[ "${migration_succeeded}" != true && ${#stopped_containers[@]} -gt 0 ]]; then
    docker start "${stopped_containers[@]}" >/dev/null 2>&1 || true
  fi
}
trap restore_on_failure EXIT
for container_name in xiaomachi-bot xiaomachi-llbot xiaomachi-napcat; do
  if [[ "$(docker inspect -f '{{.State.Running}}' "${container_name}" 2>/dev/null || true)" == true ]]; then
    docker stop --time 30 "${container_name}" >/dev/null
    stopped_containers+=("${container_name}")
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "${container_name}" 2>/dev/null || true)" == true ]]; then
    echo "Refusing migration: ${container_name} is still running." >&2
    exit 1
  fi
done

docker run --rm -i -v xiaomachi-bot-data:/data python:3.12-slim python - <<'PY'
from pathlib import Path
import sqlite3

database = Path("/data/bot.db")
if database.exists():
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
        result = connection.execute("PRAGMA integrity_check").fetchone()
    if not result or result[0] != "ok":
        raise SystemExit(f"bot.db integrity check failed: {result!r}")
PY
docker run --rm \
  -v xiaomachi-bot-data:/source:ro \
  -v "${backup_dir}:/backup" \
  python:3.12-slim sh -eu -c '
    tar -C /source -czf /backup/xiaomachi-bot-data.tar.gz .
    cd /source
    find . -type f -print0 | sort -z | xargs -0 -r sha256sum > /backup/xiaomachi-bot-data.sha256
  '

backup_tree "${WSL_DIR}/runtime/llbot/data" "llbot-data"
backup_tree "${WSL_DIR}/runtime/napcat/config" "napcat-config"
backup_tree "${WSL_DIR}/runtime/napcat/logs" "napcat-logs"
backup_tree "${WSL_DIR}/runtime/napcat/ntqq" "napcat-ntqq"
backup_tree "${WSL_DIR}/runtime/logs" "xiaomachi-logs"
if [[ -f "${WSL_DIR}/.env" ]]; then
  install -m 600 "${WSL_DIR}/.env" "${backup_dir}/xiaomachi.env"
fi

copy_tree_once "${WSL_DIR}/runtime/llbot/data" "xiaomachi-llbot-data"
for mapping in \
  "${WSL_DIR}/runtime/napcat/config|xiaomachi-napcat-config" \
  "${WSL_DIR}/runtime/napcat/logs|xiaomachi-napcat-logs" \
  "${WSL_DIR}/runtime/napcat/cache|xiaomachi-napcat-cache" \
  "${WSL_DIR}/runtime/napcat/ntqq|xiaomachi-napcat-ntqq"; do
  source_dir="${mapping%%|*}"
  volume_name="${mapping#*|}"
  if [[ -d "${source_dir}" ]]; then
    copy_tree_once "${source_dir}" "${volume_name}"
    mark_volume_complete "${volume_name}"
  fi
done

# Merge historical bot logs into the existing Linux business-data volume.
if [[ -d "${WSL_DIR}/runtime/logs" ]]; then
  docker run --rm \
    -v "${WSL_DIR}/runtime/logs:/source:ro" \
    -v xiaomachi-bot-data:/target \
    python:3.12-slim sh -eu -c 'mkdir -p /target/logs; cp -an /source/. /target/logs/'
fi

# Validate LLBot only when its source configuration exists.  A clean install is
# initialized later by bootstrap_llbot_runtime.py; NapCat-only deployments do
# not require an LLBot file at all.
bot_qq="$(sed -n 's/^[[:space:]]*BOT_QQ[[:space:]]*=[[:space:]]*//p' "${WSL_DIR}/.env" | tail -n 1 | tr -d '\r')"
source_llbot_config="${WSL_DIR}/runtime/llbot/data/config_${bot_qq}.json"
if [[ -f "${source_llbot_config}" ]]; then
  if [[ ! "${bot_qq}" =~ ^[0-9]+$ ]]; then
    echo "BOT_QQ is invalid; cannot validate migrated LLBot data." >&2
    exit 1
  fi
  docker run --rm -v xiaomachi-llbot-data:/data python:3.12-slim \
    python -m json.tool "/data/config_${bot_qq}.json" >/dev/null
  mark_volume_complete xiaomachi-llbot-data
fi

migration_succeeded=true
trap - EXIT
echo "Linux-volume migration verified. Backup: ${backup_dir}"
