#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${WSL_DIR}"
if [[ ! -f .env ]]; then
  echo "Missing ${WSL_DIR}/.env. Run bootstrap_wsl.sh and fill required secrets."
  exit 1
fi

start_keepalive() {
  local runtime_dir="${WSL_DIR}/runtime"
  local flag_file="${runtime_dir}/keepalive.enabled"
  local pid_file="${runtime_dir}/keepalive.pid"
  local existing_pid=""

  mkdir -p "${runtime_dir}"
  touch "${flag_file}"

  if [[ -f "${pid_file}" ]]; then
    existing_pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
      return 0
    fi
  fi

  nohup setsid bash -c 'exec -a xiaomachi-wsl-keepalive bash "$@"' _ \
    "${WSL_DIR}/scripts/keepalive.sh" \
    "${flag_file}" \
    "${pid_file}" \
    >/dev/null 2>&1 &
}

start_keepalive
docker compose up -d
bash "${WSL_DIR}/scripts/status.sh"
