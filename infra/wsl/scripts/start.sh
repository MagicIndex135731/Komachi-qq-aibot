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

open_napcat_login_if_needed() {
  local launcher_windows=""

  for _ in $(seq 1 30); do
    if curl -fsS --max-time 2 http://127.0.0.1:6099/ >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done

  if ! curl -fsS --max-time 2 http://127.0.0.1:6099/ >/dev/null 2>&1; then
    echo "NapCat WebUI is not ready; automatic login page was skipped."
    return 0
  fi
  if ! command -v powershell.exe >/dev/null 2>&1; then
    echo "Windows PowerShell is unavailable; automatic login page was skipped."
    return 0
  fi

  launcher_windows="$(wslpath -w "${WSL_DIR}/scripts/open_napcat_webui.ps1")"
  powershell.exe -NoProfile -ExecutionPolicy Bypass \
    -File "${launcher_windows}" -OnlyWhenLoginRequired >/dev/null 2>&1 || true
}

start_keepalive
docker compose up -d
open_napcat_login_if_needed
bash "${WSL_DIR}/scripts/status.sh"
