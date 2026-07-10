#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${WSL_DIR}"
compose_exit=0
docker compose down || compose_exit=$?

stop_keepalive() {
  local runtime_dir="${WSL_DIR}/runtime"
  local flag_file="${runtime_dir}/keepalive.enabled"
  local pid_file="${runtime_dir}/keepalive.pid"
  local existing_pid=""
  local command_line=""

  rm -f "${flag_file}"

  if [[ -f "${pid_file}" ]]; then
    existing_pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ "${existing_pid}" =~ ^[0-9]+$ ]]; then
      if [[ -r "/proc/${existing_pid}/cmdline" ]]; then
        command_line="$(tr '\0' ' ' <"/proc/${existing_pid}/cmdline" 2>/dev/null || true)"
      fi
      if [[ "${command_line}" == *xiaomachi-wsl-keepalive* ]]; then
        kill -- "-${existing_pid}" 2>/dev/null || kill "${existing_pid}" 2>/dev/null || true
        for _ in $(seq 1 20); do
          kill -0 "${existing_pid}" 2>/dev/null || break
          sleep 0.1
        done
      fi
    fi
    rm -f "${pid_file}"
  fi

  pkill -f xiaomachi-wsl-keepalive 2>/dev/null || true
  rm -f \
    "${runtime_dir}/onebot-watchdog.json" \
    "${runtime_dir}/onebot-watchdog.json.tmp" \
    "${runtime_dir}/onebot-watchdog.json.lock"
}

stop_keepalive
exit "${compose_exit}"
