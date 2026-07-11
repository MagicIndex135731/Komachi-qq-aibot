#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WSL_DIR}"

compose_exit=0
docker compose -f docker-compose.yml down --remove-orphans || compose_exit=$?
docker compose -f docker-compose.llbot.yml down --remove-orphans || compose_exit=$?

runtime_dir="${WSL_DIR}/runtime"
flag_file="${runtime_dir}/keepalive.enabled"
pid_file="${runtime_dir}/keepalive.pid"
rm -f "${flag_file}"
if [[ -f "${pid_file}" ]]; then
  existing_pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]]; then
    command_line="$(tr '\0' ' ' <"/proc/${existing_pid}/cmdline" 2>/dev/null || true)"
    if [[ "${command_line}" == *xiaomachi-wsl-keepalive* ]]; then
      kill -- "-${existing_pid}" 2>/dev/null || kill "${existing_pid}" 2>/dev/null || true
    fi
  fi
  rm -f "${pid_file}"
fi
pkill -f xiaomachi-wsl-keepalive 2>/dev/null || true
rm -f "${runtime_dir}"/onebot-watchdog-*.json{,.tmp,.lock}
exit "${compose_exit}"
