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
watchdog_pid_file="${pid_file}.watchdog"
rm -f "${flag_file}"
watchdog_pid="$(cat "${watchdog_pid_file}" 2>/dev/null || true)"
if [[ "${watchdog_pid}" =~ ^[0-9]+$ ]]; then
  watchdog_command="$(tr '\0' ' ' <"/proc/${watchdog_pid}/cmdline" 2>/dev/null || true)"
  if [[ "${watchdog_command}" == *onebot_watchdog.py* ]]; then
    kill -TERM "${watchdog_pid}" 2>/dev/null || true
  fi
fi
if [[ -f "${pid_file}" ]]; then
  existing_pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]]; then
    command_line="$(tr '\0' ' ' <"/proc/${existing_pid}/cmdline" 2>/dev/null || true)"
    if [[ "${command_line}" == *xiaomachi-wsl-keepalive* ]]; then
      kill -- "-${existing_pid}" 2>/dev/null || kill "${existing_pid}" 2>/dev/null || true
      for _ in $(seq 1 50); do
        kill -0 "${existing_pid}" 2>/dev/null || break
        sleep 0.1
      done
      if kill -0 "${existing_pid}" 2>/dev/null; then
        kill -KILL "${existing_pid}" 2>/dev/null || true
        for _ in $(seq 1 20); do
          kill -0 "${existing_pid}" 2>/dev/null || break
          sleep 0.1
        done
      fi
    fi
  fi
fi
pkill -TERM -f xiaomachi-wsl-keepalive 2>/dev/null || true
for _ in $(seq 1 50); do
  pgrep -f xiaomachi-wsl-keepalive >/dev/null 2>&1 || break
  sleep 0.1
done
if [[ "${watchdog_pid}" =~ ^[0-9]+$ ]] && kill -0 "${watchdog_pid}" 2>/dev/null; then
  kill -KILL "${watchdog_pid}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "${watchdog_pid}" 2>/dev/null || break
    sleep 0.1
  done
fi
# Do not clear these markers until the supervisor has forwarded SIGTERM to the
# daemon and both processes have released the process-level lock.
rm -f "${pid_file}"
rm -f "${watchdog_pid_file}"
rm -f "${runtime_dir}"/onebot-watchdog-*.json{,.tmp,.lock}
exit "${compose_exit}"
