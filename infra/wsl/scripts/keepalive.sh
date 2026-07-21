#!/usr/bin/env bash
set -euo pipefail

flag_file="${1:?flag file required}"
pid_file="${2:?pid file required}"
platform="${3:-napcat}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
wsl_dir="$(cd "${script_dir}/.." && pwd)"
watchdog_python="${XIAOMACHI_WATCHDOG_PYTHON:-/opt/xiaomachi/current/.venv-wsl/bin/python}"

if [[ "${platform}" == "llbot" ]]; then
  compose_file="${wsl_dir}/docker-compose.llbot.yml"
  service_name="llbot"
  llbot_ws_port="$(sed -n 's/^[[:space:]]*LLBOT_WS_PORT[[:space:]]*=[[:space:]]*//p' "${wsl_dir}/.env" | tail -n 1 | tr -d '\r')"
  llbot_ws_port="${llbot_ws_port:-3002}"
  onebot_ws_url="ws://127.0.0.1:${llbot_ws_port}"
else
  compose_file="${wsl_dir}/docker-compose.yml"
  service_name="napcat"
  onebot_ws_url="ws://127.0.0.1:3001"
fi

mkdir -p "$(dirname "${pid_file}")"
echo "$$" >"${pid_file}"
watchdog_pid_file="${pid_file}.watchdog"
watchdog_pid=""

cleanup() {
  rm -f "${pid_file}"
  rm -f "${watchdog_pid_file}"
}

stop_watchdog() {
  if [[ "${watchdog_pid}" =~ ^[0-9]+$ ]] && kill -0 "${watchdog_pid}" 2>/dev/null; then
    kill -TERM "${watchdog_pid}" 2>/dev/null || true
    wait "${watchdog_pid}" 2>/dev/null || true
  fi
  watchdog_pid=""
  rm -f "${watchdog_pid_file}"
}

shutdown() {
  stop_watchdog
  exit 0
}

trap cleanup EXIT
trap shutdown INT TERM

# This script is only a crash supervisor.  The watchdog itself owns the
# long-lived event loop and its 60-second probe cadence; we only retry it when
# it exits abnormally, with a bounded backoff to avoid restart storms.
backoff_seconds=5
while [[ -f "${flag_file}" ]]; do
  if [[ ! -x "${watchdog_python}" ]]; then
    exit 1
  fi
  "${watchdog_python}" "${script_dir}/onebot_watchdog.py" --daemon \
      --ws-url "${onebot_ws_url}" \
      --compose-file "${compose_file}" --service-name "${service_name}" \
      --state-file "${wsl_dir}/runtime/onebot-watchdog-${platform}.json" \
      --log-file "${wsl_dir}/runtime/logs/onebot-watchdog-${platform}.log" \
      --platform "${platform}" &
  watchdog_pid="$!"
  echo "${watchdog_pid}" >"${watchdog_pid_file}"
  if wait "${watchdog_pid}"; then
    watchdog_pid=""
    rm -f "${watchdog_pid_file}"
    # A clean daemon exit is an intentional stop, not an excuse to relaunch it.
    exit 0
  fi
  watchdog_pid=""
  rm -f "${watchdog_pid_file}"
  [[ -f "${flag_file}" ]] || break
  sleep "${backoff_seconds}" & wait "$!" || true
  if (( backoff_seconds < 60 )); then
    backoff_seconds=$((backoff_seconds * 2))
    (( backoff_seconds > 60 )) && backoff_seconds=60
  fi
done
