#!/usr/bin/env bash
set -euo pipefail

flag_file="${1:?flag file required}"
pid_file="${2:?pid file required}"
platform="${3:-napcat}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
wsl_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${wsl_dir}/../.." && pwd)"
watchdog_python="${repo_root}/.venv-wsl/bin/python"

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
trap 'rm -f "${pid_file}"' EXIT INT TERM

while [[ -f "${flag_file}" ]]; do
  sleep 60 & wait "$!" || true
  [[ -f "${flag_file}" ]] || break
  if [[ -x "${watchdog_python}" ]]; then
    "${watchdog_python}" "${script_dir}/onebot_watchdog.py" --once \
      --ws-url "${onebot_ws_url}" \
      --compose-file "${compose_file}" --service-name "${service_name}" \
      --state-file "${wsl_dir}/runtime/onebot-watchdog-${platform}.json" \
      --log-file "${wsl_dir}/runtime/logs/onebot-watchdog-${platform}.log" \
      --platform "${platform}" || true
  fi
done
