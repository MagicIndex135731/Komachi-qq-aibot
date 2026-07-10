#!/usr/bin/env bash
set -euo pipefail

flag_file="${1:?flag file required}"
pid_file="${2:?pid file required}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
wsl_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${wsl_dir}/../.." && pwd)"
watchdog_python="${repo_root}/.venv-wsl/bin/python"
watchdog_script="${script_dir}/onebot_watchdog.py"

mkdir -p "$(dirname "${pid_file}")"
echo "$$" >"${pid_file}"

cleanup() {
  rm -f "${pid_file}"
}
trap cleanup EXIT INT TERM

while [[ -f "${flag_file}" ]]; do
  sleep 60 &
  wait "$!" || true
  [[ -f "${flag_file}" ]] || break
  if [[ -x "${watchdog_python}" ]]; then
    "${watchdog_python}" "${watchdog_script}" --once || true
  fi
done
