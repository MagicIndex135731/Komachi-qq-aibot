#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
runtime_dir="${WSL_DIR}/runtime"
flag_file="${runtime_dir}/keepalive.enabled"
pid_file="${runtime_dir}/keepalive.pid"
lock_file="${runtime_dir}/keepalive.anchor.lock"

mkdir -p "${runtime_dir}"
exec 9>"${lock_file}"
flock -n 9 || exit 0

for _ in $(seq 1 120); do
  [[ -f "${flag_file}" ]] && break
  sleep 1
done
[[ -f "${flag_file}" ]] || exit 0

if [[ -f "${pid_file}" ]]; then
  existing_pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    command_line="$(tr '\0' ' ' <"/proc/${existing_pid}/cmdline" 2>/dev/null || true)"
    [[ "${command_line}" == *xiaomachi-wsl-keepalive* ]] && exit 0
  fi
fi

platform="$(sed -n 's/^[[:space:]]*QQ_PLATFORM[[:space:]]*=[[:space:]]*//p' "${WSL_DIR}/.env" | tail -n 1 | tr -d '\r' | tr '[:upper:]' '[:lower:]')"
platform="${platform:-napcat}"

exec -a xiaomachi-wsl-keepalive bash "${SCRIPT_DIR}/keepalive.sh" \
  "${flag_file}" "${pid_file}" "${platform}"
