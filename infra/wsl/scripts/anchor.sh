#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
runtime_dir="${WSL_DIR}/runtime"
flag_file="${runtime_dir}/keepalive.enabled"
pid_file="${runtime_dir}/keepalive.pid"
lock_file="${runtime_dir}/keepalive.anchor.lock"

mode="${1:-anchor}"
if [[ "${mode}" != "anchor" && "${mode}" != "watchdog" ]]; then
  echo "Usage: $0 [anchor|watchdog]" >&2
  exit 2
fi

if [[ "${mode}" == "anchor" ]]; then
  # Optional foreground anchor for diagnostics. systemd owns service recovery;
  # normal operation uses the manual Windows BAT entries.
  install_root="${XIAOMACHI_INSTALL_ROOT:-/opt/xiaomachi}"
  if [[ "${WSL_DIR}" != "${install_root}/current/infra/wsl" ]]; then
    echo "Refusing to anchor a Windows-mounted source tree: ${WSL_DIR}" >&2
    exit 1
  fi
  mkdir -p /run/lock
  exec 8>/run/lock/xiaomachi-wsl-anchor.lock
  flock -n 8 || exit 0
  systemctl start xiaomachi-stack.service xiaomachi-watchdog.service
  trap 'exit 0' INT TERM
  while systemctl is-active --quiet xiaomachi-stack.service \
      && systemctl is-active --quiet xiaomachi-watchdog.service; do
    sleep 5 &
    wait "$!" || true
  done
  exit 0
fi

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
