#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
case "${ACTION}" in
  start|stop|status|anchor|install) ;;
  *)
    echo "Usage: $0 {start|stop|status|anchor|install}" >&2
    exit 2
    ;;
esac

install_root="${XIAOMACHI_INSTALL_ROOT:-/opt/xiaomachi}"
runtime_entry="${install_root}/current/infra/wsl/scripts/${ACTION}.sh"

if [[ "${ACTION}" == "install" ]]; then
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "${script_dir}/install_linux_runtime.sh" ]]; then
    exec bash "${script_dir}/install_linux_runtime.sh"
  fi
  for base in /mnt/d /mnt/e /mnt/c; do
    [[ -d "${base}" ]] || continue
    while IFS= read -r installer; do
      repo_root="${installer%/infra/wsl/scripts/install_linux_runtime.sh}"
      if [[ -f "${repo_root}/pyproject.toml" ]]; then
        echo "Installing from ${repo_root}"
        exec bash "${installer}"
      fi
    done < <(
      find "${base}" -mindepth 4 -maxdepth 7 \
        -path '*/infra/wsl/scripts/install_linux_runtime.sh' -type f 2>/dev/null
    )
  done
  echo "Cannot find a Xiaomachi source tree for first-time installation." >&2
  exit 1
fi

if [[ ! -x /usr/local/bin/xiaomachi-wsl-entry || ! -d "${install_root}/current" ]]; then
  echo "Linux runtime is not installed. Run '$0 install' once from the source tree." >&2
  exit 1
fi

run_systemd_with_output() {
  local action="${1:?systemd action required}"
  local unit="${2:?systemd unit required}"
  local since command_pid journal_pid command_exit=0

  since="$(date '+%Y-%m-%d %H:%M:%S')"
  systemctl "${action}" "${unit}" &
  command_pid="$!"
  journalctl --no-pager --follow --output=cat --since "${since}" --unit "${unit}" &
  journal_pid="$!"

  wait "${command_pid}" || command_exit=$?
  sleep 0.2
  kill "${journal_pid}" 2>/dev/null || true
  wait "${journal_pid}" 2>/dev/null || true
  return "${command_exit}"
}

case "${ACTION}" in
  start)
    echo "=== Starting Xiaomachi stack ==="
    run_systemd_with_output start xiaomachi-stack.service
    systemctl start xiaomachi-watchdog.service
    echo "=== Systemd supervision ==="
    systemctl --no-pager --full status \
      xiaomachi-stack.service xiaomachi-watchdog.service || true
    ;;
  stop)
    echo "=== Stopping Xiaomachi watchdog ==="
    systemctl stop xiaomachi-watchdog.service || true
    echo "=== Stopping Xiaomachi stack ==="
    run_systemd_with_output stop xiaomachi-stack.service
    echo "=== Final service state ==="
    systemctl --no-pager --full status \
      xiaomachi-stack.service xiaomachi-watchdog.service || true
    ;;
  status)
    systemctl --no-pager --full status xiaomachi-stack.service xiaomachi-watchdog.service || true
    if ! systemctl is-active --quiet xiaomachi-stack.service \
        || ! systemctl is-active --quiet xiaomachi-watchdog.service; then
      echo "Xiaomachi systemd supervision is not active." >&2
      exit 1
    fi
    exec bash "${runtime_entry}"
    ;;
  anchor)
    exec bash "${runtime_entry}"
    ;;
esac
