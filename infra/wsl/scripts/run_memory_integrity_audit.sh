#!/usr/bin/env bash
set -euo pipefail

install_root="${XIAOMACHI_INSTALL_ROOT:-/opt/xiaomachi}"
runtime_logs="${install_root}/shared/runtime/logs"
last_report="${runtime_logs}/memory-integrity-audit.latest.json"
history_report="${runtime_logs}/memory-integrity-audit.jsonl"
bot_container="xiaomachi-bot"

mkdir -p "${runtime_logs}"

container_status="$(docker inspect --format '{{.State.Status}}' "${bot_container}" 2>/dev/null || true)"
if [[ "${container_status}" != "running" ]]; then
  echo "memory_integrity_audit_skipped reason=bot_not_running status=${container_status:-missing}"
  exit 0
fi

temporary_report="$(mktemp "${runtime_logs}/memory-integrity-audit.XXXXXX")"
trap 'rm -f "${temporary_report}"' EXIT

audit_status=0
docker exec "${bot_container}" python -m scripts.maintain_memory_integrity audit \
  --database /workspace/data/bot.db \
  --fail-on-critical >"${temporary_report}" || audit_status=$?

if [[ -s "${temporary_report}" ]]; then
  python3 -m json.tool "${temporary_report}" >/dev/null
  install -m 0640 "${temporary_report}" "${last_report}.next"
  mv -f "${last_report}.next" "${last_report}"
  tr -d '\n' <"${temporary_report}" >>"${history_report}"
  printf '\n' >>"${history_report}"
  chmod 0640 "${history_report}"
fi

if (( audit_status == 2 )); then
  echo "memory_integrity_audit_alert report=${last_report}" >&2
elif (( audit_status != 0 )); then
  echo "memory_integrity_audit_failed exit_code=${audit_status}" >&2
fi
exit "${audit_status}"
