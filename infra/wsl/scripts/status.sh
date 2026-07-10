#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${WSL_DIR}/../.." && pwd)"

cd "${WSL_DIR}"
docker compose ps

health_status() {
  docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' xiaomachi-napcat 2>/dev/null || true
}

run_onebot_probe() {
  local probe_output
  probe_output="$(mktemp)"
  trap 'rm -f "${probe_output}"' RETURN

  echo "OneBot probe:"
  if [[ ! -x "${REPO_ROOT}/.venv-wsl/bin/python" ]]; then
    echo "Probe skipped: ${REPO_ROOT}/.venv-wsl/bin/python not found. Run infra/wsl/scripts/bootstrap_wsl.sh first."
    return 0
  fi

  echo "Waiting for OneBot websocket..."
  for _ in $(seq 1 30); do
    if "${REPO_ROOT}/.venv-wsl/bin/python" "${WSL_DIR}/scripts/onebot_probe.py" --ws-url ws://127.0.0.1:3001 >"${probe_output}" 2>&1; then
      cat "${probe_output}"
      return 0
    fi
    sleep 5
  done

  echo "OneBot probe failed after waiting; last probe output follows."
  sed -n '1,40p' "${probe_output}"
  return 1
}

wait_bot_heartbeat() {
  local heartbeat_file="${WSL_DIR}/runtime/logs/group.heartbeat.json"
  local heartbeat_output
  heartbeat_output="$(mktemp)"
  trap 'rm -f "${heartbeat_output}"' RETURN

  echo
  echo "xiaomachi bot heartbeat:"
  echo "Waiting for xiaomachi bot heartbeat..."
  for _ in $(seq 1 60); do
    if python3 - "${heartbeat_file}" >"${heartbeat_output}" 2>&1 <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
updated = datetime.fromisoformat(str(payload.get("updated_at", "")).replace("Z", "+00:00"))
if updated.tzinfo is None:
    updated = updated.replace(tzinfo=timezone.utc)
age = (datetime.now(timezone.utc) - updated.astimezone(timezone.utc)).total_seconds()
state = str(payload.get("state", ""))
pid = payload.get("pid")
print(f"state={state} pid={pid} heartbeat_age_seconds={age:.1f}")
if state != "alive" or age > 20:
    raise SystemExit(1)
PY
    then
      cat "${heartbeat_output}"
      return 0
    fi
    sleep 5
  done

  echo "xiaomachi bot heartbeat did not become fresh; last heartbeat check follows."
  sed -n '1,40p' "${heartbeat_output}"
  return 1
}

status="$(health_status)"
if [[ "${status}" == "starting" || "${status}" == "" ]]; then
  echo
  echo "Waiting for NapCat healthcheck..."
  for _ in $(seq 1 18); do
    sleep 5
    status="$(health_status)"
    if [[ "${status}" == "healthy" || "${status}" == "running" ]]; then
      break
    fi
  done
  echo
  docker compose ps
fi

status="$(health_status)"
echo
if [[ "${status}" == "healthy" || "${status}" == "running" ]]; then
  if ! run_onebot_probe; then
    echo
    echo "Recent logs follow."
    echo
    docker compose logs --tail=80 napcat
    echo
    docker compose logs --tail=80 xiaomachi
    exit 1
  fi
  if ! wait_bot_heartbeat; then
    echo
    echo "Recent logs follow."
    echo
    docker compose logs --tail=80 xiaomachi
    exit 1
  fi
else
  echo "NapCat is not healthy after waiting; recent logs follow."
  echo
  docker compose logs --tail=80 napcat
  echo
  docker compose logs --tail=80 xiaomachi
fi
