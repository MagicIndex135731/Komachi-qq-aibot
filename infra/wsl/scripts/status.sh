#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${WSL_DIR}/../.." && pwd)"
cd "${WSL_DIR}"

platform="$(sed -n 's/^[[:space:]]*QQ_PLATFORM[[:space:]]*=[[:space:]]*//p' .env | tail -n 1 | tr -d '\r' | tr '[:upper:]' '[:lower:]')"
platform="${platform:-napcat}"
if [[ "${platform}" == "llbot" ]]; then
  compose_file="docker-compose.llbot.yml"
  service_name="llbot"
  container_name="xiaomachi-llbot"
else
  compose_file="docker-compose.yml"
  service_name="napcat"
  container_name="xiaomachi-napcat"
fi

llbot_ws_port="$(sed -n 's/^[[:space:]]*LLBOT_WS_PORT[[:space:]]*=[[:space:]]*//p' .env | tail -n 1 | tr -d '\r')"
llbot_ws_port="${llbot_ws_port:-3002}"
if ! [[ "${llbot_ws_port}" =~ ^[0-9]+$ ]] || (( llbot_ws_port < 1 || llbot_ws_port > 65535 )); then
  echo "LLBOT_WS_PORT must be between 1 and 65535."
  exit 1
fi
if [[ "${platform}" == "llbot" ]]; then
  onebot_ws_url="ws://127.0.0.1:${llbot_ws_port}"
else
  onebot_ws_url="ws://127.0.0.1:3001"
fi

docker compose -f "${compose_file}" ps
status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "${container_name}" 2>/dev/null || true)"
if [[ "${status}" != "healthy" && "${status}" != "running" ]]; then
  echo "Waiting for ${service_name} container..."
  for _ in $(seq 1 24); do
    if [[ "${platform}" == "llbot" ]] \
        && docker logs --tail 80 "${container_name}" 2>&1 \
          | grep -Fq -e "replay protection unavailable" -e "sign 未初始化"; then
      echo "LLBot signing backend is unavailable; quick login and QR login cannot proceed yet."
      exit 1
    fi
    sleep 5
    status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "${container_name}" 2>/dev/null || true)"
    [[ "${status}" == "healthy" || "${status}" == "running" ]] && break
  done
fi

if [[ "${platform}" == "llbot" ]] \
    && docker logs --tail 200 "${container_name}" 2>&1 \
      | grep -Fq -e "replay protection unavailable" -e "sign 未初始化"; then
  echo "LLBot signing backend is unavailable; quick login and QR login cannot proceed yet."
  exit 1
fi

if [[ "${status}" != "healthy" && "${status}" != "running" ]]; then
  docker compose -f "${compose_file}" logs --tail=80 "${service_name}"
  exit 1
fi

bot_container_name="xiaomachi-bot"
bot_status="$(docker inspect --format '{{.State.Status}}' "${bot_container_name}" 2>/dev/null || true)"
if [[ "${bot_status}" != "running" ]]; then
  echo "${bot_container_name} is not running (status=${bot_status:-missing})."
  docker compose -f "${compose_file}" logs --tail=80 xiaomachi
  exit 1
fi

if [[ "${platform}" == "llbot" ]]; then
  echo "LLBot WebUI probe:"
  webui_ok=false
  for attempt in $(seq 1 12); do
    if curl -fsS --max-time 8 http://127.0.0.1:3080/ >/dev/null; then
      echo "webui=http://127.0.0.1:3080/ ok"
      webui_ok=true
      break
    fi
    echo "  waiting for LLBot WebUI (${attempt}/12)"
    sleep 5
  done
  if [[ "${webui_ok}" != true ]]; then
    echo "LLBot WebUI did not become ready. Check the ${service_name} logs."
    docker compose -f "${compose_file}" logs --tail=80 "${service_name}"
    exit 1
  fi
fi

echo "OneBot probe (${platform}):"
probe_output="$(mktemp)"
trap 'rm -f "${probe_output}"' EXIT
probe_ok=false
for attempt in $(seq 1 12); do
  watchdog_python="${XIAOMACHI_WATCHDOG_PYTHON:-/opt/xiaomachi/current/.venv-wsl/bin/python}"
  if "${watchdog_python}" "${SCRIPT_DIR}/onebot_probe.py" --ws-url "${onebot_ws_url}" --request-timeout 8 >"${probe_output}" 2>&1; then
    cat "${probe_output}"
    probe_ok=true
    break
  fi
  echo "  waiting for OneBot (${attempt}/12)"
  sleep 5
done
if [[ "${probe_ok}" != true ]]; then
  sed -n '1,40p' "${probe_output}"
  echo "OneBot did not become ready. Check the ${service_name} logs and WebUI."
  exit 1
fi

echo "Waiting for xiaomachi bot heartbeat..."
for _ in $(seq 1 60); do
  heartbeat_payload="$(docker exec "${bot_container_name}" cat /workspace/data/logs/group.heartbeat.json 2>/dev/null || true)"
  if python3 - "${heartbeat_payload}" <<'PY'
import json, sys
from datetime import datetime, timezone
if not sys.argv[1]: raise SystemExit(1)
d = json.loads(sys.argv[1])
t = datetime.fromisoformat(str(d.get("updated_at", "")).replace("Z", "+00:00"))
if t.tzinfo is None: t = t.replace(tzinfo=timezone.utc)
age = (datetime.now(timezone.utc) - t.astimezone(timezone.utc)).total_seconds()
print(f"state={d.get('state')} pid={d.get('pid')} heartbeat_age_seconds={age:.1f}")
raise SystemExit(0 if d.get("state") == "alive" and age <= 20 else 1)
PY
  then exit 0; fi
  sleep 5
done
docker compose -f "${compose_file}" logs --tail=80 xiaomachi
exit 1
