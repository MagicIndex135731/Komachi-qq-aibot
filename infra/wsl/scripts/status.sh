#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${WSL_DIR}/../.." && pwd)"
cd "${WSL_DIR}"

strip_optional_env_quotes() {
  local value="${1:-}"
  value="${value#\"}"
  value="${value%\"}"
  value="${value#\'}"
  value="${value%\'}"
  printf '%s' "${value}"
}

xiaomachi_proxy="$(sed -n 's/^[[:space:]]*XIAOMACHI_HTTPS_PROXY[[:space:]]*=[[:space:]]*//p' .env | tail -n 1 | tr -d '\r')"
if [[ "${xiaomachi_proxy}" == "http://127.0.0.1:7897" ]]; then
  echo "Mihomo provider proxy probe:"
  systemctl is-active --quiet xiaomachi-mihomo.service || {
    echo "xiaomachi-mihomo.service is not active."
    exit 1
  }
  proxy_status="$(curl -x "${xiaomachi_proxy}" -sS -o /dev/null \
    --connect-timeout 5 --max-time 15 -w '%{http_code}' https://ai.novacode.top/ || true)"
  if [[ "${proxy_status}" == "000" || -z "${proxy_status}" ]]; then
    echo "Nova is unreachable through local Mihomo."
    exit 1
  fi
  echo "service=active nova_route=reachable http_status=${proxy_status}"
fi

platform="$(sed -n 's/^[[:space:]]*QQ_PLATFORM[[:space:]]*=[[:space:]]*//p' .env | tail -n 1 | tr -d '\r' | tr '[:upper:]' '[:lower:]')"
platform="${platform:-napcat}"
memory_embedding_provider="$(sed -n 's/^[[:space:]]*MEMORY_EMBEDDING_PROVIDER[[:space:]]*=[[:space:]]*//p' .env | tail -n 1 | tr -d '\r' | tr '[:upper:]' '[:lower:]')"
memory_embedding_provider="$(strip_optional_env_quotes "${memory_embedding_provider}")"
memory_embedding_provider="${memory_embedding_provider:-local}"
memory_embedding_device="$(sed -n 's/^[[:space:]]*MEMORY_EMBEDDING_DEVICE[[:space:]]*=[[:space:]]*//p' .env | tail -n 1 | tr -d '\r' | tr '[:upper:]' '[:lower:]')"
memory_embedding_device="$(strip_optional_env_quotes "${memory_embedding_device}")"
memory_embedding_device="${memory_embedding_device:-cpu}"
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
bot_started_at="$(docker inspect --format '{{.State.StartedAt}}' "${bot_container_name}" 2>/dev/null || true)"
if [[ -z "${bot_started_at}" ]]; then
  echo "Cannot determine ${bot_container_name} start time."
  exit 1
fi

echo "Waiting for memory embedding prewarm..."
embedding_prewarm_ok=false
for attempt in $(seq 1 60); do
  embedding_prewarm_payload="$(docker exec "${bot_container_name}" cat /workspace/data/logs/memory.embedding.ready.json 2>/dev/null || true)"
  if python3 - "${embedding_prewarm_payload}" "${bot_started_at}" "${memory_embedding_provider}" "${memory_embedding_device}" <<'PY'
import json, re, sys
from datetime import datetime, timezone
if not sys.argv[1]: raise SystemExit(1)
d = json.loads(sys.argv[1])
started_text = re.sub(r"(\.\d{6})\d+", r"\1", str(sys.argv[2]))
started = datetime.fromisoformat(started_text.replace("Z", "+00:00"))
if started.tzinfo is None: started = started.replace(tzinfo=timezone.utc)
t = datetime.fromisoformat(str(d.get("updated_at", "")).replace("Z", "+00:00"))
if t.tzinfo is None: t = t.replace(tzinfo=timezone.utc)
after_start = (t.astimezone(timezone.utc) - started.astimezone(timezone.utc)).total_seconds()
state = str(d.get("state", ""))
provider = str(d.get("provider", ""))
accelerator = str(d.get("accelerator", ""))
expected_provider = str(sys.argv[3])
expected_device = str(sys.argv[4])
if after_start < -5: raise SystemExit(1)
if expected_provider == "disabled":
    if state != "disabled" or provider != "disabled": raise SystemExit(1)
else:
    if state != "ready" or provider != expected_provider: raise SystemExit(1)
    if int(d.get("dimensions", 0)) <= 0: raise SystemExit(1)
    if expected_provider == "local" and expected_device == "cuda" and accelerator != "cuda":
        raise SystemExit(1)
print(
    f"state={state} provider={provider} accelerator={accelerator} "
    f"dimensions={d.get('dimensions')} prewarm_after_container_start_seconds={after_start:.1f}"
)
PY
  then
    embedding_prewarm_ok=true
    break
  fi
  echo "  waiting for memory embedding prewarm (${attempt}/60)"
  sleep 5
done
if [[ "${embedding_prewarm_ok}" != true ]]; then
  echo "Memory embedding prewarm did not become ready."
  docker compose -f "${compose_file}" logs --tail=80 xiaomachi
  exit 1
fi

echo "Local group policy probe:"
if ! docker exec -i "${bot_container_name}" python - <<'PY'
from pathlib import Path

import yaml

path = Path("/workspace/configs/groups.local.yaml")
if not path.is_file() or not path.stat().st_size:
    raise SystemExit("local group policy is missing or empty")
payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
groups = payload.get("groups") or {}
speaking_groups = [
    entry
    for entry in groups.values()
    if isinstance(entry, dict)
    and bool(entry.get("enabled", True))
    and bool(entry.get("speak", False))
]
print(
    f"local_group_policy=ok configured_groups={len(groups)} "
    f"speaking_groups={len(speaking_groups)}"
)
if not speaking_groups:
    raise SystemExit("local group policy has no enabled speaking groups")
PY
then
  echo "Xiaomachi local group policy is missing or does not allow any group to speak."
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
  if [[ "${platform}" == "llbot" ]] \
      && ! docker logs --tail 200 "${container_name}" 2>&1 \
        | grep -Fq -e "replay protection unavailable" -e "sign 未初始化"; then
    echo "LLBot QQ is offline; leaving the stack running so the watchdog can recover after the network returns."
    exit 75
  fi
  echo "OneBot did not become ready. Check the ${service_name} logs and WebUI."
  exit 1
fi

echo "Waiting for xiaomachi bot heartbeat..."
heartbeat_ok=false
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
  then
    heartbeat_ok=true
    break
  fi
  sleep 5
done
if [[ "${heartbeat_ok}" != true ]]; then
  docker compose -f "${compose_file}" logs --tail=80 xiaomachi
  exit 1
fi

echo "Waiting for xiaomachi bot to accept messages (gateway ready)..."
ready_ok=false
for attempt in $(seq 1 60); do
  ready_payload="$(docker exec "${bot_container_name}" cat /workspace/data/logs/group.ready.json 2>/dev/null || true)"
  if python3 - "${ready_payload}" "${bot_started_at}" <<'PY'
import json, re, sys
from datetime import datetime, timezone
if not sys.argv[1]: raise SystemExit(1)
d = json.loads(sys.argv[1])
t = datetime.fromisoformat(str(d.get("updated_at", "")).replace("Z", "+00:00"))
if t.tzinfo is None: t = t.replace(tzinfo=timezone.utc)
started_text = re.sub(r"(\.\d{6})\d+", r"\1", str(sys.argv[2]))
started = datetime.fromisoformat(started_text.replace("Z", "+00:00"))
if started.tzinfo is None: started = started.replace(tzinfo=timezone.utc)
age = (datetime.now(timezone.utc) - t.astimezone(timezone.utc)).total_seconds()
after_start = (t.astimezone(timezone.utc) - started.astimezone(timezone.utc)).total_seconds()
state = str(d.get("state", ""))
print(
    f"state={state} ready_age_seconds={age:.1f} "
    f"ready_after_container_start_seconds={after_start:.1f}"
)
if state not in ("connected", "ready"): raise SystemExit(1)
# A ready marker is a state transition, not a heartbeat.  It may remain old
# for the entire lifetime of a healthy persistent connection.  Reject only a
# marker inherited from an earlier container instance; heartbeat freshness and
# the live OneBot probe above cover current liveness.
if after_start < -5: raise SystemExit(1)
PY
  then
    ready_ok=true
    break
  fi
  echo "  waiting for bot gateway readiness (${attempt}/60)"
  sleep 5
done
if [[ "${ready_ok}" != true ]]; then
  echo "Bot did not reach gateway-ready state (not yet accepting messages)."
  docker compose -f "${compose_file}" logs --tail=80 xiaomachi
  exit 1
fi

echo "Xiaomachi bot is up and accepting messages."
exit 0
