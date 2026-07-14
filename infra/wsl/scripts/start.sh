#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WSL_DIR}"

if [[ ! -f .env ]]; then
  echo "Missing ${WSL_DIR}/.env. Run bootstrap_wsl.sh and fill required secrets."
  exit 1
fi

platform="$(sed -n 's/^[[:space:]]*QQ_PLATFORM[[:space:]]*=[[:space:]]*//p' .env | tail -n 1 | tr -d '\r' | tr '[:upper:]' '[:lower:]')"
platform="${platform:-napcat}"
if [[ "${platform}" != "napcat" && "${platform}" != "llbot" ]]; then
  echo "QQ_PLATFORM must be napcat or llbot."
  exit 1
fi

if [[ "${platform}" == "llbot" ]]; then
  compose_file="docker-compose.llbot.yml"
  other_compose_file="docker-compose.yml"
  service_name="llbot"
  webui_port=3080
  launcher="open_llbot_webui.ps1"
  python3 "${SCRIPT_DIR}/bootstrap_llbot_runtime.py" --wsl-dir "${WSL_DIR}"
else
  compose_file="docker-compose.yml"
  other_compose_file="docker-compose.llbot.yml"
  service_name="napcat"
  webui_port=6099
  launcher="open_napcat_webui.ps1"
fi

start_keepalive() {
  local runtime_dir="${WSL_DIR}/runtime"
  local flag_file="${runtime_dir}/keepalive.enabled"
  local pid_file="${runtime_dir}/keepalive.pid"
  local existing_pid=""
  mkdir -p "${runtime_dir}"
  touch "${flag_file}"
  if [[ -f "${pid_file}" ]]; then
    existing_pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
      return 0
    fi
  fi
  nohup setsid bash -c 'exec -a xiaomachi-wsl-keepalive bash "$@"' _ \
    "${SCRIPT_DIR}/keepalive.sh" "${flag_file}" "${pid_file}" "${platform}" \
    >/dev/null 2>&1 &
}

open_login_page() {
  local webui_ready=false
  for attempt in $(seq 1 10); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${webui_port}/" >/dev/null 2>&1; then
      webui_ready=true
      break
    fi
    echo "Waiting for ${service_name} WebUI (${attempt}/10)..."
    sleep 2
  done
  if [[ "${webui_ready}" == true ]] && command -v powershell.exe >/dev/null 2>&1; then
    powershell.exe -NoProfile -ExecutionPolicy Bypass \
      -File "$(wslpath -w "${SCRIPT_DIR}/${launcher}")" -OnlyWhenLoginRequired >/dev/null 2>&1 || true
  else
    echo "${service_name} WebUI is not ready yet; continuing to status diagnostics."
  fi
}

docker compose -f "${other_compose_file}" down --remove-orphans || true
# Do not let Compose's `depends_on: service_healthy` block the login page.
# The bot reconnects to OneBot on its own while the QQ platform finishes login.
docker compose -f "${compose_file}" up -d "${service_name}"
open_login_page
docker compose -f "${compose_file}" up -d --no-deps xiaomachi
start_keepalive
bash "${SCRIPT_DIR}/status.sh"
