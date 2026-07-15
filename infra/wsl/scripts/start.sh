#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WSL_DIR}"

mkdir -p "${WSL_DIR}/runtime"
exec 8>"${WSL_DIR}/runtime/start.lock"
if ! flock -n 8; then
  echo "Xiaomachi startup is already in progress."
  exit 0
fi

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

startup_complete=false

cleanup_failed_start() {
  local status=$?
  trap - EXIT
  if [[ "${startup_complete}" != true ]]; then
    rm -f "${WSL_DIR}/runtime/keepalive.enabled"
    pkill -f xiaomachi-wsl-keepalive 2>/dev/null || true
    rm -f "${WSL_DIR}/runtime/keepalive.pid"
  fi
  exit "${status}"
}

trap cleanup_failed_start EXIT

enable_keepalive() {
  local runtime_dir="${WSL_DIR}/runtime"
  local flag_file="${runtime_dir}/keepalive.enabled"
  mkdir -p "${runtime_dir}"
  touch "${flag_file}"
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

enable_keepalive
docker compose -f "${other_compose_file}" down --remove-orphans || true
# This is a cache check on normal starts. Dependency installation only runs when
# the Dockerfile or requirements file changed, or when the local image is absent.
docker compose -f "${compose_file}" build xiaomachi
# Do not let Compose's `depends_on: service_healthy` block the login page.
# The bot reconnects to OneBot on its own while the QQ platform finishes login.
docker compose -f "${compose_file}" up -d "${service_name}"
open_login_page
docker compose -f "${compose_file}" up -d --no-deps xiaomachi
bash "${SCRIPT_DIR}/status.sh"
startup_complete=true
