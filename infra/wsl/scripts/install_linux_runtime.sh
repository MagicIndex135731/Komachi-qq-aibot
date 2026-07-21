#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
INSTALL_ROOT="${XIAOMACHI_INSTALL_ROOT:-/opt/xiaomachi}"
ENTRY_PATH="${XIAOMACHI_ENTRY_PATH:-/usr/local/bin/xiaomachi-wsl-entry}"
SYSTEMD_DIR="${XIAOMACHI_SYSTEMD_DIR:-/etc/systemd/system}"

if (( EUID != 0 )); then
  if ! command -v sudo >/dev/null 2>&1; then
    echo "Root privileges are required to install under ${INSTALL_ROOT}." >&2
    exit 1
  fi
  exec sudo --preserve-env=XIAOMACHI_INSTALL_ROOT,XIAOMACHI_ENTRY_PATH,XIAOMACHI_SYSTEMD_DIR \
    bash "$0" "$@"
fi

for command_name in tar python3 systemctl; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    exit 1
  fi
done

if ! systemctl show-environment >/dev/null 2>&1; then
  echo "systemd is not running in this WSL distribution." >&2
  echo "Enable systemd in /etc/wsl.conf, run 'wsl --shutdown' in Windows, then retry." >&2
  exit 1
fi

release_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
releases_dir="${INSTALL_ROOT}/releases"
shared_dir="${INSTALL_ROOT}/shared"
release_dir="${releases_dir}/${release_id}"
shared_runtime="${shared_dir}/runtime"
previous_release="$(readlink -f "${INSTALL_ROOT}/current" 2>/dev/null || true)"
if [[ ! -e "${shared_dir}/.install-complete" ]]; then
  previous_release=""
fi

mapfile -t previously_running < <(
  docker ps --format '{{.Names}}' \
    | grep -E '^(xiaomachi-bot|xiaomachi-llbot|xiaomachi-napcat)$' || true
)
install_succeeded=false
legacy_names=()
legacy_original_names=()
restore_previous_runtime() {
  if [[ "${install_succeeded}" == true ]]; then
    return
  fi
  systemctl stop xiaomachi-watchdog.service >/dev/null 2>&1 || true
  systemctl stop xiaomachi-stack.service >/dev/null 2>&1 || true
  if [[ -n "${previous_release}" && -d "${previous_release}" ]]; then
    ln -sfn "${previous_release}" "${INSTALL_ROOT}/current"
    systemctl daemon-reload >/dev/null 2>&1 || true
    systemctl start xiaomachi-stack.service >/dev/null 2>&1 || true
    systemctl start xiaomachi-watchdog.service >/dev/null 2>&1 || true
  elif (( ${#legacy_names[@]} > 0 )); then
    for original_name in "${legacy_original_names[@]}"; do
      docker rm -f "${original_name}" >/dev/null 2>&1 || true
    done
    for index in "${!legacy_names[@]}"; do
      docker rename "${legacy_names[$index]}" "${legacy_original_names[$index]}" >/dev/null 2>&1 || true
      docker start "${legacy_original_names[$index]}" >/dev/null 2>&1 || true
    done
  elif (( ${#previously_running[@]} > 0 )); then
    docker start "${previously_running[@]}" >/dev/null 2>&1 || true
  fi
}
trap restore_previous_runtime EXIT

install -d -m 0755 "${releases_dir}" "${shared_dir}" "${release_dir}"
install -d -m 0755 "${shared_runtime}"

# This is the only phase allowed to read the old Windows-backed runtime.  It
# stops writers, creates Linux-side backups, copies named volumes, and validates
# the database and LLBot configuration before the release can be activated.
if [[ ! -e "${shared_dir}/.volumes-migrated" ]]; then
  bash "${SOURCE_ROOT}/infra/wsl/scripts/migrate_runtime_to_linux_volumes.sh"
  touch "${shared_dir}/.volumes-migrated"
fi

# Copy only the runtime allowlist.  Besides being smaller than a repository
# mirror, this avoids touching unrelated or inaccessible generated directories.
tar -C "${SOURCE_ROOT}" \
  --exclude='infra/wsl/.env' \
  --exclude='infra/wsl/runtime' \
  -cf - \
  app configs infra/wsl .dockerignore pyproject.toml README.md LICENSE \
  | tar -C "${release_dir}" -xf -

if [[ -f "${SOURCE_ROOT}/infra/wsl/.env" ]]; then
  install -m 0600 "${SOURCE_ROOT}/infra/wsl/.env" "${shared_dir}/.env.next"
  mv -f "${shared_dir}/.env.next" "${shared_dir}/.env"
elif [[ ! -f "${shared_dir}/.env" ]]; then
    install -m 0600 "${release_dir}/infra/wsl/.env.example" "${shared_dir}/.env"
    echo "Created ${shared_dir}/.env from the example; fill its required secrets before starting."
fi

if [[ ! -e "${shared_dir}/.runtime-imported" && -d "${SOURCE_ROOT}/infra/wsl/runtime" ]]; then
  tar -C "${SOURCE_ROOT}/infra/wsl/runtime" \
    --exclude='keepalive.enabled' \
    --exclude='keepalive.pid*' \
    --exclude='keepalive.anchor.lock' \
    --exclude='start.lock' \
    --exclude='onebot-watchdog-*.lock' \
    -cf - . | tar -C "${shared_runtime}" -xf -
fi
touch "${shared_dir}/.runtime-imported"

ln -s "${shared_dir}/.env" "${release_dir}/infra/wsl/.env"
ln -s "${shared_runtime}" "${release_dir}/infra/wsl/runtime"

if [[ ! -x "${shared_dir}/venv/bin/python" ]]; then
  python3 -m venv "${shared_dir}/venv"
fi
"${shared_dir}/venv/bin/python" -m pip install --disable-pip-version-check --quiet websockets
ln -s "${shared_dir}/venv" "${release_dir}/.venv-wsl"

# NapCat needs an initial OneBot websocket configuration on a clean install.
mkdir -p "${shared_runtime}/napcat/config" "${shared_runtime}/napcat/logs" \
  "${shared_runtime}/napcat/cache" "${shared_runtime}/napcat/ntqq" \
  "${shared_runtime}/logs" "${shared_runtime}/cache"
if [[ ! -f "${shared_runtime}/napcat/config/onebot11.json" ]]; then
  install -m 0644 /dev/stdin "${shared_runtime}/napcat/config/onebot11.json" <<'JSON'
{
  "network": {
    "httpServers": [],
    "httpSseServers": [],
    "httpClients": [],
    "websocketServers": [
      {
        "name": "xiaomachi",
        "enable": true,
        "host": "0.0.0.0",
        "port": 3001,
        "messagePostFormat": "array",
        "reportSelfMessage": false,
        "token": ""
      }
    ],
    "websocketClients": []
  },
  "musicSignUrl": "",
  "enableLocalFile2Url": false,
  "parseMultMsg": true
}
JSON
fi

# Build and validate the immutable application image before stopping an
# already-running release.  This keeps upgrade downtime short and prevents a
# broken build from replacing the current symlink.
docker compose -f "${release_dir}/infra/wsl/docker-compose.llbot.yml" build xiaomachi

systemctl stop xiaomachi-watchdog.service >/dev/null 2>&1 || true
systemctl stop xiaomachi-stack.service >/dev/null 2>&1 || true

# On the first migration there is no previous Linux release to recreate. Keep
# the stopped Windows-bind containers intact under legacy names until the new
# stack has passed the synchronous OneBot + heartbeat checks. The new Compose
# project uses a different project name, so it cannot adopt these containers.
if [[ -z "${previous_release}" ]]; then
  for original_name in "${previously_running[@]}"; do
    if docker inspect "${original_name}" >/dev/null 2>&1; then
      legacy_name="${original_name}-legacy-${release_id,,}"
      docker rename "${original_name}" "${legacy_name}"
      legacy_original_names+=("${original_name}")
      legacy_names+=("${legacy_name}")
    fi
  done
fi

ln -s "${release_dir}" "${INSTALL_ROOT}/current.next"
mv -Tf "${INSTALL_ROOT}/current.next" "${INSTALL_ROOT}/current"

install -m 0755 "${INSTALL_ROOT}/current/infra/wsl/scripts/xiaomachi-wsl-entry.sh" "${ENTRY_PATH}"
install -m 0644 "${INSTALL_ROOT}/current/infra/wsl/systemd/xiaomachi-stack.service" \
  "${SYSTEMD_DIR}/xiaomachi-stack.service"
install -m 0644 "${INSTALL_ROOT}/current/infra/wsl/systemd/xiaomachi-watchdog.service" \
  "${SYSTEMD_DIR}/xiaomachi-watchdog.service"

systemctl daemon-reload
systemctl disable xiaomachi-watchdog.service xiaomachi-stack.service >/dev/null 2>&1 || true
systemctl start xiaomachi-stack.service
systemctl start xiaomachi-watchdog.service

# Keep the immediately previous release for rollback and discard older code.
mapfile -t old_releases < <(find "${releases_dir}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
  | sort -nr | tail -n +3 | cut -d' ' -f2-)
if (( ${#old_releases[@]} > 0 )); then
  rm -rf -- "${old_releases[@]}"
fi

echo "Installed Linux runtime at ${INSTALL_ROOT}/current."
echo "Persistent data is stored at ${shared_dir}; normal runtime does not use /mnt drives."

# Keep first-migration legacy containers until every fallible activation step
# has completed. Only then is rollback no longer needed.
if (( ${#legacy_names[@]} > 0 )); then
  docker rm -f "${legacy_names[@]}" >/dev/null
  legacy_names=()
  legacy_original_names=()
fi
touch "${shared_dir}/.install-complete"
install_succeeded=true
trap - EXIT
