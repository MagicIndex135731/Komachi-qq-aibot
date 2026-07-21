#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${WSL_DIR}/../.." && pwd)"

cd "${WSL_DIR}"
mkdir -p runtime/napcat/config runtime/napcat/logs runtime/napcat/cache runtime/napcat/ntqq runtime/logs runtime/cache runtime/pip-cache

if [[ ! -f runtime/napcat/config/onebot11.json ]]; then
  cat > runtime/napcat/config/onebot11.json <<'JSON'
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
  echo "Created ${WSL_DIR}/runtime/napcat/config/onebot11.json."
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created ${WSL_DIR}/.env from .env.example. Fill secrets before starting."
fi

cd "${REPO_ROOT}"
watchdog_venv="${XIAOMACHI_WATCHDOG_VENV:-/opt/xiaomachi/shared/venv}"
python3 -m venv "${watchdog_venv}"
"${watchdog_venv}/bin/python" -m pip install -U pip
"${watchdog_venv}/bin/python" -m pip install websockets

echo "Installed the persistent watchdog runtime at ${watchdog_venv}."
echo "The Xiaomachi application itself runs from the immutable Docker image."

echo "WSL bootstrap complete."
