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
python3 -m venv .venv-wsl
./.venv-wsl/bin/python -m pip install -U pip
./.venv-wsl/bin/python -m pip install websockets

if ./.venv-wsl/bin/python - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 12) else 1)
PY
then
  ./.venv-wsl/bin/python -m pip install -e .
else
  echo "WSL python is older than 3.12; installed probe dependencies only."
  echo "The Xiaomachi bot runs in the python:3.12-slim Docker container."
fi

echo "WSL bootstrap complete."
