#!/usr/bin/env bash
set -euo pipefail

MIHOMO_VERSION="v1.19.30"
MIHOMO_ASSET="mihomo-linux-amd64-v1-v1.19.30.gz"
MIHOMO_SHA256="cbe553d0319a414bd3a372c5976a252155b2c4882b66bce88a4d6bba9571a553"
MIHOMO_URL="https://github.com/MetaCubeX/mihomo/releases/download/${MIHOMO_VERSION}/${MIHOMO_ASSET}"
INSTALL_ROOT="${XIAOMACHI_INSTALL_ROOT:-/opt/xiaomachi}"
CONFIG_DIR="${INSTALL_ROOT}/shared/mihomo"
CONFIG_PATH="${CONFIG_DIR}/config.yaml"
CONFIG_CANDIDATE="${CONFIG_PATH}.next"
SYSTEMD_DIR="${XIAOMACHI_SYSTEMD_DIR:-/etc/systemd/system}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_SOURCE="$(cd "${SCRIPT_DIR}/.." && pwd)/systemd/xiaomachi-mihomo.service"

if (( EUID != 0 )); then
  exec sudo --preserve-env=MIHOMO_ARCHIVE,MIHOMO_DOWNLOAD_PROXY,XIAOMACHI_INSTALL_ROOT,XIAOMACHI_SYSTEMD_DIR \
    bash "$0" "$@"
fi

for command_name in curl gzip sha256sum systemctl; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "Missing required command: ${command_name}" >&2
    exit 1
  }
done
[[ -s "${CONFIG_PATH}" || -s "${CONFIG_CANDIDATE}" ]] || {
  echo "Missing private Mihomo config: ${CONFIG_PATH}" >&2
  exit 1
}
[[ -f "${UNIT_SOURCE}" ]] || {
  echo "Missing Mihomo systemd unit: ${UNIT_SOURCE}" >&2
  exit 1
}

download_path="$(mktemp)"
binary_path="$(mktemp)"
cleanup() { rm -f "${download_path}" "${binary_path}"; }
trap cleanup EXIT

install_binary=true
if [[ -x /usr/local/bin/mihomo ]] \
    && /usr/local/bin/mihomo -v 2>/dev/null | grep -Fq "${MIHOMO_VERSION}"; then
  install_binary=false
  tester=/usr/local/bin/mihomo
else
  if [[ -n "${MIHOMO_ARCHIVE:-}" ]]; then
    install -m 0600 "${MIHOMO_ARCHIVE}" "${download_path}"
  else
    curl_args=(-fL --connect-timeout 10 --max-time 180 --retry 2 --retry-all-errors)
    if [[ -n "${MIHOMO_DOWNLOAD_PROXY:-}" ]]; then
      curl_args+=(-x "${MIHOMO_DOWNLOAD_PROXY}")
    fi
    curl "${curl_args[@]}" -o "${download_path}" "${MIHOMO_URL}"
  fi
  echo "${MIHOMO_SHA256}  ${download_path}" | sha256sum -c -
  gzip -dc "${download_path}" >"${binary_path}"
  chmod 0755 "${binary_path}"
  tester="${binary_path}"
fi

install -d -m 0700 "${CONFIG_DIR}"
config_to_test="${CONFIG_PATH}"
if [[ -s "${CONFIG_CANDIDATE}" ]]; then
  config_to_test="${CONFIG_CANDIDATE}"
fi
"${tester}" -t -d "${CONFIG_DIR}" -f "${config_to_test}"
if [[ "${config_to_test}" == "${CONFIG_CANDIDATE}" ]]; then
  mv -f "${CONFIG_CANDIDATE}" "${CONFIG_PATH}"
fi
if [[ "${install_binary}" == true ]]; then
  install -m 0755 "${binary_path}" /usr/local/bin/mihomo.next
  mv -f /usr/local/bin/mihomo.next /usr/local/bin/mihomo
fi
install -m 0644 "${UNIT_SOURCE}" "${SYSTEMD_DIR}/xiaomachi-mihomo.service"
systemctl daemon-reload
systemctl enable xiaomachi-mihomo.service >/dev/null
systemctl restart xiaomachi-mihomo.service
systemctl is-active --quiet xiaomachi-mihomo.service
echo "mihomo_install=ok version=${MIHOMO_VERSION} listen=127.0.0.1:7897"
