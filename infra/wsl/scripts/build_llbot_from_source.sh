#!/usr/bin/env bash
# Rebuild the local LLBot image from an upstream git ref.
#
# The latest release (8.1.10) predates the session-expiry fix (PR #856), so the
# stack runs a source build of upstream main.  This script downloads the ref,
# builds it with infra/wsl/Dockerfile.llbot-source (which fixes the WebUI build
# order), and verifies the resulting image before it can be used.
#
# Usage:
#   LLBOT_IMAGE_TAG=xiaomachi-llbot:main-<commit> \
#     bash infra/wsl/scripts/build_llbot_from_source.sh [ref]
set -euo pipefail

llbot_ref="${1:-${LLBOT_REF:-main}}"
image_tag="${LLBOT_IMAGE_TAG:-}"
build_proxy="${BUILD_PROXY:-http://127.0.0.1:7897}"
source_dir="${LLBOT_SOURCE_DIR:-/root/llbot-main}"
upstream_url="https://github.com/LLOneBot/LuckyLilliaBot.git"

if [[ -z "${image_tag}" ]]; then
  echo "Set LLBOT_IMAGE_TAG, for example xiaomachi-llbot:main-\$(git-commit-prefix)." >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
dockerfile="${repo_root}/infra/wsl/Dockerfile.llbot-source"
[[ -f "${dockerfile}" ]] || { echo "Missing ${dockerfile}" >&2; exit 1; }

echo "Resolving ${llbot_ref} from ${upstream_url} ..."
commit="$(
  git -c "http.proxy=${build_proxy}" ls-remote "${upstream_url}" "refs/heads/${llbot_ref}" \
    | awk '{print $1}' | head -n 1
)"
if [[ ! "${commit}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Cannot resolve ref ${llbot_ref} (got '${commit}')." >&2
  exit 1
fi
echo "Commit: ${commit}"

tarball="$(mktemp -d)/llbot-source.tar.gz"
echo "Downloading tarball ..."
curl -sSL --max-time 600 -x "${build_proxy}" \
  "https://codeload.github.com/LLOneBot/LuckyLilliaBot/tar.gz/${commit}" \
  -o "${tarball}"
gzip -t "${tarball}"

rm -rf "${source_dir}"
mkdir -p "${source_dir}"
tar -xzf "${tarball}" -C "${source_dir}" --strip-components=1
rm -rf "$(dirname "${tarball}")"
# The directory is gitignored upstream but docker/Dockerfile.local copies it.
mkdir -p "${source_dir}/.yarn"
install -m 0644 "${dockerfile}" "${source_dir}/docker/Dockerfile.local.patched"

echo "Building ${image_tag} ..."
(cd "${source_dir}" && docker build \
  -f docker/Dockerfile.local.patched \
  --network=host \
  --build-arg "BUILD_PROXY=${build_proxy}" \
  -t "${image_tag}" .)

echo "Verifying ${image_tag} ..."
docker run --rm --entrypoint sh "${image_tag}" -c '
  set -e
  test -s /app/llbot/webui/index.html
  test -s /app/llbot/llbot.js
  ls /app/llbot/sign-proxy.*-musl.node >/dev/null
  node --check /app/llbot/llbot.js
  grep -q "session-expired" /app/llbot/llbot.js
  echo "image_ok"
'

echo "Built ${image_tag} from ${llbot_ref} (${commit})."
echo "Update infra/wsl/docker-compose.llbot.yml to this tag, then run:"
echo "  docker compose -f infra/wsl/docker-compose.llbot.yml up -d --no-deps --force-recreate llbot"
