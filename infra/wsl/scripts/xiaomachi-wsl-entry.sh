#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
case "${ACTION}" in
  start|stop|status) ;;
  *)
    echo "Usage: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac

for base in /mnt/d /mnt/e /mnt/c; do
  [[ -d "${base}" ]] || continue
  while IFS= read -r script_path; do
    repo_root="${script_path%/infra/wsl/scripts/${ACTION}.sh}"
    if [[ -f "${repo_root}/pyproject.toml" ]]; then
      echo "Repo=${repo_root}"
      cd "${repo_root}"
      exec bash "infra/wsl/scripts/${ACTION}.sh"
    fi
  done < <(
    find "${base}" \
      -mindepth 4 \
      -maxdepth 6 \
      -path "*/infra/wsl/scripts/${ACTION}.sh" \
      -type f \
      2>/dev/null
  )
done

echo "Cannot find xiaomachi repo under /mnt/d, /mnt/e, or /mnt/c." >&2
exit 1
