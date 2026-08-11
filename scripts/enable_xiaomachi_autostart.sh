#!/usr/bin/env bash
set -euo pipefail

for unit in \
  /etc/systemd/system/xiaomachi-stack.service \
  /etc/systemd/system/xiaomachi-watchdog.service; do
  # Remove any corrupted literal-n lines left by broken printf quoting.
  sed -i '/^n\[Install\]n/d; /^nWantedBy=multi-user.targetn$/d' "$unit"
  if ! grep -q '^\[Install\]' "$unit"; then
    printf '\n[Install]\nWantedBy=multi-user.target\n' >> "$unit"
  fi
done

systemctl daemon-reload
systemctl enable xiaomachi-stack.service
systemctl enable xiaomachi-watchdog.service
echo "AUTOSTART_ENABLED_OK"
