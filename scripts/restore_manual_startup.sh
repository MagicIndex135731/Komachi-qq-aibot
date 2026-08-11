#!/usr/bin/env bash
set -euo pipefail

systemctl disable xiaomachi-stack.service xiaomachi-watchdog.service >/dev/null 2>&1 || true

for unit in \
  /etc/systemd/system/xiaomachi-stack.service \
  /etc/systemd/system/xiaomachi-watchdog.service; do
  sed -i '/^\[Install\]$/d; /^WantedBy=multi-user.target$/d' "$unit"
done

systemctl daemon-reload
echo "RESTORED"
grep -c "Install" /etc/systemd/system/xiaomachi-stack.service /etc/systemd/system/xiaomachi-watchdog.service || true
if ls /etc/systemd/system/multi-user.target.wants/ | grep -q xiaomachi; then
  echo "AUTOSTART_SYMLINKS_STILL_PRESENT"
else
  echo "NO_AUTOSTART_SYMLINKS"
fi
