#!/usr/bin/env bash
set -u

echo "=== timers ==="
systemctl list-timers --all --no-pager 2>&1 | head -20

echo "=== crontabs ==="
crontab -l 2>&1 | head -20
ls /etc/cron.d/ 2>/dev/null
cat /etc/cron.d/* 2>/dev/null | head -40

echo "=== xiaomachi services ==="
systemctl list-units --all --no-pager 2>&1 | grep -i xiaomachi

echo "=== watchdog/keepalive processes ==="
ps aux | grep -E "onebot_watchdog|keepalive|anchor|start\.sh" | grep -v grep || true

echo "=== systemctl calls in scripts/app ==="
grep -rn "systemctl" /opt/xiaomachi/current/infra/wsl/scripts/ 2>/dev/null | head -20
grep -rn "systemctl" /opt/xiaomachi/current/app/ 2>/dev/null | head -10

echo "=== anchor lock ==="
ls -la /run/lock/xiaomachi-wsl-anchor.lock 2>&1
