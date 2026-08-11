#!/usr/bin/env bash
set -u

echo "=== uptime ==="
uptime

echo "=== stack service start/fail history ==="
journalctl -u xiaomachi-stack.service --no-pager --since "12:30" \
  | grep -E "Starting|Started|Failed|Deactivated|Finished|Main process" | tail -40

echo "=== start.sh processes ==="
ps aux | grep -E "start\.sh|keepalive|anchor\.sh" | grep -v grep || true

echo "=== all xiaomachi-bot containers ==="
docker ps -a --filter name=xiaomachi-bot --format "{{.ID}} {{.Status}} {{.CreatedAt}}" 2>/dev/null || docker ps -a --filter name=xiaomachi-bot
