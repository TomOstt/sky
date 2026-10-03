#!/bin/bash
# Usage: benchlock.sh <command...>
# Runs a timing-sensitive command while holding a machine-wide lock, so that
# only one benchmark runs at a time. Stale locks (dead holder) are cleared.
LOCK="$(cd "$(dirname "$0")" && pwd)/bench.lock"
while ! mkdir "$LOCK" 2>/dev/null; do
  pid=$(cat "$LOCK/pid" 2>/dev/null)
  if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then rm -rf "$LOCK"; continue; fi
  sleep 3
done
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT
"$@"
