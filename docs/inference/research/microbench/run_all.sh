#!/bin/bash
# Runs every microbenchmark, each under the machine-wide bench lock (each run < ~2 min).
S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad
cd $S/research/microbench
for b in "$@"; do
  echo "[$(date +%T)] waiting for lock: $b"
  start=$(date +%s)
  bash $S/benchlock.sh bash -c "echo LOCK_ACQUIRED \$(date +%T); /usr/bin/time -p ./bin/$b" > results/$b.txt 2>&1
  echo "[$(date +%T)] done $b rc=$? (total $(( $(date +%s) - start ))s)"
done
