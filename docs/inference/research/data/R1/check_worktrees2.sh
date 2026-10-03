#!/bin/bash
# v2: same as v1, but deletes the sky-* fingerprints between worktrees, because cargo hashes path
# packages relative to the workspace root and judges freshness by mtime: with a shared target dir the
# (older) worktree sources were wrongly considered fresh in v1. Read-only for the worktrees.
S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad
OUT=$S/research/r1/wtcheck2
mkdir -p $OUT
export PATH=$HOME/.rustup/toolchains/1.85.1-aarch64-apple-darwin/bin:$PATH
export CARGO_TARGET_DIR=$S/research/r1/wtcheck-target
export CARGO_TERM_COLOR=never
for wt in $HOME/sky/.claude/worktrees/agent-*; do
  n=$(basename $wt)
  start=$(date +%s)
  bash $S/benchlock.sh bash -c "rm -rf $CARGO_TARGET_DIR/debug/.fingerprint/sky-* && cargo check --manifest-path $wt/Cargo.toml --workspace --tests --locked --message-format=short" > $OUT/$n.log 2>&1
  rc=$?
  end=$(date +%s)
  checked=$(grep -c 'Checking sky-' $OUT/$n.log)
  errs=$(grep -c ': error' $OUT/$n.log)
  warns=$(grep -c ': warning' $OUT/$n.log)
  echo "$n rc=$rc sky_crates_checked=$checked errors=$errs warnings=$warns secs=$((end-start))" >> $OUT/summary.txt
done
echo DONE >> $OUT/summary.txt
