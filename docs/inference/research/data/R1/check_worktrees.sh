#!/bin/bash
# Compile-checks main and each agent worktree (read-only: separate target dir, --locked).
S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad
OUT=$S/research/r1/wtcheck
mkdir -p $OUT
export PATH=$HOME/.rustup/toolchains/1.85.1-aarch64-apple-darwin/bin:$PATH
export CARGO_TARGET_DIR=$S/research/r1/wtcheck-target
export CARGO_TERM_COLOR=never
for wt in $HOME/sky $HOME/sky/.claude/worktrees/agent-*; do
  n=$(basename $wt)
  start=$(date +%s)
  bash $S/benchlock.sh cargo check --manifest-path $wt/Cargo.toml --workspace --tests --locked --message-format=short > $OUT/$n.log 2>&1
  rc=$?
  end=$(date +%s)
  errs=$(grep -c '^error' $OUT/$n.log)
  warns=$(grep -c ': warning' $OUT/$n.log)
  echo "$n rc=$rc errors=$errs warnings=$warns secs=$((end-start))" >> $OUT/summary.txt
done
echo DONE >> $OUT/summary.txt
