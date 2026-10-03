#!/bin/bash
# v3: for the worktrees that failed v2, (a) libs+bins only, (b) tests with --keep-going to count all errors.
S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad
OUT=$S/research/r1/wtcheck3
mkdir -p $OUT
export PATH=$HOME/.rustup/toolchains/1.85.1-aarch64-apple-darwin/bin:$PATH
export CARGO_TARGET_DIR=$S/research/r1/wtcheck-target
export CARGO_TERM_COLOR=never
for n in agent-a1dadc02119fe9b29 agent-ad1ea58eab8e832fb agent-afa55e105ab8ee772; do
  wt=$HOME/sky/.claude/worktrees/$n
  bash $S/benchlock.sh bash -c "rm -rf $CARGO_TARGET_DIR/debug/.fingerprint/sky-* && cargo check --manifest-path $wt/Cargo.toml --workspace --locked --message-format=short" > $OUT/$n.lib.log 2>&1
  rc1=$?
  bash $S/benchlock.sh bash -c "cargo check --manifest-path $wt/Cargo.toml --workspace --tests --keep-going --locked --message-format=short" > $OUT/$n.tests.log 2>&1
  rc2=$?
  echo "$n libs_bins rc=$rc1 errors=$(grep -c ': error' $OUT/$n.lib.log) | tests(keep-going) rc=$rc2 errors=$(grep -c ': error' $OUT/$n.tests.log) failed_targets=$(grep -c 'could not compile' $OUT/$n.tests.log)" >> $OUT/summary.txt
done
echo DONE >> $OUT/summary.txt
