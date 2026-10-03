#!/bin/bash
# Builds and runs sky-jit's `kernels` bench (sky vs nochk vs scalar vs C/Rust) from ~/sky main,
# into a scratch target dir (read-only for ~/sky).
S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad
OUT=$S/research/r1/kernel_bench
mkdir -p $OUT
export PATH=$HOME/.rustup/toolchains/1.85.1-aarch64-apple-darwin/bin:$PATH
export CARGO_TARGET_DIR=$S/research/r1/bench-target
export CARGO_TERM_COLOR=never
cd $HOME/sky
bash $S/benchlock.sh cargo bench -p sky-jit --bench kernels --locked --no-run > $OUT/build.log 2>&1
echo "build rc=$?" >> $OUT/build.log
bash $S/benchlock.sh bash -c "echo \"# \$(date) \$(uptime)\"; cargo bench -p sky-jit --bench kernels --locked; echo \"# end \$(date) \$(uptime)\"" > $OUT/run.log 2>&1
echo "run rc=$?" >> $OUT/run.log
