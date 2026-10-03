#!/bin/bash
# Runs the F (model overhead) benchmarks one by one, each under the bench lock.
S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad
cd $S/research
PY=$S/venv/bin/python
out=r3_out_F_1.txt
: > $out
for cfg in "tiny 36 128" "tiny 64 128" "real 8 128" "real32 3 128"; do
  bash $S/benchlock.sh $PY r3_bench_model_overhead.py $cfg >> $out 2>&1
done
for cfg in "tiny 64 128" "real 8 128"; do
  echo "## env MLX_MAX_OPS_PER_BUFFER=100000 MLX_MAX_MB_PER_BUFFER=100000" >> $out
  MLX_MAX_OPS_PER_BUFFER=100000 MLX_MAX_MB_PER_BUFFER=100000 bash $S/benchlock.sh $PY r3_bench_model_overhead.py $cfg >> $out 2>&1
done
echo "ALLDONE" >> $out
