#!/bin/bash
# R5 measurement G: cost of one decode call with n = 1..8 (and 16) tokens = speculative verification cost curve
S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad
B=$S/src/llama.cpp/build/bin
R=$S/research/R5-raw
OUT=$R/G-smallbatch.md
bash $S/benchlock.sh bash -c "
  echo \"# \$(date) load: \$(uptime | sed 's/.*load/load/')\" > $OUT
  for M in Qwen3-8B-Q4_0 Qwen3-8B-Q4_K_M Qwen3-0.6B-Q4_0; do
    echo \"## $B/llama-bench -m $S/models/\$M.gguf -p 1,2,3,4,5,6,7,8,16 -n 0 -fa on -r 10 -o md\" >> $OUT
    $B/llama-bench -m $S/models/\$M.gguf -p 1,2,3,4,5,6,7,8,16 -n 0 -fa on -r 10 -o md >> $OUT 2>/dev/null
  done
  echo \"# end \$(date)\" >> $OUT"
echo ALLDONE
