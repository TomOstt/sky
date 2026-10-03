#!/bin/bash
# R5 measurement F: ablation of Metal-backend features on decode (tg128) - env toggles that exist in the code
S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad
B=$S/src/llama.cpp/build/bin
R=$S/research/R5-raw
CFGS="BASE GGML_METAL_FUSION_DISABLE=1 GGML_METAL_CONCURRENCY_DISABLE=1 GGML_METAL_GRAPH_OPTIMIZE_DISABLE=1 LLAMA_GRAPH_REUSE_DISABLE=1"
abl() { # model reps out
  local M=$1 REPS=$2 OUT=$3
  bash $S/benchlock.sh bash -c "
    echo \"# \$(date) load: \$(uptime | sed 's/.*load/load/')\" >> $OUT
    for C in $CFGS; do
      if [ \$C = BASE ]; then E=''; else E=\$C; fi
      echo \"## env: \$C\" >> $OUT
      env \$E $B/llama-bench -m $S/models/$M.gguf -p 0 -n 128 -fa on -r $REPS -o md >> $OUT 2>/dev/null
    done
    echo \"# end \$(date)\" >> $OUT"
}
abl Qwen3-0.6B-Q4_0 5 $R/F-ablation-0.6B.md
abl Qwen3-8B-Q4_0 3 $R/F-ablation-8B.md
echo ALLDONE
