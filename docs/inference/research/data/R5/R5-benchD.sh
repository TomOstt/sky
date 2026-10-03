#!/bin/bash
# R5 measurement D: dump the Metal encode stream (per node, concurrency, fusion) for a few decode steps
S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad
B=$S/src/llama.cpp/build/bin
R=$S/research/R5-raw
for M in Qwen3-8B-Q4_0 Qwen3-0.6B-Q4_0; do
bash $S/benchlock.sh bash -c "GGML_METAL_GRAPH_DEBUG=1 GGML_METAL_FUSION_DEBUG=2 $B/llama-completion -m $S/models/$M.gguf -p 'The capital of France is' -n 4 --temp 0 -ngl 99 -c 4096 -no-cnv -lv 5 > $R/D-graphdebug-$M.log 2>&1; echo exit=\$? >> $R/D-graphdebug-$M.log"
done
echo ALLDONE
