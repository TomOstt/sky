#!/bin/bash
# R5 measurement C: speculative decoding, Qwen3-8B Q4_0 target + Qwen3-0.6B Q4_0 draft, greedy, one code prompt
S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad
B=$S/src/llama.cpp/build/bin
R=$S/research/R5-raw
T=$S/models/Qwen3-8B-Q4_0.gguf
D=$S/models/Qwen3-0.6B-Q4_0.gguf
P='Here is a quick sort implementation in C++. Just code, no comments:\n\n#include'
COMMON="-p \"$P\" -n 256 --temp 0 --top-k 1 --seed 42 -ngl 99 -c 4096 -e"
run() { local name=$1; shift
  bash $S/benchlock.sh bash -c "echo \"# \$(date) load: \$(uptime | sed 's/.*load/load/')\" > $R/$name.log; echo '# cmd: $*' >> $R/$name.log; $* >> $R/$name.log 2>&1; echo \"# exit=\$? end \$(date)\" >> $R/$name.log"
}
run C0-baseline-completion "$B/llama-completion -m $T $COMMON -no-cnv"
run C1-spec-draft-simple-n7 "$B/llama-speculative-simple -m $T -md $D -ngld 99 --spec-type draft-simple --spec-draft-n-max 7 $COMMON"
run C2-spec-draft-simple-n3 "$B/llama-speculative-simple -m $T -md $D -ngld 99 --spec-type draft-simple --spec-draft-n-max 3 $COMMON"
run C3-spec-ngram-simple "$B/llama-speculative-simple -m $T --spec-type ngram-simple $COMMON"
echo ALLDONE
