#!/bin/bash
# R5 measurement A: llama-bench, Qwen3 0.6B/8B Q4_0, pp128/512/2048 + tg128, FA off/on, r=5
S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad
B=$S/src/llama.cpp/build/bin
R=$S/research/R5-raw
run() { # name, args...
  local name=$1; shift
  bash $S/benchlock.sh bash -c "echo \"# \$(date) load: \$(uptime | sed 's/.*load/load/')\" > $R/$name.md; echo '# cmd: $B/llama-bench $*' >> $R/$name.md; $B/llama-bench $* -oe json >> $R/$name.md 2> $R/$name.err; echo \"# exit=\$? end \$(date)\" >> $R/$name.md"
}
run A1-0.6B-fa01 -m $S/models/Qwen3-0.6B-Q4_0.gguf -p 128,512,2048 -n 128 -fa off,on -r 5 -o md
run A2-8B-Q4_0-fa0 -m $S/models/Qwen3-8B-Q4_0.gguf -p 128,512,2048 -n 128 -fa off -r 5 -o md
run A3-8B-Q4_0-fa1 -m $S/models/Qwen3-8B-Q4_0.gguf -p 128,512,2048 -n 128 -fa on -r 5 -o md
echo ALLDONE
