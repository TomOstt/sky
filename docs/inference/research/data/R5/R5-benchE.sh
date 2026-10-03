#!/bin/bash
# R5 measurement E: kernel-level GEMV/GEMM throughput via test-backend-ops perf (m=4096, k=14336 = llama-8B FFN shape)
S=/private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad
B=$S/src/llama.cpp/build/bin
R=$S/research/R5-raw
bash $S/benchlock.sh bash -c "echo \"# \$(date) load: \$(uptime | sed 's/.*load/load/')\" > $R/E-mulmat-perf.log; timeout 170 $B/test-backend-ops perf -b MTL0 -o MUL_MAT -p 'type_a=(q4_0|q4_K|q8_0|mxfp4|q6_K|f16),type_b=f32,m=4096,n=(1|2|3|4|5|8|512),k=14336' >> $R/E-mulmat-perf.log 2>&1; echo \"# exit=\$? end \$(date)\" >> $R/E-mulmat-perf.log"
echo ALLDONE
