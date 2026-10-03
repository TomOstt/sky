cd /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/micro
clang -O3 -march=native idx.c -o idx_c
T="python3 /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/timeit.py -n 7 -w 2"
SKY=/Users/talperez/sky/target/release/sky
$T --label "c idx 2.5e5" -- ./idx_c 250000
$T --label "c idx 1e6" -- ./idx_c 1000000
$T --label "sky native idx 2.5e5" -- $SKY run --engine=native idx2_5e5.sky
$T --label "sky native idx 1e6" -- $SKY run --engine=native idx1e6.sky
$T --label "sky sir idx 2.5e5" -- $SKY run --engine=sir idx2_5e5.sky
$T --label "sky sir idx 1e6" -- $SKY run --engine=sir idx1e6.sky
