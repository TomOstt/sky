T="python3 /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/timeit.py -n 7 -w 1"
SKY=/Users/talperez/sky/target/release/sky
$T --label "check elem1e7 (front end)" -- $SKY check /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/sky/elementwise-1e7.sky
$T --label "native elem1e7 R=0" -- $SKY run --engine=native /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/var/elem1e7-r0.sky
$T --label "native elem1e7 build-only" -- $SKY run --engine=native /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/var/elem1e7-build-only.sky
$T --label "native T1 elem1e7 build-only" -- $SKY run --engine=native --threads=1 /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/var/elem1e7-build-only.sky
$T --label "check chol32 (front end)" -- $SKY check /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/sky/cholesky-32.sky
$T --label "check matmul512 (front end)" -- $SKY check /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/sky/matmul-512.sky
