T="python3 /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/timeit.py -n 9 -w 2"
SKY=/Users/talperez/sky/target/release/sky
$T --label "sky --help (process floor)" -- $SKY --help
$T --label "sky check floor.sky" -- $SKY check /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/var/floor.sky
$T --label "sky check nbody" -- $SKY check /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/sky/nbody.sky
$T --label "sky check --emit=clif nbody" -- $SKY check --emit=clif /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/sky/nbody.sky
$T --label "sky run native nbody steps=1" -- $SKY run --engine=native /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/var/nbody-1.sky
SKY_SLP=off $T --label "sky run native nbody steps=1 SLP=off" -- $SKY run --engine=native /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/var/nbody-1.sky
$T --label "sky run sir nbody steps=1" -- $SKY run --engine=sir /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/var/nbody-1.sky
$T --label "sky run native fib(2)" -- $SKY run --engine=native /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/var/fib-2.sky
$T --label "sky build obj nbody" -- $SKY build --emit=obj -o /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/nbody_tmp.o /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r10/sky/nbody.sky
