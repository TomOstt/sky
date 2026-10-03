#!/bin/bash
cd /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r1/probe
SKY=/Users/talperez/sky/target/release/sky
echo "# $(date) uptime: $(uptime)"
for p in p9_op32_0 p9_op32 p9_op8 p9_scalar; do
  python3 timeit.py 5 $SKY run --engine=native $p.sky
done
python3 timeit.py 3 $SKY run --engine=sir p9_op32.sky
echo "# end $(date)"
