#!/bin/bash
cd /private/tmp/claude-501/-Users-talperez/9a6d07d9-2631-4597-beed-20bd8b102afe/scratchpad/research/r1/probe
SKY=/Users/talperez/sky/target/release/sky
echo "# $(date) uptime: $(uptime)"
for args in "--engine=native" "--engine=native --numerics=relaxed" "--engine=native --threads=1" "--engine=sir"; do
  python3 timeit.py 5 $SKY run $args p8_setup.sky
  python3 timeit.py 5 $SKY run $args p8_gemv.sky
done
echo "# end $(date)"
