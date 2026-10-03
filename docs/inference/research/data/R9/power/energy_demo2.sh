#!/bin/bash
# Run under benchlock: idle baseline (10 s) -> generation -> tail (8 s), with macmon sampling every 200 ms.
W=$(cd "$(dirname "$0")" && pwd)
S=$(cd "$W/../../.." && pwd)
notifyutil -g com.apple.system.thermalpressurelevel > $W/demo_thermal_before.txt
$W/macmon pipe -i 200 > $W/demo_macmon.jsonl 2> $W/demo_macmon.err &
MM=$!
python3 -c 'import time;print(time.time())' > $W/demo_idle_start.txt
sleep 10
python3 -c 'import time;print(time.time())' > $W/demo_idle_end.txt
$S/venv/bin/python $W/energy_demo2.py mlx-community/Qwen3-0.6B-4bit 3000 12 > $W/demo_out.txt 2> $W/demo_err.txt
sleep 8
kill $MM
notifyutil -g com.apple.system.thermalpressurelevel > $W/demo_thermal_after.txt
