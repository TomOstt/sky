#!/usr/bin/env python3
"""Run a command N times (after 1 warmup), print each wall time and the median/min in ms."""
import subprocess, sys, time, statistics, os
n = int(sys.argv[1]); cmd = sys.argv[2:]
subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
ts = []
out = None
for _ in range(n):
    t = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    ts.append((time.perf_counter() - t) * 1000)
    out = (r.returncode, r.stdout.strip()[-80:], r.stderr.strip()[-200:])
print(f"cmd={' '.join(cmd)}")
print(f"runs_ms={[round(x,2) for x in ts]}")
print(f"median_ms={statistics.median(ts):.2f} min_ms={min(ts):.2f} rc={out[0]} stdout={out[1]!r} stderr_tail={out[2]!r}")
print(f"load={os.getloadavg()}")
