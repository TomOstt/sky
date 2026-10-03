#!/usr/bin/env python3
"""timeit.py [-n N] [-w W] [--label L] -- cmd args...
Runs cmd W times untimed, then N times timed (wall clock, subprocess, stdout captured).
Prints: label median_ms min_ms max_ms last_stdout_line"""
import subprocess, sys, time, statistics
args = sys.argv[1:]
n, w, label = 7, 2, None
while args and args[0] != '--':
    a = args.pop(0)
    if a == '-n': n = int(args.pop(0))
    elif a == '-w': w = int(args.pop(0))
    elif a == '--label': label = args.pop(0)
args.pop(0)
label = label or ' '.join(args)[-60:]
out = ''
for _ in range(w):
    p = subprocess.run(args, capture_output=True, text=True)
ts = []
for _ in range(n):
    t0 = time.perf_counter()
    p = subprocess.run(args, capture_output=True, text=True)
    ts.append((time.perf_counter() - t0) * 1e3)
    if p.returncode != 0:
        print(f"{label}: FAILED rc={p.returncode} {p.stderr[-300:]}"); sys.exit(1)
    out = [l for l in p.stdout.splitlines() if l.strip()][-1] if p.stdout.strip() else ''
print(f"{label:<40} median {statistics.median(ts):9.2f} ms  min {min(ts):9.2f}  max {max(ts):9.2f}  n={n}  out: {out[:80]}")
