# The linear elementwise chain x = (x + y) * 0.25 + (x - y) * 0.125 over M x K
# elements, R passes (each feeds the next). NumPy, one temporary per operation,
# as written idiomatically. usage: linear_chain.py M K R
import sys

import numpy as np


def val(seed, n):
    # Deterministic input data shared with every other language: multiples of 1/64.
    i = np.arange(n, dtype=np.int64)
    return ((i * (2 * seed + 5) + 7 * seed + 1) % 64) / 64.0


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    x = np.outer(val(0, m), val(1, k))
    y = np.outer(val(2, m), val(3, k))
    for _ in range(reps):
        x = (x + y) * 0.25 + (x - y) * 0.125
    print(repr(float(x.sum())), repr(float(x[0, 0])), repr(float(x[m - 1, k - 1])))


main()
