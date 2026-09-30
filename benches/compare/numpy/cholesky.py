# Cholesky solve K x = b, K = c1 c1' + c2 c2' + n I (n x n, dense, symmetric
# positive definite), R times with a different load each time. NumPy has no
# triangular solver (SciPy does), so the two triangular solves go through the
# general np.linalg.solve: an honest cost of staying in NumPy alone.
# usage: cholesky.py N R
import sys

import numpy as np


def val(seed, n):
    # Deterministic input data shared with every other language: multiples of 1/64.
    i = np.arange(n, dtype=np.int64)
    return ((i * (2 * seed + 5) + 7 * seed + 1) % 64) / 64.0


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    c1, c2, f = val(0, n), val(1, n), val(2, n)
    K = np.outer(c1, c1) + np.outer(c2, c2) + n * np.eye(n)
    acc = 0.0
    x = np.zeros(n)
    for r in range(reps):
        L = np.linalg.cholesky(K)
        y = np.linalg.solve(L, (1.0 + r) * f)
        x = np.linalg.solve(L.T, y)
        acc = acc + x[0] + x[n - 1] + x.sum()
    print(repr(float(acc)), repr(float(x[0])), repr(float(x[n - 1])))


main()
