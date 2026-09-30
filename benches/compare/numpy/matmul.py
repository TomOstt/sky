# Dense matmul C = A B (n x n), repeated R times, each repetition rescaling the
# product so the next depends on it. NumPy (BLAS underneath). usage: matmul.py N R
import sys

import numpy as np


def val(seed, n):
    # Deterministic input data shared with every other language: multiples of 1/64.
    i = np.arange(n, dtype=np.int64)
    return ((i * (2 * seed + 5) + 7 * seed + 1) % 64) / 64.0


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    A = np.outer(val(0, n), val(1, n)) + np.eye(n)
    B = np.outer(val(2, n), val(3, n)) + np.eye(n)
    s = 1.0
    for _ in range(reps):
        C = A @ B
        s = C.sum()
        A = C / s
    print(repr(float(s)), repr(float(A[0, 0])), repr(float(A[n - 1, n - 1])))


main()
