# Cholesky solve K x = b, K = c1 c1' + c2 c2' + n I (n x n, dense, symmetric
# positive definite), R times with a different load each time. Pure Python.
# usage: cholesky.py N R
import math
import sys


def val(seed, i):
    # Deterministic input data shared with every other language: multiples of 1/64.
    return ((i * (2 * seed + 5) + 7 * seed + 1) % 64) / 64.0


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    c1 = [val(0, i) for i in range(n)]
    c2 = [val(1, i) for i in range(n)]
    f = [val(2, i) for i in range(n)]
    K = [[c1[i] * c1[j] + c2[i] * c2[j] + (n if i == j else 0.0) for j in range(n)] for i in range(n)]
    acc = 0.0
    x = [0.0] * n
    for r in range(reps):
        L = [row[:] for row in K]
        for j in range(n):
            Lj = L[j]
            s = Lj[j] - sum(t * t for t in Lj[:j])
            d = math.sqrt(s)
            Lj[j] = d
            head = Lj[:j]
            for i in range(j + 1, n):
                Li = L[i]
                Li[j] = (Li[j] - sum(a * b for a, b in zip(Li, head))) / d
        y = [0.0] * n
        for i in range(n):
            Li = L[i]
            y[i] = ((1.0 + r) * f[i] - sum(a * b for a, b in zip(Li[:i], y))) / Li[i]
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            t = y[i]
            for q in range(i + 1, n):
                t -= L[q][i] * x[q]
            x[i] = t / L[i][i]
        acc = acc + x[0] + x[n - 1] + sum(x)
    print(repr(acc), repr(x[0]), repr(x[n - 1]))


main()
