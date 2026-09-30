# The elementwise chain x = sqrt(x*x + y*y) * 0.5 + 0.25 over M x K elements,
# R passes (each feeds the next). Pure Python lists. usage: elementwise.py M K R
import math
import sys


def val(seed, i):
    # Deterministic input data shared with every other language: multiples of 1/64.
    return ((i * (2 * seed + 5) + 7 * seed + 1) % 64) / 64.0


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    r0, c1 = [val(0, i) for i in range(m)], [val(1, j) for j in range(k)]
    r2, c3 = [val(2, i) for i in range(m)], [val(3, j) for j in range(k)]
    x = [a * b for a in r0 for b in c1]
    y = [a * b for a in r2 for b in c3]
    sqrt = math.sqrt
    for _ in range(reps):
        x = [sqrt(a * a + b * b) * 0.5 + 0.25 for a, b in zip(x, y)]
    print(repr(math.fsum(x)), repr(x[0]), repr(x[-1]))


main()
