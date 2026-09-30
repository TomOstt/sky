# Dense matmul C = A B (n x n), repeated R times, each repetition rescaling the
# product so the next depends on it. Pure Python lists. usage: matmul.py N R
import sys


def val(seed, i):
    # Deterministic input data shared with every other language: multiples of 1/64.
    return ((i * (2 * seed + 5) + 7 * seed + 1) % 64) / 64.0


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    u0, v1 = [val(0, i) for i in range(n)], [val(1, i) for i in range(n)]
    u2, v3 = [val(2, i) for i in range(n)], [val(3, i) for i in range(n)]
    A = [[u0[i] * v1[j] + (1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]
    B = [[u2[i] * v3[j] + (1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]
    s = 1.0
    for _ in range(reps):
        C = []
        for i in range(n):
            row = [0.0] * n
            for aik, bk in zip(A[i], B):
                row = [c + aik * b for c, b in zip(row, bk)]
            C.append(row)
        s = 0.0
        for row in C:
            s += sum(row)
        A = [[c / s for c in row] for row in C]
    print(repr(s), repr(A[0][0]), repr(A[n - 1][n - 1]))


main()
