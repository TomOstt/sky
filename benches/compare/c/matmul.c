// Dense matmul C = A B (n x n), repeated R times, each repetition rescaling the
// product so the next depends on it. usage: matmul N R
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

int main(int argc, char **argv) {
    long n = argc > 1 ? atol(argv[1]) : 256;
    long reps = argc > 2 ? atol(argv[2]) : 1;
    double *A = malloc(n * n * sizeof(double));
    double *B = malloc(n * n * sizeof(double));
    double *C = malloc(n * n * sizeof(double));
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++) {
            A[i * n + j] = val(0, i) * val(1, j) + (i == j ? 1.0 : 0.0);
            B[i * n + j] = val(2, i) * val(3, j) + (i == j ? 1.0 : 0.0);
        }
    double s = 1.0;
    for (long r = 0; r < reps; r++) {
        for (long i = 0; i < n * n; i++) C[i] = 0.0;
        for (long i = 0; i < n; i++)
            for (long k = 0; k < n; k++) {
                double a = A[i * n + k];
                for (long j = 0; j < n; j++) C[i * n + j] += a * B[k * n + j];
            }
        s = 0.0;
        for (long i = 0; i < n * n; i++) s += C[i];
        for (long i = 0; i < n * n; i++) A[i] = C[i] / s;
    }
    printf("%.17g %.17g %.17g\n", s, A[0], A[n * n - 1]);
    return 0;
}
