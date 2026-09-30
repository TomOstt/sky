// The matmul workload through a hand-tuned BLAS: Apple Accelerate on macOS,
// OpenBLAS elsewhere. Same data and same driver as ../c/matmul.c; only the
// product is cblas_dgemm. usage: matmul N R
#include <stdio.h>
#include <stdlib.h>

#include "../c/common.h"

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

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
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)n, (int)n, (int)n, 1.0, A,
                    (int)n, B, (int)n, 0.0, C, (int)n);
        s = 0.0;
        for (long i = 0; i < n * n; i++) s += C[i];
        for (long i = 0; i < n * n; i++) A[i] = C[i] / s;
    }
    printf("%.17g %.17g %.17g\n", s, A[0], A[n * n - 1]);
    return 0;
}
