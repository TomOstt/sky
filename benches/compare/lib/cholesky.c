// The Cholesky workload through LAPACK (dpotrf + dpotrs): Apple Accelerate on
// macOS, OpenBLAS elsewhere. Same data and driver as ../c/cholesky.c.
// usage: cholesky N R
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../c/common.h"

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
typedef __LAPACK_int lapack_int_t;
#else
typedef int lapack_int_t;
extern void dpotrf_(const char *uplo, const lapack_int_t *n, double *a, const lapack_int_t *lda,
                    lapack_int_t *info);
extern void dpotrs_(const char *uplo, const lapack_int_t *n, const lapack_int_t *nrhs,
                    const double *a, const lapack_int_t *lda, double *b, const lapack_int_t *ldb,
                    lapack_int_t *info);
#endif

int main(int argc, char **argv) {
    long n = argc > 1 ? atol(argv[1]) : 256;
    long reps = argc > 2 ? atol(argv[2]) : 1;
    double *K = malloc(n * n * sizeof(double));
    double *L = malloc(n * n * sizeof(double));
    double *f = malloc(n * sizeof(double));
    double *x = malloc(n * sizeof(double));
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++)
            K[i * n + j] = val(0, i) * val(0, j) + val(1, i) * val(1, j) + (i == j ? (double)n : 0.0);
        f[i] = val(2, i);
    }
    double acc = 0.0;
    lapack_int_t ni = (lapack_int_t)n, one = 1, info = 0;
    for (long r = 0; r < reps; r++) {
        memcpy(L, K, n * n * sizeof(double));
        for (long i = 0; i < n; i++) x[i] = (1.0 + r) * f[i];
        // K is symmetric, so row-major and column-major layouts agree.
        dpotrf_("L", &ni, L, &ni, &info);
        if (info != 0) {
            fprintf(stderr, "dpotrf failed: %d\n", (int)info);
            return 1;
        }
        dpotrs_("L", &ni, &one, L, &ni, x, &ni, &info);
        if (info != 0) {
            fprintf(stderr, "dpotrs failed: %d\n", (int)info);
            return 1;
        }
        double sum = 0.0;
        for (long i = 0; i < n; i++) sum += x[i];
        acc = acc + x[0] + x[n - 1] + sum;
    }
    printf("%.17g %.17g %.17g\n", acc, x[0], x[n - 1]);
    return 0;
}
