// Cholesky solve K x = b, K = c1 c1' + c2 c2' + n I (n x n, dense, symmetric
// positive definite), R times with a different load each time.
// usage: cholesky N R
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

int main(int argc, char **argv) {
    long n = argc > 1 ? atol(argv[1]) : 256;
    long reps = argc > 2 ? atol(argv[2]) : 1;
    double *K = malloc(n * n * sizeof(double));
    double *L = malloc(n * n * sizeof(double));
    double *f = malloc(n * sizeof(double));
    double *y = malloc(n * sizeof(double));
    double *x = malloc(n * sizeof(double));
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++)
            K[i * n + j] = val(0, i) * val(0, j) + val(1, i) * val(1, j) + (i == j ? (double)n : 0.0);
        f[i] = val(2, i);
    }
    double acc = 0.0;
    for (long r = 0; r < reps; r++) {
        memcpy(L, K, n * n * sizeof(double));
        // K = L L' (lower triangle, row-oriented).
        for (long j = 0; j < n; j++) {
            double s = L[j * n + j];
            for (long k = 0; k < j; k++) s -= L[j * n + k] * L[j * n + k];
            double d = sqrt(s);
            L[j * n + j] = d;
            for (long i = j + 1; i < n; i++) {
                double t = L[i * n + j];
                for (long k = 0; k < j; k++) t -= L[i * n + k] * L[j * n + k];
                L[i * n + j] = t / d;
            }
        }
        // L y = b, then L' x = y.
        for (long i = 0; i < n; i++) {
            double t = (1.0 + r) * f[i];
            for (long k = 0; k < i; k++) t -= L[i * n + k] * y[k];
            y[i] = t / L[i * n + i];
        }
        for (long i = n - 1; i >= 0; i--) {
            double t = y[i];
            for (long k = i + 1; k < n; k++) t -= L[k * n + i] * x[k];
            x[i] = t / L[i * n + i];
        }
        double sum = 0.0;
        for (long i = 0; i < n; i++) sum += x[i];
        acc = acc + x[0] + x[n - 1] + sum;
    }
    printf("%.17g %.17g %.17g\n", acc, x[0], x[n - 1]);
    return 0;
}
