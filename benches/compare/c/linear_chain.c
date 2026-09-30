// The linear elementwise chain x = (x + y) * 0.25 + (x - y) * 0.125 over M x K
// elements, R passes (each feeds the next). usage: linear_chain M K R
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

int main(int argc, char **argv) {
    long m = argc > 1 ? atol(argv[1]) : 1000;
    long k = argc > 2 ? atol(argv[2]) : 1000;
    long reps = argc > 3 ? atol(argv[3]) : 1;
    long len = m * k;
    double *x = malloc(len * sizeof(double));
    double *y = malloc(len * sizeof(double));
    for (long i = 0; i < m; i++)
        for (long j = 0; j < k; j++) {
            x[i * k + j] = val(0, i) * val(1, j);
            y[i * k + j] = val(2, i) * val(3, j);
        }
    for (long r = 0; r < reps; r++)
        for (long i = 0; i < len; i++) x[i] = (x[i] + y[i]) * 0.25 + (x[i] - y[i]) * 0.125;
    double sum = 0.0;
    for (long i = 0; i < len; i++) sum += x[i];
    printf("%.17g %.17g %.17g\n", sum, x[0], x[len - 1]);
    return 0;
}
