// The same baselines in C, for `clang -O3 -march=native -ffp-contract=off
// -fno-math-errno` (no contraction, so the bits are Sky's; no errno so that
// sqrt vectorizes; neither changes any result). Output lines as baseline.rs:
// `c <kernel> <variant> <n> <ns per element>`.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static inline int finite(double r) { return fabs(r) <= 1.7976931348623157e308; }

#define RESTRICT __restrict__

__attribute__((noinline)) static void axpy(double *RESTRICT out, const double *RESTRICT x,
                                           const double *RESTRICT y, double a, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = a * x[i] + y[i];
}

__attribute__((noinline)) static int axpy_checked(double *RESTRICT out, const double *RESTRICT x,
                                                  const double *RESTRICT y, double a, size_t n) {
    int ok = 1;
    for (size_t i = 0; i < n; i++) {
        double r = a * x[i] + y[i];
        ok &= finite(r);
        out[i] = r;
    }
    return ok;
}

__attribute__((noinline)) static void hypot_k(double *RESTRICT out, const double *RESTRICT x,
                                              const double *RESTRICT y, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = sqrt(x[i] * x[i] + y[i] * y[i]) + 0.0;
}

__attribute__((noinline)) static int hypot_checked(double *RESTRICT out, const double *RESTRICT x,
                                                   const double *RESTRICT y, size_t n) {
    int ok = 1;
    for (size_t i = 0; i < n; i++) {
        double r = sqrt(x[i] * x[i] + y[i] * y[i]) + 0.0;
        ok &= finite(r);
        out[i] = r;
    }
    return ok;
}

__attribute__((noinline)) static void chain6(double *RESTRICT out, const double *RESTRICT x,
                                             const double *RESTRICT y, double s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        double t2 = x[i] * s + y[i];
        double t4 = x[i] * x[i] + 1.5;
        out[i] = t2 / t4 - y[i];
    }
}

__attribute__((noinline)) static int chain6_checked(double *RESTRICT out, const double *RESTRICT x,
                                                    const double *RESTRICT y, double s, size_t n) {
    int ok = 1;
    for (size_t i = 0; i < n; i++) {
        double t2 = x[i] * s + y[i];
        double t4 = x[i] * x[i] + 1.5;
        ok &= finite(t4);
        double r = t2 / t4 - y[i];
        ok &= finite(r);
        out[i] = r;
    }
    return ok;
}

static volatile int sink;

#define BENCH(kernel, variant, call)                                  \
    do {                                                              \
        size_t reps = 20000000 / n;                                   \
        if (reps < 1) reps = 1;                                       \
        call;                                                         \
        double best = 1e300;                                          \
        for (int b = 0; b < 9; b++) {                                 \
            double t = now();                                         \
            for (size_t r = 0; r < reps; r++) call;                   \
            double e = (now() - t) / (double)(reps * n);              \
            if (e < best) best = e;                                   \
        }                                                             \
        printf("c %s %s %zu %.4f\n", kernel, variant, n, best * 1e9); \
    } while (0)

int main(void) {
    size_t sizes[] = {1000, 100000, 10000000};
    for (int k = 0; k < 3; k++) {
        size_t n = sizes[k];
        double *x = malloc(n * 8), *y = malloc(n * 8), *out = malloc(n * 8);
        for (size_t i = 0; i < n; i++) {
            x[i] = (double)(i % 1000) * 0.001 + 0.5;
            y[i] = (double)(i % 777) * 0.002 - 0.7;
            out[i] = 0.0;
        }
        volatile double sv = 2.5;
        double s = sv;
        BENCH("axpy", "plain", axpy(out, x, y, s, n));
        BENCH("axpy", "checked", sink = axpy_checked(out, x, y, s, n));
        BENCH("hypot", "plain", hypot_k(out, x, y, n));
        BENCH("hypot", "checked", sink = hypot_checked(out, x, y, n));
        BENCH("chain6", "plain", chain6(out, x, y, s, n));
        BENCH("chain6", "checked", sink = chain6_checked(out, x, y, s, n));
        sink = (int)out[n / 2];
        free(x);
        free(y);
        free(out);
    }
    return 0;
}
