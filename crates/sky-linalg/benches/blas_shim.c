/* The vendor-BLAS side of `benches/blas.rs`: times dgemm, dpotrf and dgesv
 * on the same sizes and with the same method as the Rust side, and prints
 * one line per measurement:
 *
 *     <library> <op> <n> <gflops>
 *
 * macOS:  /usr/bin/clang -O3 -framework Accelerate blas_shim.c
 * Linux:  cc -O3 blas_shim.c -lopenblas   (OpenBLAS with its LAPACK)
 *
 * Only built and run when the toolchain and library are there. Threads are
 * chosen by the environment (VECLIB_MAXIMUM_THREADS / OPENBLAS_NUM_THREADS).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
#define LIBNAME "accelerate"
#include <Accelerate/Accelerate.h>
#else
#define LIBNAME "openblas"
#include <cblas.h>
extern void dpotrf_(const char *, const int *, double *, const int *, int *);
extern void dgesv_(const int *, const int *, double *, const int *, int *,
                   double *, const int *, int *);
#endif

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static uint64_t state = 0x9E3779B97F4A7C15ull;
static double rnd(void) { /* uniform in [-0.5, 0.5) */
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return (double)(state >> 11) / 9007199254740992.0 - 0.5;
}

/* Best (minimum) time per call over many short samples of about 3 ms: on a
 * machine doing other work, the minimum finds the undisturbed windows. */
typedef void (*body_fn)(void);
static double best_time(body_fn body) {
  body();
  long reps = 1;
  for (;;) {
    double t0 = now();
    for (long i = 0; i < reps; i++) body();
    double dt = now() - t0;
    if (dt >= 0.003 || reps >= (1L << 24)) break;
    reps *= 2;
  }
  double best = 1e30, start = now();
  for (int s = 0; s < 400; s++) {
    double t0 = now();
    for (long i = 0; i < reps; i++) body();
    double dt = (now() - t0) / (double)reps;
    if (dt < best) best = dt;
    if (s >= 5 && now() - start > 1.5) break;
  }
  return best;
}

static int N;
static double *A, *B, *C, *W, *V, *SPD;
static int *PIV;

static void run_gemm(void) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B,
              N, 0.0, C, N);
}

static void run_potrf(void) {
  memcpy(W, SPD, sizeof(double) * (size_t)N * (size_t)N);
  char uplo = 'L';
  int n = N, info = 0;
  dpotrf_(&uplo, &n, W, &n, &info);
}

static void run_gesv(void) {
  memcpy(W, A, sizeof(double) * (size_t)N * (size_t)N);
  memcpy(V, B, sizeof(double) * (size_t)N);
  int n = N, nrhs = 1, info = 0;
  dgesv_(&n, &nrhs, W, &n, PIV, V, &n, &info);
}

int main(int argc, char **argv) {
  for (int a = 1; a < argc; a++) {
    int n = atoi(argv[a]);
    N = n;
    size_t nn = (size_t)n * (size_t)n;
    A = malloc(nn * sizeof(double));
    B = malloc(nn * sizeof(double));
    C = malloc(nn * sizeof(double));
    W = malloc(nn * sizeof(double));
    SPD = malloc(nn * sizeof(double));
    V = malloc((size_t)n * sizeof(double));
    PIV = malloc((size_t)n * sizeof(int));
    for (size_t i = 0; i < nn; i++) {
      A[i] = 2.0 * rnd();
      B[i] = 2.0 * rnd();
    }
    for (int i = 0; i < n; i++)
      for (int j = 0; j <= i; j++) {
        double x = rnd();
        SPD[(size_t)i * n + j] = SPD[(size_t)j * n + i] = x;
      }
    for (int i = 0; i < n; i++) SPD[(size_t)i * n + i] += (double)n;
    double nd = (double)n;
    printf("%s gemm %d %.4f\n", LIBNAME, n, 2.0 * nd * nd * nd / best_time(run_gemm) / 1e9);
    printf("%s potrf %d %.4f\n", LIBNAME, n, (nd * nd * nd / 3.0) / best_time(run_potrf) / 1e9);
    printf("%s gesv %d %.4f\n", LIBNAME, n,
           (2.0 / 3.0 * nd * nd * nd + 2.0 * nd * nd) / best_time(run_gesv) / 1e9);
    fflush(stdout);
    free(A); free(B); free(C); free(W); free(SPD); free(V); free(PIV);
  }
  return 0;
}
