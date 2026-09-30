// Shared by the C programs: the deterministic input data every language builds
// the same way. val(seed, i) is a multiple of 1/64 in [0, 1), so products of two
// values and their sums are exact in double precision and the inputs are
// bit-identical everywhere. seed < 0 gives all ones.
#ifndef SKY_COMPARE_COMMON_H
#define SKY_COMPARE_COMMON_H

static inline double val(long seed, long i) {
    if (seed < 0) return 1.0;
    return (double)((i * (2 * seed + 5) + 7 * seed + 1) % 64) / 64.0;
}

#endif
