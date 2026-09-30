//! Relaxed dot products and norms: reassociated, fused, and (for very long
//! vectors) split across threads.
//!
//! The reproducible [`crate::dot`] fixes eight partial sums and a
//! combining tree so that every machine adds the same terms in the same
//! order. That is slower than the hardware allows: an FMA has a latency of
//! about four cycles and two to four can issue per cycle, so peak needs 8 to
//! 16 independent accumulators of the machine's vector width, and the
//! reproducible eight scalar-order lanes never reach it.
//!
//! The relaxed kernel here uses whatever the machine has: on aarch64 eight
//! two-lane NEON accumulators (16 partial sums) updated with `fmla`; on
//! x86-64 with FMA3 four four-lane AVX2 accumulators (also 16); elsewhere it
//! is the reproducible kernel. The partial sums are then combined by a
//! fixed tree, so a given machine and thread count always gives the same
//! bits, but different machines (and different thread counts, see
//! [`dot_relaxed_with`]) need not.
//!
//! Every term is fused into its accumulator (`acc = fma(a, b, acc)`), so the
//! result also differs from the reproducible one by the roundings of the
//! products.

use crate::threads;

/// Vectors shorter than this use the reproducible kernel: the fused loop's
/// setup costs more than it saves.
const MIN_LEN: usize = 32;

/// Vectors at least this long are split across threads (each thread sums a
/// contiguous piece, the pieces are added in order). Below it the kernel is
/// bound by one core's memory bandwidth anyway.
const PARALLEL_MIN_LEN: usize = 1 << 20;

/// The fused, reassociated dot product of one piece.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
fn dot_piece(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::aarch64::*;
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    // SAFETY: NEON is part of the aarch64 base ISA. Every load reads two
    // elements at an offset `o` with `o + 2 <= n` (the loops below stop
    // before that bound), from slices of length `n`.
    unsafe {
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [vdupq_n_f64(0.0); 8];
        let mut i = 0;
        while i + 16 <= n {
            for (v, s) in acc.iter_mut().enumerate() {
                *s = vfmaq_f64(
                    *s,
                    vld1q_f64(pa.add(i + 2 * v)),
                    vld1q_f64(pb.add(i + 2 * v)),
                );
            }
            i += 16;
        }
        while i + 2 <= n {
            acc[0] = vfmaq_f64(acc[0], vld1q_f64(pa.add(i)), vld1q_f64(pb.add(i)));
            i += 2;
        }
        // (0+4)+(2+6) and (1+5)+(3+7), then across.
        let t = vaddq_f64(
            vaddq_f64(vaddq_f64(acc[0], acc[4]), vaddq_f64(acc[2], acc[6])),
            vaddq_f64(vaddq_f64(acc[1], acc[5]), vaddq_f64(acc[3], acc[7])),
        );
        let mut s = vaddvq_f64(t);
        if i < n {
            s = a[i].mul_add(b[i], s);
        }
        s
    }
}

/// The fused, reassociated dot product of one piece (FMA3 and AVX2).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
unsafe fn dot_piece_fma3(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = [_mm256_setzero_pd(); 4];
    let mut i = 0;
    // SAFETY: AVX2 and FMA3 are enabled on this function (and checked by
    // its caller); every load reads four elements at an offset `o` with
    // `o + 4 <= n`, from slices of length `n`.
    while i + 16 <= n {
        for (v, s) in acc.iter_mut().enumerate() {
            *s = _mm256_fmadd_pd(
                _mm256_loadu_pd(pa.add(i + 4 * v)),
                _mm256_loadu_pd(pb.add(i + 4 * v)),
                *s,
            );
        }
        i += 16;
    }
    while i + 4 <= n {
        acc[0] = _mm256_fmadd_pd(
            _mm256_loadu_pd(pa.add(i)),
            _mm256_loadu_pd(pb.add(i)),
            acc[0],
        );
        i += 4;
    }
    let t = _mm256_add_pd(_mm256_add_pd(acc[0], acc[2]), _mm256_add_pd(acc[1], acc[3]));
    let mut lanes = [0.0_f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), t);
    let mut s = (lanes[0] + lanes[2]) + (lanes[1] + lanes[3]);
    while i < n {
        s = a[i].mul_add(b[i], s);
        i += 1;
    }
    s
}

/// One piece, on the best kernel the machine has.
fn piece(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        dot_piece(a, b)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if crate::fma_available() {
            // SAFETY: `fma_available` just confirmed AVX2 and FMA3.
            #[allow(unsafe_code)]
            return unsafe { dot_piece_fma3(a, b) };
        }
        crate::dot(a, b)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        crate::dot(a, b)
    }
}

/// `Σ a[i]·b[i]` with **relaxed numerics**: fused multiply-adds into 16
/// partial sums, combined by a fixed tree, and (from about a million terms)
/// split across up to `threads` threads whose pieces are added in order.
///
/// The result depends on the machine's vector width and on `threads`; for a
/// given machine and thread count it is deterministic. It differs from
/// [`crate::dot`] by rounding error only: the relative error is at most
/// about `n·ε` of `Σ|a[i]·b[i]|` (as for any order of summation, and far
/// less in practice). Panics if the lengths differ.
pub fn dot_relaxed_with(a: &[f64], b: &[f64], threads: usize) -> f64 {
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    let n = a.len();
    if n < MIN_LEN {
        return crate::dot(a, b);
    }
    if threads <= 1 || n < PARALLEL_MIN_LEN {
        return piece(a, b);
    }
    let per = n.div_ceil(threads).next_multiple_of(16);
    let ranges: Vec<(usize, usize)> = (0..n).step_by(per).map(|s| (s, (s + per).min(n))).collect();
    threads::map_collect(threads, ranges, |(s, e)| piece(&a[s..e], &b[s..e]))
        .into_iter()
        .fold(0.0, |acc, p| acc + p)
}

/// [`dot_relaxed_with`] on the default number of threads.
pub fn dot_relaxed(a: &[f64], b: &[f64]) -> f64 {
    dot_relaxed_with(a, b, threads::default_threads())
}

/// `Σ x[i]²` with relaxed numerics (see [`dot_relaxed_with`]): the sum
/// behind `norm`.
pub fn sum_squares_relaxed(x: &[f64]) -> f64 {
    dot_relaxed(x, x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect()
    }

    #[test]
    fn agrees_with_the_reproducible_dot_to_rounding_error() {
        for n in [0, 1, 7, 31, 32, 33, 100, 1000, 4097, 100_000] {
            let (a, b) = (data(n, 1), data(n, 2));
            let want = crate::dot(&a, &b);
            let scale: f64 = a.iter().zip(&b).map(|(x, y)| (x * y).abs()).sum();
            for threads in [1, 3] {
                let got = dot_relaxed_with(&a, &b, threads);
                assert!(
                    (got - want).abs() <= 1e-14 * scale.max(1e-300),
                    "n = {n}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn long_vectors_split_across_threads_agree_too() {
        let n = PARALLEL_MIN_LEN + 12345;
        let (a, b) = (data(n, 5), data(n, 6));
        let want = crate::dot(&a, &b);
        let scale: f64 = a.iter().zip(&b).map(|(x, y)| (x * y).abs()).sum();
        for threads in [1, 2, 3, 8] {
            let got = dot_relaxed_with(&a, &b, threads);
            assert!(
                (got - want).abs() <= 1e-14 * scale,
                "{threads}: {got} {want}"
            );
            // The same machine and thread count give the same bits.
            assert_eq!(got.to_bits(), dot_relaxed_with(&a, &b, threads).to_bits());
        }
    }

    #[test]
    fn exact_small_integers_are_exact() {
        let a: Vec<f64> = (0..1000).map(f64::from).collect();
        let b = vec![1.0; 1000];
        assert_eq!(dot_relaxed_with(&a, &b, 1), 499_500.0);
        assert_eq!(sum_squares_relaxed(&[3.0; 100]), 900.0);
    }
}
