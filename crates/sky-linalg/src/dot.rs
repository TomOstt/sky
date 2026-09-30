//! Dot products and sums of squares: reductions with a fixed order.
//!
//! A single sum cannot be vectorized in sequential order, so Sky fixes a
//! different order that any hardware can follow:
//!
//! * fewer than [`LANES`] (8) terms: left to right, `((0 + p₀) + p₁) + …`,
//!   which is the naive loop (short vectors such as 3-vectors keep the sum
//!   everyone expects);
//! * otherwise: eight partial sums `s₀ … s₇`, all starting at `+0.0`, where
//!   term `pᵢ` is added to `s[i mod 8]` in index order, combined by the
//!   pairwise tree `((s₀+s₄) + (s₂+s₆)) + ((s₁+s₅) + (s₃+s₇))`.
//!
//! Terms are products rounded to `f64` first (no fused multiply-add).

/// Interleaved partial sums used for a reduction of eight or more terms.
pub const LANES: usize = 8;

/// The pairwise tree over the eight partial sums.
#[inline(always)]
fn combine(s: [f64; LANES]) -> f64 {
    let t0 = s[0] + s[4];
    let t1 = s[1] + s[5];
    let t2 = s[2] + s[6];
    let t3 = s[3] + s[7];
    (t0 + t2) + (t1 + t3)
}

/// The reduction of `f(a[i], b[i])` over the elements, in the fixed order
/// above. `a` and `b` have the same length.
#[inline(always)]
fn reduce2<F: Fn(f64, f64) -> f64>(a: &[f64], b: &[f64], f: F) -> f64 {
    if a.len() < LANES {
        let mut s = 0.0;
        for (&x, &y) in a.iter().zip(b) {
            s += f(x, y);
        }
        return s;
    }
    let mut s = [0.0_f64; LANES];
    let (ca, cb) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for l in 0..LANES {
            s[l] += f(x[l], y[l]);
        }
    }
    for (l, (&x, &y)) in ra.iter().zip(rb).enumerate() {
        s[l] += f(x, y);
    }
    combine(s)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn dot_avx2(a: &[f64], b: &[f64]) -> f64 {
    reduce2(a, b, |x, y| x * y)
}

/// `Σ a[i]·b[i]` in the fixed order of this module. Panics if the lengths
/// differ.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        if a.len() >= LANES && std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was detected at run time; the function is safe
            // code compiled with that feature enabled.
            #[allow(unsafe_code)]
            return unsafe { dot_avx2(a, b) };
        }
    }
    reduce2(a, b, |x, y| x * y)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn sum_squares_avx2(x: &[f64]) -> f64 {
    reduce2(x, x, |x, y| x * y)
}

/// `Σ x[i]²` in the fixed order of this module.
pub fn sum_squares(x: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if x.len() >= LANES && std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: as in `dot`.
            #[allow(unsafe_code)]
            return unsafe { sum_squares_avx2(x) };
        }
    }
    reduce2(x, x, |x, y| x * y)
}

/// `Σ (x[i]/scale)²` in the fixed order of this module: the scaled sum of
/// squares behind `norm` for entries of extreme magnitude. Each term is
/// `y * y` with `y = x / scale`.
pub fn sum_squares_scaled(x: &[f64], scale: f64) -> f64 {
    reduce2(x, x, |x, _| {
        let y = x / scale;
        y * y
    })
}
