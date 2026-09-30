//! The relaxed kernels: what `sky run --numerics=relaxed` calls.
//!
//! Each function here does the same job as its reproducible namesake
//! (`gemm_relaxed_with` is `gemm_with`, and so on) and returns the same
//! *kind* of answer, but is allowed to
//!
//! * fuse each multiply into its add (one rounding instead of two),
//! * add the terms of a sum in a different order (a dot product), and that
//!   order may depend on the machine and on the thread count,
//! * call the platform's BLAS/LAPACK (Accelerate on macOS; nothing else
//!   yet: without a vendor library the fused kernels of this crate run).
//!
//! The results agree with the reproducible ones to rounding error, not bit
//! for bit (for a well-conditioned problem, to a relative error of about
//! `1e-12`; the tests pin it).
//!
//! # Errors are Sky's, not the vendor's
//!
//! A vendor factorization reports failure by its own rules (LAPACK: an
//! exactly zero pivot; Sky: a pivot no larger than `n·ε·max|aᵢⱼ|`, and a
//! Cholesky pivot that is not positive). So a vendor result is trusted only
//! if the answer is *clearly* on the success side of Sky's test, and
//! whenever the vendor reports `info != 0`, or a pivot within a safety
//! factor of Sky's threshold, or anything non-finite, the reproducible
//! kernel runs instead and decides. So a singular or not-positive-definite
//! matrix is rejected with Sky's own diagnostic (`Singular`,
//! `NotPositiveDefinite { row }`) at the same place, and a matrix that is
//! ill-conditioned but fine costs only the time of the second attempt.
//! Symmetry (`NotSymmetric`) is checked by Sky's own test before any vendor
//! call.

use crate::{syrk_fused_with, threads, LinalgError};

/// Below this many multiply-adds a product is not worth a call into the
/// vendor library (measured: the call and its thread wake-up cost more than
/// our kernels' whole run).
#[cfg(target_os = "macos")]
const VENDOR_MIN_MADDS: usize = 1 << 15;

/// Products with a side shorter than this (a matrix times a thin matrix, a
/// matrix-vector product) stay on the fused kernels, which are already
/// memory-bound there.
#[cfg(target_os = "macos")]
const VENDOR_MIN_SIDE: usize = 4;

/// Factorizations and solves of at most this order stay on the fused
/// kernels.
#[cfg(target_os = "macos")]
const VENDOR_MIN_ORDER: usize = 24;

/// How many times Sky's singularity threshold a vendor pivot must exceed to
/// be trusted (see the module docs). Generous on purpose: a fall back costs
/// time, a wrong verdict costs correctness.
#[cfg(target_os = "macos")]
const PIVOT_MARGIN: f64 = 1024.0;

/// `a * b` through the vendor library alone: `Some` on macOS (whatever the
/// size), `None` elsewhere. For tests and benchmarks; use
/// [`gemm_relaxed_with`].
pub fn gemm_vendor(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Option<Vec<f64>> {
    #[cfg(target_os = "macos")]
    {
        let mut c = vec![0.0; m * n];
        if m == 0 || n == 0 || k == 0 {
            return Some(c);
        }
        crate::accelerate::gemm(a, m, k, b, n, &mut c).then_some(c)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (a, m, k, b, n);
        None
    }
}

/// `a * b` (`a` is `m` by `k`, `b` is `k` by `n`, both row-major) with
/// **relaxed numerics**: the vendor library where there is one and the
/// product is big enough to pay for the call, else the fused kernels
/// ([`crate::gemm_into_fused_with`]). Panics if the lengths do not match
/// the shape.
pub fn gemm_relaxed_with(
    a: &[f64],
    m: usize,
    k: usize,
    b: &[f64],
    n: usize,
    threads: usize,
) -> Vec<f64> {
    assert_eq!((a.len(), b.len()), (m * k, k * n), "gemm: shape mismatch");
    #[cfg(target_os = "macos")]
    if m.min(n).min(k) >= VENDOR_MIN_SIDE && m * n * k >= VENDOR_MIN_MADDS {
        if let Some(c) = gemm_vendor(a, m, k, b, n) {
            return c;
        }
    }
    let mut c = vec![0.0; m * n];
    crate::gemm_into_fused_with(a, m, k, b, n, &mut c, threads);
    c
}

/// [`gemm_relaxed_with`] on the default number of threads.
pub fn gemm_relaxed(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    gemm_relaxed_with(a, m, k, b, n, threads::default_threads())
}

/// The Gram matrix `b · bᵀ` with relaxed numerics (see [`crate::syrk_with`]):
/// exactly symmetric, as always (the lower triangle is computed and
/// mirrored).
pub fn syrk_relaxed_with(b: &[f64], m: usize, k: usize, threads: usize) -> Vec<f64> {
    assert_eq!(b.len(), m * k, "syrk: shape mismatch");
    #[cfg(target_os = "macos")]
    if m.min(k) >= VENDOR_MIN_SIDE && m * m * k / 2 >= VENDOR_MIN_MADDS {
        let mut c = vec![0.0; m * m];
        if crate::accelerate::syrk_lower(b, m, k, &mut c) {
            for i in 0..m {
                for j in i + 1..m {
                    c[i * m + j] = c[j * m + i];
                }
            }
            return c;
        }
    }
    syrk_fused_with(b, m, k, threads)
}

/// [`syrk_relaxed_with`] on the default number of threads.
pub fn syrk_relaxed(b: &[f64], m: usize, k: usize) -> Vec<f64> {
    syrk_relaxed_with(b, m, k, threads::default_threads())
}

/// The reason to leave the vendor result: see the module docs.
#[cfg(target_os = "macos")]
enum Vendor<T> {
    /// A trusted answer (success, or a verdict Sky's own test made).
    Done(T),
    /// Not trusted: let the reproducible kernel decide.
    Fallback,
}

/// Cholesky through LAPACK's `dpotrf`, checked as the module docs say.
#[cfg(target_os = "macos")]
fn cholesky_vendor(a: &[f64], n: usize) -> Vendor<Result<Vec<f64>, LinalgError>> {
    if !crate::is_symmetric(a, n) {
        return Vendor::Done(Err(LinalgError::NotSymmetric));
    }
    let mut w = a.to_vec();
    if crate::accelerate::potrf_lower_rowmajor(&mut w, n) != Some(0) {
        return Vendor::Fallback;
    }
    // A pivot within rounding of zero is Sky's call to make: `l[j,j]²` is the
    // reduced diagonal `d` of Sky's algorithm.
    let margin = PIVOT_MARGIN * n as f64 * f64::EPSILON;
    for j in 0..n {
        let l = w[j * n + j];
        if !l.is_finite() || l * l <= margin * a[j * n + j] {
            return Vendor::Fallback;
        }
    }
    for i in 0..n {
        w[i * n + i + 1..(i + 1) * n].fill(0.0);
    }
    if w.iter().all(|x| x.is_finite()) {
        Vendor::Done(Ok(w))
    } else {
        Vendor::Fallback
    }
}

/// The lower-triangular Cholesky factor with **relaxed numerics**: LAPACK's
/// `dpotrf` on macOS (for matrices of order [`VENDOR_MIN_ORDER`] and up),
/// else the fused kernels. The errors are Sky's ([`LinalgError::NotSymmetric`]
/// first, then [`LinalgError::NotPositiveDefinite`] at the first row whose
/// pivot is not positive), decided by Sky's kernel whenever the vendor's
/// answer is not clearly a success.
pub fn cholesky_relaxed_with(a: &[f64], n: usize, threads: usize) -> Result<Vec<f64>, LinalgError> {
    assert_eq!(a.len(), n * n, "cholesky: shape mismatch");
    #[cfg(target_os = "macos")]
    if n >= VENDOR_MIN_ORDER {
        return match cholesky_vendor(a, n) {
            Vendor::Done(r) => r,
            Vendor::Fallback => crate::cholesky_with(a, n, threads),
        };
    }
    crate::cholesky_fused_with(a, n, threads)
}

/// [`cholesky_relaxed_with`] on the default number of threads.
pub fn cholesky_relaxed(a: &[f64], n: usize) -> Result<Vec<f64>, LinalgError> {
    cholesky_relaxed_with(a, n, threads::default_threads())
}

/// The transpose of an `r` by `c` row-major matrix.
#[cfg(target_os = "macos")]
fn transpose(x: &[f64], r: usize, c: usize) -> Vec<f64> {
    let mut t = vec![0.0; r * c];
    for i in 0..r {
        for j in 0..c {
            t[j * r + i] = x[i * c + j];
        }
    }
    t
}

/// Solves `A·X = B` for a Cholesky factor with **relaxed numerics**
/// (`dpotrs` on macOS for order [`VENDOR_MIN_ORDER`] and up, else Sky's
/// solve). `l` is the factor from [`cholesky_relaxed_with`] (or
/// [`crate::cholesky_with`]), `B` is `n` by `k` row-major. Zeros are `+0.0`.
pub fn cholesky_solve_relaxed_with(
    l: &[f64],
    n: usize,
    b: &[f64],
    k: usize,
    threads: usize,
) -> Vec<f64> {
    assert_eq!((l.len(), b.len()), (n * n, n * k), "cholesky_solve: shape");
    #[cfg(target_os = "macos")]
    if n >= VENDOR_MIN_ORDER && k >= 1 {
        // LAPACK wants `B` by columns: a vector already is, a matrix is
        // transposed in and out.
        let mut x = if k == 1 {
            b.to_vec()
        } else {
            transpose(b, n, k)
        };
        if crate::accelerate::potrs_lower_rowmajor(l, n, &mut x, k) == Some(0) {
            if k > 1 {
                x = transpose(&x, k, n);
            }
            if x.iter().all(|v| v.is_finite()) {
                for v in &mut x {
                    *v += 0.0;
                }
                return x;
            }
        }
    }
    crate::cholesky_solve_with(l, n, b, k, threads)
}

/// [`cholesky_solve_relaxed_with`] on the default number of threads.
pub fn cholesky_solve_relaxed(l: &[f64], n: usize, b: &[f64], k: usize) -> Vec<f64> {
    cholesky_solve_relaxed_with(l, n, b, k, threads::default_threads())
}

/// Sky's singularity threshold for `a`: `n · ε · max|aᵢⱼ|`.
#[cfg(target_os = "macos")]
fn singular_tolerance(a: &[f64], n: usize) -> f64 {
    n as f64 * f64::EPSILON * a.iter().fold(0.0_f64, |s, x| s.max(x.abs()))
}

/// Whether the diagonal of a column-major LU (`n` by `n`) has a pivot Sky
/// might call zero, or not a number.
#[cfg(target_os = "macos")]
fn pivot_in_doubt(lu: &[f64], n: usize, tol: f64) -> bool {
    (0..n).any(|i| {
        let d = lu[i * n + i].abs();
        d.is_nan() || d <= PIVOT_MARGIN * tol
    })
}

/// Solves `A·X = B` (`gesv`) with **relaxed numerics**: LAPACK's `dgesv` on
/// macOS for order [`VENDOR_MIN_ORDER`] and up, else an LU on the fused
/// kernels. `A` is `n` by `n`, `B` is `n` by `k`, both row-major; zeros are
/// `+0.0`. [`LinalgError::Singular`] under Sky's test (a pivot no larger
/// than `n·ε·max|aᵢⱼ|`), decided by Sky's kernel whenever the vendor's
/// pivots are not clearly larger.
pub fn solve_relaxed_with(
    a: &[f64],
    n: usize,
    b: &[f64],
    k: usize,
    threads: usize,
) -> Result<Vec<f64>, LinalgError> {
    assert_eq!((a.len(), b.len()), (n * n, n * k), "solve: shape mismatch");
    #[cfg(target_os = "macos")]
    if n >= VENDOR_MIN_ORDER && k >= 1 {
        let tol = singular_tolerance(a, n);
        let mut lu = transpose(a, n, n);
        let mut x = if k == 1 {
            b.to_vec()
        } else {
            transpose(b, n, k)
        };
        if crate::accelerate::gesv(&mut lu, n, &mut x, k) == Some(0) && !pivot_in_doubt(&lu, n, tol)
        {
            if k > 1 {
                x = transpose(&x, k, n);
            }
            if x.iter().all(|v| v.is_finite()) {
                for v in &mut x {
                    *v += 0.0;
                }
                return Ok(x);
            }
        }
        // Not trusted: Sky's own factorization decides (and, if it accepts
        // the matrix, gives the answer).
        return Ok(crate::lu_with(a, n, threads)?.solve_with(b, k, threads));
    }
    Ok(crate::lu_fused_with(a, n, threads)?.solve_with(b, k, threads))
}

/// [`solve_relaxed_with`] on the default number of threads.
pub fn solve_relaxed(a: &[f64], n: usize, b: &[f64], k: usize) -> Result<Vec<f64>, LinalgError> {
    solve_relaxed_with(a, n, b, k, threads::default_threads())
}

/// Solves `Z·A = X` (`Z = X·A⁻¹`) for `X` (`m` by `n`, row-major) with
/// relaxed numerics: `dgetrf` and `dgetrs` on the transposed system on
/// macOS for order [`VENDOR_MIN_ORDER`] and up (no copy of `X`: a row-major
/// `X` is `Xᵀ` by columns), else an LU on the fused kernels. Errors and
/// fallbacks as [`solve_relaxed_with`].
pub fn solve_right_relaxed_with(
    a: &[f64],
    n: usize,
    x: &[f64],
    m: usize,
    threads: usize,
) -> Result<Vec<f64>, LinalgError> {
    assert_eq!((a.len(), x.len()), (n * n, m * n), "solve_right: shape");
    #[cfg(target_os = "macos")]
    if n >= VENDOR_MIN_ORDER && m >= 1 {
        let tol = singular_tolerance(a, n);
        let mut lu = transpose(a, n, n);
        if let Some(f) = crate::accelerate::getrf(&mut lu, n) {
            if f.info == 0 && !pivot_in_doubt(&lu, n, tol) {
                let mut z = x.to_vec();
                if crate::accelerate::getrs_transposed(&lu, n, &f.ipiv, &mut z, m) == Some(0)
                    && z.iter().all(|v| v.is_finite())
                {
                    for v in &mut z {
                        *v += 0.0;
                    }
                    return Ok(z);
                }
            }
        }
        return Ok(crate::lu_with(a, n, threads)?.solve_right_with(x, m, threads));
    }
    Ok(crate::lu_fused_with(a, n, threads)?.solve_right_with(x, m, threads))
}

/// [`solve_right_relaxed_with`] on the default number of threads.
pub fn solve_right_relaxed(
    a: &[f64],
    n: usize,
    x: &[f64],
    m: usize,
) -> Result<Vec<f64>, LinalgError> {
    solve_right_relaxed_with(a, n, x, m, threads::default_threads())
}
