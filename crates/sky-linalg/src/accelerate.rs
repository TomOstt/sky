//! Apple's Accelerate framework (macOS only): `cblas_dgemm`, `cblas_dsyrk`,
//! and LAPACK's `dpotrf`/`dpotrs`, `dgesv` and `dgetrf`/`dgetrs`.
//!
//! Accelerate is what the relaxed numerics call on macOS. Its GEMM runs on
//! the AMX matrix coprocessor of Apple silicon (about 14 times the
//! throughput of the best portable kernel on an M2 at large sizes) and
//! reorders sums and fuses multiply-adds freely, so it can never serve
//! the reproducible path.
//!
//! The declarations are the classic (LP64, 32-bit `int`) interface, which
//! every macOS since 10.x exports; newer SDKs mark the header deprecated but
//! the symbols remain. This module is only the FFI and the layout facts
//! (row-major buffers of Sky, column-major LAPACK); the decisions (when to
//! call, what to check afterwards) are in `crate::relaxed`.
//!
//! # Layout
//!
//! Sky matrices are row-major. `cblas_dgemm` and `cblas_dsyrk` take an
//! order argument, so they are called row-major. LAPACK is column-major
//! only: a row-major `n` by `n` buffer *is* its transpose in column-major,
//! which the callers use (a symmetric matrix is its own transpose; a lower
//! triangle stored by rows is an upper one stored by columns) or undo with an
//! explicit transposed copy.
//!
//! # Unsafe
//!
//! Only the FFI calls, each with the buffer lengths its contract needs
//! checked right before it.

#![allow(unsafe_code)]

use std::os::raw::{c_char, c_int};

const CBLAS_ROW_MAJOR: c_int = 101;
const CBLAS_NO_TRANS: c_int = 111;
const CBLAS_LOWER: c_int = 122;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_dgemm(
        order: c_int,
        trans_a: c_int,
        trans_b: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f64,
        a: *const f64,
        lda: c_int,
        b: *const f64,
        ldb: c_int,
        beta: f64,
        c: *mut f64,
        ldc: c_int,
    );
    fn cblas_dsyrk(
        order: c_int,
        uplo: c_int,
        trans: c_int,
        n: c_int,
        k: c_int,
        alpha: f64,
        a: *const f64,
        lda: c_int,
        beta: f64,
        c: *mut f64,
        ldc: c_int,
    );
    fn dpotrf_(
        uplo: *const c_char,
        n: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        info: *mut c_int,
    );
    fn dpotrs_(
        uplo: *const c_char,
        n: *const c_int,
        nrhs: *const c_int,
        a: *const f64,
        lda: *const c_int,
        b: *mut f64,
        ldb: *const c_int,
        info: *mut c_int,
    );
    fn dgesv_(
        n: *const c_int,
        nrhs: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        ipiv: *mut c_int,
        b: *mut f64,
        ldb: *const c_int,
        info: *mut c_int,
    );
    fn dgetrf_(
        m: *const c_int,
        n: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        ipiv: *mut c_int,
        info: *mut c_int,
    );
    fn dgetrs_(
        trans: *const c_char,
        n: *const c_int,
        nrhs: *const c_int,
        a: *const f64,
        lda: *const c_int,
        ipiv: *const c_int,
        b: *mut f64,
        ldb: *const c_int,
        info: *mut c_int,
    );
}

/// A dimension as the `int` the interface takes, or `None` if it does not
/// fit (the caller then uses its own kernel).
fn int(x: usize) -> Option<c_int> {
    c_int::try_from(x).ok()
}

/// `c = a * b` (all row-major; `a` is `m` by `k`, `b` is `k` by `n`).
/// `false` if a dimension does not fit an `int`.
pub(crate) fn gemm(a: &[f64], m: usize, k: usize, b: &[f64], n: usize, c: &mut [f64]) -> bool {
    assert_eq!((a.len(), b.len(), c.len()), (m * k, k * n, m * n));
    let (Some(mi), Some(ni), Some(ki)) = (int(m), int(n), int(k)) else {
        return false;
    };
    if m == 0 || n == 0 || k == 0 {
        return false;
    }
    // SAFETY: the slices have exactly `m*k`, `k*n` and `m*n` elements
    // (asserted), which is what a row-major `m` by `k` times `k` by `n`
    // product with leading dimensions `k`, `n`, `n` reads and writes; `c`
    // is a unique borrow and does not overlap `a` or `b`.
    unsafe {
        cblas_dgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            mi,
            ni,
            ki,
            1.0,
            a.as_ptr(),
            ki,
            b.as_ptr(),
            ni,
            0.0,
            c.as_mut_ptr(),
            ni,
        );
    }
    true
}

/// The lower triangle of `b · bᵀ` (`b` is `m` by `k`, row-major) into `c`
/// (`m` by `m`); the upper triangle is left as it was. `false` if a
/// dimension does not fit an `int`.
pub(crate) fn syrk_lower(b: &[f64], m: usize, k: usize, c: &mut [f64]) -> bool {
    assert_eq!((b.len(), c.len()), (m * k, m * m));
    let (Some(mi), Some(ki)) = (int(m), int(k)) else {
        return false;
    };
    if m == 0 || k == 0 {
        return false;
    }
    // SAFETY: `b` has `m*k` elements (leading dimension `k`) and `c` has
    // `m*m` (leading dimension `m`), as a row-major rank-k update of an
    // `m` by `m` matrix reads and writes; `c` is a unique borrow.
    unsafe {
        cblas_dsyrk(
            CBLAS_ROW_MAJOR,
            CBLAS_LOWER,
            CBLAS_NO_TRANS,
            mi,
            ki,
            1.0,
            b.as_ptr(),
            ki,
            0.0,
            c.as_mut_ptr(),
            mi,
        );
    }
    true
}

/// Cholesky of the row-major symmetric `n` by `n` matrix `a`, in place:
/// `dpotrf` on the column-major upper triangle, which is the row-major
/// lower one, leaves `L` (with `A = L·Lᵀ`) in the lower triangle. Returns
/// LAPACK's `info` (0 ok, `i > 0` the leading minor of order `i` is not
/// positive definite), or `None` if `n` does not fit an `int`. The strict
/// upper triangle is not touched.
pub(crate) fn potrf_lower_rowmajor(a: &mut [f64], n: usize) -> Option<i32> {
    assert_eq!(a.len(), n * n);
    let ni = int(n)?;
    if n == 0 {
        return None;
    }
    let uplo = b'U' as c_char;
    let mut info: c_int = 0;
    // SAFETY: `a` has `n*n` elements with leading dimension `n`; `uplo`,
    // `ni` and `info` are live locals for the call.
    unsafe { dpotrf_(&uplo, &ni, a.as_mut_ptr(), &ni, &mut info) };
    Some(info)
}

/// Solves `A·X = B` given the factor from [`potrf_lower_rowmajor`] (`l`,
/// row-major `n` by `n`) and `B` as a column-major `n` by `nrhs` buffer
/// (overwritten with `X`). Returns `info`.
pub(crate) fn potrs_lower_rowmajor(l: &[f64], n: usize, b: &mut [f64], nrhs: usize) -> Option<i32> {
    assert_eq!((l.len(), b.len()), (n * n, n * nrhs));
    let (ni, ri) = (int(n)?, int(nrhs)?);
    if n == 0 || nrhs == 0 {
        return None;
    }
    let uplo = b'U' as c_char;
    let mut info: c_int = 0;
    // SAFETY: `l` has `n*n` elements (leading dimension `n`) and `b` has
    // `n*nrhs` (leading dimension `n`); the scalars are live locals.
    unsafe {
        dpotrs_(
            &uplo,
            &ni,
            &ri,
            l.as_ptr(),
            &ni,
            b.as_mut_ptr(),
            &ni,
            &mut info,
        )
    };
    Some(info)
}

/// `dgesv` on column-major `a` (`n` by `n`, overwritten with its LU
/// factors) and `b` (`n` by `nrhs`, overwritten with the solution).
/// Returns `info`.
pub(crate) fn gesv(a: &mut [f64], n: usize, b: &mut [f64], nrhs: usize) -> Option<i32> {
    assert_eq!((a.len(), b.len()), (n * n, n * nrhs));
    let (ni, ri) = (int(n)?, int(nrhs)?);
    if n == 0 || nrhs == 0 {
        return None;
    }
    let mut ipiv: Vec<c_int> = vec![0; n];
    let mut info: c_int = 0;
    // SAFETY: `a` has `n*n` elements, `b` has `n*nrhs`, `ipiv` has `n`, all
    // with the leading dimensions `n` passed; the scalars are live locals.
    unsafe {
        dgesv_(
            &ni,
            &ri,
            a.as_mut_ptr(),
            &ni,
            ipiv.as_mut_ptr(),
            b.as_mut_ptr(),
            &ni,
            &mut info,
        )
    };
    Some(info)
}

/// LU factors of a column-major matrix and its pivots.
pub(crate) struct Getrf {
    pub ipiv: Vec<c_int>,
    pub info: i32,
}

/// `dgetrf` on column-major `a` (`n` by `n`, overwritten with `L` and `U`).
pub(crate) fn getrf(a: &mut [f64], n: usize) -> Option<Getrf> {
    assert_eq!(a.len(), n * n);
    let ni = int(n)?;
    if n == 0 {
        return None;
    }
    let mut ipiv: Vec<c_int> = vec![0; n];
    let mut info: c_int = 0;
    // SAFETY: `a` has `n*n` elements (leading dimension `n`) and `ipiv` has
    // `n`; the scalars are live locals.
    unsafe { dgetrf_(&ni, &ni, a.as_mut_ptr(), &ni, ipiv.as_mut_ptr(), &mut info) };
    Some(Getrf { ipiv, info })
}

/// `dgetrs` with the transposed factors: solves `Aᵀ·X = B` for the
/// column-major `b` (`n` by `nrhs`, overwritten) given [`getrf`]'s output.
pub(crate) fn getrs_transposed(
    lu: &[f64],
    n: usize,
    ipiv: &[c_int],
    b: &mut [f64],
    nrhs: usize,
) -> Option<i32> {
    assert_eq!((lu.len(), ipiv.len(), b.len()), (n * n, n, n * nrhs));
    let (ni, ri) = (int(n)?, int(nrhs)?);
    if n == 0 || nrhs == 0 {
        return None;
    }
    let trans = b'T' as c_char;
    let mut info: c_int = 0;
    // SAFETY: `lu` has `n*n` elements, `ipiv` has `n`, `b` has `n*nrhs`,
    // with the leading dimensions `n` passed; the scalars are live locals.
    unsafe {
        dgetrs_(
            &trans,
            &ni,
            &ri,
            lu.as_ptr(),
            &ni,
            ipiv.as_ptr(),
            b.as_mut_ptr(),
            &ni,
            &mut info,
        )
    };
    Some(info)
}
