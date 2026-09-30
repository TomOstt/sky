//! Cholesky factorization and its solve.
//!
//! The factor is defined entry by entry (`L` lower, `A = L·Lᵀ`):
//!
//! ```text
//! d   = a[j,j];  for k in 0..j { d = d - l[j,k]*l[j,k] }   // must be > 0
//! l[j,j] = sqrt(d)
//! s   = a[i,j];  for k in 0..j { s = s - l[i,k]*l[j,k] }
//! l[i,j] = s / l[j,j]                                       // i > j
//! ```
//!
//! The blocked right-looking algorithm below computes exactly those chains:
//! a block column's diagonal block and panel finish the sums for `k` inside
//! the block (they arrive already reduced by every earlier block), and the
//! trailing matrix is updated `c = c - l·l` for the block's `k`, in
//! ascending order, by the packed GEMM kernel. So the rounding of every
//! entry is the naive algorithm's, but almost all the work runs in the fast
//! kernel.

use crate::gemm::{gemm_view, Op, View};
use crate::threads;
use crate::{tri, LinalgError};

/// The largest block of columns.
const NB: usize = 64;

/// Up to this order the plain algorithm beats the blocked one.
const SMALL: usize = 40;

/// Whether the `n` by `n` matrix `a` equals its transpose entry for entry.
pub fn is_symmetric(a: &[f64], n: usize) -> bool {
    (0..n).all(|i| (i + 1..n).all(|j| a[i * n + j] == a[j * n + i]))
}

/// Factors the diagonal block `j0..j1` of `w` in place. The block is held
/// transposed (`t[j][i]` is entry `(i, j)`), so that a column of `L` is a
/// contiguous vector and the updates run along it.
fn factor_diagonal(w: &mut [f64], n: usize, j0: usize, j1: usize) -> Result<(), LinalgError> {
    let nb = j1 - j0;
    let mut t = [0.0_f64; NB * NB];
    for i in 0..nb {
        for j in 0..=i {
            t[j * NB + i] = w[(j0 + i) * n + j0 + j];
        }
    }
    let mut result = Ok(());
    for j in 0..nb {
        let mut d = t[j * NB + j];
        for k in 0..j {
            d -= t[k * NB + j] * t[k * NB + j];
        }
        if d.is_nan() || d <= 0.0 || !d.is_finite() {
            result = Err(LinalgError::NotPositiveDefinite { row: j0 + j });
            break;
        }
        let ljj = d.sqrt();
        t[j * NB + j] = ljj;
        // Column j below the diagonal: s = a[i,j] - Σ_{k<j} l[i,k]·l[j,k].
        let (done, rest) = t.split_at_mut(j * NB);
        let col = &mut rest[j + 1..nb];
        for k in 0..j {
            let ljk = done[k * NB + j];
            for (x, &lik) in col.iter_mut().zip(&done[k * NB + j + 1..k * NB + nb]) {
                *x -= lik * ljk;
            }
        }
        for x in col.iter_mut() {
            *x /= ljj;
        }
    }
    for i in 0..nb {
        for j in 0..=i {
            w[(j0 + i) * n + j0 + j] = t[j * NB + i];
        }
    }
    result
}

/// Solves the panel rows (rows `j1..n`, columns `j0..j1` of `w`) against the
/// factored diagonal block: for each row `i`,
/// `l[i,j] = (a[i,j] - Σ_{k<j} l[i,k]·l[j,k]) / l[j,j]`. That is a lower
/// triangular solve with `L11` for the transposed panel, whose columns are
/// the rows, and the triangular kernel runs those in register blocks.
fn factor_panel(w: &mut [f64], n: usize, j0: usize, j1: usize, threads: usize) {
    let (nb, rest) = (j1 - j0, n - j1);
    let mut l11 = vec![0.0; nb * nb];
    for j in 0..nb {
        l11[j * nb..j * nb + j + 1].copy_from_slice(&w[(j0 + j) * n + j0..][..j + 1]);
    }
    let mut pt = vec![0.0; nb * rest];
    for i in 0..rest {
        for (k, &v) in w[(j1 + i) * n + j0..][..nb].iter().enumerate() {
            pt[k * rest + i] = v;
        }
    }
    tri::solve_in_place(
        View::row_major(&l11, nb),
        nb,
        false,
        true,
        &mut pt,
        rest,
        threads,
        threads::par_min_madds() / 8,
    );
    for i in 0..rest {
        for (k, v) in w[(j1 + i) * n + j0..][..nb].iter_mut().enumerate() {
            *v = pt[k * rest + i];
        }
    }
}

/// The lower-triangular Cholesky factor of a symmetric positive-definite
/// `n` by `n` matrix (row-major, zeros above the diagonal), using up to
/// `threads` workers. Errors: [`LinalgError::NotSymmetric`] unless
/// `a[i,j] == a[j,i]` exactly, and [`LinalgError::NotPositiveDefinite`] at
/// the first row whose pivot is not a positive finite number.
pub fn cholesky_with(a: &[f64], n: usize, threads: usize) -> Result<Vec<f64>, LinalgError> {
    cholesky_impl(a, n, threads, false)
}

/// [`cholesky_with`] with the trailing updates (nearly all the work) on the
/// fused kernel: relaxed numerics without a vendor library. The errors are
/// the same kinds, decided on the same tests (the pivots are rounded
/// differently, so a matrix within rounding error of not being positive
/// definite may go either way).
pub fn cholesky_fused_with(a: &[f64], n: usize, threads: usize) -> Result<Vec<f64>, LinalgError> {
    cholesky_impl(a, n, threads, crate::fma_available())
}

fn cholesky_impl(a: &[f64], n: usize, threads: usize, fma: bool) -> Result<Vec<f64>, LinalgError> {
    assert_eq!(a.len(), n * n, "cholesky: shape mismatch");
    if n <= SMALL {
        // The plain loops are the definition, and at this size the fastest.
        return crate::reference::cholesky(a, n);
    }
    if !is_symmetric(a, n) {
        return Err(LinalgError::NotSymmetric);
    }
    let mut w = a.to_vec();
    let bs = crate::tuning::cholesky_nb(n).clamp(1, NB);
    let mut j0 = 0;
    while j0 < n {
        let j1 = (j0 + bs).min(n);
        let nb = j1 - j0;
        factor_diagonal(&mut w, n, j0, j1)?;
        if j1 < n {
            factor_panel(&mut w, n, j0, j1, threads);
            // The trailing update, lower triangle only: A22 -= L21·L21ᵀ.
            let rest = n - j1;
            let mut panel = vec![0.0; rest * nb];
            for i in 0..rest {
                panel[i * nb..(i + 1) * nb].copy_from_slice(&w[(j1 + i) * n + j0..][..nb]);
            }
            let l21 = View::row_major(&panel, nb);
            gemm_view(
                l21,
                l21.transposed(),
                &mut w[j1 * n + j1..],
                n,
                rest,
                rest,
                nb,
                Op::subtract(0).fused(fma),
                threads,
            );
        }
        j0 = j1;
    }
    for i in 0..n {
        w[i * n + i + 1..(i + 1) * n].fill(0.0);
    }
    Ok(w)
}

/// [`cholesky_with`] on the default number of threads.
pub fn cholesky(a: &[f64], n: usize) -> Result<Vec<f64>, LinalgError> {
    cholesky_with(a, n, threads::default_threads())
}

/// Solves `A·X = B` given the Cholesky factor `l` of `A` (`n` by `n`) and
/// `B` (`n` by `k`, row-major): `L·Y = B` by forward substitution, then
/// `Lᵀ·X = Y` by back substitution, each entry `x[i]` subtracting its
/// terms in ascending `j` and then dividing by `l[i,i]`. A vector is the
/// `k = 1` case. Every entry of the result is `+0.0` rather than `-0.0`
/// when it is a zero.
pub fn cholesky_solve_with(l: &[f64], n: usize, b: &[f64], k: usize, threads: usize) -> Vec<f64> {
    assert_eq!((l.len(), b.len()), (n * n, n * k), "cholesky_solve: shape");
    let mut x = b.to_vec();
    let lv = View::row_major(l, n);
    tri::solve_in_place(
        lv,
        n,
        false,
        true,
        &mut x,
        k,
        threads,
        threads::par_min_madds(),
    );
    // Lᵀ is read through a physical transpose, so its rows are contiguous.
    let lt = transpose(l, n);
    tri::solve_in_place(
        View::row_major(&lt, n),
        n,
        false,
        false,
        &mut x,
        k,
        threads,
        threads::par_min_madds(),
    );
    for v in &mut x {
        *v += 0.0;
    }
    x
}

/// [`cholesky_solve_with`] on the default number of threads.
pub fn cholesky_solve(l: &[f64], n: usize, b: &[f64], k: usize) -> Vec<f64> {
    cholesky_solve_with(l, n, b, k, threads::default_threads())
}

fn transpose(a: &[f64], n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = a[i * n + j];
        }
    }
    t
}
