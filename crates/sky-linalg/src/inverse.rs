//! The inverse by Gauss-Jordan elimination with partial pivoting.
//!
//! Sky's inverse is *defined* by this algorithm (not by "an inverse that is
//! close"), because a program's printed digits and its singular verdict
//! are part of its meaning: for `[A | I]`, column by column, pick the
//! largest entry of the column at or below the diagonal (the first on a
//! tie), call the matrix singular if it is NaN or no larger than
//! `n · ε · max|aᵢⱼ|`, swap it up, divide its row by it, and subtract
//! `f × (pivot row)` from every other row, `f` being that row's entry in the
//! column (rows with `f == 0` are left alone).
//!
//! Two implementations compute exactly that, bit for bit:
//!
//! * the plain one does the operations on whole rows, skipping only what
//!   cannot matter (the eliminated half of `A` is never read again, so its
//!   row updates start at the pivot column) so that each update is a vector
//!   operation; small matrices, and any matrix in which an entry overflows,
//!   use it;
//! * the blocked one (GEMM-based, for larger matrices) takes the columns a
//!   block at a time: it eliminates the block's columns on a transposed
//!   strip of `A`, recording each row's multipliers, then applies the block
//!   to the rest of `[A | I]`: the pivot rows by a triangular solve and the
//!   other rows by one GEMM (`row -= f₀·u₀; row -= f₁·u₁; …` in the order of
//!   the steps, which is what the GEMM kernel's accumulation is). It gives
//!   up, so that the plain one decides, if a pivot row ever holds an
//!   infinity or a NaN (where skipping a zero multiplier and multiplying by
//!   it could differ).
//!
//! If `A` is exactly symmetric the result is mirrored from its upper
//! triangle, so it is bitwise symmetric.

use crate::gemm::{gemm_view, Op, View};
use crate::{threads, tri, LinalgError};

/// Matrices of at least this order try the blocked implementation.
const BLOCKED_MIN_N: usize = 96;

/// A chunk of rows of `m` and `inv`, and the global index of its first row.
type Rows<'a> = (usize, (&'a mut [f64], &'a mut [f64]));

/// `row[j] -= f * pivot[j]`, a lane per `j`.
#[inline(always)]
fn axpy_sub(row: &mut [f64], f: f64, pivot: &[f64]) {
    for (x, &p) in row.iter_mut().zip(pivot) {
        *x -= f * p;
    }
}

/// Eliminates column `col` from the rows in `rows` (whole rows of `m`, then
/// of `inv`, starting at global row `row0`), given the normalized pivot
/// rows.
fn eliminate(
    m: &mut [f64],
    inv: &mut [f64],
    n: usize,
    col: usize,
    row0: usize,
    pm: &[f64],
    pinv: &[f64],
) {
    for (r, (mrow, irow)) in m.chunks_mut(n).zip(inv.chunks_mut(n)).enumerate() {
        if row0 + r == col {
            continue;
        }
        let f = mrow[col];
        if f == 0.0 {
            continue;
        }
        mrow[col] = 0.0;
        axpy_sub(&mut mrow[col + 1..], f, &pm[col + 1..]);
        axpy_sub(irow, f, pinv);
    }
}

fn identity(n: usize) -> Vec<f64> {
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    inv
}

/// The plain implementation: whole rows, one column at a time.
fn plain(a: &[f64], n: usize, tol: f64, threads: usize) -> Result<Vec<f64>, LinalgError> {
    let mut m = a.to_vec();
    let mut inv = identity(n);
    let par = threads > 1 && n >= 256;
    let mut pm = vec![0.0; n];
    let mut pinv = vec![0.0; n];
    for col in 0..n {
        let mut pivot = col;
        for r in col + 1..n {
            if m[r * n + col].abs() > m[pivot * n + col].abs() {
                pivot = r;
            }
        }
        let size = m[pivot * n + col].abs();
        if size.is_nan() || size <= tol {
            return Err(LinalgError::Singular);
        }
        if pivot != col {
            for j in col..n {
                m.swap(pivot * n + j, col * n + j);
            }
            for j in 0..n {
                inv.swap(pivot * n + j, col * n + j);
            }
        }
        let d = m[col * n + col];
        for j in col..n {
            m[col * n + j] /= d;
        }
        for j in 0..n {
            inv[col * n + j] /= d;
        }
        pm.copy_from_slice(&m[col * n..(col + 1) * n]);
        pinv.copy_from_slice(&inv[col * n..(col + 1) * n]);
        if par {
            let rows_per = n.div_ceil(threads * 2).max(16);
            let items: Vec<Rows> = m
                .chunks_mut(rows_per * n)
                .zip(inv.chunks_mut(rows_per * n))
                .enumerate()
                .map(|(c, rows)| (c * rows_per, rows))
                .collect();
            threads::for_each(threads, items, |(row0, (mc, ic))| {
                eliminate(mc, ic, n, col, row0, &pm, &pinv)
            });
        } else {
            eliminate(&mut m, &mut inv, n, col, 0, &pm, &pinv);
        }
    }
    Ok(inv)
}

/// Columns eliminated per block.
fn block_size(n: usize) -> usize {
    if n >= 256 {
        64
    } else {
        32
    }
}

/// The blocked implementation; `None` means "let [`plain`] decide" (an
/// infinity or NaN appeared where the two could differ).
fn blocked(a: &[f64], n: usize, tol: f64, threads: usize) -> Option<Result<Vec<f64>, LinalgError>> {
    if !a.iter().all(|x| x.is_finite()) {
        return None;
    }
    let nb_max = block_size(n);
    let mut m = a.to_vec();
    let mut inv = identity(n);
    // st[j * n + r]: row r of the strip of columns being eliminated;
    // ft[c * n + r]: the multiplier row r had at step c of the block.
    let mut st = vec![0.0; nb_max * n];
    let mut ft = vec![0.0; nb_max * n];
    let mut dv = vec![0.0; nb_max];
    let mut c0 = 0;
    while c0 < n {
        let c1 = (c0 + nb_max).min(n);
        let nb = c1 - c0;
        for r in 0..n {
            for j in 0..nb {
                st[j * n + r] = m[r * n + c0 + j];
            }
        }
        // The block's steps, on the strip only.
        for c in 0..nb {
            let gc = c0 + c;
            let col = &st[c * n..(c + 1) * n];
            let mut pivot = gc;
            for r in gc + 1..n {
                if col[r].abs() > col[pivot].abs() {
                    pivot = r;
                }
            }
            let size = col[pivot].abs();
            if size.is_nan() || size <= tol {
                return Some(Err(LinalgError::Singular));
            }
            if pivot != gc {
                for j in 0..nb {
                    st.swap(j * n + gc, j * n + pivot);
                }
                for cc in 0..c {
                    ft.swap(cc * n + gc, cc * n + pivot);
                }
                for j in c1..n {
                    m.swap(gc * n + j, pivot * n + j);
                }
                for j in 0..n {
                    inv.swap(gc * n + j, pivot * n + j);
                }
            }
            let d = st[c * n + gc];
            dv[c] = d;
            for j in c..nb {
                st[j * n + gc] /= d;
            }
            // The multipliers are the strip's column, except that the pivot
            // row is not eliminated from itself.
            let (fcol, _) = ft[c * n..].split_at_mut(n);
            fcol.copy_from_slice(&st[c * n..(c + 1) * n]);
            fcol[gc] = 0.0;
            for j in c + 1..nb {
                let colj = &mut st[j * n..(j + 1) * n];
                let p = colj[gc];
                for (x, &f) in colj.iter_mut().zip(fcol.iter()) {
                    if f != 0.0 {
                        *x -= f * p;
                    }
                }
            }
        }
        // Apply the block to the rest of [A | I]: the columns `c1..n` of `m`
        // and all of `inv`, `w` columns in all.
        let cm = n - c1;
        let w = cm + n;
        let mut u = vec![0.0; nb * w];
        for c in 0..nb {
            let gc = c0 + c;
            u[c * w..c * w + cm].copy_from_slice(&m[gc * n + c1..(gc + 1) * n]);
            u[c * w + cm..(c + 1) * w].copy_from_slice(&inv[gc * n..(gc + 1) * n]);
        }
        // The normalized pivot rows as they stood at their own steps:
        // u_c = (row_c - Σ_{c' < c} f[c, c']·u_c') / d_c.
        let mut lower = vec![0.0; nb * nb];
        for c in 0..nb {
            for cc in 0..c {
                lower[c * nb + cc] = ft[cc * n + c0 + c];
            }
            lower[c * nb + c] = dv[c];
        }
        tri::solve_in_place(
            View::row_major(&lower, nb),
            nb,
            false,
            true,
            &mut u,
            w,
            threads,
            threads::par_min_madds() / 8,
        );
        if !u.iter().all(|x| x.is_finite()) {
            return None;
        }
        // Every other row: row -= f₀·u₀; row -= f₁·u₁; … in step order.
        let multipliers = View {
            data: &ft,
            rs: 1,
            cs: n,
        };
        let u_m = View {
            data: &u,
            rs: w,
            cs: 1,
        };
        let u_inv = View {
            data: &u[cm..],
            rs: w,
            cs: 1,
        };
        for (r0, r1) in [(0, c0), (c1, n)] {
            if r1 > r0 {
                let rows = r1 - r0;
                let f = multipliers.offset(r0, 0);
                gemm_view(
                    f,
                    u_m,
                    &mut m[r0 * n + c1..],
                    n,
                    rows,
                    cm,
                    nb,
                    Op::subtract(Op::ALL),
                    threads,
                );
                gemm_view(
                    f,
                    u_inv,
                    &mut inv[r0 * n..],
                    n,
                    rows,
                    n,
                    nb,
                    Op::subtract(Op::ALL),
                    threads,
                );
            }
        }
        // The pivot rows, which the later steps of the block also updated:
        // row_c = u_c - Σ_{c'' > c} f[c, c'']·u_c'' (with `f == 0` skipped,
        // as in the plain loops).
        let mut acc = vec![0.0; w];
        for c in 0..nb {
            let gc = c0 + c;
            acc.copy_from_slice(&u[c * w..(c + 1) * w]);
            for cc in c + 1..nb {
                let f = ft[cc * n + gc];
                if f != 0.0 {
                    axpy_sub(&mut acc, f, &u[cc * w..(cc + 1) * w]);
                }
            }
            m[gc * n + c1..(gc + 1) * n].copy_from_slice(&acc[..cm]);
            inv[gc * n..(gc + 1) * n].copy_from_slice(&acc[cm..]);
        }
        c0 = c1;
    }
    Some(Ok(inv))
}

/// The inverse of the `n` by `n` matrix `a` (row-major). If `a` is exactly
/// symmetric and `mirror` is set, the result is exactly symmetric too.
/// Fails with [`LinalgError::Singular`] as described in the module docs.
pub fn inverse_with(
    a: &[f64],
    n: usize,
    mirror: bool,
    threads: usize,
) -> Result<Vec<f64>, LinalgError> {
    assert_eq!(a.len(), n * n, "inverse: shape mismatch");
    let scale = a.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
    let tol = n as f64 * f64::EPSILON * scale;
    let done = if n >= BLOCKED_MIN_N {
        blocked(a, n, tol, threads)
    } else {
        None
    };
    let mut inv = match done {
        Some(result) => result?,
        None => plain(a, n, tol, threads)?,
    };
    // `x + 0.0` turns a `-0.0` (from `0 / negative`) into `0.0`.
    for x in &mut inv {
        *x += 0.0;
    }
    if mirror && crate::is_symmetric(a, n) {
        for i in 0..n {
            for j in i + 1..n {
                inv[j * n + i] = inv[i * n + j];
            }
        }
    }
    Ok(inv)
}

/// [`inverse_with`] mirroring symmetric results, on the default number of
/// threads.
pub fn inverse(a: &[f64], n: usize) -> Result<Vec<f64>, LinalgError> {
    inverse_with(a, n, true, threads::default_threads())
}
