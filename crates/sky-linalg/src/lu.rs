//! LU factorization with partial pivoting, and the solves it gives.
//!
//! Sky's LU is the forward phase of the Gauss-Jordan [`crate::inverse`],
//! operation for operation (see [`crate::reference::lu`], which is the
//! definition): at step `c` it picks the pivot (largest magnitude at or
//! below the diagonal, first on a tie; singular if NaN or no larger than
//! `n · ε · max|aᵢⱼ|`), exchanges whole rows, divides the pivot row right of
//! the diagonal by the pivot `d` (the diagonal keeps `d`), and for each
//! later row with multiplier `f = a[r,c]` (kept in place) reduces the
//! entries right of column `c` by `f` times the normalized pivot row. So a
//! matrix is singular for the solve exactly when it is for the inverse.
//!
//! Every entry therefore evolves as a chain of `a = a - f·u` with `c`
//! ascending: the recursive algorithm below (factor the left half of the
//! columns, solve for the block row of the normalized `U` with a triangular
//! solve, update the rest of the right half with a GEMM, factor the right
//! half) computes exactly those chains and spends nearly all its time in the
//! packed GEMM. It skips no zero multipliers (the plain loops skip `f == 0`),
//! which cannot change a finite result, so if an infinity or NaN appears in
//! a block of `U` it gives up and the plain loops decide.

use crate::gemm::{gemm_view, Op, View};
use crate::{threads, tri, LinalgError};

/// Panels this narrow (or narrower) are factored column by column.
const LEAF: usize = 16;

/// The result of [`lu`]: `A = P⁻¹·L·U'` with a unit upper `U'`.
///
/// `lu` holds, row-major, the normalized rows of `U'` right of the diagonal,
/// the elimination multipliers left of it, and the pivots `d` on it (the
/// unit diagonal of `U'` is implied). `swaps[c]` is the row exchanged with
/// row `c` at step `c`. Whole rows are exchanged, so the multipliers sit at
/// their final positions.
#[derive(Clone, Debug, PartialEq)]
pub struct Lu {
    /// Order of the matrix.
    pub n: usize,
    /// The factors, row-major, as described above.
    pub lu: Vec<f64>,
    /// The row exchanged with row `c` at step `c` (`swaps[c] >= c`).
    pub swaps: Vec<usize>,
}

/// Why the recursion stopped early.
enum Stop {
    Singular,
    /// An infinity or NaN in a block of `U`: let the plain loops decide.
    NonFinite,
}

/// Everything the recursion shares.
struct Factor<'a> {
    w: &'a mut [f64],
    n: usize,
    tol: f64,
    swaps: &'a mut [usize],
    threads: usize,
}

impl Factor<'_> {
    /// A narrow panel: columns `c0..c1`, rows `c0..n`, one column at a time.
    /// It is worked on transposed (a column of the panel is a contiguous
    /// vector over the rows), so the updates of a column are vector
    /// operations. Rows are swapped whole (the panel's part here, the rest
    /// in `w`), so the columns outside the panel stay in step.
    fn panel(&mut self, c0: usize, c1: usize) -> Result<(), Stop> {
        let n = self.n;
        let (rows, width) = (n - c0, c1 - c0);
        // pt[j * rows + i] is entry (c0 + i, c0 + j).
        let mut pt = vec![0.0; width * rows];
        for i in 0..rows {
            for (j, &v) in self.w[(c0 + i) * n + c0..][..width].iter().enumerate() {
                pt[j * rows + i] = v;
            }
        }
        for c in 0..width {
            let mut pivot = c;
            let col = &pt[c * rows..(c + 1) * rows];
            for i in c + 1..rows {
                if col[i].abs() > col[pivot].abs() {
                    pivot = i;
                }
            }
            let size = col[pivot].abs();
            if size.is_nan() || size <= self.tol {
                return Err(Stop::Singular);
            }
            self.swaps[c0 + c] = c0 + pivot;
            if pivot != c {
                for j in 0..width {
                    pt.swap(j * rows + c, j * rows + pivot);
                }
                let (r0, r1) = ((c0 + c) * n, (c0 + pivot) * n);
                for j in (0..c0).chain(c1..n) {
                    self.w.swap(r0 + j, r1 + j);
                }
            }
            let d = pt[c * rows + c];
            for j in c + 1..width {
                pt[j * rows + c] /= d;
            }
            let (left, right) = pt.split_at_mut((c + 1) * rows);
            let mult = &left[c * rows + c + 1..];
            for colj in right.chunks_exact_mut(rows) {
                let u = colj[c];
                for (x, &f) in colj[c + 1..].iter_mut().zip(mult) {
                    if f != 0.0 {
                        *x -= f * u;
                    }
                }
            }
        }
        for i in 0..rows {
            for (j, v) in self.w[(c0 + i) * n + c0..][..width].iter_mut().enumerate() {
                *v = pt[j * rows + i];
            }
        }
        Ok(())
    }

    /// The block row of the normalized `U` for the columns `mid..c1`: for
    /// each row `r` in `c0..mid` and every column `j` there,
    /// `u[r,j] = (a[r,j] - Σ_{k<r} f[r,k]·u[k,j]) / d[r]` with `k` ascending
    /// from `c0`, i.e. a triangular solve whose diagonal holds the pivots.
    fn block_row(&mut self, c0: usize, mid: usize, c1: usize) -> Result<(), Stop> {
        let n = self.n;
        let (nb, cols) = (mid - c0, c1 - mid);
        let mut l11 = vec![0.0; nb * nb];
        for r in 0..nb {
            l11[r * nb..r * nb + r + 1].copy_from_slice(&self.w[(c0 + r) * n + c0..][..r + 1]);
        }
        let mut block = vec![0.0; nb * cols];
        for r in 0..nb {
            block[r * cols..(r + 1) * cols].copy_from_slice(&self.w[(c0 + r) * n + mid..][..cols]);
        }
        tri::solve_in_place(
            View::row_major(&l11, nb),
            nb,
            false,
            true,
            &mut block,
            cols,
            self.threads,
            threads::PARALLEL_MIN_MADDS / 8,
        );
        if !block.iter().all(|x| x.is_finite()) {
            return Err(Stop::NonFinite);
        }
        for r in 0..nb {
            self.w[(c0 + r) * n + mid..][..cols].copy_from_slice(&block[r * cols..(r + 1) * cols]);
        }
        Ok(())
    }

    /// Factors the columns `c0..c1` (rows `c0..n`), given that every update
    /// from the columns before `c0` has been applied to them.
    fn columns(&mut self, c0: usize, c1: usize) -> Result<(), Stop> {
        if c1 - c0 <= LEAF {
            return self.panel(c0, c1);
        }
        let mid = c0 + (c1 - c0) / 2;
        self.columns(c0, mid)?;
        self.block_row(c0, mid, c1)?;
        // A22 -= F21·U12 on the rows below `mid` and the columns `mid..c1`
        // (a `k` chain of `mid - c0` terms per entry).
        let n = self.n;
        let (rows, k, cols) = (n - mid, mid - c0, c1 - mid);
        let mut f21 = vec![0.0; rows * k];
        for i in 0..rows {
            f21[i * k..(i + 1) * k].copy_from_slice(&self.w[(mid + i) * n + c0..][..k]);
        }
        let mut u12 = vec![0.0; k * cols];
        for r in 0..k {
            u12[r * cols..(r + 1) * cols].copy_from_slice(&self.w[(c0 + r) * n + mid..][..cols]);
        }
        gemm_view(
            View::row_major(&f21, k),
            View::row_major(&u12, cols),
            &mut self.w[mid * n + mid..],
            n,
            rows,
            cols,
            k,
            Op::subtract(Op::ALL),
            self.threads,
        );
        self.columns(mid, c1)
    }
}

/// Factors the `n` by `n` matrix `a` (row-major) on up to `threads` workers.
pub fn lu_with(a: &[f64], n: usize, threads: usize) -> Result<Lu, LinalgError> {
    assert_eq!(a.len(), n * n, "lu: shape mismatch");
    let mut w = a.to_vec();
    let scale = a.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
    let mut swaps = vec![0; n];
    let mut f = Factor {
        w: &mut w,
        n,
        tol: n as f64 * f64::EPSILON * scale,
        swaps: &mut swaps,
        threads,
    };
    match f.columns(0, n) {
        Ok(()) => Ok(Lu { n, lu: w, swaps }),
        Err(Stop::Singular) => Err(LinalgError::Singular),
        Err(Stop::NonFinite) => crate::reference::lu(a, n),
    }
}

/// [`lu_with`] on the default number of threads.
pub fn lu(a: &[f64], n: usize) -> Result<Lu, LinalgError> {
    lu_with(a, n, threads::default_threads())
}

impl Lu {
    /// The view of the factors whose entry `(i, j)` is `lu[j][i]`: the
    /// transpose, in which `U'ᵀ` is the strict lower triangle, the
    /// multipliers are the strict upper triangle, and the diagonal holds the
    /// pivots.
    fn transposed(&self) -> View<'_> {
        View {
            data: &self.lu,
            rs: 1,
            cs: self.n,
        }
    }

    /// Solves `A·X = B` for `B` (`n` by `k`, row-major): the row exchanges
    /// applied to `B`, then `L·Y = P·B` with each entry subtracting its
    /// terms in ascending `j` and dividing by its pivot, then `U'·X = Y`
    /// with the unit upper triangle. Zeros are `+0.0`.
    pub fn solve_with(&self, b: &[f64], k: usize, threads: usize) -> Vec<f64> {
        let n = self.n;
        assert_eq!(b.len(), n * k, "lu solve: shape mismatch");
        let mut x = b.to_vec();
        for (c, &p) in self.swaps.iter().enumerate() {
            if p != c {
                for j in 0..k {
                    x.swap(p * k + j, c * k + j);
                }
            }
        }
        let v = View::row_major(&self.lu, n);
        let par = threads::PARALLEL_MIN_MADDS;
        tri::solve_in_place(v, n, false, true, &mut x, k, threads, par);
        tri::solve_in_place(v, n, true, false, &mut x, k, threads, par);
        if !x.iter().all(|v| v.is_finite()) {
            // The plain loops skip a zero multiplier; multiplying it by an
            // infinity is a NaN. Let them decide.
            return crate::reference::lu_solve(self, b, k);
        }
        for v in &mut x {
            *v += 0.0;
        }
        x
    }

    /// [`Lu::solve_with`] on the default number of threads.
    pub fn solve(&self, b: &[f64], k: usize) -> Vec<f64> {
        self.solve_with(b, k, threads::default_threads())
    }

    /// Solves `Z·A = X` (`Z = X·A⁻¹`) for `X` (`m` by `n`, row-major)
    /// without forming `A⁻¹` or `Aᵀ`: `Y·U' = X` by forward substitution
    /// (each `yⱼ = xⱼ - Σ_{i<j} yᵢ·u'ᵢⱼ`, `i` ascending), then the
    /// elimination's operations applied to columns, last step first
    /// (`zᶜ = (yᶜ - Σ_{r>c} f[r,c]·yʳ) / d[c]`, `r` ascending), then the
    /// column exchanges. The rows of `X` are independent, so they run as the
    /// right-hand sides of two triangular solves with the transposed factors.
    pub fn solve_right_with(&self, x: &[f64], m: usize, threads: usize) -> Vec<f64> {
        let n = self.n;
        assert_eq!(x.len(), m * n, "lu solve_right: shape mismatch");
        let mut t = vec![0.0; n * m];
        for i in 0..m {
            for j in 0..n {
                t[j * m + i] = x[i * n + j];
            }
        }
        let v = self.transposed();
        let par = threads::PARALLEL_MIN_MADDS;
        tri::solve_in_place(v, n, true, true, &mut t, m, threads, par);
        tri::solve_in_place(v, n, false, false, &mut t, m, threads, par);
        if !t.iter().all(|v| v.is_finite()) {
            return crate::reference::lu_solve_right(self, x, m);
        }
        for (c, &p) in self.swaps.iter().enumerate().rev() {
            if p != c {
                for j in 0..m {
                    t.swap(p * m + j, c * m + j);
                }
            }
        }
        let mut z = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                z[i * n + j] = t[j * m + i] + 0.0;
            }
        }
        z
    }

    /// [`Lu::solve_right_with`] on the default number of threads.
    pub fn solve_right(&self, x: &[f64], m: usize) -> Vec<f64> {
        self.solve_right_with(x, m, threads::default_threads())
    }
}

/// Solves `A·X = B` (`gesv`): [`lu`] then [`Lu::solve`].
pub fn solve(a: &[f64], n: usize, b: &[f64], k: usize) -> Result<Vec<f64>, LinalgError> {
    Ok(lu(a, n)?.solve(b, k))
}
