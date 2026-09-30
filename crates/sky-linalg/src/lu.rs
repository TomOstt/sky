//! LU factorization with partial pivoting, and the solve it gives.
//!
//! `P·A = L·U`, `L` unit lower triangular. Column `c` picks the entry of
//! largest magnitude at or below the diagonal (the first on a tie), swaps
//! that row up, stores the multipliers `l[r,c] = a[r,c] / a[c,c]`
//! (a division) and updates the rest with `a[r,j] = a[r,j] - l[r,c]·a[c,j]`.
//! Every entry therefore evolves as a chain of `c = c - l·u` with `k`
//! ascending: the recursive algorithm below (factor the left half of the
//! columns, solve for the block row of `U` with a triangular solve, update
//! the rest of the right half with a GEMM, factor the right half) computes
//! exactly those chains, and spends nearly all its time in the packed GEMM.
//!
//! A pivot no larger than `n · ε · max|aᵢⱼ|` (or NaN) means the matrix is
//! singular, the same threshold as [`crate::inverse`].

use crate::gemm::{gemm_view, Op, View};
use crate::{threads, tri, LinalgError};

/// Panels this narrow (or narrower) are factored column by column.
const LEAF: usize = 16;

/// The result of [`lu`]: `L` and `U` packed in one matrix (the unit
/// diagonal of `L` is implicit) and the row permutation.
#[derive(Clone, Debug, PartialEq)]
pub struct Lu {
    /// Order of the matrix.
    pub n: usize,
    /// `L` strictly below the diagonal, `U` on and above it, row-major.
    pub lu: Vec<f64>,
    /// Row `i` of `P·A` is row `perm[i]` of `A`.
    pub perm: Vec<usize>,
    /// The number of row swaps that moved a row (the parity gives the sign
    /// of the determinant).
    pub swaps: usize,
}

/// Everything the recursion shares.
struct Factor<'a> {
    w: &'a mut [f64],
    n: usize,
    tol: f64,
    perm: &'a mut [usize],
    swaps: usize,
    threads: usize,
}

impl Factor<'_> {
    /// A narrow panel: columns `c0..c1`, rows `c0..n`, one column at a time.
    /// It is worked on transposed (a column of the panel is a contiguous
    /// vector over the rows), so the multipliers and the updates of a column
    /// are vector operations. Rows are swapped whole (the panel's part here,
    /// the rest in `w`), so the columns outside the panel stay in step.
    fn panel(&mut self, c0: usize, c1: usize) -> Result<(), LinalgError> {
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
                return Err(LinalgError::Singular);
            }
            if pivot != c {
                for j in 0..width {
                    pt.swap(j * rows + c, j * rows + pivot);
                }
                let (r0, r1) = ((c0 + c) * n, (c0 + pivot) * n);
                for j in (0..c0).chain(c1..n) {
                    self.w.swap(r0 + j, r1 + j);
                }
                self.perm.swap(c0 + c, c0 + pivot);
                self.swaps += 1;
            }
            let (left, right) = pt.split_at_mut((c + 1) * rows);
            let lcol = &mut left[c * rows..];
            let d = lcol[c];
            for l in &mut lcol[c + 1..] {
                *l /= d;
            }
            for colj in right.chunks_exact_mut(rows) {
                let u = colj[c];
                for (x, &l) in colj[c + 1..].iter_mut().zip(&lcol[c + 1..]) {
                    *x -= l * u;
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

    /// The block row of `U` for the columns `mid..c1`: for each row `r` in
    /// `c0..mid` and every column `j` there, `u[r,j] = a[r,j] - Σ_{k<r}
    /// l[r,k]·u[k,j]` with `k` ascending from `c0`, i.e. a unit-lower
    /// triangular solve.
    fn block_row(&mut self, c0: usize, mid: usize, c1: usize) {
        let n = self.n;
        let (nb, cols) = (mid - c0, c1 - mid);
        let mut l11 = vec![0.0; nb * nb];
        for r in 0..nb {
            l11[r * nb..r * nb + r].copy_from_slice(&self.w[(c0 + r) * n + c0..][..r]);
        }
        let mut block = vec![0.0; nb * cols];
        for r in 0..nb {
            block[r * cols..(r + 1) * cols].copy_from_slice(&self.w[(c0 + r) * n + mid..][..cols]);
        }
        tri::solve_in_place(
            View::row_major(&l11, nb),
            nb,
            true,
            true,
            &mut block,
            cols,
            self.threads,
            threads::PARALLEL_MIN_MADDS / 8,
        );
        for r in 0..nb {
            self.w[(c0 + r) * n + mid..][..cols].copy_from_slice(&block[r * cols..(r + 1) * cols]);
        }
    }

    /// Factors the columns `c0..c1` (rows `c0..n`), given that every update
    /// from the columns before `c0` has been applied to them.
    fn columns(&mut self, c0: usize, c1: usize) -> Result<(), LinalgError> {
        if c1 - c0 <= LEAF {
            return self.panel(c0, c1);
        }
        let mid = c0 + (c1 - c0) / 2;
        self.columns(c0, mid)?;
        self.block_row(c0, mid, c1);
        // A22 -= L21·U12 on the rows below `mid` and the columns `mid..c1`
        // (a `k` chain of `mid - c0` terms per entry).
        let n = self.n;
        let (rows, k, cols) = (n - mid, mid - c0, c1 - mid);
        let mut l21 = vec![0.0; rows * k];
        for i in 0..rows {
            l21[i * k..(i + 1) * k].copy_from_slice(&self.w[(mid + i) * n + c0..][..k]);
        }
        let mut u12 = vec![0.0; k * cols];
        for r in 0..k {
            u12[r * cols..(r + 1) * cols].copy_from_slice(&self.w[(c0 + r) * n + mid..][..cols]);
        }
        gemm_view(
            View::row_major(&l21, k),
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
    let mut perm: Vec<usize> = (0..n).collect();
    let mut f = Factor {
        w: &mut w,
        n,
        tol: n as f64 * f64::EPSILON * scale,
        perm: &mut perm,
        swaps: 0,
        threads,
    };
    f.columns(0, n)?;
    let swaps = f.swaps;
    Ok(Lu {
        n,
        lu: w,
        perm,
        swaps,
    })
}

/// [`lu_with`] on the default number of threads.
pub fn lu(a: &[f64], n: usize) -> Result<Lu, LinalgError> {
    lu_with(a, n, threads::default_threads())
}

impl Lu {
    /// Solves `A·X = B` for `B` (`n` by `k`, row-major): `L·Y = P·B` then
    /// `U·X = Y`, each entry subtracting its terms in ascending `j` and
    /// (for `U`) dividing by the diagonal. Zeros are `+0.0`.
    pub fn solve_with(&self, b: &[f64], k: usize, threads: usize) -> Vec<f64> {
        let n = self.n;
        assert_eq!(b.len(), n * k, "lu solve: shape mismatch");
        let mut x = vec![0.0; n * k];
        for (i, &p) in self.perm.iter().enumerate() {
            x[i * k..(i + 1) * k].copy_from_slice(&b[p * k..(p + 1) * k]);
        }
        let v = View::row_major(&self.lu, n);
        let par = threads::PARALLEL_MIN_MADDS;
        tri::solve_in_place(v, n, true, true, &mut x, k, threads, par);
        tri::solve_in_place(v, n, false, false, &mut x, k, threads, par);
        for v in &mut x {
            *v += 0.0;
        }
        x
    }

    /// [`Lu::solve_with`] on the default number of threads.
    pub fn solve(&self, b: &[f64], k: usize) -> Vec<f64> {
        self.solve_with(b, k, threads::default_threads())
    }
}

/// Solves `A·X = B` (`gesv`): [`lu`] then [`Lu::solve`].
pub fn solve(a: &[f64], n: usize, b: &[f64], k: usize) -> Result<Vec<f64>, LinalgError> {
    Ok(lu(a, n)?.solve(b, k))
}
