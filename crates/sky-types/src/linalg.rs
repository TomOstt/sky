//! Dense linear algebra on row-major `f64` slices.
//!
//! Plain loops, no dependencies (`faer` comes later). Everything here is
//! pure numerics: no spans, no dimensions, no values. The interpreter's
//! value layer (`sky_interp::array`) checks shapes, attaches dimensions and
//! turns the failures below into runtime errors.
//!
//! The code lives in `sky-types` (and `sky-interp` re-exports it as
//! `sky_interp::linalg`) so the checker and the interpreter share one
//! implementation: the checker's numerical warnings ([`crate::numerics`])
//! call [`inverse`], so "this matrix is singular" at check time is by
//! construction the verdict `sky run` gives.
//!
//! Matrices are `n` by `n` (or `m` by
//! `k`, as the signature says) in row-major order: entry `(i, j)` of an
//! `m` by `n` matrix is `a[i * n + j]`.
//!
//! Sums accumulate in index order starting from `+0.0`, so results are
//! deterministic and never contain `-0.0` from an empty or all-zero sum.

/// Why a factorization or inverse failed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LinalgError {
    /// No pivot large enough: the matrix has no inverse (or is so close to
    /// singular that its inverse would be noise).
    Singular,
    /// Cholesky needs `aᵢⱼ == aⱼᵢ` exactly.
    NotSymmetric,
    /// Cholesky found a pivot that is not positive, at this row.
    NotPositiveDefinite { row: usize },
}

/// `a * b` for `a` (m by k) and `b` (k by n): an m by n matrix.
pub fn matmul(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    debug_assert_eq!((a.len(), b.len()), (m * k, k * n));
    let mut out = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            out.push(s);
        }
    }
    out
}

/// Whether the `n` by `n` matrix `a` equals its transpose entry for entry.
pub fn is_symmetric(a: &[f64], n: usize) -> bool {
    (0..n).all(|i| (i + 1..n).all(|j| a[i * n + j] == a[j * n + i]))
}

/// The inverse of an `n` by `n` matrix by Gauss-Jordan elimination with
/// partial pivoting (the largest entry of the column, the first on a tie).
///
/// A pivot no larger than `n · ε · max|aᵢⱼ|` counts as zero, so an exactly
/// singular matrix (`[[1, 2], [2, 4]]`) and a numerically singular one
/// (`[[1, 2, 3], [4, 5, 6], [7, 8, 9]]`, whose elimination leaves a pivot of
/// rounding size) are both [`LinalgError::Singular`].
///
/// If `a` is exactly symmetric the result is too: the inverse of a
/// symmetric matrix is symmetric, but two elimination orders round
/// `(A⁻¹)ᵢⱼ` and `(A⁻¹)ⱼᵢ` differently, so the upper triangle is computed
/// and mirrored ([`crate::needs_mirroring`]).
pub fn inverse(a: &[f64], n: usize) -> Result<Vec<f64>, LinalgError> {
    debug_assert_eq!(a.len(), n * n);
    let mut m = a.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    let scale = a.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
    let tol = n as f64 * f64::EPSILON * scale;
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
            for j in 0..n {
                m.swap(pivot * n + j, col * n + j);
                inv.swap(pivot * n + j, col * n + j);
            }
        }
        let d = m[col * n + col];
        for j in 0..n {
            m[col * n + j] /= d;
            inv[col * n + j] /= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = m[r * n + col];
            if f == 0.0 {
                continue;
            }
            for j in 0..n {
                m[r * n + j] -= f * m[col * n + j];
                inv[r * n + j] -= f * inv[col * n + j];
            }
        }
    }
    // `x + 0.0` turns a `-0.0` (from `0 / negative`) into `0.0`.
    for x in &mut inv {
        *x += 0.0;
    }
    if crate::needs_mirroring(crate::TagOp::Inverse) && is_symmetric(a, n) {
        for i in 0..n {
            for j in i + 1..n {
                inv[j * n + i] = inv[i * n + j];
            }
        }
    }
    Ok(inv)
}

/// The lower-triangular Cholesky factor `L` (row-major, zeros above the
/// diagonal) of a symmetric positive-definite `n` by `n` matrix: `A = L·Lᵀ`.
/// Only exactly symmetric matrices are factored, and every pivot must be
/// positive.
pub fn cholesky(a: &[f64], n: usize) -> Result<Vec<f64>, LinalgError> {
    debug_assert_eq!(a.len(), n * n);
    if !is_symmetric(a, n) {
        return Err(LinalgError::NotSymmetric);
    }
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= l[j * n + k] * l[j * n + k];
        }
        if d.is_nan() || d <= 0.0 || !d.is_finite() {
            return Err(LinalgError::NotPositiveDefinite { row: j });
        }
        let ljj = d.sqrt();
        l[j * n + j] = ljj;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / ljj;
        }
    }
    Ok(l)
}

/// Solves `A·X = B` given the Cholesky factor `l` of `A` (`n` by `n`) and `B`
/// (`n` by `k`, row-major): forward substitution `L·Y = B`, then back
/// substitution `Lᵀ·X = Y`. A vector is the `k = 1` case.
pub fn cholesky_solve(l: &[f64], n: usize, b: &[f64], k: usize) -> Vec<f64> {
    debug_assert_eq!((l.len(), b.len()), (n * n, n * k));
    let mut x = b.to_vec();
    for c in 0..k {
        // L·y = b
        for i in 0..n {
            let mut s = x[i * k + c];
            for j in 0..i {
                s -= l[i * n + j] * x[j * k + c];
            }
            x[i * k + c] = s / l[i * n + i];
        }
        // Lᵀ·x = y
        for i in (0..n).rev() {
            let mut s = x[i * k + c];
            for j in i + 1..n {
                s -= l[j * n + i] * x[j * k + c];
            }
            x[i * k + c] = s / l[i * n + i];
        }
    }
    for v in &mut x {
        *v += 0.0;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= tol, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn matmul_known_products() {
        // [[1, 2, 3], [4, 5, 6]] * [[7, 8], [9, 10], [11, 12]]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        assert_eq!(matmul(&a, 2, 3, &b, 2), [58.0, 64.0, 139.0, 154.0]);
        // matrix times a column, and a row times a column (a dot product)
        assert_eq!(matmul(&a, 2, 3, &[1.0, 0.0, -1.0], 1), [-2.0, -2.0]);
        assert_eq!(matmul(&[1.0, 2.0, 3.0], 1, 3, &[4.0, 5.0, 6.0], 1), [32.0]);
    }

    #[test]
    fn inverse_of_a_2x2_is_exact() {
        // [[4, 7], [2, 6]]⁻¹ = 1/10 [[6, -7], [-2, 4]]
        let inv = inverse(&[4.0, 7.0, 2.0, 6.0], 2).unwrap();
        close(&inv, &[0.6, -0.7, -0.2, 0.4], 1e-15);
    }

    #[test]
    fn inverse_needs_pivoting() {
        // A zero in the leading position: only row exchange finds the inverse.
        let inv = inverse(&[0.0, 1.0, 1.0, 0.0], 2).unwrap();
        assert_eq!(inv, [0.0, 1.0, 1.0, 0.0]);
        let a = [0.0, 2.0, 1.0, 1.0, 0.0, 3.0, 4.0, 5.0, 0.0];
        let inv = inverse(&a, 3).unwrap();
        let back = matmul(&a, 3, 3, &inv, 3);
        close(&back, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 1e-14);
    }

    #[test]
    fn singular_matrices_are_rejected() {
        assert_eq!(
            inverse(&[1.0, 2.0, 2.0, 4.0], 2),
            Err(LinalgError::Singular)
        );
        assert_eq!(inverse(&[0.0; 4], 2), Err(LinalgError::Singular));
        // Numerically singular: elimination leaves a rounding-size pivot.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq!(inverse(&a, 3), Err(LinalgError::Singular));
    }

    #[test]
    fn the_inverse_of_a_symmetric_matrix_is_bitwise_symmetric() {
        // A matrix whose two triangles round differently without mirroring.
        let a = [
            2.0, -1.0, 0.3, 0.1, //
            -1.0, 2.0, -1.0, 0.7, //
            0.3, -1.0, 2.0, -1.0, //
            0.1, 0.7, -1.0, 2.0,
        ];
        assert!(is_symmetric(&a, 4));
        let inv = inverse(&a, 4).unwrap();
        assert!(is_symmetric(&inv, 4));
        let back = matmul(&a, 4, 4, &inv, 4);
        let eye: Vec<f64> = (0..16).map(|i| f64::from(i % 5 == 0)).collect();
        close(&back, &eye, 1e-14);
    }

    #[test]
    fn a_diagonal_inverse_has_no_negative_zeros() {
        let inv = inverse(&[-2.0, 0.0, 0.0, 4.0], 2).unwrap();
        assert_eq!(inv, [-0.5, 0.0, 0.0, 0.25]);
        assert!(inv.iter().all(|x| !(x.to_bits() >> 63 == 1 && *x == 0.0)));
    }

    #[test]
    fn cholesky_factor_and_solve() {
        // The classic example: A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
        // has L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]].
        let a = [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
        let l = cholesky(&a, 3).unwrap();
        assert_eq!(l, [2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0]);
        // A·x = b with x = [1, 2, 3]: b = [4+24-48, 12+74-129, -16-86+294]
        let b = [-20.0, -43.0, 192.0];
        assert_eq!(cholesky_solve(&l, 3, &b, 1), [1.0, 2.0, 3.0]);
        // Several right-hand sides at once: the columns of A·I are A's columns.
        let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let inv = cholesky_solve(&l, 3, &eye, 3);
        let back = matmul(&a, 3, 3, &inv, 3);
        close(&back, &eye, 1e-12);
    }

    #[test]
    fn cholesky_rejects_what_is_not_positive_definite() {
        assert_eq!(
            cholesky(&[1.0, 2.0, 2.0, 1.0], 2),
            Err(LinalgError::NotPositiveDefinite { row: 1 })
        );
        assert_eq!(
            cholesky(&[0.0, 0.0, 0.0, 1.0], 2),
            Err(LinalgError::NotPositiveDefinite { row: 0 })
        );
        assert_eq!(
            cholesky(&[-1.0], 1),
            Err(LinalgError::NotPositiveDefinite { row: 0 })
        );
        assert_eq!(
            cholesky(&[2.0, 1.0, 1.5, 2.0], 2),
            Err(LinalgError::NotSymmetric)
        );
    }
}
