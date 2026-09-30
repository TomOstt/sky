//! Dense linear algebra on row-major `f64` slices.
//!
//! The kernels live in the `sky-linalg` crate (packed, blocked, vectorized
//! and threaded, with a fixed order of operations that makes every engine
//! and machine give the same bits; see its docs and the spec's "Arrays:
//! reproducible arithmetic"). This module is the checker's and interpreter's door to
//! them: `sky-types` re-exports them here and `sky-interp` re-exports this
//! module as `sky_interp::linalg`, so the checker and the interpreter share
//! one implementation. That matters for the checker's numerical warnings
//! ([`crate::numerics`]), which call [`inverse`]: "this matrix is singular"
//! at check time is by construction the verdict `sky run` gives.
//!
//! Everything here is pure numerics: no spans, no dimensions, no values. The
//! interpreter's value layer (`sky_interp::array`) checks shapes, attaches
//! dimensions and turns the failures below into runtime errors.
//!
//! Matrices are `n` by `n` (or `m` by `k`, as the signature says) in
//! row-major order: entry `(i, j)` of an `m` by `n` matrix is `a[i * n + j]`.
//!
//! Sums accumulate in the order the spec fixes, starting from `+0.0`, so
//! results are deterministic and never contain `-0.0` from an empty or
//! all-zero sum.

pub use sky_linalg::{
    cholesky, cholesky_solve, dot, is_symmetric, sum_squares, sum_squares_scaled, LinalgError,
};

/// `a * b` for `a` (m by k) and `b` (k by n): an m by n matrix. Each entry
/// is accumulated in the naive order (`c = 0; c = c + a*b`, `k` ascending)
/// with no fused multiply-add.
pub fn matmul(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    debug_assert_eq!((a.len(), b.len()), (m * k, k * n));
    sky_linalg::gemm(a, m, k, b, n)
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
    sky_linalg::inverse_with(
        a,
        n,
        crate::needs_mirroring(crate::TagOp::Inverse),
        sky_linalg::default_threads(),
    )
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
