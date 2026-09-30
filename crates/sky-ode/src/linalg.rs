//! Dense LU factorisation with partial pivoting: the only linear algebra the
//! stiff solver needs (no new dependencies, and the systems are small).
//!
//! Row-major storage, Doolittle elimination with row exchanges, multipliers
//! kept in place below the diagonal (the LAPACK `getrf`/`getrs` layout).
//! Partial pivoting (choosing the largest entry in the column) keeps the
//! multipliers at most 1 in magnitude, which is what makes Gaussian
//! elimination backward stable in practice (Golub & Van Loan, *Matrix
//! Computations*, 4th ed., section 3.4).

/// An LU factorisation `P A = L U` of a square matrix.
pub(crate) struct Lu {
    n: usize,
    lu: Vec<f64>,
    /// `piv[k]` is the row swapped with row `k` at elimination step `k`.
    piv: Vec<usize>,
}

impl Lu {
    /// Factors the `n x n` row-major matrix `a`. `None` if a pivot is zero
    /// or not finite (a singular or corrupted matrix): the caller reacts by
    /// shrinking the step, which makes `I/(gamma h) - J` diagonally dominant.
    pub(crate) fn factor(mut a: Vec<f64>, n: usize) -> Option<Lu> {
        debug_assert_eq!(a.len(), n * n);
        let mut piv = vec![0usize; n];
        for k in 0..n {
            let mut p = k;
            let mut best = a[k * n + k].abs();
            for i in k + 1..n {
                let v = a[i * n + k].abs();
                if v > best {
                    best = v;
                    p = i;
                }
            }
            // Zero, NaN and infinite pivots all mean the matrix is unusable.
            if best.is_nan() || best <= 0.0 || best.is_infinite() {
                return None;
            }
            piv[k] = p;
            if p != k {
                for j in 0..n {
                    a.swap(k * n + j, p * n + j);
                }
            }
            let d = a[k * n + k];
            for i in k + 1..n {
                let m = a[i * n + k] / d;
                a[i * n + k] = m;
                if m != 0.0 {
                    for j in k + 1..n {
                        a[i * n + j] -= m * a[k * n + j];
                    }
                }
            }
        }
        Some(Lu { n, lu: a, piv })
    }

    /// Solves `A x = b` in place (`b` becomes `x`).
    pub(crate) fn solve(&self, b: &mut [f64]) {
        let n = self.n;
        debug_assert_eq!(b.len(), n);
        for k in 0..n {
            b.swap(k, self.piv[k]);
        }
        for i in 1..n {
            let mut s = b[i];
            for j in 0..i {
                s -= self.lu[i * n + j] * b[j];
            }
            b[i] = s;
        }
        for i in (0..n).rev() {
            let mut s = b[i];
            for j in i + 1..n {
                s -= self.lu[i * n + j] * b[j];
            }
            b[i] = s / self.lu[i * n + i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matvec(a: &[f64], x: &[f64]) -> Vec<f64> {
        let n = x.len();
        (0..n)
            .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum())
            .collect()
    }

    #[test]
    fn solves_a_system_that_needs_pivoting() {
        // A zero in the leading position: without row exchanges this fails.
        let a = vec![0.0, 1.0, 1.0, 1.0, 0.0, 2.0, 3.0, 1.0, 0.0];
        let lu = Lu::factor(a.clone(), 3).unwrap();
        let x_true = [1.0, -2.0, 3.0];
        let mut b = matvec(&a, &x_true);
        lu.solve(&mut b);
        for (x, e) in b.iter().zip(x_true) {
            assert!((x - e).abs() < 1e-13, "{x} vs {e}");
        }
    }

    #[test]
    fn solves_pseudo_random_systems_to_small_residual() {
        // A fixed LCG so the test is deterministic without a dependency.
        let mut s = 12345u64;
        let mut rnd = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        for n in [1usize, 2, 5, 12, 30] {
            let a: Vec<f64> = (0..n * n).map(|_| rnd()).collect();
            let x_true: Vec<f64> = (0..n).map(|_| rnd()).collect();
            let b0 = matvec(&a, &x_true);
            let lu = Lu::factor(a.clone(), n).unwrap();
            let mut x = b0.clone();
            lu.solve(&mut x);
            let r = matvec(&a, &x);
            let res = r
                .iter()
                .zip(&b0)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0, f64::max);
            assert!(res < 1e-11, "n = {n}: residual {res}");
        }
    }

    #[test]
    fn singular_and_non_finite_matrices_are_rejected() {
        assert!(Lu::factor(vec![1.0, 2.0, 2.0, 4.0], 2).is_none());
        assert!(Lu::factor(vec![f64::NAN, 0.0, 0.0, 1.0], 2).is_none());
        assert!(Lu::factor(vec![0.0], 1).is_none());
    }
}
