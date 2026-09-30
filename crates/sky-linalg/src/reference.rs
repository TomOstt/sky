//! The plain loops the fast kernels must equal **bit for bit**.
//!
//! These are the definitions: short, sequential, and free of blocking,
//! vectors and threads. The property tests in `tests/reference.rs` compare
//! every kernel with them on random shapes (including awkward sizes and
//! values near overflow), and the benchmarks use them as the "naive"
//! baseline.

use crate::{is_symmetric, LinalgError, Lu, LANES};

/// `a * b`, `a` (m by k) and `b` (k by n): each entry `c = 0; c = c + a*b`
/// for `k` ascending.
pub fn matmul(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
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

/// The Gram matrix `b · bᵀ` (`b` is m by k): the upper triangle by the
/// naive sums, mirrored.
pub fn gram(b: &[f64], m: usize, k: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * m];
    for i in 0..m {
        for j in i..m {
            let mut s = 0.0;
            for p in 0..k {
                s += b[i * k + p] * b[j * k + p];
            }
            out[i * m + j] = s;
            out[j * m + i] = s;
        }
    }
    out
}

/// The dot product in the fixed order of [`crate::dot`]'s module docs.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    if n < LANES {
        let mut s = 0.0;
        for i in 0..n {
            s += a[i] * b[i];
        }
        return s;
    }
    let mut s = [0.0_f64; LANES];
    for i in 0..n {
        s[i % LANES] += a[i] * b[i];
    }
    ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]))
}

/// Gauss-Jordan with partial pivoting, every operation on the whole
/// augmented rows.
pub fn inverse(a: &[f64], n: usize, mirror: bool) -> Result<Vec<f64>, LinalgError> {
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
    for x in &mut inv {
        *x += 0.0;
    }
    if mirror && is_symmetric(a, n) {
        for i in 0..n {
            for j in i + 1..n {
                inv[j * n + i] = inv[i * n + j];
            }
        }
    }
    Ok(inv)
}

/// The left-looking Cholesky factor, entry by entry.
pub fn cholesky(a: &[f64], n: usize) -> Result<Vec<f64>, LinalgError> {
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

/// Forward then back substitution, a column of `B` at a time.
pub fn cholesky_solve(l: &[f64], n: usize, b: &[f64], k: usize) -> Vec<f64> {
    let mut x = b.to_vec();
    for c in 0..k {
        for i in 0..n {
            let mut s = x[i * k + c];
            for j in 0..i {
                s -= l[i * n + j] * x[j * k + c];
            }
            x[i * k + c] = s / l[i * n + i];
        }
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

/// LU as Sky defines it: the forward phase of the Gauss-Jordan `inverse`,
/// operation for operation. At step `col`: the pivot search and test of the
/// inverse; the whole rows exchanged; the row's entries right of the
/// diagonal divided by the pivot `d` (the diagonal keeps `d`); and for each
/// later row with multiplier `f = m[r][col] != 0` (kept in place) the
/// entries right of the column reduced by `f` times the normalized pivot
/// row. So a matrix is singular for the solve exactly when it is for the
/// inverse.
pub fn lu(a: &[f64], n: usize) -> Result<Lu, LinalgError> {
    let mut m = a.to_vec();
    let mut swaps = Vec::with_capacity(n);
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
            }
        }
        swaps.push(pivot);
        let d = m[col * n + col];
        for j in col + 1..n {
            m[col * n + j] /= d;
        }
        for r in col + 1..n {
            let f = m[r * n + col];
            if f == 0.0 {
                continue;
            }
            for j in col + 1..n {
                m[r * n + j] -= f * m[col * n + j];
            }
        }
    }
    Ok(Lu { n, lu: m, swaps })
}

/// `A·X = B` from a factorization: the row exchanges applied to `B`, the
/// forward phase of the elimination on the right-hand sides (divide row
/// `col` by its pivot, then subtract `f` times it from each later row with
/// `f != 0`), then back substitution with the unit upper triangle.
pub fn lu_solve(f: &Lu, b: &[f64], k: usize) -> Vec<f64> {
    let n = f.n;
    let mut x = b.to_vec();
    for (c, &p) in f.swaps.iter().enumerate() {
        if p != c {
            for j in 0..k {
                x.swap(p * k + j, c * k + j);
            }
        }
    }
    for col in 0..n {
        let d = f.lu[col * n + col];
        for j in 0..k {
            x[col * k + j] /= d;
        }
        for r in col + 1..n {
            let m = f.lu[r * n + col];
            if m == 0.0 {
                continue;
            }
            for j in 0..k {
                x[r * k + j] -= m * x[col * k + j];
            }
        }
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            let u = f.lu[i * n + j];
            for c in 0..k {
                x[i * k + c] -= u * x[j * k + c];
            }
        }
    }
    for v in &mut x {
        *v += 0.0;
    }
    x
}

/// `Z·A = X` (that is, `X·A⁻¹`) for `X` of `m` rows, without forming
/// `A⁻¹`: a forward solve against the unit upper triangle, then the
/// elimination's operations applied to columns, last step first, then the
/// column exchanges.
pub fn lu_solve_right(f: &Lu, x: &[f64], m: usize) -> Vec<f64> {
    let n = f.n;
    let mut z = x.to_vec();
    for row in 0..m {
        let y = &mut z[row * n..(row + 1) * n];
        for j in 0..n {
            let mut s = y[j];
            for (i, &yi) in y[..j].iter().enumerate() {
                s -= yi * f.lu[i * n + j];
            }
            y[j] = s;
        }
    }
    for col in (0..n).rev() {
        for row in 0..m {
            let y = &mut z[row * n..(row + 1) * n];
            let mut s = y[col];
            for r in col + 1..n {
                let mult = f.lu[r * n + col];
                if mult == 0.0 {
                    continue;
                }
                s -= mult * y[r];
            }
            y[col] = s / f.lu[col * n + col];
        }
    }
    for (c, &p) in f.swaps.iter().enumerate().rev() {
        if p != c {
            for row in 0..m {
                z.swap(row * n + p, row * n + c);
            }
        }
    }
    for v in &mut z {
        *v += 0.0;
    }
    z
}
