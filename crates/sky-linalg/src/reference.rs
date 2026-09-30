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

/// Unblocked right-looking LU with partial pivoting.
pub fn lu(a: &[f64], n: usize) -> Result<Lu, LinalgError> {
    let mut w = a.to_vec();
    let scale = a.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
    let tol = n as f64 * f64::EPSILON * scale;
    let mut perm: Vec<usize> = (0..n).collect();
    let mut swaps = 0;
    for c in 0..n {
        let mut pivot = c;
        for r in c + 1..n {
            if w[r * n + c].abs() > w[pivot * n + c].abs() {
                pivot = r;
            }
        }
        let size = w[pivot * n + c].abs();
        if size.is_nan() || size <= tol {
            return Err(LinalgError::Singular);
        }
        if pivot != c {
            for j in 0..n {
                w.swap(pivot * n + j, c * n + j);
            }
            perm.swap(pivot, c);
            swaps += 1;
        }
        let d = w[c * n + c];
        for r in c + 1..n {
            let l = w[r * n + c] / d;
            w[r * n + c] = l;
            for j in c + 1..n {
                w[r * n + j] -= l * w[c * n + j];
            }
        }
    }
    Ok(Lu {
        n,
        lu: w,
        perm,
        swaps,
    })
}

/// The solve with a factorization from [`lu`], a column at a time.
pub fn lu_solve(f: &Lu, b: &[f64], k: usize) -> Vec<f64> {
    let n = f.n;
    let mut x = vec![0.0; n * k];
    for (i, &p) in f.perm.iter().enumerate() {
        x[i * k..(i + 1) * k].copy_from_slice(&b[p * k..(p + 1) * k]);
    }
    for c in 0..k {
        for i in 0..n {
            let mut s = x[i * k + c];
            for j in 0..i {
                s -= f.lu[i * n + j] * x[j * k + c];
            }
            x[i * k + c] = s;
        }
        for i in (0..n).rev() {
            let mut s = x[i * k + c];
            for j in i + 1..n {
                s -= f.lu[i * n + j] * x[j * k + c];
            }
            x[i * k + c] = s / f.lu[i * n + i];
        }
    }
    for v in &mut x {
        *v += 0.0;
    }
    x
}
