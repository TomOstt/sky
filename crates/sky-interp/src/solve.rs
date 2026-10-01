//! The algorithm variants of `A⁻¹ @ b`, `x @ A⁻¹` and `matmul`: the numerics
//! and the value layer for what the algorithm-choosing rewrites emit.
//!
//! Like [`crate::array`], this is shared by every executor of Sky: the
//! tree-walker (for a chain the checker planned) and the SIR interpreter
//! (for `solve`, `solve.right`, `matmul[...]` and `diag.mul`) call the same
//! functions, so the same rewritten program gives the same bits and the same
//! errors on both.
//!
//! # Which are exact
//!
//! * [`diag_mul`] and [`matmul_ex`] (transposes and the Gram product) give
//!   **exactly the bits of `matmul`** on the same operands. Each function
//!   says why.
//! * [`solve`] and [`solve_right`] do **not** give the bits of `inverse`
//!   followed by `matmul`. They solve the system instead of forming the
//!   inverse, which rounds differently and is at least as accurate. The
//!   spec's "Algorithm choice" section permits it.
//!
//! # LU
//!
//! [`lu_factor`] runs the forward phase of [`linalg::inverse`]'s Gauss-Jordan
//! elimination, operation for operation: the same pivot search, the same
//! pivot test, the same normalization of the pivot row and the same
//! elimination below it. So the trailing entries, and hence the pivots, are
//! the same numbers, and a matrix is singular for the solve exactly when it
//! is for the inverse (`solve_is_singular_exactly_when_inverse_is` pins it).
//! Only the back phase differs: back substitution with the unit upper
//! triangle instead of elimination above the pivots. The factorization and
//! both solves are `sky-linalg`'s (`sky_linalg::Lu`), which runs them
//! blocked and threaded and gives the bits of the plain loops
//! (`sky_linalg::reference`); this module only turns failures into
//! runtime errors.

use sky_diag::Span;
use sky_units::Dimension;

use crate::arith::{overflow, R};
use crate::array::{cholesky_solve, shape_text, transpose};
use crate::linalg::{self, LinalgError};
use crate::{RuntimeError, Value};
use sky_types::{Algorithm, Side, SolveMethod};

/// The factorization [`lu_factor`] returns: see `sky_linalg::Lu`.
pub use crate::linalg::Lu;

/// Factors the `n` by `n` matrix `a`. [`LinalgError::Singular`] under the
/// same test as [`linalg::inverse`]: a pivot no larger than `n·ε·max|aᵢⱼ|`.
pub fn lu_factor(a: &[f64], n: usize) -> Result<Lu, LinalgError> {
    debug_assert_eq!(a.len(), n * n);
    linalg::lu(a, n)
}

// ----------------------------------------------------------------------
// The value layer.

fn exponent_overflow(whole: Span) -> RuntimeError {
    RuntimeError::new("dimension exponent too large", whole)
}

fn all_finite(data: Vec<f64>, whole: Span) -> R<Vec<f64>> {
    match data.iter().all(|x| x.is_finite()) {
        true => Ok(data),
        false => Err(overflow(whole)),
    }
}

/// The error `inverse` gives for a singular matrix, word for word.
fn singular(whole: Span, operand: Span, dims: &[usize]) -> RuntimeError {
    RuntimeError::new("the matrix is singular", whole)
        .with_label("this matrix has no inverse")
        .with_secondary(operand, format!("this is {}", shape_text(dims)))
        .with_note(
            "a matrix is singular when a pivot of the elimination is zero, or too small to tell from rounding error",
        )
}

fn promise(method: SolveMethod) -> &'static str {
    match method {
        SolveMethod::Cholesky => {
            "the solve is a Cholesky factorization because the checker knows `A` is PositiveDefinite"
        }
        SolveMethod::Diagonal => {
            "the solve is a division of each entry because the checker knows `A` is Diagonal"
        }
        SolveMethod::Lu => "the solve is an LU factorization",
    }
}

/// The note of a broken parameter promise.
const PARAMETER_PROMISE: &str =
    "a tag on a parameter is checked when the function is called, for the matrix passed";

/// Checks, at a call, the tags a parameter declares (`declared`) of the
/// matrix `v` passed for it, beyond those the caller already guarantees
/// (`known`; the tree-walker passes none). The rewrites that rely on a tag
/// are not checked where they run (`Kᵀ` of a Symmetric `K` is `K` itself),
/// so every engine checks the promise here, once per call, with this one
/// message, at the argument's span:
///
/// * Diagonal: every off-diagonal entry is zero (this implies Symmetric);
/// * else Symmetric (also implied by PositiveDefinite): entry `[i, j]` is
///   entry `[j, i]`, bit for bit, as the checker's own Symmetric results
///   are (`sky_types::needs_mirroring`).
///
/// Positive definiteness is left to the Cholesky solve that relies on it,
/// which checks it (and costs a factorization to check), and Orthogonal to
/// the checker, which requires the argument to be known Orthogonal. A value
/// that is not a square matrix has no promise to check.
pub fn check_parameter_tags(
    declared: sky_types::TagSet,
    known: sky_types::TagSet,
    v: &Value,
    span: Span,
) -> R<()> {
    use sky_types::Tag;
    let (want, have) = (declared.closure(), known.closure());
    let Value::Array { dims, data, .. } = v else {
        return Ok(());
    };
    let n = match dims[..] {
        [r, c] if r == c => r,
        _ => return Ok(()),
    };
    let needs = |t: Tag| want.contains(t) && !have.contains(t);
    if want.contains(Tag::Diagonal) {
        if !needs(Tag::Diagonal) {
            return Ok(());
        }
        for i in 0..n {
            for j in 0..n {
                if i != j && data[i * n + j] != 0.0 {
                    return Err(RuntimeError::new("the matrix is not diagonal", span)
                        .with_label(format!(
                            "declared Diagonal, but entry [{i}, {j}] is not zero"
                        ))
                        .with_note(PARAMETER_PROMISE));
                }
            }
        }
        return Ok(());
    }
    if !needs(Tag::Symmetric) {
        return Ok(());
    }
    let declared = if want.contains(Tag::PositiveDefinite) {
        "declared PositiveDefinite, which implies symmetric"
    } else {
        "declared Symmetric"
    };
    for i in 0..n {
        for j in i + 1..n {
            if data[i * n + j] != data[j * n + i] {
                return Err(RuntimeError::new("the matrix is not symmetric", span)
                    .with_label(format!(
                        "{declared}, but entries [{i}, {j}] and [{j}, {i}] differ"
                    ))
                    .with_note(PARAMETER_PROMISE));
            }
        }
    }
    Ok(())
}

/// `A` as an `n` by `n` matrix: its size and entries, or the error a solve
/// gives for anything else.
fn square(a: &Value, whole: Span, aspan: Span) -> R<(usize, &[f64], Dimension)> {
    match a {
        Value::Array { dims, data, dim } => match dims[..] {
            [n, m] if n == m => Ok((n, data, *dim)),
            _ => Err(RuntimeError::new(
                format!(
                    "the inverse needs a square matrix, found {}",
                    shape_text(dims)
                ),
                aspan,
            )),
        },
        other => Err(RuntimeError::new(
            format!("the inverse needs a matrix, found {}", other.kind()),
            whole,
        )),
    }
}

/// The columns of the right-hand side of `A⁻¹ @ b` (`b` a vector of `n`
/// entries, or `n` rows of `k` columns), or the error `matmul` gives.
fn rhs_columns(
    a: &Value,
    b: &Value,
    n: usize,
    whole: Span,
    aspan: Span,
    bspan: Span,
) -> R<(usize, &'static str)> {
    let dims_of = |v: &Value| match v {
        Value::Array { dims, .. } => dims.clone(),
        _ => Vec::new(),
    };
    let (da, db) = (dims_of(a), dims_of(b));
    match db[..] {
        [len] if len == n => Ok((1, "")),
        [rows, cols] if rows == n => Ok((cols, "")),
        _ => Err(RuntimeError::new(
            format!(
                "cannot multiply {} and {}: the inner dimensions differ",
                shape_text(&da),
                shape_text(&db)
            ),
            whole,
        )
        .with_secondary(aspan, shape_text(&da))
        .with_secondary(bspan, shape_text(&db))),
    }
}

/// Checks that a Diagonal promise holds: every off-diagonal entry is zero.
fn check_diagonal(a: &[f64], n: usize, aspan: Span) -> R<()> {
    for i in 0..n {
        for j in 0..n {
            if i != j && a[i * n + j] != 0.0 {
                return Err(RuntimeError::new("the matrix is not diagonal", aspan)
                    .with_label(format!(
                        "declared Diagonal, but entry [{i}, {j}] is not zero"
                    ))
                    .with_note(promise(SolveMethod::Diagonal)));
            }
        }
    }
    Ok(())
}

/// The diagonal of a Diagonal matrix, or the errors of a broken promise or a
/// singular one (a pivot no larger than `n·ε·max|aᵢⱼ|`, as `inverse`).
fn diagonal_of(a: &[f64], n: usize, whole: Span, aspan: Span, dims: &[usize]) -> R<Vec<f64>> {
    check_diagonal(a, n, aspan)?;
    let d: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    let scale = a.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
    let tol = n as f64 * f64::EPSILON * scale;
    if d.iter().any(|x| x.is_nan() || x.abs() <= tol) {
        return Err(singular(whole, aspan, dims));
    }
    Ok(d)
}

fn cholesky_factor(a: &[f64], n: usize, aspan: Span) -> R<Vec<f64>> {
    let note = promise(SolveMethod::Cholesky);
    linalg::cholesky(a, n).map_err(|e| match e {
        LinalgError::NotSymmetric => RuntimeError::new("the matrix is not symmetric", aspan)
            .with_label("declared PositiveDefinite, which implies symmetric")
            .with_note(note),
        LinalgError::NotPositiveDefinite { row } => {
            RuntimeError::new("the matrix is not positive definite", aspan)
                .with_label(format!(
                    "the factorization failed at row {row}: its pivot is not positive"
                ))
                .with_note(note)
        }
        LinalgError::Singular => RuntimeError::new("the matrix is singular", aspan),
    })
}

/// `A⁻¹ @ b` by `method`, without forming `A⁻¹`: `b` is a vector or a matrix
/// of right-hand sides, the result has `b`'s shape and the dimension of `b`
/// over `A`'s. `whole` is the expression, `aspan` and `bspan` the operands.
///
/// Not the bits of `inverse` then `matmul` (see the module documentation),
/// except that a singular `A` is reported exactly as `inverse` reports it,
/// and a broken tag promise (`Diagonal`, `PositiveDefinite`) is an error
/// naming the fact.
pub fn solve(
    method: SolveMethod,
    a: &Value,
    b: &Value,
    whole: Span,
    aspan: Span,
    bspan: Span,
) -> R<Value> {
    if method == SolveMethod::Cholesky {
        return cholesky_solve(a, b, whole, aspan, bspan);
    }
    let (n, ad, adim) = square(a, whole, aspan)?;
    let Value::Array {
        dims: db,
        data: bd,
        dim: bdim,
    } = b
    else {
        return Err(RuntimeError::new(
            format!("cannot solve with {} and {}", a.kind(), b.kind()),
            whole,
        ));
    };
    let (k, _) = rhs_columns(a, b, n, whole, aspan, bspan)?;
    let dim = bdim
        .checked_div(adim)
        .map_err(|_| exponent_overflow(whole))?;
    let adims = [n, n];
    let x = match method {
        SolveMethod::Lu => {
            linalg::lu_solve(ad, n, bd, k).map_err(|_| singular(whole, aspan, &adims))?
        }
        SolveMethod::Diagonal => {
            let d = diagonal_of(ad, n, whole, aspan, &adims)?;
            let mut x = bd.clone();
            for i in 0..n {
                for j in 0..k {
                    x[i * k + j] = bd[i * k + j] / d[i] + 0.0;
                }
            }
            x
        }
        SolveMethod::Cholesky => unreachable!("handled above"),
    };
    Ok(Value::array(db.clone(), all_finite(x, whole)?, dim))
}

/// `x @ A⁻¹` by `method`, without forming `A⁻¹` (a transposed solve): `x` is
/// a vector of `n` entries or a matrix of `n` columns, the result has `x`'s
/// shape and the dimension of `x` over `A`'s. See [`solve`].
pub fn solve_right(
    method: SolveMethod,
    x: &Value,
    a: &Value,
    whole: Span,
    xspan: Span,
    aspan: Span,
) -> R<Value> {
    let (n, ad, adim) = square(a, whole, aspan)?;
    let Value::Array {
        dims: dx,
        data: xd,
        dim: xdim,
    } = x
    else {
        return Err(RuntimeError::new(
            format!("cannot solve with {} and {}", x.kind(), a.kind()),
            whole,
        ));
    };
    let m = match dx[..] {
        [len] if len == n => 1,
        [rows, cols] if cols == n => rows,
        _ => {
            return Err(RuntimeError::new(
                format!(
                    "cannot multiply {} and {}: the inner dimensions differ",
                    shape_text(dx),
                    shape_text(&[n, n])
                ),
                whole,
            )
            .with_secondary(xspan, shape_text(dx))
            .with_secondary(aspan, shape_text(&[n, n])))
        }
    };
    let dim = xdim
        .checked_div(adim)
        .map_err(|_| exponent_overflow(whole))?;
    let adims = [n, n];
    let z = match method {
        SolveMethod::Lu => {
            linalg::lu_solve_right(ad, n, xd, m).map_err(|_| singular(whole, aspan, &adims))?
        }
        SolveMethod::Diagonal => {
            let d = diagonal_of(ad, n, whole, aspan, &adims)?;
            let mut z = xd.clone();
            for i in 0..m {
                for j in 0..n {
                    z[i * n + j] = xd[i * n + j] / d[j] + 0.0;
                }
            }
            z
        }
        // `A` is symmetric, so `x·A⁻¹ = (A⁻¹·xᵀ)ᵀ`: solve for the rows of `x`
        // as columns. The factor and the two substitutions are `A⁻¹ @ b`'s.
        SolveMethod::Cholesky => {
            let l = cholesky_factor(ad, n, aspan)?;
            let mut xt = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    xt[j * m + i] = xd[i * n + j];
                }
            }
            let zt = linalg::cholesky_solve(&l, n, &xt, m);
            let mut z = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    z[i * n + j] = zt[j * m + i];
                }
            }
            z
        }
    };
    Ok(Value::array(dx.clone(), all_finite(z, whole)?, dim))
}

/// `D @ X` (`Side::Left`) or `X @ D` (`Side::Right`) for a Diagonal `D`: a
/// scaling of the rows or of the columns of `X`, in O(entries of `X`).
///
/// **Bit-identical to [`crate::array::matmul`]**: entry `(i, j)` of the
/// product is a sum, started at `+0.0` and accumulated in index order, of
/// `dᵢₚ·xₚⱼ`; every term but `p = i` multiplies a zero, giving `±0.0`
/// (all entries are finite), and adding `±0.0` to a sum leaves it unchanged
/// unless it is `-0.0` (which the `+0.0` start rules out) or `+0.0` (where
/// `+0.0 + -0.0` is `+0.0` again). So the sum is `0.0 + dᵢ·xᵢⱼ`, which is
/// what this computes, including the sign of a zero. An overflow anywhere is
/// the same `arithmetic overflow` at `whole`.
///
/// The Diagonal tag is a promise; a nonzero off-diagonal entry is a
/// run-time error naming it.
pub fn diag_mul(
    side: Side,
    d: &Value,
    x: &Value,
    whole: Span,
    dspan: Span,
    xspan: Span,
) -> R<Value> {
    let (l, r) = match side {
        Side::Left => (d, x),
        Side::Right => (x, d),
    };
    // `matmul`'s shape rules and errors: reuse it for anything unusual.
    let (n, dd, ddim) = match d {
        Value::Array { dims, data, dim } if dims.len() == 2 && dims[0] == dims[1] => {
            (dims[0], data, *dim)
        }
        _ => {
            let (lspan, rspan) = if side == Side::Left {
                (dspan, xspan)
            } else {
                (xspan, dspan)
            };
            return crate::array::matmul(l, r, whole, lspan, rspan);
        }
    };
    let Value::Array {
        dims: dx,
        data: xd,
        dim: xdim,
    } = x
    else {
        let (lspan, rspan) = if side == Side::Left {
            (dspan, xspan)
        } else {
            (xspan, dspan)
        };
        return crate::array::matmul(l, r, whole, lspan, rspan);
    };
    let fits = match (side, &dx[..]) {
        (Side::Left, [rows] | [rows, _]) => *rows == n,
        (Side::Right, [len]) => *len == n,
        (Side::Right, [_, cols]) => *cols == n,
        _ => false,
    };
    if !fits {
        let (lspan, rspan) = if side == Side::Left {
            (dspan, xspan)
        } else {
            (xspan, dspan)
        };
        return crate::array::matmul(l, r, whole, lspan, rspan);
    }
    check_diagonal(dd, n, dspan)?;
    let dim = match side {
        Side::Left => ddim.checked_mul(*xdim),
        Side::Right => xdim.checked_mul(ddim),
    }
    .map_err(|_| exponent_overflow(whole))?;
    // A vector on the left of `D @ x` is a column (entry `i` scales by
    // `dᵢ`); on the right of `x @ D` it is a row (entry `j` by `dⱼ`).
    let cols = match (side, dx.len()) {
        (Side::Left, 1) => 1,
        _ => *dx.last().unwrap_or(&1),
    };
    let out: Vec<f64> = xd
        .iter()
        .enumerate()
        .map(|(idx, &v)| {
            let (i, j) = (idx / cols, idx % cols);
            match side {
                Side::Left => 0.0 + dd[i * n + i] * v,
                Side::Right => 0.0 + v * dd[j * n + j],
            }
        })
        .collect();
    Ok(Value::array(dx.clone(), all_finite(out, whole)?, dim))
}

/// `matmul[ta,tb,alg=...]`: `l @ r` with the flagged operands transposed and
/// the algorithm class recorded.
///
/// **Bit-identical to `matmul(transpose(l), r)` and friends**, by
/// construction: the transpose is [`crate::array::transpose`] (the function
/// `transpose` calls) and the product is [`crate::array::matmul`]. A fast
/// kernel that reads the operand with swapped strides and accumulates `p`
/// in index order gives the same bits.
///
/// [`Algorithm::Syrk`] computes one triangle of the Gram product and mirrors
/// it, which is also bit-identical: entry `(i, j)` sums `bᵢₚ·bⱼₚ` and entry
/// `(j, i)` sums `bⱼₚ·bᵢₚ`, the same products in the same order (IEEE
/// multiplication commutes). The other classes name the shape and change
/// nothing here.
#[allow(clippy::too_many_arguments)]
pub fn matmul_ex(
    ta: bool,
    tb: bool,
    algorithm: Algorithm,
    l: &Value,
    r: &Value,
    whole: Span,
    lspan: Span,
    rspan: Span,
) -> R<Value> {
    let flip = |v: &Value, on: bool| -> R<Value> {
        if on {
            transpose(v, whole)
        } else {
            Ok(v.clone())
        }
    };
    let (l2, r2) = (flip(l, ta)?, flip(r, tb)?);
    if algorithm == Algorithm::Syrk && ta != tb {
        if let Some(v) = gram(&l2, &r2, whole)? {
            return Ok(v);
        }
    }
    crate::array::matmul(&l2, &r2, whole, lspan, rspan)
}

/// `b @ bᵀ` for `l = b` and `r = bᵀ` (both given, `r` the transpose of
/// `l`), computing the upper triangle and mirroring it. `None` if the
/// operands are not a matrix and its transpose (then the caller multiplies).
fn gram(l: &Value, r: &Value, whole: Span) -> R<Option<Value>> {
    let (
        Value::Array {
            dims: dl,
            data: b,
            dim: dim_l,
        },
        Value::Array {
            dims: dr,
            data: bt,
            dim: dim_r,
        },
    ) = (l, r)
    else {
        return Ok(None);
    };
    let (&[m, k], &[k2, m2]) = (&dl[..], &dr[..]) else {
        return Ok(None);
    };
    // `r` must be `lᵀ` bit for bit; a caller that broke the contract gets
    // the plain product instead of a wrong mirror.
    if (m, k) != (m2, k2) || (0..m).any(|i| (0..k).any(|p| b[i * k + p] != bt[p * m + i])) {
        return Ok(None);
    }
    let dim = dim_l
        .checked_mul(*dim_r)
        .map_err(|_| exponent_overflow(whole))?;
    let out = linalg::syrk(b, m, k);
    Ok(Some(Value::array(vec![m, m], all_finite(out, whole)?, dim)))
}
