//! Array values: the rules for vectors and matrices that don't depend on how
//! a program is represented.
//!
//! Like [`crate::arith`], this is shared by every executor of Sky: elementwise
//! `+` and `-`, scaling, `@`, the inverse, the Cholesky solve the checker
//! promises for `A⁻¹ @ b`, indexing, and the builtins `zeros`, `identity`
//! and `len`. The numerics are in [`crate::linalg`]; this layer checks
//! shapes and dimensions, attaches them to the result, checks that every
//! result is finite, and turns failures into [`RuntimeError`]s.
//!
//! The checker has already checked shapes and dimensions, so most errors
//! here are defensive (the same rule as scalars: a runtime error, never a
//! panic). The ones a checked program can reach are a singular matrix, a
//! matrix that is not positive definite where the checker was told it is,
//! an index out of bounds (a variable one), and overflow.

use sky_diag::Span;
use sky_types::MAX_ARRAY_ELEMENTS;
use sky_units::Dimension;

use crate::arith::{describe, div_zero, overflow, ArithOp, Operand, R};
use crate::linalg::{self, LinalgError};
use crate::{format_number, RuntimeError, Value};

/// `Vector[3]`, `Matrix[2,3]`: the shape as the checker prints it.
pub fn shape_text(dims: &[usize]) -> String {
    let name = match dims.len() {
        1 => "Vector",
        2 => "Matrix",
        _ => "Tensor",
    };
    let list: Vec<String> = dims.iter().map(usize::to_string).collect();
    format!("{name}[{}]", list.join(","))
}

fn exponent_overflow(whole: Span) -> RuntimeError {
    RuntimeError::new("dimension exponent too large", whole)
}

/// All entries of `data` are finite, or an overflow error.
fn all_finite(data: Vec<f64>, whole: Span) -> R<Vec<f64>> {
    match data.iter().all(|x| x.is_finite()) {
        true => Ok(data),
        false => Err(overflow(whole)),
    }
}

/// `l op r` where at least one operand is an array. Called by
/// [`crate::arith::arith`]; see [`arith_owned`].
pub fn arith(op: ArithOp, l: &Value, r: &Value, whole: Span, lspan: Span, rspan: Span) -> R<Value> {
    arith_owned(
        op,
        Operand::Borrowed(l),
        Operand::Borrowed(r),
        whole,
        lspan,
        rspan,
    )
}

/// The entries of an array value (none for anything else, which the callers
/// have ruled out).
fn entries(v: &Value) -> &[f64] {
    match v {
        Value::Array { data, .. } => data,
        _ => &[],
    }
}

/// Scales (or divides) every entry of the array operand `arr` by the number
/// already folded into `f`: in place if `arr` is owned, into a new array
/// otherwise. Entries are computed exactly as `x op s` in both cases.
fn scale(arr: Operand<'_>, out: Dimension, f: impl Fn(f64) -> f64, whole: Span) -> R<Value> {
    match arr {
        Operand::Owned(Value::Array { dims, mut data, .. }) => {
            for x in &mut data {
                *x = f(*x);
            }
            Ok(Value::array(dims, all_finite(data, whole)?, out))
        }
        other => match other.get() {
            Value::Array { dims, data, .. } => Ok(Value::array(
                dims.clone(),
                all_finite(data.iter().map(|&x| f(x)).collect(), whole)?,
                out,
            )),
            v => Err(RuntimeError::new(
                format!("expected an array, found {}", v.kind()),
                whole,
            )),
        },
    }
}

/// [`arith`] on operands that may be owned: an owned array whose shape is the
/// result's holds the result, so no array is allocated (`v + dt * a` with `v`
/// dead afterwards writes into `v`). Entries are the same computation
/// either way, so the bits are identical; the checks come first, so an error
/// is the same too. If both are owned and equally good, the left one is used.
///
/// * `+` and `-`: two arrays of the same shape and dimension, elementwise.
/// * `*`, `/`: an array and a number, scaling every entry (a number cannot
///   be divided by an array, and two arrays have no elementwise product).
pub fn arith_owned(
    op: ArithOp,
    l: Operand<'_>,
    r: Operand<'_>,
    whole: Span,
    lspan: Span,
    rspan: Span,
) -> R<Value> {
    match (op, l.get(), r.get()) {
        (
            ArithOp::Add | ArithOp::Sub,
            Value::Array {
                dims: d1, dim: da, ..
            },
            Value::Array {
                dims: d2, dim: db, ..
            },
        ) => {
            let (verb, prep) = if op == ArithOp::Add {
                ("add", "and")
            } else {
                ("subtract", "from")
            };
            if d1 != d2 {
                let (x, y) = if op == ArithOp::Add {
                    (d1, d2)
                } else {
                    (d2, d1)
                };
                return Err(RuntimeError::new(
                    format!("cannot {verb} {} {prep} {}", shape_text(x), shape_text(y)),
                    whole,
                )
                .with_label("mismatched shapes")
                .with_secondary(lspan, shape_text(d1))
                .with_secondary(rspan, shape_text(d2)));
            }
            if da != db {
                let message = if op == ArithOp::Add {
                    format!("cannot add {} and {}", describe(da), describe(db))
                } else {
                    format!("cannot subtract {} from {}", describe(db), describe(da))
                };
                return Err(RuntimeError::new(message, whole)
                    .with_label("mismatched dimensions")
                    .with_secondary(lspan, describe(da))
                    .with_secondary(rspan, describe(db)));
            }
            let dim = *da;
            let f = if op == ArithOp::Add {
                |x: f64, y: f64| x + y
            } else {
                |x: f64, y: f64| x - y
            };
            match (l, r) {
                (Operand::Owned(Value::Array { dims, mut data, .. }), other) => {
                    for (x, &y) in data.iter_mut().zip(entries(other.get())) {
                        *x = f(*x, y);
                    }
                    Ok(Value::array(dims, all_finite(data, whole)?, dim))
                }
                (other, Operand::Owned(Value::Array { dims, mut data, .. })) => {
                    for (y, &x) in data.iter_mut().zip(entries(other.get())) {
                        *y = f(x, *y);
                    }
                    Ok(Value::array(dims, all_finite(data, whole)?, dim))
                }
                (l, r) => {
                    let data = entries(l.get())
                        .iter()
                        .zip(entries(r.get()))
                        .map(|(&x, &y)| f(x, y))
                        .collect();
                    let dims = match l.get() {
                        Value::Array { dims, .. } => dims.clone(),
                        _ => Vec::new(),
                    };
                    Ok(Value::array(dims, all_finite(data, whole)?, dim))
                }
            }
        }
        (ArithOp::Mul, Value::Number { value: s, dim: ds }, Value::Array { dim, .. })
        | (ArithOp::Mul, Value::Array { dim, .. }, Value::Number { value: s, dim: ds }) => {
            let out = dim.checked_mul(*ds).map_err(|_| exponent_overflow(whole))?;
            let s = *s;
            let arr = if matches!(l.get(), Value::Array { .. }) {
                l
            } else {
                r
            };
            scale(arr, out, |x| x * s, whole)
        }
        (ArithOp::Div, Value::Array { dim, .. }, Value::Number { value: s, dim: ds }) => {
            let out = dim.checked_div(*ds).map_err(|_| exponent_overflow(whole))?;
            if *s == 0.0 {
                return Err(div_zero(whole, rspan));
            }
            let s = *s;
            scale(l, out, |x| x / s, whole)
        }
        (ArithOp::Div, Value::Number { .. }, Value::Array { dims, .. }) => Err(RuntimeError::new(
            format!("cannot divide a number by a {}", noun(dims)),
            whole,
        )
        .with_help("for the matrix inverse write `A⁻¹`")),
        (ArithOp::Mul | ArithOp::Div, Value::Array { .. }, Value::Array { .. }) => {
            let verb = if op == ArithOp::Mul {
                "multiply"
            } else {
                "divide"
            };
            let sym = if op == ArithOp::Mul { "*" } else { "/" };
            Err(
                RuntimeError::new(format!("cannot {verb} two arrays with `{sym}`"), whole)
                    .with_note("v0.1 has no elementwise `*` or `/` between arrays")
                    .with_help("use `@` for the matrix product"),
            )
        }
        (op, l, r) => {
            let (verb, prep) = match op {
                ArithOp::Add => ("add", "and"),
                ArithOp::Sub => ("subtract", "from"),
                ArithOp::Mul => ("multiply", "and"),
                ArithOp::Div => ("divide", "and"),
            };
            // `subtract b from a`: name the right operand first.
            let (x, y) = if op == ArithOp::Sub { (r, l) } else { (l, r) };
            Err(RuntimeError::new(
                format!("cannot {verb} {} {prep} {}", x.kind(), y.kind()),
                whole,
            ))
        }
    }
}

fn noun(dims: &[usize]) -> &'static str {
    if dims.len() == 1 {
        "vector"
    } else {
        "matrix"
    }
}

/// `-v` for an array: every entry negated. `0 - x`, not `-x`, so an entry of
/// `0` stays `0` instead of becoming `-0`.
pub fn negate(dims: &[usize], data: &[f64], dim: Dimension) -> Value {
    Value::array(dims.to_vec(), data.iter().map(|&x| 0.0 - x).collect(), dim)
}

/// `l @ r`: matrix times matrix, matrix times vector, vector times matrix,
/// and the dot product of two vectors (a number).
pub fn matmul(l: &Value, r: &Value, whole: Span, lspan: Span, rspan: Span) -> R<Value> {
    let (
        Value::Array {
            dims: d1,
            data: a,
            dim: da,
        },
        Value::Array {
            dims: d2,
            data: b,
            dim: db,
        },
    ) = (l, r)
    else {
        return Err(RuntimeError::new(
            format!("`@` needs two arrays, found {} and {}", l.kind(), r.kind()),
            whole,
        ));
    };
    let dim = da.checked_mul(*db).map_err(|_| exponent_overflow(whole))?;
    // Row and column counts: a vector is a row on the left, a column on the right.
    let (m, k) = match d1[..] {
        [k] => (None, k),
        [m, k] => (Some(m), k),
        _ => unreachable!("arrays have rank 1 or 2"),
    };
    let (k2, n) = match d2[..] {
        [k2] => (k2, None),
        [k2, n] => (k2, Some(n)),
        _ => unreachable!("arrays have rank 1 or 2"),
    };
    if k != k2 {
        return Err(RuntimeError::new(
            format!(
                "cannot multiply {} and {}: the inner dimensions {k} and {k2} differ",
                shape_text(d1),
                shape_text(d2)
            ),
            whole,
        )
        .with_secondary(lspan, shape_text(d1))
        .with_secondary(rspan, shape_text(d2)));
    }
    // Vector @ vector is a dot product, a single reduction, in the fixed
    // order of `linalg::dot`; every other product accumulates each entry in
    // the naive order (see the spec's "Arrays: reproducible arithmetic").
    let data = all_finite(
        match (m, n) {
            (None, None) => vec![linalg::dot(a, b)],
            _ => linalg::matmul(a, m.unwrap_or(1), k, b, n.unwrap_or(1)),
        },
        whole,
    )?;
    Ok(match (m, n) {
        (None, None) => Value::number(data[0], dim),
        (Some(m), None) => Value::array(vec![m], data, dim),
        (None, Some(n)) => Value::array(vec![n], data, dim),
        (Some(m), Some(n)) => Value::array(vec![m, n], data, dim),
    })
}

/// `Aᵀ`: a number is its own transpose, and an `m` by `n` matrix becomes
/// `n` by `m`. A vector or anything else is an error (the checker's rule:
/// v0.1 has no row/column-vector distinction). Entries only move, so the
/// dimension and every value are unchanged. `Aᵀ` and `transpose(A)` in
/// `sky run` and SIR's `transpose` all call this, its one implementation.
pub fn transpose(v: &Value, whole: Span) -> R<Value> {
    match v {
        Value::Number { .. } => Ok(v.clone()),
        Value::Array { dims, data, dim } => match dims[..] {
            [m, n] => {
                let mut out = Vec::with_capacity(data.len());
                for j in 0..n {
                    out.extend((0..m).map(|i| data[i * n + j]));
                }
                Ok(Value::array(vec![n, m], out, *dim))
            }
            _ => Err(RuntimeError::new(
                format!(
                    "cannot transpose {}: transpose needs a matrix",
                    shape_text(dims)
                ),
                whole,
            )),
        },
        other => Err(RuntimeError::new(
            format!("cannot transpose {}", other.kind()),
            whole,
        )),
    }
}

/// `A⁻¹` for a square matrix, by Gauss-Jordan elimination with partial
/// pivoting. The result's dimension is the inverse of `A`'s. A singular
/// matrix is a runtime error, and the inverse of an exactly symmetric matrix
/// is exactly symmetric ([`linalg::inverse`]).
pub fn inverse(v: &Value, whole: Span, operand: Span) -> R<Value> {
    let Value::Array { dims, data, dim } = v else {
        return Err(RuntimeError::new(
            format!("the inverse needs a matrix, found {}", v.kind()),
            operand,
        ));
    };
    let &[n, m] = &dims[..] else {
        return Err(RuntimeError::new(
            format!(
                "the inverse needs a square matrix, found {}",
                shape_text(dims)
            ),
            operand,
        ));
    };
    if n != m {
        return Err(RuntimeError::new(
            format!(
                "the inverse needs a square matrix, found {}",
                shape_text(dims)
            ),
            operand,
        ));
    }
    let out_dim = dim.inverse().map_err(|_| exponent_overflow(whole))?;
    match linalg::inverse(data, n) {
        Ok(inv) => Ok(Value::array(dims.clone(), all_finite(inv, whole)?, out_dim)),
        Err(_) => Err(RuntimeError::new("the matrix is singular", whole)
            .with_label("this matrix has no inverse")
            .with_secondary(operand, format!("this is {}", shape_text(dims)))
            .with_note(
                "a matrix is singular when a pivot of the elimination is zero, or too small to tell from rounding error",
            )),
    }
}

/// `A⁻¹ @ b` for a positive-definite `A`, the rewrite the checker records:
/// factor `A = L·Lᵀ`, solve `L·y = b`, then `Lᵀ·x = y`. `A⁻¹` is never
/// formed. `b` is a vector or a matrix of right-hand sides; the result has
/// `b`'s shape and the dimension of `b` over `A`'s.
///
/// The checker takes a declared PositiveDefinite on trust for an input, so
/// the promise is verified here: a matrix that is not exactly symmetric, or
/// that has a non-positive pivot, is a runtime error naming the fact.
pub fn cholesky_solve(a: &Value, b: &Value, whole: Span, aspan: Span, bspan: Span) -> R<Value> {
    let (
        Value::Array {
            dims: da,
            data: ad,
            dim: adim,
        },
        Value::Array {
            dims: db,
            data: bd,
            dim: bdim,
        },
    ) = (a, b)
    else {
        return Err(RuntimeError::new(
            format!("cannot solve with {} and {}", a.kind(), b.kind()),
            whole,
        ));
    };
    let n = match da[..] {
        [n, m] if n == m => n,
        _ => {
            return Err(RuntimeError::new(
                format!(
                    "the inverse needs a square matrix, found {}",
                    shape_text(da)
                ),
                aspan,
            ))
        }
    };
    let k = match db[..] {
        [len] if len == n => 1,
        [rows, cols] if rows == n => cols,
        _ => {
            return Err(RuntimeError::new(
                format!(
                    "cannot multiply {} and {}: the inner dimensions differ",
                    shape_text(da),
                    shape_text(db)
                ),
                whole,
            )
            .with_secondary(aspan, shape_text(da))
            .with_secondary(bspan, shape_text(db)))
        }
    };
    let dim = bdim
        .checked_div(*adim)
        .map_err(|_| exponent_overflow(whole))?;
    let promise = "`A⁻¹ @ b` is solved by a Cholesky factorization because the checker knows `A` is PositiveDefinite";
    let l = linalg::cholesky(ad, n).map_err(|e| match e {
        LinalgError::NotSymmetric => RuntimeError::new("the matrix is not symmetric", aspan)
            .with_label("declared PositiveDefinite, which implies symmetric")
            .with_note(promise),
        LinalgError::NotPositiveDefinite { row } => {
            RuntimeError::new("the matrix is not positive definite", aspan)
                .with_label(format!(
                    "the factorization failed at row {row}: its pivot is not positive"
                ))
                .with_note(promise)
        }
        LinalgError::Singular => RuntimeError::new("the matrix is singular", aspan),
    })?;
    let x = all_finite(linalg::cholesky_solve(&l, n, bd, k), whole)?;
    Ok(Value::array(db.clone(), x, dim))
}

/// One item of an index, evaluated: a position, or a slice with its bounds.
#[derive(Clone, Copy, Debug)]
pub enum Ix {
    At {
        i: usize,
        span: Span,
    },
    Slice {
        start: Option<usize>,
        stop: Option<usize>,
        span: Span,
    },
}

/// The position `v` names: a dimensionless, non-negative integer.
pub fn to_index(v: &Value, span: Span) -> R<usize> {
    let Value::Number { value, dim } = v else {
        return Err(RuntimeError::new(
            format!("an index must be a number, found {}", v.kind()),
            span,
        ));
    };
    if !dim.is_dimensionless() {
        return Err(RuntimeError::new("an index must be dimensionless", span)
            .with_label(format!("this is {}", describe(dim))));
    }
    if value.fract() != 0.0 {
        return Err(RuntimeError::new(
            format!(
                "an index must be an integer, found {}",
                format_number(*value)
            ),
            span,
        ));
    }
    if *value < 0.0 {
        return Err(RuntimeError::new("an index cannot be negative", span)
            .with_note("v0.1 counts from 0 and has no negative indices"));
    }
    // Saturates for huge values, which are then out of bounds.
    Ok(*value as usize)
}

/// The error for position `i` on axis `axis` of length `len`.
fn out_of_bounds(
    i: usize,
    axis: usize,
    len: usize,
    span: Span,
    base_span: Span,
    dims: &[usize],
) -> RuntimeError {
    RuntimeError::new(
        format!("index {i} is out of bounds for axis {axis} with length {len}"),
        span,
    )
    .with_label(format!("valid indices are 0 to {}", len - 1))
    .with_secondary(base_span, format!("this is {}", shape_text(dims)))
}

/// `base[items]` for an array: a scalar index takes its axis away and a
/// slice keeps it; trailing axes without an item stay whole. All the
/// indices were evaluated first; bounds are checked here.
pub fn index(base: &Value, items: &[Ix], whole: Span, base_span: Span) -> R<Value> {
    let Value::Array { dims, data, dim } = base else {
        return Err(RuntimeError::new(
            format!("cannot index {}", base.kind()),
            base_span,
        ));
    };
    if items.len() > dims.len() {
        return Err(RuntimeError::new(
            format!(
                "too many indices: {} has {} axis{}, found {} indices",
                shape_text(dims),
                dims.len(),
                if dims.len() == 1 { "" } else { "es" },
                items.len()
            ),
            whole,
        ));
    }
    // `v[i]` and `A[i, j]`: one entry, found without building ranges.
    if items.len() == dims.len() && items.iter().all(|it| matches!(it, Ix::At { .. })) {
        let mut offset = 0;
        for (axis, (&len, item)) in dims.iter().zip(items).enumerate() {
            let Ix::At { i, span } = *item else {
                continue;
            };
            if i >= len {
                return Err(out_of_bounds(i, axis, len, span, base_span, dims));
            }
            offset = offset * len + i;
        }
        return Ok(Value::number(data[offset], *dim));
    }
    // The range of each axis, and whether it stays in the result.
    let mut ranges = Vec::with_capacity(dims.len());
    let mut kept = Vec::new();
    for (axis, &len) in dims.iter().enumerate() {
        match items.get(axis) {
            None => {
                ranges.push(0..len);
                kept.push(len);
            }
            Some(&Ix::At { i, span }) => {
                if i >= len {
                    return Err(out_of_bounds(i, axis, len, span, base_span, dims));
                }
                ranges.push(i..i + 1);
            }
            Some(&Ix::Slice { start, stop, span }) => {
                let (lo, hi) = (start.unwrap_or(0), stop.unwrap_or(len));
                if hi > len || lo > len {
                    return Err(RuntimeError::new(
                        format!(
                            "slice {}..{} is out of bounds for axis {axis} with length {len}",
                            lo, hi
                        ),
                        span,
                    )
                    .with_secondary(base_span, format!("this is {}", shape_text(dims))));
                }
                if lo >= hi {
                    return Err(RuntimeError::new("this slice is empty", span)
                        .with_label(format!("{lo} is not less than {hi}"))
                        .with_note("arrays are never empty in v0.1"));
                }
                ranges.push(lo..hi);
                kept.push(hi - lo);
            }
        }
    }
    let cols = *dims.last().unwrap_or(&1);
    let mut out = Vec::new();
    match ranges.as_slice() {
        [rows] => out.extend(rows.clone().map(|i| data[i])),
        [rows, columns] => {
            for i in rows.clone() {
                out.extend(columns.clone().map(|j| data[i * cols + j]));
            }
        }
        _ => unreachable!("arrays have rank 1 or 2"),
    }
    Ok(if kept.is_empty() {
        Value::number(out[0], *dim)
    } else {
        Value::array(kept, out, *dim)
    })
}

/// A size argument of `zeros` or `identity`: a dimensionless positive
/// integer. (The checker requires a constant; the interpreter accepts any
/// value that is one.)
pub fn size_arg(name: &str, v: &Value, span: Span) -> R<usize> {
    let bad = |what: String| RuntimeError::new(format!("`{name}` needs a size, {what}"), span);
    let Value::Number { value, dim } = v else {
        return Err(bad(format!("found {}", v.kind())));
    };
    if !dim.is_dimensionless() {
        return Err(bad(format!("found {}", describe(dim))));
    }
    if value.fract() != 0.0 || *value < 1.0 {
        return Err(bad(format!(
            "a positive integer, found {}",
            format_number(*value)
        )));
    }
    if *value > MAX_ARRAY_ELEMENTS as f64 {
        return Err(too_large(name, span));
    }
    Ok(*value as usize)
}

fn too_large(name: &str, span: Span) -> RuntimeError {
    RuntimeError::new(format!("the array from `{name}` is too large"), span)
        .with_label(format!("more than {MAX_ARRAY_ELEMENTS} elements"))
}

/// `zeros(n)` or `zeros(m, n)`: dimensionless.
pub fn zeros(sizes: &[usize], whole: Span) -> R<Value> {
    let total = sizes
        .iter()
        .try_fold(1u64, |acc, &n| acc.checked_mul(n as u64))
        .filter(|&t| t <= MAX_ARRAY_ELEMENTS)
        .ok_or_else(|| too_large("zeros", whole))?;
    Ok(Value::array(
        sizes.to_vec(),
        vec![0.0; total as usize],
        Dimension::DIMENSIONLESS,
    ))
}

/// `identity(n)`: the `n` by `n` identity, dimensionless.
pub fn identity(n: usize, whole: Span) -> R<Value> {
    let total = (n as u64)
        .checked_mul(n as u64)
        .filter(|&t| t <= MAX_ARRAY_ELEMENTS)
        .ok_or_else(|| too_large("identity", whole))?;
    let mut data = vec![0.0; total as usize];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    Ok(Value::array(vec![n, n], data, Dimension::DIMENSIONLESS))
}

/// `len(x)`: the length of the first axis, as a dimensionless number.
pub fn len(v: &Value, span: Span) -> R<Value> {
    match v {
        Value::Array { dims, .. } => Ok(Value::scalar(dims[0] as f64)),
        other => Err(RuntimeError::new(
            format!("`len` needs a vector or a matrix, found {}", other.kind()),
            span,
        )),
    }
}

/// Builds a literal from evaluated rows: every entry a number, all of one
/// dimension, every row the same length. `rows` is one row for a vector
/// literal (`vector` true), or one per row of a matrix literal. Each entry
/// comes with its span, for errors. Defensive: the checker has verified all
/// of this.
pub fn literal(rows: Vec<Vec<(Value, Span)>>, vector: bool, whole: Span) -> R<Value> {
    let mut dim: Option<Dimension> = None;
    let cols = rows.first().map_or(0, Vec::len);
    let mut data = Vec::with_capacity(rows.len() * cols);
    for row in &rows {
        if row.len() != cols {
            return Err(RuntimeError::new(
                format!(
                    "matrix rows must have the same length: expected {cols}, found {}",
                    row.len()
                ),
                whole,
            ));
        }
        for (v, span) in row {
            let Value::Number { value, dim: d } = v else {
                return Err(RuntimeError::new(
                    format!("an array element must be a number, found {}", v.kind()),
                    *span,
                ));
            };
            match dim {
                None => dim = Some(*d),
                Some(want) if want != *d => {
                    return Err(RuntimeError::new(
                        format!(
                            "array elements must have the same dimension: expected {}, found {}",
                            describe(&want),
                            describe(d)
                        ),
                        *span,
                    ))
                }
                Some(_) => {}
            }
            data.push(*value);
        }
    }
    let dim = dim.unwrap_or(Dimension::DIMENSIONLESS);
    let dims = if vector {
        vec![cols]
    } else {
        vec![rows.len(), cols]
    };
    if data.is_empty() {
        return Err(RuntimeError::new(
            "an array literal needs at least one element",
            whole,
        ));
    }
    Ok(Value::array(dims, all_finite(data, whole)?, dim))
}
