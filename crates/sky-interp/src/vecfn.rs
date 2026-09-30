//! The vector builtins `norm(x)` and `dot(a, b)`: the runtime twin of
//! `sky_types::vecfn` (which gives the dimension rules).
//!
//! * `norm(x)`: the Euclidean length of a vector (the Frobenius norm of a
//!   matrix, the absolute value of a scalar), in the dimension of `x`. Entries
//!   beyond `1e±100` are scaled by the largest, so a vector of `1e200`s has a
//!   norm and the squares of tiny entries do not vanish.
//! * `dot(a, b)`: the inner product of two vectors of one length, in the
//!   product of their dimensions.
//!
//! Like everything the interpreter computes, a result is finite or a
//! [`RuntimeError`].

use sky_ast::Expr;
use sky_diag::Span;

use crate::arith::{arity_mismatch, describe, finite, overflow, R};
use crate::eval::Interp;
use crate::{RuntimeError, Value};

impl Interp<'_> {
    /// `norm(x)` or `dot(a, b)`, with the call and its arguments.
    pub(crate) fn vector_builtin(&mut self, name: &str, call: &Expr, args: &[Expr]) -> R<Value> {
        let want = if name == "norm" { 1 } else { 2 };
        if args.len() != want {
            return Err(arity_mismatch(name, want, args.len(), call.span));
        }
        let vals = args
            .iter()
            .map(|a| self.eval(a))
            .collect::<R<Vec<Value>>>()?;
        match vals.as_slice() {
            [x] => norm(x, call.span, args[0].span),
            [a, b] => dot(a, b, call.span, args[0].span, args[1].span),
            _ => Err(arity_mismatch(name, want, args.len(), call.span)),
        }
    }
}

/// The Euclidean norm of a number, vector or matrix.
pub fn norm(v: &Value, whole: Span, arg: Span) -> R<Value> {
    let (data, dim): (&[f64], _) = match v {
        Value::Number { value, dim } => (std::slice::from_ref(value), *dim),
        Value::Array { data, dim, .. } => (data, *dim),
        other => {
            return Err(RuntimeError::new(
                format!("`norm` needs a vector, found {}", other.kind()),
                arg,
            ));
        }
    };
    let scale = data.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    if scale == 0.0 {
        return Ok(Value::number(0.0, dim));
    }
    // Squares of ordinary numbers cannot overflow or vanish, and give the
    // correctly rounded `5` for `[3, 4]`; only extreme entries are scaled.
    if (1e-100..1e100).contains(&scale) {
        let sum: f64 = data.iter().map(|x| x * x).sum();
        return Ok(Value::number(sum.sqrt(), dim));
    }
    let sum: f64 = data.iter().map(|x| (x / scale) * (x / scale)).sum();
    Ok(Value::number(finite(scale * sum.sqrt(), whole)?, dim))
}

/// The inner product of two vectors of one length.
pub fn dot(a: &Value, b: &Value, whole: Span, aspan: Span, bspan: Span) -> R<Value> {
    let (
        Value::Array {
            dims: da,
            data: xs,
            dim: dima,
        },
        Value::Array {
            dims: db,
            data: ys,
            dim: dimb,
        },
    ) = (a, b)
    else {
        let bad = if matches!(a, Value::Array { .. }) {
            b
        } else {
            a
        };
        return Err(RuntimeError::new(
            format!("`dot` needs two vectors, found {}", bad.kind()),
            whole,
        ));
    };
    if da.len() != 1 || db.len() != 1 {
        return Err(
            RuntimeError::new("`dot` needs two vectors", whole).with_help("for matrices, use `@`")
        );
    }
    if da != db {
        return Err(RuntimeError::new(
            format!(
                "`dot` needs vectors of one length, found {} and {}",
                crate::array::shape_text(da),
                crate::array::shape_text(db)
            ),
            whole,
        )
        .with_secondary(aspan, crate::array::shape_text(da))
        .with_secondary(bspan, crate::array::shape_text(db)));
    }
    let dim = dima.checked_mul(*dimb).map_err(|_| {
        RuntimeError::new(
            format!(
                "the dimension of `dot` ({} times {}) overflows",
                describe(dima),
                describe(dimb)
            ),
            whole,
        )
    })?;
    let sum: f64 = xs.iter().zip(ys).map(|(x, y)| x * y).sum();
    if !sum.is_finite() {
        return Err(overflow(whole));
    }
    Ok(Value::number(sum, dim))
}
