//! Evaluation of the built-in math functions (`sqrt`, `exp`, `sin`, ...).
//!
//! Which functions exist, their arity and their dimension rules are
//! `sky_units::MathFn`, shared with the checker; this file is the value
//! side, public so the SIR reference interpreter can call the same code and
//! agree on every number and message (see `arith`).
//!
//! # Rules
//!
//! Like the operators, a math function never yields `NaN` or `inf`. Where
//! the mathematical function has no real value the call is a runtime error
//! naming the domain: `sqrt(-1)`, `log(0)`, `log(-2)`, `asin(2)`, and
//! `atan2(0, 0)` (which C returns 0 for, but the angle of the zero vector
//! is undefined, and a silent 0 hides a bug). A finite input whose result
//! does not fit an `f64` (`exp(1000)`, `hypot` of two huge numbers) is the
//! usual overflow error. Underflow to zero (`exp(-1000)`) is fine: zero is
//! a correct rounding, not a lost value.
//!
//! `round` is half away from zero (Rust's `f64::round`, C's `round`), not
//! banker's rounding. A result of `-0` is normalised to `0`, so `round(-0.4)`
//! prints `0`. The dimension rules are re-checked here because the checker
//! lets calls through it cannot type (see the crate docs).

use sky_diag::Span;
use sky_units::{MathDimError, MathFn};

use crate::arith::{arity_mismatch, describe, finite, R};
use crate::{RuntimeError, Value};

/// `f(args)` for already evaluated arguments. `whole` is the call's span
/// and `spans` the arguments' (same length as `args`; an IR op with one
/// span passes it for all).
pub fn apply(f: MathFn, args: &[Value], whole: Span, spans: &[Span]) -> R<Value> {
    if args.len() != f.arity() {
        return Err(arity_mismatch(f.name(), f.arity(), args.len(), whole));
    }
    let name = f.name();
    let mut nums = Vec::with_capacity(args.len());
    let mut dims = Vec::with_capacity(args.len());
    for (i, v) in args.iter().enumerate() {
        let Value::Number { value, dim } = v else {
            return Err(RuntimeError::new(
                format!("`{name}` needs a number, found {}", v.kind()),
                spans.get(i).copied().unwrap_or(whole),
            ));
        };
        nums.push(*value);
        dims.push(*dim);
    }
    let span_of = |i: usize| spans.get(i).copied().unwrap_or(whole);
    let dim = f.result_dim(&dims).map_err(|e| match e {
        MathDimError::NotDimensionless => RuntimeError::new(
            format!(
                "the argument of `{name}` must be dimensionless, found {}",
                describe(&dims[0])
            ),
            span_of(0),
        )
        .with_label(format!("this is {}", describe(&dims[0]))),
        MathDimError::Mismatch => RuntimeError::new(
            format!(
                "the arguments of `{name}` must have the same dimension, found {} and {}",
                describe(&dims[0]),
                describe(&dims[1])
            ),
            whole,
        ),
        MathDimError::Overflow => RuntimeError::new("dimension exponent too large", whole),
    })?;
    let x = nums[0];
    let domain = |what: &str| {
        RuntimeError::new(format!("`{name}` of {what} has no real value"), whole)
            .with_label(format!("`{name}` is undefined here"))
            .with_note("Sky numbers never become NaN or inf")
    };
    let r = match f {
        MathFn::Sqrt if x < 0.0 => {
            return Err(domain("a negative number").with_secondary(span_of(0), "this is negative"))
        }
        MathFn::Sqrt => x.sqrt(),
        MathFn::Cbrt => x.cbrt(),
        MathFn::Abs => x.abs(),
        MathFn::Exp => x.exp(),
        MathFn::Log | MathFn::Log10 | MathFn::Log2 if x <= 0.0 => {
            let what = if x == 0.0 {
                "zero"
            } else {
                "a negative number"
            };
            return Err(domain(what).with_secondary(span_of(0), format!("this is {what}")));
        }
        MathFn::Log => x.ln(),
        MathFn::Log10 => x.log10(),
        MathFn::Log2 => x.log2(),
        MathFn::Sin => x.sin(),
        MathFn::Cos => x.cos(),
        MathFn::Tan => x.tan(),
        MathFn::Asin | MathFn::Acos if x.abs() > 1.0 => {
            return Err(domain("a number outside [-1, 1]")
                .with_secondary(span_of(0), "this is outside [-1, 1]"))
        }
        MathFn::Asin => x.asin(),
        MathFn::Acos => x.acos(),
        MathFn::Atan => x.atan(),
        MathFn::Atan2 if x == 0.0 && nums[1] == 0.0 => {
            return Err(domain("the zero vector").with_note("the angle of (0, 0) is undefined"))
        }
        MathFn::Atan2 => x.atan2(nums[1]),
        MathFn::Hypot => x.hypot(nums[1]),
        MathFn::Min => x.min(nums[1]),
        MathFn::Max => x.max(nums[1]),
        MathFn::Floor => x.floor(),
        MathFn::Ceil => x.ceil(),
        MathFn::Round => x.round(),
    };
    // `+ 0.0` turns `-0.0` into `0.0`.
    Ok(Value::number(finite(r, whole)? + 0.0, dim))
}
