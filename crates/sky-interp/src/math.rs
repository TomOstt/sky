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
//! usual overflow error. The same holds for the added functions: `log1p(x)`
//! for `x <= -1`, `acosh(x)` for `x < 1` and `atanh(x)` for `|x| >= 1` are
//! domain errors (the last two limits are infinite), and `expm1`, `sinh` and
//! `cosh` overflow for large arguments. `erf` (see `erf`), `tanh`, `asinh`
//! and `sign` cannot fail. `sign(0)` is 0, and `sign` accepts any
//! dimension and returns a plain number.
//! Underflow to zero (`exp(-1000)`) is fine: zero is a correct rounding, not a
//! lost value.
//!
//! `round` is half away from zero (Rust's `f64::round`, C's `round`), not
//! banker's rounding. A result of `-0` is normalised to `0`, so `round(-0.4)`
//! prints `0`. The dimension rules are re-checked here because the checker
//! lets calls through it cannot type (see the crate docs).

use sky_diag::Span;
use sky_units::{MathDimError, MathFn};

use crate::arith::{arity_mismatch, describe, finite, R};
use crate::erf::erf;
use crate::{RuntimeError, Value};

/// `f(args)` for already evaluated arguments. `whole` is the call's span
/// and `spans` the arguments' (same length as `args`; an IR op that
/// recorded none passes its own span for each).
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
    Ok(Value::number(
        apply_f64(
            f,
            nums[0],
            nums.get(1).copied().unwrap_or(0.0),
            whole,
            span_of(0),
        )?,
        dim,
    ))
}

/// `f(x)`, or `f(x, y)` for the two-argument functions (`y` is ignored by
/// the others), on finite `f64`s, after the dimensions were checked: the
/// numeric core of [`apply`], with the same domain errors, finiteness check
/// and `-0` normalisation. `whole` is the call's span and `first` the first
/// argument's. Native code calls it for the functions it does not inline, so
/// it computes the same bits as `apply`.
pub fn apply_f64(f: MathFn, x: f64, y: f64, whole: Span, first: Span) -> R<f64> {
    let name = f.name();
    let domain = |what: &str| {
        RuntimeError::new(format!("`{name}` of {what} has no real value"), whole)
            .with_label(format!("`{name}` is undefined here"))
            .with_note("Sky numbers never become NaN or inf")
    };
    let r = match f {
        MathFn::Sqrt if x < 0.0 => {
            return Err(domain("a negative number").with_secondary(first, "this is negative"))
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
            return Err(domain(what).with_secondary(first, format!("this is {what}")));
        }
        MathFn::Log => x.ln(),
        MathFn::Log10 => x.log10(),
        MathFn::Log2 => x.log2(),
        MathFn::Sin => x.sin(),
        MathFn::Cos => x.cos(),
        MathFn::Tan => x.tan(),
        MathFn::Asin | MathFn::Acos if x.abs() > 1.0 => {
            return Err(
                domain("a number outside [-1, 1]").with_secondary(first, "this is outside [-1, 1]")
            )
        }
        MathFn::Asin => x.asin(),
        MathFn::Acos => x.acos(),
        MathFn::Atan => x.atan(),
        MathFn::Atan2 if x == 0.0 && y == 0.0 => {
            return Err(domain("the zero vector").with_note("the angle of (0, 0) is undefined"))
        }
        MathFn::Atan2 => x.atan2(y),
        MathFn::Hypot => x.hypot(y),
        MathFn::Min => x.min(y),
        MathFn::Max => x.max(y),
        MathFn::Floor => x.floor(),
        MathFn::Ceil => x.ceil(),
        MathFn::Round => x.round(),
        MathFn::Expm1 => x.exp_m1(),
        MathFn::Log1p if x <= -1.0 => {
            let what = if x == -1.0 { "-1" } else { "a number below -1" };
            return Err(domain(what).with_secondary(first, format!("this is {what}")));
        }
        MathFn::Log1p => x.ln_1p(),
        MathFn::Sinh => x.sinh(),
        MathFn::Cosh => x.cosh(),
        MathFn::Tanh => x.tanh(),
        MathFn::Asinh => asinh(x),
        MathFn::Acosh if x < 1.0 => {
            return Err(domain("a number below 1").with_secondary(first, "this is below 1"))
        }
        MathFn::Acosh => acosh(x),
        MathFn::Atanh if x.abs() >= 1.0 => {
            return Err(domain("a number outside (-1, 1)")
                .with_secondary(first, "this is outside (-1, 1)")
                .with_note("atanh(x) grows without bound as x approaches 1 or -1"))
        }
        MathFn::Atanh => x.atanh(),
        MathFn::Erf => erf(x),
        MathFn::Sign if x == 0.0 => 0.0,
        MathFn::Sign => x.signum(),
    };
    // `+ 0.0` turns `-0.0` into `0.0`.
    Ok(finite(r, whole)? + 0.0)
}

/// Above this the `+ 1` under the square root of `asinh`/`acosh` is below
/// half an ulp and squaring `x` would overflow near `f64::MAX`, so the
/// asymptotic `ln(2x)` is used (the same switch as fdlibm's).
const LARGE: f64 = 268_435_456.0; // 2^28

/// `asinh(x)` that stays finite for every finite `x` (the standard
/// library's forms `x + x/(...)` overflow for `|x|` near `f64::MAX`).
fn asinh(x: f64) -> f64 {
    let a = x.abs();
    let r = if a > LARGE {
        a.ln() + std::f64::consts::LN_2
    } else {
        a.asinh()
    };
    r.copysign(x)
}

/// `acosh(x)` for `x >= 1`, finite for every finite `x`.
fn acosh(x: f64) -> f64 {
    if x > LARGE {
        x.ln() + std::f64::consts::LN_2
    } else {
        x.acosh()
    }
}

#[cfg(test)]
mod can_fail_agrees {
    use super::*;
    use sky_units::Dimension;

    /// `MathFn::can_fail` (which makes a SIR op fallible) must match this
    /// evaluator: a function marked infallible never fails on any input,
    /// and one marked fallible fails on at least one of these edge cases.
    #[test]
    fn can_fail_matches_the_evaluator() {
        let edges = [
            0.0, -0.0, 1.0, -1.0, 0.5, 2.0, -2.0, 1e308, 1.5e308, -1e308, 1e-308, 710.0,
        ];
        for f in MathFn::ALL {
            let mut failed = false;
            for &a in &edges {
                for &b in &edges {
                    let args: Vec<Value> = [a, b][..f.arity()]
                        .iter()
                        .map(|&x| Value::number(x, Dimension::DIMENSIONLESS))
                        .collect();
                    let spans = vec![Span::point(0); args.len()];
                    failed |= apply(f, &args, Span::point(0), &spans).is_err();
                }
            }
            assert_eq!(
                failed,
                f.can_fail(),
                "{f:?}: can_fail says {}",
                f.can_fail()
            );
        }
    }
}
