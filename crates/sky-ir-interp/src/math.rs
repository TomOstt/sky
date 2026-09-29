//! `OpKind::Math`: the mathematical functions, with the finite rules of
//! `sky_interp::arith`. A domain error or an overflow is a run-time error,
//! never NaN or infinity. The dimension rules are those documented on
//! [`sky_ir::MathFn`], checked dynamically here as everywhere in this
//! interpreter.

use sky_diag::Span;
use sky_interp::arith::{describe, finite};
use sky_interp::{RuntimeError, Value};
use sky_ir::MathFn;
use sky_units::{Dimension, Rational};

type R<T> = Result<T, RuntimeError>;

/// `f(args)`; `args` has `f.arity()` entries (the caller checks).
pub(crate) fn eval(f: MathFn, args: &[&Value], span: Span) -> R<Value> {
    let name = f.name();
    let mut nums = Vec::with_capacity(2);
    for v in args {
        match v {
            Value::Number { value, dim } => nums.push((*value, *dim)),
            other => {
                return Err(RuntimeError::new(
                    format!("`{name}` needs a number, found {}", other.kind()),
                    span,
                ))
            }
        }
    }
    let dimensionless = |dim: &Dimension| -> R<()> {
        if dim.is_dimensionless() {
            Ok(())
        } else {
            Err(
                RuntimeError::new(format!("`{name}` needs a dimensionless number"), span)
                    .with_label(format!("this is {}", describe(dim))),
            )
        }
    };
    let root = |dim: &Dimension, n: i64| -> R<Dimension> {
        (*dim)
            .pow_rat(Rational::new(1, n))
            .map_err(|_| RuntimeError::new("dimension exponent too large", span))
    };

    if f.arity() == 1 {
        let (x, dim) = nums[0];
        let (y, out) = match f {
            MathFn::Sqrt => {
                if x < 0.0 {
                    return Err(
                        domain(span, "the square root of a negative number is not real")
                            .with_note("Sky numbers are real; `sqrt` never produces NaN"),
                    );
                }
                (x.sqrt(), root(&dim, 2)?)
            }
            MathFn::Cbrt => (x.cbrt(), root(&dim, 3)?),
            MathFn::Abs => (x.abs(), dim),
            MathFn::Floor => (x.floor(), dim),
            MathFn::Ceil => (x.ceil(), dim),
            MathFn::Round => (x.round(), dim),
            _ => {
                dimensionless(&dim)?;
                let y = match f {
                    MathFn::Exp => x.exp(),
                    MathFn::Log | MathFn::Log10 | MathFn::Log2 => {
                        if x <= 0.0 {
                            return Err(domain(
                                span,
                                format!("the logarithm of {} is not defined", zero_or_neg(x)),
                            )
                            .with_note("Sky numbers never become -inf or NaN"));
                        }
                        match f {
                            MathFn::Log => x.ln(),
                            MathFn::Log10 => x.log10(),
                            _ => x.log2(),
                        }
                    }
                    MathFn::Sin => x.sin(),
                    MathFn::Cos => x.cos(),
                    MathFn::Tan => x.tan(),
                    MathFn::Atan => x.atan(),
                    MathFn::Asin | MathFn::Acos => {
                        if !(-1.0..=1.0).contains(&x) {
                            return Err(domain(
                                span,
                                format!("`{name}` needs an operand between -1 and 1"),
                            )
                            .with_note("Sky numbers never become NaN"));
                        }
                        if f == MathFn::Asin {
                            x.asin()
                        } else {
                            x.acos()
                        }
                    }
                    _ => unreachable!("arity 1 functions are all handled"),
                };
                (y, Dimension::DIMENSIONLESS)
            }
        };
        return Ok(Value::number(finite(y, span)?, out));
    }

    let ((a, da), (b, db)) = (nums[0], nums[1]);
    if da != db {
        return Err(RuntimeError::new(
            format!("`{name}` needs operands of the same dimension"),
            span,
        )
        .with_label(format!("{} and {}", describe(&da), describe(&db))));
    }
    let (y, out) = match f {
        MathFn::Atan2 => (a.atan2(b), Dimension::DIMENSIONLESS),
        MathFn::Hypot => (a.hypot(b), da),
        MathFn::Min => (if b < a { b } else { a }, da),
        MathFn::Max => (if b > a { b } else { a }, da),
        _ => unreachable!("arity 2 functions are all handled"),
    };
    Ok(Value::number(finite(y, span)?, out))
}

fn domain(span: Span, message: impl Into<String>) -> RuntimeError {
    RuntimeError::new(message, span).with_label("outside the function's domain")
}

fn zero_or_neg(x: f64) -> &'static str {
    if x == 0.0 {
        "zero"
    } else {
        "a negative number"
    }
}
