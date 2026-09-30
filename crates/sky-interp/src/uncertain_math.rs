//! The math functions on uncertain numbers (`sqrt(x)` with `x = 4 ± 0.1`).
//!
//! The nominal result is computed by the exact rules ([`crate::math::apply`]
//! on the nominal arguments), so the dimension rules, the domain errors and
//! their messages are the exact functions'. What this file adds is the
//! derivative `∂f/∂xᵢ` at the nominal point of each uncertain argument, the
//! weights of [`Lin::combine`] (the chain rule of first-order propagation,
//! see [`crate::uncertain`]).
//!
//! The derivatives are the same ones the automatic differentiation of SIR
//! uses (the tangent rules of `sky-ad`'s `math_rules`, spec "Derivatives"),
//! with the same conventions where a derivative is not a single number:
//! `floor`, `ceil`, `round` and `sign` are constant between their jumps, so
//! their slope is zero and the result has no uncertainty; `abs` takes the
//! right-hand slope at 0 (`+1`), and `min` and `max` take the slope of the
//! operand they return (the first when the operands are equal). `sky-ad`
//! builds SIR operations rather than numbers, so its rules cannot be called
//! here; the tests check every formula below against a numerical central
//! difference instead.
//!
//! Where the slope is infinite at the nominal point (`sqrt` at zero, `asin`
//! at one) and the argument is uncertain, the call is a runtime error, as the
//! derivative of the same expression is; an exact argument never needs a
//! slope, so `sqrt(0)` is still `0`.

use sky_diag::Span;
use sky_units::MathFn;

use crate::arith::R;
use crate::uncertain::{infinite_slope, Lin};
use crate::{format_number, Value};

/// `f(args)` where at least one argument is uncertain. `whole` is the call's
/// span and `spans` the arguments'.
pub fn apply(f: MathFn, args: &[Value], whole: Span, spans: &[Span]) -> R<Value> {
    let nominal: Vec<Value> = args.iter().map(nominal_of).collect();
    let result = crate::math::apply(f, &nominal, whole, spans)?;
    // `min` and `max` return one of their arguments, uncertainty and all.
    if matches!(f, MathFn::Min | MathFn::Max) {
        if let [a, b] = args {
            let (x, y) = (value_of(a), value_of(b));
            let second = match f {
                MathFn::Min => y < x,
                _ => y > x,
            };
            return Ok(if second { b.clone() } else { a.clone() });
        }
    }
    let Value::Number { value, dim } = result else {
        return Ok(result);
    };
    let xs: Vec<f64> = args.iter().map(value_of).collect();
    let mut parts: Vec<(&[crate::uncertain::Term], f64)> = Vec::new();
    for (i, a) in args.iter().enumerate() {
        match a {
            // An exact argument (or one whose sources have cancelled) needs
            // no slope, so `sqrt(0 ± 0)` is `0`, not an error.
            Value::Uncertain(u) if !u.lin.is_exact() => {
                parts.push((u.lin.terms(), slope(f, &xs, value, i, whole)?));
            }
            _ => {}
        }
    }
    let lin = Lin::combine(&parts, whole)?;
    Ok(Value::uncertain(value, dim, lin))
}

fn nominal_of(v: &Value) -> Value {
    match v {
        Value::Uncertain(u) => Value::number(u.value, u.dim),
        other => other.clone(),
    }
}

fn value_of(v: &Value) -> f64 {
    match v {
        Value::Number { value, .. } => *value,
        Value::Uncertain(u) => u.value,
        _ => 0.0,
    }
}

/// `∂f/∂xᵢ` at the nominal arguments `x`, where `r = f(x)`.
pub(crate) fn slope(f: MathFn, x: &[f64], r: f64, i: usize, whole: Span) -> R<f64> {
    let a = x[0];
    let infinite = || {
        let at = format_number(x[i]);
        infinite_slope(&format!("`{}`", f.name()), &at, whole)
    };
    let v = match f {
        MathFn::Sqrt => 0.5 / r,
        MathFn::Cbrt => 1.0 / (3.0 * r * r),
        MathFn::Abs => {
            if a < 0.0 {
                -1.0
            } else {
                1.0
            }
        }
        MathFn::Exp => r,
        MathFn::Expm1 => r + 1.0,
        MathFn::Log => 1.0 / a,
        MathFn::Log10 => 1.0 / (a * std::f64::consts::LN_10),
        MathFn::Log2 => 1.0 / (a * std::f64::consts::LN_2),
        MathFn::Log1p => 1.0 / (1.0 + a),
        MathFn::Sin => a.cos(),
        MathFn::Cos => -a.sin(),
        MathFn::Tan => 1.0 + r * r,
        MathFn::Asin => 1.0 / ((1.0 - a) * (1.0 + a)).sqrt(),
        MathFn::Acos => -1.0 / ((1.0 - a) * (1.0 + a)).sqrt(),
        MathFn::Atan => 1.0 / (1.0 + a * a),
        MathFn::Sinh => a.cosh(),
        MathFn::Cosh => a.sinh(),
        MathFn::Tanh => 1.0 - r * r,
        MathFn::Asinh => 1.0 / a.hypot(1.0),
        MathFn::Acosh => 1.0 / ((a - 1.0) * (a + 1.0)).sqrt(),
        MathFn::Atanh => 1.0 / ((1.0 - a) * (1.0 + a)),
        MathFn::Erf => std::f64::consts::FRAC_2_SQRT_PI * (-a * a).exp(),
        // atan2(y, x): ∂y = x/(x²+y²), ∂x = -y/(x²+y²), scaled by the
        // hypotenuse so the squares do not overflow.
        MathFn::Atan2 => {
            let h = x[0].hypot(x[1]);
            if i == 0 {
                (x[1] / h) / h
            } else {
                -(x[0] / h) / h
            }
        }
        MathFn::Hypot => x[i] / r,
        // Constant between their jumps.
        MathFn::Floor | MathFn::Ceil | MathFn::Round | MathFn::Sign => 0.0,
        // The operand `apply` returns has slope 1, the other 0.
        MathFn::Min | MathFn::Max => {
            let second = if f == MathFn::Min {
                x[1] < x[0]
            } else {
                x[1] > x[0]
            };
            if (i == 1) == second {
                1.0
            } else {
                0.0
            }
        }
    };
    if v.is_finite() {
        Ok(v)
    } else {
        Err(infinite())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Central-difference check of every slope, on points inside every
    /// function's domain and away from its kinks.
    #[test]
    fn slopes_match_a_numerical_derivative() {
        let points: &[f64] = &[0.3, 0.7, 1.5, 2.5, 4.0];
        for f in MathFn::ALL {
            if matches!(
                f,
                MathFn::Floor
                    | MathFn::Ceil
                    | MathFn::Round
                    | MathFn::Sign
                    | MathFn::Min
                    | MathFn::Max
            ) {
                continue;
            }
            for &p in points {
                let xs: Vec<f64> = match f.arity() {
                    1 => vec![p],
                    _ => vec![p, 1.7],
                };
                // Outside the domain of the inverse functions: skip.
                let dom = |v: &[f64]| exact_eval(f, v);
                let Some(r) = dom(&xs) else { continue };
                for i in 0..xs.len() {
                    let Ok(s) = slope(f, &xs, r, i, Span::point(0)) else {
                        continue;
                    };
                    let h = 1e-6 * xs[i].abs().max(1.0);
                    let (mut lo, mut hi) = (xs.clone(), xs.clone());
                    lo[i] -= h;
                    hi[i] += h;
                    let (Some(a), Some(b)) = (dom(&lo), dom(&hi)) else {
                        continue;
                    };
                    let numeric = (b - a) / (2.0 * h);
                    assert!(
                        (s - numeric).abs() <= 1e-6 * s.abs().max(1.0),
                        "{f:?} at {xs:?} argument {i}: slope {s}, numeric {numeric}"
                    );
                }
            }
        }
    }

    fn exact_eval(f: MathFn, xs: &[f64]) -> Option<f64> {
        let args: Vec<Value> = xs.iter().map(|&x| Value::scalar(x)).collect();
        let spans = vec![Span::point(0); xs.len()];
        match crate::math::apply(f, &args, Span::point(0), &spans) {
            Ok(Value::Number { value, .. }) => Some(value),
            _ => None,
        }
    }
}
