//! Reading literals: the one place a number literal's text and a unit
//! literal's factors become SI `f64` values.
//!
//! Shared by the tree-walking evaluator and by lowering to SIR
//! (`sky-lower`), so a `36 km/h` constant in SIR is bit-for-bit the number
//! the interpreter computes. Do not duplicate this arithmetic elsewhere.

use sky_ast::ast::span_text;
use sky_ast::UnitExpr;
use sky_diag::Span;
use sky_units::{lookup, Dimension, Rational};

use crate::arith::{overflow, R};
use crate::RuntimeError;

/// Reads a numeric literal's text (`1_000`, `2.5e-3`) as a finite `f64`.
pub fn number_literal(src: &str, span: Span) -> R<f64> {
    let text = span_text(src, span).unwrap_or("").replace('_', "");
    match text.parse::<f64>() {
        Ok(x) if x.is_finite() => Ok(x),
        Ok(_) => Err(RuntimeError::new("number too large", span)
            .with_label("does not fit in a 64-bit float")
            .with_note("the largest Sky number is about 1.8e308")),
        Err(_) => Err(RuntimeError::new("invalid number", span)),
    }
}

/// The SI value and dimension of the unit literal whose number is at
/// `value` and whose unit is `unit` (`whole` is the literal's span, for the
/// overflow error). The number is read with [`number_literal`] and the unit
/// applied factor by factor.
pub fn unit_literal_si(
    src: &str,
    value: Span,
    unit: &UnitExpr,
    whole: Span,
) -> R<(f64, Dimension)> {
    let x = number_literal(src, value)?;
    let (si, dim) = apply_unit(src, unit, x)?;
    if !si.is_finite() {
        return Err(overflow(whole));
    }
    Ok((si, dim))
}

/// Applies unit `u` to the value `v`, factor by factor: `36 km/h` is
/// `36 * 1000 / 3600`, which is exactly 10, where multiplying by a
/// precomputed `1000/3600` would not be. A single unit is `v * scale`,
/// as the spec says. Returns the SI value and the dimension.
fn apply_unit(src: &str, u: &UnitExpr, v: f64) -> R<(f64, Dimension)> {
    let overflow_dim = || RuntimeError::new("dimension exponent too large", u.span());
    match u {
        UnitExpr::Name(id) => {
            let name = span_text(src, id.span).unwrap_or("");
            match lookup(name) {
                Some(unit) => Ok((v * unit.scale, unit.dim)),
                None => Err(RuntimeError::new(format!("unknown unit `{name}`"), id.span)),
            }
        }
        UnitExpr::Mul { lhs, rhs, .. } => {
            let (v, d1) = apply_unit(src, lhs, v)?;
            let (v, d2) = apply_unit(src, rhs, v)?;
            Ok((v, d1.checked_mul(d2).map_err(|_| overflow_dim())?))
        }
        UnitExpr::Div { lhs, rhs, .. } => {
            let (v, d1) = apply_unit(src, lhs, v)?;
            let (scale, d2) = apply_unit(src, rhs, 1.0)?;
            Ok((v / scale, d1.checked_div(d2).map_err(|_| overflow_dim())?))
        }
        UnitExpr::Pow { base, num, den, .. } => {
            let (scale, d) = apply_unit(src, base, 1.0)?;
            let exp = i64::try_from(*den)
                .ok()
                .and_then(|den| Rational::checked_new(*num, den).ok())
                .ok_or_else(overflow_dim)?;
            let dim = d.pow_rat(exp).map_err(|_| overflow_dim())?;
            let factor = if *den == 1 {
                // Integer exponent: multiply or divide, `s^-2` is `/ s^2`.
                let k = i32::try_from(num.unsigned_abs()).map_err(|_| overflow_dim())?;
                let p = scale.powi(k);
                if *num < 0 {
                    return Ok((v / p, dim));
                }
                p
            } else {
                scale.powf(*num as f64 / *den as f64)
            };
            Ok((v * factor, dim))
        }
    }
}
