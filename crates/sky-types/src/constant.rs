//! Constants the checker and the interpreter must read the same way.
//!
//! The checker proves matrix properties of a literal from its entries (see
//! [`crate::array`]), and the proof is only honest if it reads every entry
//! as the *same* `f64` the interpreter will hold. So the number and unit
//! conversions live here once, next to [`crate::const_exponent`]:
//! [`parse_number`] reads a numeric literal's text, [`apply_unit`] converts a
//! value to SI factor by factor, and [`entry_value`] reads a whole constant
//! entry. Nothing here evaluates user code: it is arithmetic on the numbers
//! the source spells out.

use sky_ast::ast::span_text;
use sky_ast::{Expr, ExprKind, Literal, UnaryOp, UnitExpr};
use sky_diag::Span;
use sky_units::{lookup, Dimension, Rational};

use crate::check::strip_parens;

/// Most elements an array may have (16 Mi, 128 MiB of `f64`s). A constant
/// size beyond it (`zeros(100000, 100000)`) is a check error, and the
/// interpreter refuses the same sizes, so a program cannot ask for more
/// memory than that in one array.
pub const MAX_ARRAY_ELEMENTS: u64 = 1 << 24;

/// Why a numeric literal has no `f64` value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NumberError {
    /// Too large: the value would be infinite.
    TooLarge,
    /// Not a number (the lexer normally rules this out).
    Invalid,
}

/// Reads a numeric literal's text (`1_000`, `2.5e-3`) as a finite `f64`.
pub fn parse_number(text: &str) -> Result<f64, NumberError> {
    match text.replace('_', "").parse::<f64>() {
        Ok(x) if x.is_finite() => Ok(x),
        Ok(_) => Err(NumberError::TooLarge),
        Err(_) => Err(NumberError::Invalid),
    }
}

/// Why a unit could not be applied.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnitError {
    /// A name that is not in the unit table.
    Unknown(Span),
    /// A dimension exponent that does not fit.
    Exponent(Span),
}

/// Applies unit `u` to the value `v`, factor by factor: `36 km/h` is
/// `36 * 1000 / 3600`, which is exactly 10, where multiplying by a
/// precomputed `1000/3600` would not be. A single unit is `v * scale`.
/// Returns the SI value (not checked for overflow) and the dimension.
pub fn apply_unit(src: &str, u: &UnitExpr, v: f64) -> Result<(f64, Dimension), UnitError> {
    let overflow = || UnitError::Exponent(u.span());
    match u {
        UnitExpr::Name(id) => match lookup(span_text(src, id.span).unwrap_or("")) {
            Some(unit) => Ok((v * unit.scale, unit.dim)),
            None => Err(UnitError::Unknown(id.span)),
        },
        UnitExpr::Mul { lhs, rhs, .. } => {
            let (v, d1) = apply_unit(src, lhs, v)?;
            let (v, d2) = apply_unit(src, rhs, v)?;
            Ok((v, d1.checked_mul(d2).map_err(|_| overflow())?))
        }
        UnitExpr::Div { lhs, rhs, .. } => {
            let (v, d1) = apply_unit(src, lhs, v)?;
            let (scale, d2) = apply_unit(src, rhs, 1.0)?;
            Ok((v / scale, d1.checked_div(d2).map_err(|_| overflow())?))
        }
        UnitExpr::Pow { base, num, den, .. } => {
            let (scale, d) = apply_unit(src, base, 1.0)?;
            let exp = i64::try_from(*den)
                .ok()
                .and_then(|den| Rational::checked_new(*num, den).ok())
                .ok_or_else(overflow)?;
            let dim = d.pow_rat(exp).map_err(|_| overflow())?;
            let factor = if *den == 1 {
                // Integer exponent: multiply or divide, `s^-2` is `/ s^2`.
                let k = i32::try_from(num.unsigned_abs()).map_err(|_| overflow())?;
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

/// The SI value of a constant array entry, exactly as the interpreter
/// computes it: a numeric literal or a number with a unit, optionally
/// negated or parenthesized (`2`, `-1.5`, `3 N/m`, `-(2 m)`). `None` for
/// anything else, or a value that is not finite.
pub fn entry_value(src: &str, e: &Expr) -> Option<f64> {
    let e = strip_parens(e);
    let x = match &e.kind {
        ExprKind::Literal(Literal::Int(s) | Literal::Float(s)) => {
            parse_number(span_text(src, *s)?).ok()?
        }
        ExprKind::UnitLiteral {
            value: Literal::Int(s) | Literal::Float(s),
            unit,
        } => {
            let v = parse_number(span_text(src, *s)?).ok()?;
            apply_unit(src, unit, v).ok()?.0
        }
        ExprKind::Unary {
            op: UnaryOp::Neg,
            operand,
            ..
        } => -entry_value(src, operand)?,
        ExprKind::Unary {
            op: UnaryOp::Pos,
            operand,
            ..
        } => entry_value(src, operand)?,
        _ => return None,
    };
    x.is_finite().then_some(x)
}
