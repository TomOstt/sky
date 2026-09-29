//! Reading literals: the one place a number literal's text and a unit
//! literal's factors become SI `f64` values.
//!
//! Shared by the tree-walking evaluator and by lowering to SIR
//! (`sky-lower`), so a `36 km/h` constant in SIR is bit-for-bit the number
//! the interpreter computes. The arithmetic itself is `sky_types::constant`
//! (the checker needs it too, to prove properties of constant matrices, and
//! can't depend on this crate); this module only turns its errors into
//! runtime diagnostics. Do not duplicate the arithmetic elsewhere.

use sky_ast::ast::span_text;
use sky_ast::UnitExpr;
use sky_diag::Span;
use sky_types::{NumberError, UnitError};
use sky_units::Dimension;

use crate::arith::{overflow, R};
use crate::RuntimeError;

/// A number literal that does not fit in an `f64`, at `span`.
pub fn number_too_large(span: Span) -> RuntimeError {
    RuntimeError::new("number too large", span)
        .with_label("does not fit in a 64-bit float")
        .with_note("the largest Sky number is about 1.8e308")
}

/// Reads a numeric literal's text (`1_000`, `2.5e-3`) as a finite `f64`.
pub fn number_literal(src: &str, span: Span) -> R<f64> {
    match sky_types::parse_number(span_text(src, span).unwrap_or("")) {
        Ok(x) => Ok(x),
        Err(NumberError::TooLarge) => Err(number_too_large(span)),
        Err(NumberError::Invalid) => Err(RuntimeError::new("invalid number", span)),
    }
}

/// A unit the table cannot apply, as a runtime error.
pub(crate) fn unit_error(src: &str, e: UnitError) -> RuntimeError {
    match e {
        UnitError::Unknown(span) => RuntimeError::new(
            format!("unknown unit `{}`", span_text(src, span).unwrap_or("")),
            span,
        ),
        UnitError::Exponent(span) => RuntimeError::new("dimension exponent too large", span),
    }
}

/// The SI value and dimension of the unit literal whose number is at
/// `value` and whose unit is `unit` (`whole` is the literal's span, for the
/// overflow error). The unit is applied factor by factor
/// (`sky_types::apply_unit`): `36 km/h` is `36 * 1000 / 3600`, exactly 10.
pub fn unit_literal_si(
    src: &str,
    value: Span,
    unit: &UnitExpr,
    whole: Span,
) -> R<(f64, Dimension)> {
    let x = number_literal(src, value)?;
    let (si, dim) = sky_types::apply_unit(src, unit, x).map_err(|e| unit_error(src, e))?;
    if !si.is_finite() {
        return Err(overflow(whole));
    }
    Ok((si, dim))
}
