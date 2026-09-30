//! `v as unit`: writing a value in a chosen unit (spec "Showing values in a
//! unit").
//!
//! Values are stored in SI, so showing one in `km/h` divides by the unit's
//! factors. That arithmetic is `sky_types::express_in_unit`, the inverse of
//! the `apply_unit` the literals use, over the same unit table; nothing here
//! knows a unit's factor. Numbers are then written by [`format_number`], as
//! `print` writes SI values.
//!
//! One extra rule keeps the display honest: the number shown is the
//! shortest one that *reads back as the stored value*. `100 km/h` is stored
//! as `27.77777777777778 m/s`; dividing that by the factors gives
//! `100.00000000000001`, which no reader meant. The shortest decimal `d`
//! with `d km/h` converting to exactly the stored SI value is `100`, so that
//! is what is shown. When no decimal of at most 17 digits reads back
//! exactly, the computed number is shown as is.
//!
//! [`show_value`] is what the tree-walking interpreter calls, with the
//! unit from the AST. [`show_unit_text`] is the same function for a unit
//! given as text (`km/h`), which is how the `show` operation in SIR carries
//! its unit; both share every line of the conversion, so `sky run` and the
//! SIR interpreter cannot print different digits.

use std::fmt::Write as _;

use sky_ast::ast::span_text;
use sky_ast::{parse_unit_text, UnitExpr};
use sky_diag::Span;
use sky_types::{a_dimension, apply_unit, express_in_unit};

use crate::arith::{overflow, R};
use crate::literal::unit_error;
use crate::value::{format_number, write_array};
use crate::{RuntimeError, Value};

/// The display value of `v` written in `unit` (parsed from `src`). `span` is
/// the whole `v as unit` expression, for errors. `v` must be a number or an
/// array whose dimension is the unit's.
pub fn show_value(src: &str, unit: &UnitExpr, v: &Value, span: Span) -> R<Value> {
    let label = span_text(src, unit.span()).unwrap_or("");
    let dim = match v {
        Value::Number { dim, .. } | Value::Array { dim, .. } => *dim,
        Value::Uncertain(u) => u.dim,
        other => {
            return Err(RuntimeError::new(
                format!("cannot show {} in `{label}`", other.kind()),
                span,
            )
            .with_label("only numbers and arrays of numbers can be shown in a unit"));
        }
    };
    let (_, unit_dim) = apply_unit(src, unit, 1.0).map_err(|e| unit_error(src, e))?;
    if unit_dim != dim {
        return Err(RuntimeError::new(
            format!("cannot show {} in `{label}`", a_dimension(&dim)),
            span,
        )
        .with_label(format!("`{label}` measures {unit_dim}, this is {dim}")));
    }
    // Convert every entry before writing, so an overflow is an error and
    // not a half-written string.
    let mut text = String::new();
    match v {
        Value::Array { dims, data, .. } => {
            let shown = data
                .iter()
                .map(|&si| shown_number(src, unit, si, span))
                .collect::<R<Vec<f64>>>()?;
            let _ = write_array(&mut text, dims, &shown, |f, x| {
                write!(f, "{} {label}", format_number(x))
            });
        }
        Value::Number { value, .. } => {
            let x = shown_number(src, unit, *value, span)?;
            let _ = write!(text, "{} {label}", format_number(x));
        }
        // Both parts scale by the unit's factor (units here are pure
        // factors), and the digits follow the uncertainty, so no readback
        // search: `5.31 ± 0.02 kg as g` is `5310 ± 20 g`.
        Value::Uncertain(u) => {
            let x = express(src, unit, u.value, span)?;
            let s = express(src, unit, u.std_dev(), span)?;
            let _ = write!(text, "{} {label}", crate::plus_minus_text(x, s));
        }
        _ => {}
    }
    Ok(Value::Shown(text))
}

/// [`show_value`] for a unit given as its source text, `km/h`. The text
/// must be a unit as [`sky_ast`] reads one after a number; anything else is
/// an error (SIR that names an unreadable unit is malformed).
pub fn show_unit_text(text: &str, v: &Value, span: Span) -> R<Value> {
    match parse_unit_text(text) {
        Some(u) => show_value(&u.src, &u.unit, v, span),
        None => Err(RuntimeError::new(format!("`{text}` is not a unit"), span)),
    }
}

/// `si` (in SI) in `unit`, exactly converted.
fn express(src: &str, unit: &UnitExpr, si: f64, span: Span) -> R<f64> {
    let (x, _) = express_in_unit(src, unit, si).map_err(|e| unit_error(src, e))?;
    if x.is_finite() {
        Ok(x)
    } else {
        Err(overflow(span))
    }
}

/// The number to write for the SI value `si` in `unit`: the exact
/// conversion, replaced by the shortest decimal that converts back to `si`
/// (see the module docs).
fn shown_number(src: &str, unit: &UnitExpr, si: f64, span: Span) -> R<f64> {
    let (x, _) = express_in_unit(src, unit, si).map_err(|e| unit_error(src, e))?;
    if !x.is_finite() {
        return Err(overflow(span));
    }
    let reads_back = |c: f64| matches!(apply_unit(src, unit, c), Ok((back, _)) if back == si);
    for digits in 1..=17usize {
        let candidate = format!("{x:.*e}", digits - 1);
        if let Ok(c) = candidate.parse::<f64>() {
            if c.is_finite() && reads_back(c) {
                return Ok(c);
            }
        }
    }
    Ok(x)
}
