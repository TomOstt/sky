//! Runtime values and how they print.

use std::fmt::{self, Write as _};

use sky_units::{BaseDim, Dimension, Rational, UNITS};

/// A runtime value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// A number in SI with its dimension. Always finite. Dimensionless
    /// numbers use [`Dimension::DIMENSIONLESS`].
    ///
    /// Arrays will be a sibling variant holding the element dimension and a
    /// shape; this is the rank-0 case.
    Number {
        value: f64,
        dim: Dimension,
    },
    Bool(bool),
    Str(String),
    /// What `print(...)` returns. The REPL does not echo it.
    None,
}

impl Value {
    pub fn number(value: f64, dim: Dimension) -> Self {
        Value::Number { value, dim }
    }

    /// A dimensionless number.
    pub fn scalar(value: f64) -> Self {
        Value::Number {
            value,
            dim: Dimension::DIMENSIONLESS,
        }
    }

    /// Short noun for error messages.
    pub fn kind(&self) -> &'static str {
        match self {
            Value::Number { .. } => "a number",
            Value::Bool(_) => "a Bool",
            Value::Str(_) => "a string",
            Value::None => "none",
        }
    }

    /// The text `print` writes: like [`fmt::Display`], but a string is raw
    /// (unquoted, unescaped), as in Python.
    pub fn print_text(&self) -> String {
        match self {
            Value::Str(s) => s.clone(),
            other => other.to_string(),
        }
    }
}

impl fmt::Display for Value {
    /// Sky source for the value, in SI: `9.81 m/s^2`, `true`, `"a\n"`. A
    /// REPL echo can be pasted back as input.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Number { value, dim } => {
                f.write_str(&format_number(*value))?;
                match si_unit(dim) {
                    Some(u) => write!(f, " {u}"),
                    None => Ok(()),
                }
            }
            Value::Bool(b) => f.write_str(if *b { "true" } else { "false" }),
            Value::Str(s) => {
                f.write_char('"')?;
                for c in s.chars() {
                    match c {
                        '\\' => f.write_str("\\\\")?,
                        '"' => f.write_str("\\\"")?,
                        '\n' => f.write_str("\\n")?,
                        '\t' => f.write_str("\\t")?,
                        '\r' => f.write_str("\\r")?,
                        '\0' => f.write_str("\\0")?,
                        c => f.write_char(c)?,
                    }
                }
                f.write_char('"')
            }
            Value::None => f.write_str("none"),
        }
    }
}

/// Formats an `f64` so that it parses back to the same value and is valid
/// Sky source: Rust's shortest round-trip digits, in plain decimal for
/// `1e-4 <= |x| < 1e16` (so `0.01`, `10000`) and in `1.5e-11` form outside
/// that range (Python's thresholds), so `6.674e-11` does not print as
/// eleven zeros. Non-finite values never occur in a run; they print as
/// Rust does.
pub fn format_number(x: f64) -> String {
    let a = x.abs();
    if a != 0.0 && x.is_finite() && !(1e-4..1e16).contains(&a) {
        format!("{x:e}")
    } else {
        format!("{x}")
    }
}

/// The SI unit text for `dim` as valid Sky unit syntax, or `None` if the
/// dimension is dimensionless (a bare number).
///
/// If a unit in the table has exactly this dimension and scale 1 (`N`,
/// `J`, `W`, `Pa`, `Hz`, and whatever the table gains), that symbol is
/// used. Otherwise base units are composed in SI order (kg, m, s, K):
/// `m/s^2`, `kg*m^2/s^2`, `m^-1`, `m^(1/2)`.
pub fn si_unit(dim: &Dimension) -> Option<String> {
    if dim.is_dimensionless() {
        return None;
    }
    if let Some(u) = UNITS.iter().find(|u| u.dim == *dim && u.scale == 1.0) {
        return Some(u.name.to_string());
    }
    // Current, amount and luminosity have no table unit yet, so these
    // symbols are not valid Sky source; no v0.1 value has such a dimension.
    let symbol = |b: BaseDim| match b {
        BaseDim::Mass => "kg",
        BaseDim::Length => "m",
        BaseDim::Time => "s",
        BaseDim::Temperature => "K",
        BaseDim::Current => "A",
        BaseDim::Amount => "mol",
        BaseDim::Luminosity => "cd",
    };
    let power = |sym: &str, e: Rational| {
        if e == Rational::ONE {
            sym.to_string()
        } else if e.is_integer() {
            format!("{sym}^{}", e.numer())
        } else {
            format!("{sym}^({}/{})", e.numer(), e.denom())
        }
    };
    let order = [
        BaseDim::Mass,
        BaseDim::Length,
        BaseDim::Time,
        BaseDim::Temperature,
        BaseDim::Current,
        BaseDim::Amount,
        BaseDim::Luminosity,
    ];
    let (mut pos, mut neg) = (Vec::new(), Vec::new());
    for b in order {
        let e = dim.exponent(b);
        if e.is_zero() {
            continue;
        }
        if e.numer() > 0 {
            pos.push(power(symbol(b), e));
        } else {
            neg.push((symbol(b), e));
        }
    }
    if pos.is_empty() {
        // No numerator: write the negative exponents, `s^-1*K^-1`.
        let parts: Vec<String> = neg.iter().map(|&(s, e)| power(s, e)).collect();
        return Some(parts.join("*"));
    }
    let mut out = pos.join("*");
    for (s, e) in neg {
        out.push('/');
        // `e` is negative, so it can be negated unless it is i64::MIN,
        // which the checker's exponent limits keep out of any value.
        out.push_str(&power(s, e.checked_neg().unwrap_or(e)));
    }
    Some(out)
}
