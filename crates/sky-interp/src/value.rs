//! Runtime values and how they print.

use std::collections::BTreeMap;
use std::fmt::{self, Write as _};

use std::sync::Arc;

use sky_ast::Block;
use sky_diag::Span;
use sky_units::{BaseDim, Dimension, Rational, UNITS};

/// A runtime value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// A number in SI with its dimension. Always finite. Dimensionless
    /// numbers use [`Dimension::DIMENSIONLESS`].
    ///
    /// Arrays are the sibling variant [`Value::Array`]; this is the rank-0
    /// case.
    Number {
        value: f64,
        dim: Dimension,
    },
    /// A vector (one entry in `dims`) or matrix (two) of numbers that share
    /// one dimension, row-major: entry `(i, j)` of an `m` by `n` matrix is
    /// `data[i * n + j]`. Every entry is finite, `data.len()` is the product
    /// of `dims`, and no axis is empty. Arrays are immutable values (v0.1
    /// has no element writes). Build one with [`Value::array`].
    Array {
        dims: Vec<usize>,
        data: Vec<f64>,
        dim: Dimension,
    },
    Bool(bool),
    Str(String),
    /// A number or array written in a chosen unit (`v as km/h`): its text,
    /// `90 km/h` or `[100 mm, 200 mm]`, already formatted. It is not a
    /// number: it can only be printed or echoed (the checker rejects any
    /// other use). Built by [`crate::show`].
    Shown(String),
    /// What `print(...)` and a function without a `return` value give.
    /// The REPL does not echo it.
    None,
    /// A function defined with `def`.
    Func(Arc<Function>),
}

/// A user function: what a call needs to run it. The body is a copy of the
/// checked AST (the interpreter walks the tree; there is no compiled form
/// yet).
pub struct Function {
    pub name: String,
    /// The span of the function's name in its `def`: which `def` of the
    /// program this is (two `def`s may share a name), the key the checker
    /// and lowering use.
    pub def_span: Span,
    pub params: Vec<String>,
    pub body: Block,
    /// The functions in scope where the `def` ran, by name. A call inside
    /// the body of a name that is not a local refers to the function here
    /// (or to the builtin, if the name is not here), whatever the name means
    /// by the time of the call: calls are bound where a function is
    /// *defined*, as the checker types them (spec, "Functions"). The
    /// function itself is not in the map (a recursive call is to it).
    pub captured: BTreeMap<String, Arc<Function>>,
}

impl fmt::Debug for Function {
    /// Without `captured`, whose functions capture each other's ancestors:
    /// printing them all is exponential in the number of `def`s.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Function")
            .field("name", &self.name)
            .field("def_span", &self.def_span)
            .field("params", &self.params)
            .field("body", &self.body)
            .field("captured", &self.captured.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl PartialEq for Function {
    /// The same `def` with the same captured functions (by identity: a
    /// structural comparison of `captured` is exponential).
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.def_span == other.def_span
            && self.params == other.params
            && self.body == other.body
            && self.captured.len() == other.captured.len()
            && self
                .captured
                .iter()
                .zip(&other.captured)
                .all(|((ka, a), (kb, b))| ka == kb && Arc::ptr_eq(a, b))
    }
}

impl Value {
    pub fn number(value: f64, dim: Dimension) -> Self {
        Value::Number { value, dim }
    }

    /// An array of shape `dims` (rank 1 or 2) holding `data` row-major. The
    /// caller upholds the invariants on [`Value::Array`]; they are only
    /// checked in debug builds.
    pub fn array(dims: Vec<usize>, data: Vec<f64>, dim: Dimension) -> Self {
        debug_assert!((1..=2).contains(&dims.len()));
        debug_assert!(dims.iter().all(|&d| d > 0));
        debug_assert_eq!(dims.iter().product::<usize>(), data.len());
        debug_assert!(data.iter().all(|x| x.is_finite()));
        Value::Array { dims, data, dim }
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
            Value::Array { dims, .. } if dims.len() == 1 => "a vector",
            Value::Array { .. } => "a matrix",
            Value::Bool(_) => "a Bool",
            Value::Str(_) => "a string",
            Value::Shown(_) => "a display value",
            Value::None => "none",
            Value::Func(_) => "a function",
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
            Value::Array { dims, data, dim } => {
                // Every entry is written with the unit, so the text is
                // valid Sky: `[1 m, 2 m]`, `[[1, 2], [3, 4]]`.
                let unit = si_unit(dim);
                write_array(f, dims, data, |f, x| {
                    f.write_str(&format_number(x))?;
                    match &unit {
                        Some(u) => write!(f, " {u}"),
                        None => Ok(()),
                    }
                })
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
            Value::Shown(text) => f.write_str(text),
            Value::None => f.write_str("none"),
            Value::Func(func) => write!(f, "<function {}>", func.name),
        }
    }
}

/// Writes a vector or matrix as a Sky array literal, row by row, with each
/// entry written by `entry`. Shared by `Display` (entries in SI) and by
/// `v as unit` (entries in the chosen unit), so both lay arrays out alike.
pub(crate) fn write_array<W: fmt::Write>(
    f: &mut W,
    dims: &[usize],
    data: &[f64],
    mut entry: impl FnMut(&mut W, f64) -> fmt::Result,
) -> fmt::Result {
    let cols = dims.last().copied().unwrap_or(1).max(1);
    let mut write_row = |f: &mut W, row: &[f64]| -> fmt::Result {
        f.write_char('[')?;
        for (j, &x) in row.iter().enumerate() {
            if j > 0 {
                f.write_str(", ")?;
            }
            entry(f, x)?;
        }
        f.write_char(']')
    };
    if dims.len() == 1 {
        return write_row(f, data);
    }
    f.write_char('[')?;
    for (i, row) in data.chunks(cols).enumerate() {
        if i > 0 {
            f.write_str(", ")?;
        }
        write_row(f, row)?;
    }
    f.write_char(']')
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
