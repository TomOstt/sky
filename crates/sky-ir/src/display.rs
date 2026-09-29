//! Text for types and tokens, shared by the printer and the verifier's
//! messages (`num[L; (3, 3); {Symmetric}]`), and the token-level rules the
//! text format is built from. See [`crate::print`] for the format.

use std::fmt::{self, Write as _};

use sky_types::ShapeDim;

use crate::ir::{NumberType, Repr, Type};

/// True if `s` can be printed after `%` or `@` as a bare token.
pub(crate) fn is_bare_name(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_alphanumeric() || c == '_')
}

/// A name after a sigil: bare if possible, else a quoted string. Function
/// names may also use `<`, `>` and `.` bare (`@<main>`).
pub(crate) fn write_name(out: &mut String, name: &str, function: bool) {
    let bare = is_bare_name(name)
        || (function
            && !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_alphanumeric() || matches!(c, '_' | '<' | '>' | '.')));
    if bare {
        out.push_str(name);
    } else {
        write_string(out, name);
    }
}

/// A string literal with escapes for `"`, `\`, control characters and
/// anything the diagnostics renderer treats as display-unsafe.
pub(crate) fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() || sky_diag::is_display_unsafe(c) => {
                let _ = write!(out, "\\u{{{:x}}}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A float's bits as text: shortest round-trip decimal if finite, else hex.
pub(crate) fn float_text(bits: u64) -> String {
    let x = f64::from_bits(bits);
    if !x.is_finite() {
        return format!("{bits:#018x}");
    }
    let s = format!("{x:?}");
    match s.strip_suffix(".0") {
        Some(int) => int.to_string(),
        None => s,
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Number(n) => n.fmt(f),
            Type::Bool => f.write_str("bool"),
            Type::Str => f.write_str("str"),
            Type::Nothing => f.write_str("nothing"),
            Type::Unknown => f.write_str("unknown"),
        }
    }
}

impl fmt::Display for NumberType {
    /// `num[L; (3, N#0)]`, with `; {Symmetric}` and `; f64` appended when
    /// there are tags or a chosen representation.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "num[{}; (", self.dim)?;
        for (i, d) in self.shape.dims().iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            match d {
                ShapeDim::Known(n) => write!(f, "{n}")?,
                ShapeDim::Var(v) => {
                    let mut name = String::new();
                    write_name(&mut name, v.name(), false);
                    write!(f, "{name}#{}", v.id())?;
                }
            }
        }
        f.write_str(")")?;
        if !self.tags.is_empty() {
            write!(f, "; {}", self.tags)?;
        }
        match self.repr {
            Repr::Unchosen => {}
            Repr::Exact => f.write_str("; exact")?,
            Repr::F32 => f.write_str("; f32")?,
            Repr::F64 => f.write_str("; f64")?,
        }
        f.write_str("]")
    }
}
