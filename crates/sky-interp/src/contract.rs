//! Contracts at run time: `assert`, `requires` and `ensures` (spec
//! "Contracts").
//!
//! The tree-walker evaluates them in `eval.rs`; the SIR interpreter runs the
//! `assert` operation. Both build their failure from the functions here, so
//! the two engines give the same message, the same span and the same
//! wording: [`condition_text`] and [`condition_names`] read the source once,
//! at lowering (SIR) or at the failure (tree-walker), and [`failure`] turns
//! them and the values of the names into the error.

use sky_ast::ast::span_text;
use sky_ast::{AssertKind, Expr, ExprKind, IndexItem};
use sky_diag::{display_capped, Span};
use sky_syntax::lexer::unescape_str;

use crate::{RuntimeError, Value};

/// The condition as the message quotes it: the source text with each run of
/// white space (a condition may span lines inside parentheses) as one space.
pub fn condition_text(src: &str, cond: Span) -> String {
    let text = span_text(src, cond).unwrap_or("");
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The text of a message given as a string literal token (`"..."`).
pub fn message_text(src: &str, literal: Span) -> String {
    unescape_str(span_text(src, literal).unwrap_or(""))
}

/// The names a failure message shows the values of: the names read in the
/// condition, each once, in order of first appearance. The name of a called
/// function (`sqrt` in `sqrt(x) > 1`) is not read as a value, so it is not
/// one of them; a name that turns out to hold no value (a function, or a
/// name that is not bound) is left out by the caller.
pub fn condition_names<'s>(cond: &Expr, src: &'s str) -> Vec<&'s str> {
    let mut out: Vec<&'s str> = Vec::new();
    collect(cond, src, &mut out);
    out
}

fn collect<'s>(e: &Expr, src: &'s str, out: &mut Vec<&'s str>) {
    match &e.kind {
        ExprKind::Name(id) => {
            let name = span_text(src, id.span).unwrap_or("");
            if !name.is_empty() && !out.contains(&name) {
                out.push(name);
            }
        }
        ExprKind::Literal(_) | ExprKind::UnitLiteral { .. } => {}
        ExprKind::Unary { operand, .. }
        | ExprKind::Inverse { operand, .. }
        | ExprKind::Transpose { operand, .. } => collect(operand, src, out),
        ExprKind::Paren(inner) => collect(inner, src, out),
        ExprKind::Binary { lhs, rhs, .. } => {
            collect(lhs, src, out);
            collect(rhs, src, out);
        }
        ExprKind::BoolOp { operands, .. } | ExprKind::Array(operands) => {
            for o in operands {
                collect(o, src, out);
            }
        }
        ExprKind::IfElse { then, cond, else_ } => {
            for x in [then, cond, else_] {
                collect(x, src, out);
            }
        }
        ExprKind::As { value, .. } => collect(value, src, out),
        ExprKind::PlusMinus { value, sigma, .. } => {
            collect(value, src, out);
            collect(sigma, src, out);
        }
        ExprKind::Attribute { base, .. } => collect(base, src, out),
        ExprKind::Index { base, items } => {
            collect(base, src, out);
            for item in items {
                match item {
                    IndexItem::Expr(x) => collect(x, src, out),
                    IndexItem::Slice { start, stop, .. } => {
                        for b in [start, stop].into_iter().flatten() {
                            collect(b, src, out);
                        }
                    }
                }
            }
        }
        ExprKind::Call { callee, args } => {
            let mut target = &**callee;
            while let ExprKind::Paren(inner) = &target.kind {
                target = inner;
            }
            if !matches!(target.kind, ExprKind::Name(_)) {
                collect(callee, src, out);
            }
            for a in args {
                collect(a, src, out);
            }
        }
    }
}

/// Most entries of an array a failure message writes out; a longer array is
/// named by its shape (`Vector[300] m`), so a message never costs the size of
/// the data (an array may have millions of entries).
const MAX_SHOWN_ENTRIES: usize = 16;

/// A value as a failure message shows it: Sky source (`-3.2 J`, `[1 m, 2 m]`,
/// `"a"`), or, for an array of more than [`MAX_SHOWN_ENTRIES`] entries, its
/// shape and unit.
pub fn show(v: &Value) -> String {
    match v {
        Value::Array { dims, data, dim } if data.len() > MAX_SHOWN_ENTRIES => {
            let shape = match dims[..] {
                [n] => format!("Vector[{n}]"),
                [m, n] => format!("Matrix[{m}, {n}]"),
                _ => "Tensor".to_string(),
            };
            match crate::si_unit(dim) {
                Some(u) => format!("{shape} {u}"),
                None => shape,
            }
        }
        other => other.to_string(),
    }
}

/// The condition's value as a Bool: anything else is an error on the
/// condition (v0.1 has no truthiness; the checker rejects a typed non-Bool,
/// so this covers a condition it could not type).
pub fn expect_condition(kind: AssertKind, v: &Value, span: Span) -> Result<bool, RuntimeError> {
    match v {
        Value::Bool(b) => Ok(*b),
        other => Err(RuntimeError::new(
            format!(
                "`{}` needs a Bool condition, found {}",
                kind.keyword(),
                other.kind()
            ),
            span,
        )
        .with_label("not a Bool")
        .with_note("v0.1 has no truthiness")),
    }
}

/// The message of a failed contract:
///
/// * `assertion failed: `E > 0 J` (E = -3.2 J)`
/// * `precondition of `fall_time` failed: `h >= 0 m` (h = -1 m)`
/// * `postcondition of `fall_time` failed: `result > 0 s` (result = 0 s, h = 0 m)`
///
/// With the author's message it leads: `assertion failed: energy must be
/// positive (`E > 0 J`, E = -3.2 J)`. `func` is the function a `requires` or
/// `ensures` belongs to; `shown` are the names of the condition with their
/// values (`Value`'s `Display`, Sky source), in order.
///
/// Everything from the program is capped and escaped like any quoted source
/// text, so a long condition or a long array cannot make a long message.
pub fn failure_message(
    kind: AssertKind,
    func: Option<&str>,
    text: &str,
    message: Option<&str>,
    shown: &[(&str, String)],
) -> String {
    let head = match (kind, func) {
        (AssertKind::Requires, Some(f)) => {
            format!("precondition of `{}` failed", display_capped(f))
        }
        (AssertKind::Ensures, Some(f)) => {
            format!("postcondition of `{}` failed", display_capped(f))
        }
        (AssertKind::Requires, None) => "precondition failed".to_string(),
        (AssertKind::Ensures, None) => "postcondition failed".to_string(),
        (AssertKind::Assert, _) => "assertion failed".to_string(),
    };
    let quoted = format!("`{}`", display_capped(text));
    let values: Vec<String> = shown
        .iter()
        .map(|(name, v)| format!("{} = {}", display_capped(name), display_capped(v)))
        .collect();
    match message {
        None if values.is_empty() => format!("{head}: {quoted}"),
        None => format!("{head}: {quoted} ({})", values.join(", ")),
        Some(m) => {
            let m = display_capped(m);
            if values.is_empty() {
                format!("{head}: {m} ({quoted})")
            } else {
                format!("{head}: {m} ({quoted}, {})", values.join(", "))
            }
        }
    }
}

/// The error for a failed contract at the condition's `span`.
pub fn failure(
    kind: AssertKind,
    func: Option<&str>,
    text: &str,
    message: Option<&str>,
    shown: &[(&str, String)],
    span: Span,
) -> RuntimeError {
    RuntimeError::new(failure_message(kind, func, text, message, shown), span)
        .with_label("this is false")
}
