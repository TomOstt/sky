//! A unit given as text, read with the parser's own unit sub-grammar.
//!
//! SIR carries the unit of a `v as km/h` as its source text (`km/h`), so the
//! things that read SIR (the verifier and the reference interpreter) need to
//! turn that text back into a [`UnitExpr`]. This does it by parsing the
//! unit literal `1 km/h` with the real lexer and parser, so there is no
//! second unit grammar to keep in step: whatever a unit literal accepts,
//! this accepts, and nothing else.

use crate::ast::{span_text, ExprKind, UnitExpr};
use crate::parser::parse_expr;

/// A unit parsed from text. The spans of `unit` index into `src`, which is
/// the text prefixed by `1 `, so pass both to code that reads unit names
/// (`sky_types::apply_unit(&u.src, &u.unit, x)`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitText {
    pub src: String,
    pub unit: UnitExpr,
}

/// Parses `text` as a unit (`km/h`, `kg*m^2/s^2`, `MeV`). `None` unless the
/// whole text is exactly one unit, written as it is after a number: no
/// spaces, no trailing tokens, no unit that the parser rejects.
pub fn parse_unit_text(text: &str) -> Option<UnitText> {
    let src = format!("1 {text}");
    let lexed = sky_syntax::lex(&src).ok()?;
    let (expr, errors) = parse_expr(&lexed.tokens, &src);
    if !errors.is_empty() {
        return None;
    }
    let ExprKind::UnitLiteral { unit, .. } = expr?.kind else {
        return None;
    };
    // The whole text, not a prefix of it that happened to parse.
    (span_text(&src, unit.span()) == Some(text)).then_some(UnitText { src, unit })
}
