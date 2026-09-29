//! The single binding-power table for Sky expressions.
//!
//! Lowest to highest (see `PARSER_DECISIONS.md`, needs Eng Lead sign-off):
//!
//! | level | operators                      | assoc |
//! |-------|--------------------------------|-------|
//! | -1    | `v as unit`                    | none: covers the whole expression, does not chain |
//! | 0     | `a if c else b`                | right (in the `else` arm); `c` is an `or` expression |
//! | 1     | `or`                           | left (one flat `BoolOp` node per run) |
//! | 2     | `and`                          | left (one flat `BoolOp` node per run) |
//! | 3     | prefix `not`                   | prefix; not allowed as an operand of a tighter operator (`-not x`) |
//! | 4     | `== != < <= > >=`              | none: `a < b < c` is an error |
//! | 5     | `+ -`                          | left  |
//! | 6     | `* / @`                        | left  |
//! | 7     | prefix `- + ∂ ∇`               | prefix |
//! | 8     | `^`                            | right; its right operand may be prefix (`2^-1`) |
//! | 9     | postfix `⁻¹`, `ᵀ`, call, index, `.` | postfix (tightest) |
//!
//! Infix entries are `(left_bp, right_bp)`. Left-associative operators use
//! `right = left + 1`; right-associative `^` uses `right = left - 1`.
//! `-x^2` is `-(x^2)` because `^`'s left bp (51) beats prefix bp (40).
//! Non-associativity of comparisons is enforced by the parser, not the table.
//! `not`'s operand is parsed at bp 7, so it takes whole comparisons but
//! stops at `and`/`or`.

use crate::ast::{BinaryOp, BoolOp, UnaryOp};
use sky_syntax::{Kw, TokenKind};

/// Binding power of `v as unit`: the loosest construct. `as` continues an
/// operand only where the enclosing expression accepts anything (`min_bp`
/// 0), so `a + b as km` is `(a + b) as km` and `x if c else y as km` is
/// `(x if c else y) as km`. The parser ends the expression after its unit.
pub const AS_BP: u8 = 0;

/// `and` and `or`, parsed as flat runs (see `ExprKind::BoolOp`).
/// Binding power of the conditional expression `a if c else b` (level 0):
/// looser than `or` (2). Only an operand parsed at `min_bp <= CONDITIONAL_BP`
/// may continue with `if`, so it never appears under any operator. Its
/// condition is parsed at [`CONDITIONAL_COND_BP`] (an `or` expression, no
/// nested conditional without parentheses) and its `else` arm at
/// `CONDITIONAL_BP` again, which makes it right-associative.
pub const CONDITIONAL_BP: u8 = 1;

/// Minimum binding power of a conditional's condition: `or` and tighter.
pub const CONDITIONAL_COND_BP: u8 = 2;

pub const BOOL: &[(TokenKind, BoolOp, u8, u8)] = &[
    (TokenKind::Keyword(Kw::Or), BoolOp::Or, 2, 3),
    (TokenKind::Keyword(Kw::And), BoolOp::And, 4, 5),
];

/// Binding power of prefix `not`: looser than comparisons (10), tighter
/// than `and` (4). An operand position that requires more (e.g. after `-`
/// or `+`) can't start with `not`.
pub const NOT_BP: u8 = 7;

pub const INFIX: &[(TokenKind, BinaryOp, u8, u8)] = &[
    (TokenKind::Eq, BinaryOp::Eq, 10, 11),
    (TokenKind::Ne, BinaryOp::Ne, 10, 11),
    (TokenKind::Lt, BinaryOp::Lt, 10, 11),
    (TokenKind::Le, BinaryOp::Le, 10, 11),
    (TokenKind::Gt, BinaryOp::Gt, 10, 11),
    (TokenKind::Ge, BinaryOp::Ge, 10, 11),
    (TokenKind::Plus, BinaryOp::Add, 20, 21),
    (TokenKind::Minus, BinaryOp::Sub, 20, 21),
    (TokenKind::Star, BinaryOp::Mul, 30, 31),
    (TokenKind::Slash, BinaryOp::Div, 30, 31),
    (TokenKind::At, BinaryOp::MatMul, 30, 31),
    (TokenKind::Caret, BinaryOp::Pow, 51, 50),
];

pub const PREFIX: &[(TokenKind, UnaryOp, u8)] = &[
    (TokenKind::Minus, UnaryOp::Neg, 40),
    (TokenKind::Plus, UnaryOp::Pos, 40),
    (TokenKind::Partial, UnaryOp::Partial, 40),
    (TokenKind::Nabla, UnaryOp::Nabla, 40),
    (TokenKind::Keyword(Kw::Not), UnaryOp::Not, NOT_BP),
];

/// Binding power of every postfix form: `⁻¹`, `(`, `[`, `.`.
pub const POSTFIX_BP: u8 = 60;

pub fn infix_binding_power(kind: TokenKind) -> Option<(BinaryOp, u8, u8)> {
    INFIX
        .iter()
        .find(|entry| entry.0 == kind)
        .map(|&(_, op, l, r)| (op, l, r))
}

pub fn bool_binding_power(kind: TokenKind) -> Option<(BoolOp, u8, u8)> {
    BOOL.iter()
        .find(|entry| entry.0 == kind)
        .map(|&(_, op, l, r)| (op, l, r))
}

pub fn prefix_binding_power(kind: TokenKind) -> Option<(UnaryOp, u8)> {
    PREFIX
        .iter()
        .find(|entry| entry.0 == kind)
        .map(|&(_, op, bp)| (op, bp))
}

pub fn is_postfix_start(kind: TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::Inverse
            | TokenKind::Transpose
            | TokenKind::LParen
            | TokenKind::LBracket
            | TokenKind::Dot
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_ordering_matches_the_documented_levels() {
        let lbp = |k| infix_binding_power(k).map(|(_, l, _)| l).unwrap_or(0);
        let prefix = prefix_binding_power(TokenKind::Minus)
            .map(|(_, bp)| bp)
            .unwrap_or(0);
        let bool_l = |k| bool_binding_power(k).map(|(_, l, _)| l).unwrap_or(0);
        let bool_r = |k| bool_binding_power(k).map(|(_, _, r)| r).unwrap_or(0);
        let or = TokenKind::Keyword(Kw::Or);
        let and = TokenKind::Keyword(Kw::And);
        assert!(CONDITIONAL_BP < bool_l(or) && CONDITIONAL_COND_BP <= bool_l(or));
        assert!(bool_l(or) < bool_l(and));
        // Left-assoc runs: the right operand stops at the same operator.
        assert!(bool_r(or) > bool_l(or) && bool_r(and) > bool_l(and));
        assert!(bool_r(and) <= NOT_BP && NOT_BP < lbp(TokenKind::Lt));
        assert!(lbp(TokenKind::Lt) < lbp(TokenKind::Plus));
        assert!(lbp(TokenKind::Plus) < lbp(TokenKind::Star));
        assert!(lbp(TokenKind::Star) < prefix);
        assert!(prefix < lbp(TokenKind::Caret));
        assert!(lbp(TokenKind::Caret) < POSTFIX_BP);
        // `^` is right-associative and its right side accepts prefix operators.
        let (_, l, r) = infix_binding_power(TokenKind::Caret).unwrap_or((BinaryOp::Pow, 0, 0));
        assert!(r < l && r > prefix);
    }

    #[test]
    fn every_token_has_at_most_one_infix_entry() {
        for (i, a) in INFIX.iter().enumerate() {
            for b in INFIX.iter().skip(i + 1) {
                assert_ne!(a.0, b.0);
            }
        }
    }
}
