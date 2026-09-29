//! The single binding-power table for Sky expressions.
//!
//! Lowest to highest (see `PARSER_DECISIONS.md`, needs Eng Lead sign-off):
//!
//! | level | operators                      | assoc |
//! |-------|--------------------------------|-------|
//! | 1     | `== != < <= > >=`              | none: `a < b < c` is an error |
//! | 2     | `+ -`                          | left  |
//! | 3     | `* / @`                        | left  |
//! | 4     | prefix `- + ∂ ∇`               | prefix |
//! | 5     | `^`                            | right; its right operand may be prefix (`2^-1`) |
//! | 6     | postfix `⁻¹`, call, index, `.` | postfix (tightest) |
//!
//! Infix entries are `(left_bp, right_bp)`. Left-associative operators use
//! `right = left + 1`; right-associative `^` uses `right = left - 1`.
//! `-x^2` is `-(x^2)` because `^`'s left bp (51) beats prefix bp (40).
//! Non-associativity of comparisons is enforced by the parser, not the table.

use crate::ast::{BinaryOp, UnaryOp};
use sky_syntax::TokenKind;

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
];

/// Binding power of every postfix form: `⁻¹`, `(`, `[`, `.`.
pub const POSTFIX_BP: u8 = 60;

pub fn infix_binding_power(kind: TokenKind) -> Option<(BinaryOp, u8, u8)> {
    INFIX
        .iter()
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
        TokenKind::Inverse | TokenKind::LParen | TokenKind::LBracket | TokenKind::Dot
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_ordering_matches_the_documented_levels() {
        let lbp = |k| infix_binding_power(k).map(|(_, l, _)| l).unwrap_or(0);
        let prefix = prefix_binding_power(TokenKind::Minus).map(|(_, bp)| bp).unwrap_or(0);
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
