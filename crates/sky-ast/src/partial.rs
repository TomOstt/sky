//! The shape `∂f/∂t`, which is sugar for `derivative(f, t)` when it is
//! called (spec "Derivatives").
//!
//! The parser does not treat it specially: `∂` is a prefix operator that
//! binds tighter than `/`, so `∂f/∂t` is the division of two prefix nodes,
//! `(/ (partial f) (partial t))`. The checker, the interpreter and lowering
//! recognise that shape where it is the callee of a call, through
//! [`Expr::partial_quotient`], so the meaning is one definition.
//!
//! Only two bare names count. `∂f/∂t^2` is `∂f / ∂(t^2)` and `∂f/∂t(2 s)`
//! is `∂f / ∂(t(2 s))` (a call and a power bind tighter than `∂`), so
//! neither is the form; `(∂f/∂t)(2 s)` is.

use crate::ast::{BinaryOp, Expr, ExprKind, UnaryOp};

impl Expr {
    /// `(f, t)` when this expression is exactly `∂f/∂t`: a quotient whose
    /// operands are `∂` applied to one bare name each. Parentheses around
    /// the whole quotient or an operand are not looked through: the caller
    /// strips those it accepts.
    pub fn partial_quotient(&self) -> Option<(&Expr, &Expr)> {
        let ExprKind::Binary {
            op: BinaryOp::Div,
            lhs,
            rhs,
            ..
        } = &self.kind
        else {
            return None;
        };
        Some((partial_of_name(lhs)?, partial_of_name(rhs)?))
    }
}

/// `x` for the expression `∂x` where `x` is a bare name.
fn partial_of_name(e: &Expr) -> Option<&Expr> {
    match &e.kind {
        ExprKind::Unary {
            op: UnaryOp::Partial,
            operand,
            ..
        } if matches!(operand.kind, ExprKind::Name(_)) => Some(operand),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::parse_expr;

    /// The names of `∂f/∂t` in `src` as text, if it is that shape.
    fn names(src: &str) -> Option<(String, String)> {
        let lexed = sky_syntax::lex(src).expect("lexes");
        let (expr, errors) = parse_expr(&lexed.tokens, src);
        assert!(errors.is_empty(), "{src}: {errors:?}");
        let e = expr.expect("an expression");
        let (f, t) = e.partial_quotient()?;
        Some((
            src[f.span.range()].to_string(),
            src[t.span.range()].to_string(),
        ))
    }

    #[test]
    fn two_bare_names_are_the_form() {
        assert_eq!(names("∂f/∂t"), Some(("f".into(), "t".into())));
        assert_eq!(names("∂ f / ∂ t"), Some(("f".into(), "t".into())));
    }

    #[test]
    fn anything_else_is_ordinary_arithmetic() {
        // A call or a power binds tighter than `∂`, so these are not names.
        for src in [
            "∂f/∂t(2 s)",
            "∂f/∂t^2",
            "∂f/t",
            "f/∂t",
            "∂(f)/∂t",
            "(∂f)/(∂t)",
            "∂f*∂t",
            "∇f/∇t",
            "∂f.x/∂t",
            "∂f",
        ] {
            assert_eq!(names(src), None, "{src}");
        }
    }
}
