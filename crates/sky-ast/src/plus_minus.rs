//! Building `value ± sigma` (spec "Uncertainty").
//!
//! The one rule that is not plain precedence is the *shared unit*: in
//! `5.31 ± 0.02 kg` the number before `±` has no unit and the quantity after
//! it does, and the unit belongs to both, as in a lab notebook. The parser
//! makes that explicit here, so no later phase has to know the rule: the
//! value becomes the unit literal `5.31 kg`, exactly as if it had been
//! written that way. It applies only when the value is a bare number
//! literal, optionally under unary `-` or `+` (`-5.31 ± 0.02 kg`), and the
//! sigma is a unit literal. Anything else is left alone, so `x ± 0.02 kg`
//! asks `x` to be a mass, `(5.31) ± 0.02 kg` asks for a dimensionless
//! `5.31` and is a checker error, and `5.31 kg ± 20 g` needs no help.

use sky_syntax::Span;

use crate::ast::{Expr, ExprKind, Literal, UnaryOp, UnitExpr};

/// The `ExprKind::PlusMinus` for `value ± sigma`, with the shared unit
/// applied when the shape calls for it.
#[inline(never)]
pub fn plus_minus(value: Expr, sigma: Expr, op_span: Span) -> Expr {
    let value = match &sigma.kind {
        ExprKind::UnitLiteral { unit, .. } => share_unit(value, unit, sigma.span),
        _ => value,
    };
    Expr {
        span: value.span.join(sigma.span),
        kind: ExprKind::PlusMinus {
            value: Box::new(value),
            sigma: Box::new(sigma),
            op_span,
        },
    }
}

/// `value` with `unit` (a clone of the sigma's) attached, if `value` is a
/// bare numeric literal under any run of unary `-`/`+`; otherwise `value`
/// unchanged. The new nodes' spans run from the number to the end of the
/// sigma, which is where their unit is.
fn share_unit(value: Expr, unit: &UnitExpr, sigma_span: Span) -> Expr {
    let span = value.span.join(sigma_span);
    match value.kind {
        ExprKind::Literal(lit @ (Literal::Int(_) | Literal::Float(_))) => Expr {
            kind: ExprKind::UnitLiteral {
                value: lit,
                unit: unit.clone(),
            },
            span,
        },
        ExprKind::Unary {
            op: op @ (UnaryOp::Neg | UnaryOp::Pos),
            op_span,
            operand,
        } => {
            let shared = share_unit(*operand, unit, sigma_span);
            let span = if matches!(shared.kind, ExprKind::UnitLiteral { .. }) {
                span
            } else {
                // Nothing to share under the sign: the node as it was.
                op_span.join(shared.span)
            };
            Expr {
                kind: ExprKind::Unary {
                    op,
                    op_span,
                    operand: Box::new(shared),
                },
                span,
            }
        }
        kind => Expr {
            kind,
            span: value.span,
        },
    }
}
