//! `accuracy x within tol`: the error budget of a value (spec "Accuracy
//! budgets").
//!
//! The checker only checks the declaration: `x` is a name in scope that
//! holds a number, and the tolerance is a positive number literal, with a
//! unit or without. Whether the program meets the budget is the roundoff
//! analysis's question (`sky_ir::roundoff`), answered by `sky check` after
//! lowering; nothing here evaluates anything.
//!
//! * `accuracy y within 1e-12`: a **relative** error, `|computed - exact| <=
//!   1e-12 |exact|`. A bare number, so it works for a value of any unit.
//! * `accuracy y within 2 mm`: an **absolute** error, in the unit written,
//!   which must have `y`'s dimension.

use sky_ast::{Expr, ExprKind, Ident};
use sky_diag::Diagnostic;

use crate::check::{describe, strip_parens, Checker, Ty};
use crate::constant::entry_value;

impl Checker<'_> {
    pub(crate) fn accuracy_stmt(&mut self, name: &Ident, tolerance: &Expr) {
        let target = Expr {
            kind: ExprKind::Name(*name),
            span: name.span,
        };
        // Reads the name like any expression: an undefined one is reported
        // once, and so is one that may be unbound on some path.
        let t = self.expr(&target);
        // Typed whatever else is wrong, like every expression.
        let tol_ty = self.expr(tolerance);
        let want = match &t {
            Ty::Value(v) if v.is_scalar() => Some(v.dim),
            Ty::Value(v) => {
                let shown = self.quoted(name.span);
                self.error(
                    Diagnostic::error(
                        format!("`accuracy` applies to a number, but `{shown}` is {}", array_word(v.ty.shape().rank())),
                        name.span,
                    )
                    .with_label("not a single number")
                    .with_note(
                        "v0.1 states an error budget for one number at a time; declare it for an entry (`accuracy x0 within 1e-12` after `x0 = v[0]`)",
                    ),
                );
                return;
            }
            Ty::Bool | Ty::Str | Ty::Func(_) | Ty::Ode(_) => {
                let shown = self.quoted(name.span);
                let what = match &t {
                    Ty::Bool => "a Bool".to_string(),
                    Ty::Str => "a string".to_string(),
                    Ty::Ode(o) => o.describe(),
                    _ => "a function".to_string(),
                };
                self.error(
                    Diagnostic::error(
                        format!("`accuracy` applies to a number, but `{shown}` is {what}"),
                        name.span,
                    )
                    .with_label(format!("this is {what}")),
                );
                return;
            }
            Ty::Unknown => None,
        };
        let inner = strip_parens(tolerance);
        let literal = matches!(
            inner.kind,
            ExprKind::Literal(sky_ast::Literal::Int(_) | sky_ast::Literal::Float(_))
                | ExprKind::UnitLiteral { .. }
        );
        let Some(value) = literal.then(|| entry_value(self.src, tolerance)).flatten() else {
            self.error(
                Diagnostic::error("the tolerance must be a number literal", tolerance.span)
                    .with_label("not a number literal")
                    .with_help(
                        "write a relative error as `1e-12`, or an absolute one with a unit, `2 mm`",
                    ),
            );
            return;
        };
        if value <= 0.0 {
            self.error(
                Diagnostic::error("the tolerance must be positive", tolerance.span)
                    .with_label("zero or negative")
                    .with_note("a budget of zero asks for an exact result, which floating point does not promise"),
            );
        }
        // A tolerance with a unit is absolute: its dimension is the value's.
        if matches!(inner.kind, ExprKind::UnitLiteral { .. }) {
            if let (Ty::Value(tol), Some(want)) = (&tol_ty, want) {
                if tol.dim != want {
                    let shown = self.quoted(name.span);
                    self.error(
                        Diagnostic::error(
                            format!(
                                "the tolerance is {}, but `{shown}` is {}",
                                crate::a_dimension(&tol.dim),
                                crate::a_dimension(&want)
                            ),
                            tolerance.span,
                        )
                        .with_label(format!("this is {}", describe(&tol.dim)))
                        .with_secondary(name.span, format!("this is {}", describe(&want)))
                        .with_help("an absolute error is in the unit of the value; write a plain number for a relative one"),
                    );
                }
            }
        }
    }
}

fn array_word(rank: usize) -> &'static str {
    match rank {
        1 => "a vector",
        2 => "a matrix",
        _ => "an array",
    }
}
