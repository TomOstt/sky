//! `accuracy x within tol`: recording the budget for the roundoff analysis.
//!
//! Lowering emits no operation for the statement: the budget is a claim
//! about the program, not something it computes, and a pass could not be
//! trusted to keep an operation that has no result and no effect. It
//! records a [`Goal`] instead: the SSA value the name holds *at that
//! point* (so `x = ...; accuracy x within ...; x = ...` is about the first
//! `x`), the function it is in, and the tolerance in SI. The goals refer to
//! the module as lowering produces it, before any pass; `sky check` runs
//! the analysis on that module ([`sky_ir::roundoff`]).

use sky_ast::{Expr, ExprKind, Ident, Stmt};
use sky_ir::roundoff::{Goal, Tolerance};

use crate::ctx::Lower;

impl Lower<'_> {
    pub(crate) fn accuracy(&mut self, stmt: &Stmt, name: &Ident, tolerance: &Expr) {
        let n = self.text(name.span).to_string();
        // A name not in scope (a predefined constant nobody bound) has no
        // value to state a budget for: nothing to record.
        let Some(&value) = self.st.scope.get(&n) else {
            return;
        };
        let Some(t) = sky_types::entry_value(self.src, tolerance) else {
            return;
        };
        let mut inner = tolerance;
        while let ExprKind::Paren(e) = &inner.kind {
            inner = e;
        }
        let tolerance = if matches!(inner.kind, ExprKind::UnitLiteral { .. }) {
            Tolerance::Absolute(t)
        } else {
            Tolerance::Relative(t)
        };
        let function = self.st.b.name().to_string();
        self.goals.push(Goal {
            function,
            value,
            name: n,
            tolerance,
            span: stmt.span,
        });
    }
}
