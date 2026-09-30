//! Lowering contracts: `assert`, `requires` and `ensures` become `assert[kind]`
//! operations (`OpKind::Assert`).
//!
//! * `assert cond` lowers where it stands.
//! * `requires cond` stands at the top of the function body, so it lowers as
//!   the first operations of the function, where it stands.
//! * `ensures cond` is lowered by [`Lower::lower_body`] just before the
//!   function's `return`, on every path (early `return`s are flags, so the
//!   function has one exit), in the scope the function *started* with: the
//!   parameters as they were passed, the module names, and `result`, the
//!   value about to be returned. What the body assigned to its locals or
//!   parameters is not visible to it, as in the tree-walker, which runs the
//!   postcondition in a frame of the arguments and `result`.
//!
//! The operation carries the text and the names its failure message needs
//! (see `sky_interp::contract`, which both engines build the message with),
//! and one operand per name: the value the name has where the contract is
//! checked. A name that holds no value (the name of a function) has no
//! operand.

use sky_ast::{AssertKind, Expr, Stmt, StmtKind};
use sky_diag::Span;
use sky_interp::contract::{condition_names, condition_text, message_text};
use sky_ir::{OpKind, Type, ValueId};
use sky_units::lookup_constant;

use crate::ctx::{Lower, Scope, R};

/// Whether `stmts` (a function body) has an `ensures` line.
pub(crate) fn has_ensures(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| {
        matches!(
            s.kind,
            StmtKind::Assert {
                kind: AssertKind::Ensures,
                ..
            }
        )
    })
}

impl Lower<'_> {
    /// `kind cond, "message"` where a statement stands. An `ensures` does
    /// not: it runs at the function's end ([`Lower::lower_ensures`]).
    pub(crate) fn assert_stmt(
        &mut self,
        kind: AssertKind,
        cond: &Expr,
        message: Option<Span>,
    ) -> R<()> {
        if kind == AssertKind::Ensures {
            return Ok(());
        }
        self.assert_op(kind, cond, message)
    }

    /// The function's `ensures` lines, checked in the scope it started with
    /// (`entry`) and `result` (if the function returns a value).
    pub(crate) fn lower_ensures(
        &mut self,
        stmts: &[Stmt],
        entry: Scope,
        result: Option<ValueId>,
    ) -> R<()> {
        let saved = std::mem::replace(&mut self.st.scope, entry);
        if let Some(v) = result {
            self.st.scope.insert("result".to_string(), v);
        }
        let mut done = Ok(());
        for s in stmts {
            if let StmtKind::Assert {
                kind: AssertKind::Ensures,
                cond,
                message,
            } = &s.kind
            {
                done = self.assert_op(AssertKind::Ensures, cond, *message);
                if done.is_err() {
                    break;
                }
            }
        }
        self.st.scope = saved;
        done
    }

    fn assert_op(&mut self, kind: AssertKind, cond: &Expr, message: Option<Span>) -> R<()> {
        let c = self.expr(cond)?;
        let mut operands = vec![c];
        let mut names = Vec::new();
        for name in condition_names(cond, self.src) {
            let value = match self.st.scope.get(name) {
                Some(v) => Some(*v),
                None => lookup_constant(name)
                    .map(|k| self.const_float(k.value, Type::scalar(k.dim), cond.span)),
            };
            if let Some(v) = value {
                operands.push(v);
                names.push(name.to_string());
            }
        }
        let func =
            (kind != AssertKind::Assert).then(|| self.st.fn_name.clone().unwrap_or_default());
        let op = OpKind::Assert {
            kind,
            text: condition_text(self.src, cond.span),
            message: message.map(|m| message_text(self.src, m)),
            func,
            names,
        };
        self.st.b.op(op, operands, Vec::new(), cond.span);
        Ok(())
    }
}
