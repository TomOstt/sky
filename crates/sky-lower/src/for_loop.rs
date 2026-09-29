//! Lowering `for` loops, as a desugaring to the `while` of `stmt.rs`.
//!
//! ```text
//! for i in range(a, b, s):            <idx> = a
//!     body                    ==>     while <idx> < b:          (> if s < 0)
//!                                         i = <idx>
//!                                         <idx> = <idx> + s
//!                                         body
//! ```
//!
//! `<idx>` is a hidden loop-carried value, one per loop depth (`<i1>`,
//! `<i2>`, ...; names that start with `<` cannot clash with an identifier,
//! like the exit flags). The index is advanced *before* the body, so a
//! `continue`, which skips the rest of the body through a guard, still
//! moves on to the next item, and `i` is the visible copy: rebinding `i` in
//! the body does not change the iteration, and after a `break` it keeps the
//! value it had there, as in the interpreter.
//!
//! `range(b)` and `range(a, b)` have step 1 and start 0. The bounds and the
//! step are evaluated once, before the loop, left to right. The test is
//! `<idx> < b` for a constant positive step and `<idx> > b` for a constant
//! negative one; for a step known only at run time it is
//! `(<idx> - b) * s < 0`, which is both at once and is false for a zero
//! step. The interpreter also rejects a bound that is not an integer or
//! is larger than 2^53 in size, and a zero step; SIR has no operation that
//! stops a program with a message yet, so those checks are made by the
//! checker for constants and only by the tree walker for computed values. A
//! lowered loop with such a value counts as far as `f64` arithmetic goes
//! (`range(5 / 2)` counts 0, 1, 2), which is fine for looking at the IR and
//! wrong for running it: [`crate::lower_for_run`], which `sky run` lowers
//! with, refuses such a loop, and the tree walker runs the program.
//!
//! `for x in v` over an array needs array values, which SIR does not have
//! yet, so it is a [`LowerError`] that says so.
//!
//! Which names the loop carries out of the body follows the checker
//! ([`sky_types::range_runs_at_least_once`] is the shared rule): if the
//! loop is known to run, everything the body assigns is defined after it
//! (the loop variable included), so it is carried like the names of a
//! `while true`; otherwise only names bound before the loop are, and the
//! loop variable is not usable after it.

use std::collections::BTreeSet;

use sky_ast::{Block, Expr, ExprKind, Ident};
use sky_diag::Span;
use sky_ir::{CmpOp, OpKind, Type, ValueId};
use sky_types::{range_constants, range_runs_at_least_once};
use sky_units::Dimension;

use crate::analysis::is_forever;
use crate::ctx::{Lower, Needed, R};
use crate::expr::ARRAYS;
use crate::LowerError;

/// What starts a loop: the condition of a `while`, or a counting header.
pub(crate) enum Head<'e> {
    While(&'e Expr),
    Range(Range),
}

/// A `for` over `range(...)`, after its bounds were evaluated.
pub(crate) struct Range {
    /// The hidden index, a name that starts with `<`.
    idx: String,
    /// The loop variable.
    var: String,
    stop: ValueId,
    step: ValueId,
    /// The sign of a constant step; `None` if it is computed.
    step_sign: Option<i64>,
    /// Whether the checker counts the loop as running at least once.
    runs: bool,
    span: Span,
}

impl Head<'_> {
    /// Whether the names first assigned in the body are defined after the
    /// loop, so the loop must carry them (see the module docs).
    pub(crate) fn carries_new_names(&self) -> bool {
        match self {
            Head::While(c) => is_forever(c),
            Head::Range(r) => r.runs,
        }
    }

    /// The span of the condition, or of the whole `for` header.
    pub(crate) fn span(&self) -> Span {
        match self {
            Head::While(c) => c.span,
            Head::Range(r) => r.span,
        }
    }

    /// Adds the names the header itself assigns each iteration.
    pub(crate) fn assigned(&self, out: &mut BTreeSet<String>) {
        if let Head::Range(r) = self {
            out.insert(r.idx.clone());
            out.insert(r.var.clone());
        }
    }
}

impl Lower<'_> {
    /// The loop's condition, in the `before` region, where the carried
    /// values are in scope.
    pub(crate) fn head_cond(&mut self, head: &Head<'_>) -> R<ValueId> {
        match head {
            Head::While(cond) => self.expr(cond),
            Head::Range(r) => self.range_cond(r),
        }
    }

    /// What runs at the top of the body: for a `for`, the index becomes the
    /// loop variable and moves on to the next item.
    pub(crate) fn head_prologue(&mut self, head: &Head<'_>) -> R<()> {
        let Head::Range(r) = head else {
            return Ok(());
        };
        let Some(&idx) = self.st.scope.get(&r.idx) else {
            return self.err("internal error: a loop index is not in scope", r.span);
        };
        self.bind(&r.var, idx);
        let next = self.op1(OpKind::Add, vec![idx, r.step], index_ty(), r.span);
        self.st.scope.insert(r.idx.clone(), next);
        Ok(())
    }

    fn range_cond(&mut self, r: &Range) -> R<ValueId> {
        let Some(&idx) = self.st.scope.get(&r.idx) else {
            return self.err("internal error: a loop index is not in scope", r.span);
        };
        let cmp =
            |op: CmpOp, l: &mut Self, a, b| l.op1(OpKind::Cmp(op), vec![a, b], Type::Bool, r.span);
        Ok(match r.step_sign {
            Some(s) if s > 0 => cmp(CmpOp::Lt, self, idx, r.stop),
            Some(_) => cmp(CmpOp::Gt, self, idx, r.stop),
            None => {
                let gap = self.op1(OpKind::Sub, vec![idx, r.stop], index_ty(), r.span);
                let signed = self.op1(OpKind::Mul, vec![gap, r.step], index_ty(), r.span);
                let zero = self.const_float(0.0, index_ty(), r.span);
                cmp(CmpOp::Lt, self, signed, zero)
            }
        })
    }

    /// `for var in iter: body`.
    pub(crate) fn for_stmt(
        &mut self,
        span: Span,
        var: &Ident,
        iter: &Expr,
        body: &Block,
        needed: &Needed,
    ) -> R<()> {
        let Some(args) = self.range_args(iter) else {
            return Err(LowerError::new(
                "iteration over arrays lowers once SIR has array values",
                iter.span,
            )
            .with_note(ARRAYS));
        };
        let (start, stop, step) = match args {
            [stop] => (None, stop, None),
            [start, stop] => (Some(start), stop, None),
            [start, stop, step] => (Some(start), stop, Some(step)),
            _ => return self.err("`range` takes 1 to 3 arguments", iter.span),
        };
        // Evaluated once, before the loop, left to right.
        let start_v = match start {
            Some(a) => self.expr(a)?,
            None => self.const_float(0.0, index_ty(), iter.span),
        };
        let stop_v = self.expr(stop)?;
        let step_v = match step {
            Some(a) => self.expr(a)?,
            None => self.const_float(1.0, index_ty(), iter.span),
        };
        let constants = range_constants(self.src, args);
        // The interpreter checks a computed bound or step when the loop
        // starts (a whole number of sane size, a step that is not zero) and
        // stops with an error otherwise; only constants are checked before
        // the run. SIR cannot make that check, so a program that is to be
        // executed would run on where `sky run` must fail: it stays on the
        // tree walker until SIR can stop with a message.
        if self.for_run && constants.is_none() {
            return Err(LowerError::new(
                "a `range` bound or step computed at run time is checked by the tree-walking interpreter",
                iter.span,
            )
            .with_note(RANGE_CHECKS));
        }
        let step_sign = match (step, constants) {
            (None, _) => Some(1),
            (Some(_), Some((_, _, s))) => Some(s.signum()),
            (Some(_), None) => None,
        };
        let idx = format!("<i{}>", self.st.loops + 1);
        let range = Range {
            idx: idx.clone(),
            var: self.text(var.span).to_string(),
            stop: stop_v,
            step: step_v,
            step_sign,
            runs: constants.is_some_and(range_runs_at_least_once),
            span: iter.span,
        };
        self.st.scope.insert(idx.clone(), start_v);
        if range.runs {
            // The loop overwrites `var` before anything reads it and always
            // runs, so its old value is dead: a typed stand-in is its
            // initial value, and it comes out of the loop as an index.
            let stand_in = self.placeholder(&index_ty(), iter.span);
            self.st.scope.insert(range.var.clone(), stand_in);
        }
        self.while_loop(span, Head::Range(range), body, needed)?;
        self.st.scope.remove(&idx);
        Ok(())
    }

    /// The arguments if `iter` calls the builtin `range`: the checker never
    /// typed the callee (it types a user's own `range`).
    fn range_args<'e>(&self, iter: &'e Expr) -> Option<&'e [Expr]> {
        let ExprKind::Call { callee, args } = &iter.kind else {
            return None;
        };
        match &callee.kind {
            ExprKind::Name(id)
                if self.text(id.span) == "range"
                    && !self.checked.expr_types.contains_key(&callee.span) =>
            {
                Some(args)
            }
            _ => None,
        }
    }
}

/// Why a computed `range` bound is not lowered for execution.
const RANGE_CHECKS: &str = "`sky run` checks a computed `range` bound at run time (a whole number, not too large, a step that is not zero) and SIR has no operation to stop with that message yet";

/// The type of an index: a dimensionless scalar.
fn index_ty() -> Type {
    Type::scalar(Dimension::DIMENSIONLESS)
}
