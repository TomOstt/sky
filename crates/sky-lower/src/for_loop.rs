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
//! is larger than 2^53 in size, a bound with a unit, and a zero step. The
//! checker makes those checks for literals; for any other bound or step
//! lowering makes them at run time, as the interpreter does: an
//! `expect.range[bound]` right after each argument is evaluated, left to
//! right, and an `expect.range[step]` on the step, with the interpreter's
//! own errors on the argument's span.
//!
//! `for x in v` over an array is the same loop over its first axis, as
//! the spec defines it (`for i in range(len(v))` with `x = v[i]`):
//!
//! ```text
//! for x in v:                         <a> = v
//!     body                    ==>     <idx> = 0
//!                                     while <idx> < len(<a>):
//!                                         x = <a>[<idx>]
//!                                         <idx> = <idx> + 1
//!                                         body
//! ```
//!
//! The iterable is evaluated once, before the loop; an array is never
//! empty, so the loop runs. Over a vector `x` is an entry, over a matrix a
//! row. An iterable whose type the checker does not know (a parameter
//! without an annotation: it may be a scalar or a string, which the
//! interpreter rejects with its own error) is a [`LowerError`].
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
use sky_ir::infer::result_type;
use sky_ir::{BoolContext, CmpOp, OpKind, RangePart, Type, ValueId};
use sky_types::{range_constants, range_runs_at_least_once, ShapeDim};
use sky_units::Dimension;

use crate::analysis::is_forever;
use crate::ctx::{Lower, Needed, R};
use crate::LowerError;

/// What starts a loop: the condition of a `while`, or a counting header.
pub(crate) enum Head<'e> {
    While(&'e Expr),
    Range(Box<Range>),
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
    /// For `for x in v`: the array and the type of its entries (or rows),
    /// and the loop variable is `v[<idx>]` rather than the index.
    over: Option<(ValueId, Type)>,
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
            Head::While(cond) => {
                let c = self.expr(cond)?;
                Ok(self.bool_operand(BoolContext::While, cond, c))
            }
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
        let item = match &r.over {
            Some((array, ty)) => self.op1s(
                OpKind::Index,
                vec![*array, idx],
                vec![r.span, r.span],
                ty.clone(),
                r.span,
            ),
            None => idx,
        };
        self.bind(&r.var, item);
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
            return self.array_for(span, var, iter, body, needed);
        };
        let (start, stop, step) = match args {
            [stop] => (None, stop, None),
            [start, stop] => (Some(start), stop, None),
            [start, stop, step] => (Some(start), stop, Some(step)),
            _ => return self.err("`range` takes 1 to 3 arguments", iter.span),
        };
        // Evaluated once, before the loop, left to right; the interpreter
        // checks each one as it reads it.
        let start_v = match start {
            Some(a) => self.range_arg(a)?,
            None => self.const_float(0.0, index_ty(), iter.span),
        };
        let stop_v = self.range_arg(stop)?;
        let step_v = match step {
            Some(a) => {
                let v = self.range_arg(a)?;
                self.range_step(a, v)
            }
            None => self.const_float(1.0, index_ty(), iter.span),
        };
        let constants = range_constants(self.src, args);
        let step_sign = match (step, constants) {
            (None, _) => Some(1),
            (Some(_), Some((_, _, s))) => Some(s.signum()),
            // A constant step has a known direction even when a bound is
            // computed (`range(0, len(v), 2)`): the read is `range(step)`.
            (Some(a), None) => {
                range_constants(self.src, std::slice::from_ref(a)).map(|(_, s, _)| s.signum())
            }
        };
        let idx = format!("<i{}>", self.st.loops + 1);
        let range = Range {
            idx: idx.clone(),
            var: self.text(var.span).to_string(),
            stop: stop_v,
            step: step_v,
            step_sign,
            runs: constants.is_some_and(range_runs_at_least_once),
            over: None,
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
        self.while_loop(span, Head::Range(Box::new(range)), body, needed)?;
        self.st.scope.remove(&idx);
        Ok(())
    }

    /// `for var in iter` over an array (see the module docs).
    fn array_for(
        &mut self,
        span: Span,
        var: &Ident,
        iter: &Expr,
        body: &Block,
        needed: &Needed,
    ) -> R<()> {
        let array = self.expr(iter)?;
        let array_ty = self.st.b.ty(array).clone();
        let item_ty = match &array_ty {
            Type::Number(n) if matches!(n.shape.rank(), 1 | 2) => {
                result_type(&OpKind::Index, &[&array_ty, &index_ty()])
            }
            _ => None,
        };
        let Some(item_ty) = item_ty else {
            return Err(LowerError::new(
                "iteration over a value whose type is not known before the program runs does \
                 not lower to SIR yet",
                iter.span,
            ));
        };
        let stop = match &array_ty {
            Type::Number(n) => match n.shape.dims().first() {
                Some(ShapeDim::Known(len)) => self.const_float(*len as f64, index_ty(), iter.span),
                _ => self.op1s(
                    OpKind::Len,
                    vec![array],
                    vec![iter.span],
                    index_ty(),
                    iter.span,
                ),
            },
            _ => unreachable!("an array type was matched above"),
        };
        let start = self.const_float(0.0, index_ty(), iter.span);
        let step = self.const_float(1.0, index_ty(), iter.span);
        let idx = format!("<i{}>", self.st.loops + 1);
        let range = Range {
            idx: idx.clone(),
            var: self.text(var.span).to_string(),
            stop,
            step,
            step_sign: Some(1),
            runs: true,
            over: Some((array, item_ty.clone())),
            span: iter.span,
        };
        self.st.scope.insert(idx.clone(), start);
        // The loop always runs and overwrites `var` first: its value on
        // entry is dead, and the first item is a stand-in of the right type.
        let first = self.op1s(
            OpKind::Index,
            vec![array, start],
            vec![iter.span, iter.span],
            item_ty,
            iter.span,
        );
        self.st.scope.insert(range.var.clone(), first);
        self.while_loop(span, Head::Range(Box::new(range)), body, needed)?;
        self.st.scope.remove(&idx);
        Ok(())
    }

    /// One argument of `range`, evaluated, and checked at run time unless
    /// it is a literal (the checker did): a whole number of magnitude at
    /// most 2^53 with no unit, as an `expect.range[bound]` on the argument.
    fn range_arg(&mut self, a: &Expr) -> R<ValueId> {
        let v = self.expr(a)?;
        if sky_types::const_exponent(self.src, a).is_some() {
            return Ok(v);
        }
        Ok(self.op1(
            OpKind::ExpectRange {
                part: RangePart::Bound,
            },
            vec![v],
            index_ty(),
            a.span,
        ))
    }

    /// The step `v` of `range(.., step)`, which is not zero: an
    /// `expect.range[step]` unless the step is a literal.
    fn range_step(&mut self, a: &Expr, v: ValueId) -> ValueId {
        if sky_types::const_exponent(self.src, a).is_some() {
            return v;
        }
        self.op1(
            OpKind::ExpectRange {
                part: RangePart::Step,
            },
            vec![v],
            index_ty(),
            a.span,
        )
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

/// The type of an index: a dimensionless scalar.
fn index_ty() -> Type {
    Type::scalar(Dimension::DIMENSIONLESS)
}
