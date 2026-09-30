//! Running a planned `@` chain on the tree-walker.
//!
//! The checker plans a chain (`sky_types::chain`) and records the plan under
//! the span of the whole expression; this evaluates it. The order is the
//! plan's own, spelled out in `sky_types::chain`'s "Evaluation": every
//! factor is evaluated once, left to right (a factor a solve consumes as
//! `A⁻¹` evaluates only `A`), then the tree is computed children first.
//! Lowering emits the same operations in the same order, so the two engines
//! agree bit for bit.

use sky_ast::Expr;
use sky_types::chain::chain_factors;
use sky_types::{ChainPlan, LeafForm, Node};

use crate::arith::R;
use crate::eval::Interp;
use crate::{array, solve, Value};

impl Interp<'_> {
    /// The value of the chain `lhs @ rhs` (at `e`) by `plan`, or `None` if
    /// the plan does not describe this chain (then the caller evaluates it
    /// as written): the checker's plan must come from the source being run.
    pub(crate) fn run_chain(
        &mut self,
        lhs: &Expr,
        rhs: &Expr,
        plan: &ChainPlan,
    ) -> R<Option<Value>> {
        let exprs = chain_factors(lhs, rhs);
        if exprs.len() != plan.factors.len() {
            return Ok(None);
        }
        let forms = plan.factor_forms();
        // The factors must be what the plan says: same spans, and an `A⁻¹`
        // where it consumes one.
        let mut operands: Vec<Option<&Expr>> = Vec::with_capacity(exprs.len());
        for (i, e) in exprs.iter().enumerate() {
            let pf = &plan.factors[i];
            let operand = self.inverted_operand(e);
            if pf.span != e.span || pf.inverse_operand != operand.map(|a| a.span) {
                return Ok(None);
            }
            if forms[i] != LeafForm::Plain && operand.is_none() {
                return Ok(None);
            }
            operands.push(operand);
        }
        let mut vals: Vec<Option<Value>> = Vec::with_capacity(exprs.len());
        for (i, e) in exprs.iter().enumerate() {
            let v = match (forms[i], operands[i]) {
                (LeafForm::Plain, _) => self.eval(e)?,
                (LeafForm::Deferred, Some(a)) => self.eval(a)?,
                (LeafForm::Transposed, Some(a)) => {
                    let q = self.eval(a)?;
                    array::transpose(&q, e.span)?
                }
                _ => return Ok(None),
            };
            vals.push(Some(v));
        }
        node_value(plan, &plan.root, &mut vals).map(Some)
    }
}

/// Takes factor `t`'s value (each is used once).
fn take(vals: &mut [Option<Value>], t: usize) -> Value {
    vals[t]
        .take()
        .expect("a plan uses each factor once, so its value is still there")
}

/// The value of `n`, children first.
fn node_value(plan: &ChainPlan, n: &Node, vals: &mut Vec<Option<Value>>) -> R<Value> {
    let span = plan.span_of(n);
    let span_of = |m: &Node| plan.span_of(m);
    let operand = |t: usize| {
        plan.factors[t]
            .inverse_operand
            .unwrap_or(plan.factors[t].span)
    };
    match n {
        Node::Leaf { factor, .. } => Ok(take(vals, *factor)),
        Node::Mul { lhs, rhs } => {
            let l = node_value(plan, lhs, vals)?;
            let r = node_value(plan, rhs, vals)?;
            array::matmul(&l, &r, span, span_of(lhs), span_of(rhs))
        }
        Node::Solve { method, inv, rhs } => {
            let a = take(vals, *inv);
            let b = node_value(plan, rhs, vals)?;
            solve::solve(*method, &a, &b, span, operand(*inv), span_of(rhs))
        }
        Node::SolveRight { method, lhs, inv } => {
            let x = node_value(plan, lhs, vals)?;
            let a = take(vals, *inv);
            solve::solve_right(*method, &x, &a, span, span_of(lhs), operand(*inv))
        }
        Node::InvOfProduct { left, right } => {
            // A⁻¹ @ B⁻¹ is (B @ A)⁻¹.
            let a = take(vals, *left);
            let b = take(vals, *right);
            let product = array::matmul(&b, &a, span, operand(*right), operand(*left))?;
            array::inverse(&product, span, span)
        }
    }
}
