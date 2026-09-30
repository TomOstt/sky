//! The checker's side of chain planning ([`crate::chain`]): read the
//! factors' types, plan, record the plan for the executors, and write the
//! `--explain` note.

use std::sync::Arc;

use sky_ast::{Expr, ExprKind};
use sky_diag::{Diagnostic, Span};

use crate::chain::{
    chain_factors, nested_chain_spans, plan_chain, ChainDecision, ChainFactor, InverseFactor,
    CHAIN_RULE,
};
use crate::check::{strip_parens, Checker, Ty, MAX_STORED_REWRITES};
use crate::flops::grouped;

impl Checker<'_> {
    /// Plans the chain `lhs @ rhs` (the expression at `span`), whose
    /// operands are typed, and records the plan and its explanation. Runs
    /// after the expression's type is known and never changes it.
    ///
    /// A chain nested in the left spine of a longer one is planned first (the
    /// walk types the inside before the outside); the outer plan covers it,
    /// so the inner records are withdrawn.
    pub(crate) fn plan_chain_site(&mut self, span: Span, lhs: &Expr, rhs: &Expr) {
        self.chains.remove(&span);
        self.chain_notes.retain(|(s, _)| *s != span);
        let exprs = chain_factors(lhs, rhs);
        let Some(factors) = self.chain_factor_info(&exprs) else {
            return;
        };
        let Some(decision) = plan_chain(&factors) else {
            return;
        };
        for inner in nested_chain_spans(lhs) {
            self.chains.remove(&inner);
            self.chain_notes.retain(|(s, _)| *s != inner);
            // The per-site Cholesky rewrite of an inner `A⁻¹ @ b`.
            self.solves.remove(&inner);
            self.rewrites.retain(|r| r.span != inner);
        }
        let note = self.chain_note(span, &decision, &factors);
        self.chains.insert(span, Arc::new(decision.plan));
        if self.chain_notes.len() < MAX_STORED_REWRITES {
            self.chain_notes.push((span, note));
        } else {
            self.chain_notes_omitted += 1;
        }
    }

    /// The planner's view of each factor, or `None` if one is not a typed
    /// vector or matrix.
    fn chain_factor_info(&self, exprs: &[&Expr]) -> Option<Vec<ChainFactor>> {
        let mut out = Vec::with_capacity(exprs.len());
        for e in exprs {
            let Ty::Value(v) = self.env.resolve(self.expr_types.get(&e.span)?) else {
                return None;
            };
            let shape = self.env.shapes.apply(v.ty.shape());
            if !(1..=2).contains(&shape.rank()) {
                return None;
            }
            let inner = strip_parens(e);
            let inverted = match &inner.kind {
                ExprKind::Inverse { operand, .. } => Some(&**operand),
                ExprKind::Call { callee, args } => self.inv_call(callee, args),
                _ => None,
            };
            let inverse = match inverted {
                Some(operand) => {
                    let Ty::Value(a) = self.env.resolve(self.expr_types.get(&operand.span)?) else {
                        return None;
                    };
                    Some(InverseFactor {
                        operand_span: operand.span,
                        operand_atom: self.operand_text(operand),
                        tags: a.ty.tags().closure(),
                    })
                }
                None => None,
            };
            out.push(ChainFactor {
                span: e.span,
                text: self.quoted(e.span).to_string(),
                atom: self.operand_text(e),
                dim: v.dim,
                shape,
                tags: v.ty.tags().closure(),
                inverse,
            });
        }
        Some(out)
    }

    /// The `--explain` note for a planned chain.
    fn chain_note(&self, span: Span, d: &ChainDecision, factors: &[ChainFactor]) -> Diagnostic {
        let written = self.quoted(span);
        let mut note = Diagnostic::note(
            format!("[{CHAIN_RULE}] `{written}` is rewritten to `{}`", d.text),
            span,
        )
        .with_note("applied by the checker: the tree-walker and SIR both follow this choice");
        let list: Vec<String> = factors
            .iter()
            .map(|f| match &f.inverse {
                Some(i) => format!("{}⁻¹ is {}", i.operand_atom, f.shape),
                None => format!("{} is {}", f.atom, f.shape),
            })
            .collect();
        note = note.with_note(list.join(", "));
        for reason in &d.reasons {
            note = note.with_note(reason.clone());
        }
        if d.reassociated {
            note = note.with_note(
                "an unparenthesized `@` chain has no fixed association: the compiler picks the cheapest (spec, \"Algorithm choice\"); write parentheses to force an order",
            );
        }
        let saving = match (d.before.value(), d.after.value()) {
            (Some(b), Some(a)) if a > 0 && b > a => {
                let ratio = b as f64 / a as f64;
                if ratio < 10.0 {
                    format!(", about {ratio:.1}x fewer")
                } else {
                    format!(", about {}x fewer", grouped(ratio.round() as u128))
                }
            }
            _ => String::new(),
        };
        let unit = |v: Option<u128>| if v == Some(1) { "flop" } else { "flops" };
        note = note.with_note(format!(
            "estimated cost: {} {} as written, {} {} after the rewrite{saving}",
            d.before,
            unit(d.before.value()),
            d.after,
            unit(d.after.value()),
        ));
        note.with_note(format!(
            "not bit-identical to the written form: {} (spec, \"Algorithm choice\")",
            d.rounding.join("; ")
        ))
    }
}
