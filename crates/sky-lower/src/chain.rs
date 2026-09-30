//! Lowering a planned `@` chain.
//!
//! The checker plans a chain (`sky_types::chain`: the order, solves instead
//! of inverses, `Q⁻¹` as `Qᵀ`, `A⁻¹ @ B⁻¹` as `(B @ A)⁻¹`) and records the
//! plan under the span of the whole expression (`Checked::chains`). This
//! emits it, in the order the plan's "Evaluation" spells out and the
//! tree-walker follows (`sky_interp`'s `chain_exec`): each factor once, left
//! to right, then the tree, children first. The two engines therefore
//! compute the same operations on the same operands, and agree bit for bit.
//!
//! Types are the checker's for the whole chain (the expression's type) and
//! for a factor; an inner product has no expression, so its type is derived
//! from its operands' types: the shape rule of `@`, the product of the
//! dimensions, and no tags beyond what `@` proves.

use sky_ast::Expr;
use sky_ir::{NumberType, OpKind, Repr, Type, ValueId};
use sky_types::chain::{chain_factors, product_shape};
use sky_types::tags::{matmul_tags, preserved_by_inverse};
use sky_types::{ChainPlan, LeafForm, Node, TagSet};

use crate::ctx::{Lower, R};

impl Lower<'_> {
    /// The chain `lhs @ rhs` (at `e`) as its plan says, or `None` if the
    /// checker made no plan for it, or the plan is not for this chain (then
    /// the caller lowers it as written).
    #[inline(never)]
    pub(crate) fn chain_site(&mut self, e: &Expr, lhs: &Expr, rhs: &Expr) -> R<Option<ValueId>> {
        let Some(plan) = self.checked.chains.get(&e.span).cloned() else {
            return Ok(None);
        };
        let exprs = chain_factors(lhs, rhs);
        if exprs.len() != plan.factors.len() {
            return Ok(None);
        }
        let forms = plan.factor_forms();
        let mut operands: Vec<Option<&Expr>> = Vec::with_capacity(exprs.len());
        for (i, f) in exprs.iter().enumerate() {
            let pf = &plan.factors[i];
            let operand = self.inverted_operand(f);
            if pf.span != f.span || pf.inverse_operand != operand.map(|a| a.span) {
                return Ok(None);
            }
            if forms[i] != LeafForm::Plain && operand.is_none() {
                return Ok(None);
            }
            operands.push(operand);
        }
        let mut vals: Vec<Option<ValueId>> = Vec::with_capacity(exprs.len());
        for (i, f) in exprs.iter().enumerate() {
            let v = match (forms[i], operands[i]) {
                (LeafForm::Plain, _) => self.expr(f)?,
                (LeafForm::Deferred, Some(a)) => self.expr(a)?,
                (LeafForm::Transposed, Some(a)) => {
                    let q = self.expr(a)?;
                    let ty = self.ty_of(f)?;
                    self.op1s(OpKind::Transpose, vec![q], vec![a.span], ty, f.span)
                }
                _ => return Ok(None),
            };
            vals.push(Some(v));
        }
        let ty = self.ty_of(e)?;
        Ok(Some(self.chain_node(
            &plan,
            &plan.root,
            &mut vals,
            Some(ty),
        )))
    }

    fn number(&self, v: ValueId) -> Option<NumberType> {
        match self.st.b.ty(v) {
            Type::Number(n) => Some(n.clone()),
            _ => None,
        }
    }

    /// The type of `a @ b`.
    fn product_type(&self, a: ValueId, b: ValueId) -> Type {
        let (Some(x), Some(y)) = (self.number(a), self.number(b)) else {
            return Type::Unknown;
        };
        let (Ok(dim), Some(shape)) = (x.dim.checked_mul(y.dim), product_shape(&x.shape, &y.shape))
        else {
            return Type::Unknown;
        };
        let tags = if x.shape.rank() == 2 && y.shape.rank() == 2 {
            matmul_tags(x.tags, y.tags)
        } else {
            TagSet::EMPTY
        };
        Type::Number(NumberType {
            dim,
            shape,
            tags,
            repr: Repr::Unchosen,
        })
    }

    /// The type of `x` divided by `a`: `x`'s shape, the dimension of `x`
    /// over `a`'s, no tags.
    fn quotient_type(&self, x: ValueId, a: ValueId) -> Type {
        let (Some(x), Some(a)) = (self.number(x), self.number(a)) else {
            return Type::Unknown;
        };
        match x.dim.checked_div(a.dim) {
            Ok(dim) => Type::Number(NumberType {
                dim,
                shape: x.shape,
                tags: TagSet::EMPTY,
                repr: Repr::Unchosen,
            }),
            Err(_) => Type::Unknown,
        }
    }

    /// Emits `n`, children first. `ty` is the checker's type for the whole
    /// chain (the root); an inner node's type is derived.
    fn chain_node(
        &mut self,
        plan: &ChainPlan,
        n: &Node,
        vals: &mut Vec<Option<ValueId>>,
        ty: Option<Type>,
    ) -> ValueId {
        fn take(vals: &mut [Option<ValueId>], t: usize) -> ValueId {
            vals[t]
                .take()
                .expect("a plan uses each factor once, so its value is still there")
        }
        let span = plan.span_of(n);
        let operand = |t: usize| {
            plan.factors[t]
                .inverse_operand
                .unwrap_or(plan.factors[t].span)
        };
        match n {
            Node::Leaf { factor, .. } => take(vals, *factor),
            Node::Mul { lhs, rhs } => {
                let l = self.chain_node(plan, lhs, vals, None);
                let r = self.chain_node(plan, rhs, vals, None);
                let ty = ty.unwrap_or_else(|| self.product_type(l, r));
                let spans = vec![plan.span_of(lhs), plan.span_of(rhs)];
                self.op1s(OpKind::MatMul, vec![l, r], spans, ty, span)
            }
            Node::Solve { method, inv, rhs } => {
                let a = take(vals, *inv);
                let b = self.chain_node(plan, rhs, vals, None);
                let ty = ty.unwrap_or_else(|| self.quotient_type(b, a));
                let spans = vec![operand(*inv), plan.span_of(rhs)];
                let kind = OpKind::Solve { method: *method };
                self.op1s(kind, vec![a, b], spans, ty, span)
            }
            Node::SolveRight { method, lhs, inv } => {
                let x = self.chain_node(plan, lhs, vals, None);
                let a = take(vals, *inv);
                let ty = ty.unwrap_or_else(|| self.quotient_type(x, a));
                let spans = vec![plan.span_of(lhs), operand(*inv)];
                let kind = OpKind::SolveRight { method: *method };
                self.op1s(kind, vec![x, a], spans, ty, span)
            }
            Node::InvOfProduct { left, right } => {
                // A⁻¹ @ B⁻¹ is (B @ A)⁻¹.
                let a = take(vals, *left);
                let b = take(vals, *right);
                let product_ty = self.product_type(b, a);
                let spans = vec![operand(*right), operand(*left)];
                let product =
                    self.op1s(OpKind::MatMul, vec![b, a], spans, product_ty.clone(), span);
                let ty = ty.unwrap_or_else(|| match &product_ty {
                    Type::Number(p) => match p.dim.inverse() {
                        Ok(dim) => Type::Number(NumberType {
                            dim,
                            shape: p.shape.clone(),
                            tags: preserved_by_inverse(p.tags),
                            repr: Repr::Unchosen,
                        }),
                        Err(_) => Type::Unknown,
                    },
                    _ => Type::Unknown,
                });
                self.op1s(OpKind::Inverse, vec![product], vec![span], ty, span)
            }
        }
    }
}
