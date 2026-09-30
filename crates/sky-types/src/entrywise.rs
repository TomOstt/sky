//! Entry-by-entry arithmetic on arrays: `a * b` and `a / b` of two arrays of
//! one shape, and the math functions of arrays (`sqrt(v)`, `atan2(y, x)`).
//! The rules are in the spec's "Arrays: entry by entry".
//!
//! Only equal shapes combine: there is no broadcasting between arrays of
//! different shapes (a number scales an array, as it always did). A shape
//! mismatch is the shape debugger's error, worded for the operator, with a
//! pointer to `@` when the matrix product of the operands would be valid
//! (the one operator a reader may have meant instead).

use sky_ast::{BinaryOp, Expr};
use sky_diag::Span;
use sky_units::{Dimension, MathFn};

use crate::check::{Checker, Ty, ValueTy};
use crate::rules::{elementwise, matmul, TensorType};
use crate::shape::{Shape, ShapeContext};
use crate::tag_ops::{tag_result, TagOp};
use crate::tags::TagSet;

impl Checker<'_> {
    /// Unifies the shapes of two arrays that an entry-by-entry operator
    /// combines (`symbol` is `*`, `/` or a function's name). On a mismatch
    /// the error is reported and `None` returned; otherwise the shape, with
    /// the sizes the unification solved.
    pub(crate) fn unify_entrywise(
        &mut self,
        symbol: &str,
        op_span: Span,
        (le, lv): (&Expr, &ValueTy),
        (re, rv): (&Expr, &ValueTy),
    ) -> Option<Shape> {
        let what = format!("`{}`", self.quoted(le.span.join(re.span)));
        self.set_origin(op_span, what);
        // On an error the unifier is unchanged.
        let checked = elementwise(&mut self.env.shapes, &lv.ty, &rv.ty);
        self.env.shapes.set_origin(None);
        match checked {
            Ok(_) => Some(self.env.shapes.apply(lv.ty.shape())),
            Err(e) => {
                let e = e.in_context(ShapeContext::Operator(symbol.into()));
                let (ll, rl) = (self.this_is(lv), self.this_is(rv));
                let mut d = self
                    .shape_diagnostic(&e, op_span)
                    .with_secondary(le.span, ll)
                    .with_secondary(re.span, rl);
                if symbol == "*" || symbol == "/" {
                    d = d.with_note(format!(
                        "`{symbol}` on two arrays works entry by entry, so both need one shape"
                    ));
                }
                if symbol == "*" && self.matmul_would_fit(lv, rv) {
                    d = d.with_help("for the matrix product use `@`");
                }
                self.error(d);
                None
            }
        }
    }

    /// Whether `l @ r` is valid for these shapes (on a scratch unifier).
    fn matmul_would_fit(&self, l: &ValueTy, r: &ValueTy) -> bool {
        let mut probe = self.env.shapes.clone();
        matmul(&mut probe, &l.ty, &r.ty).is_ok()
    }

    /// `a * b` and `a / b` of two arrays: entry by entry, the dimension of
    /// each entry multiplied or divided as for numbers.
    pub(crate) fn entrywise_product(
        &mut self,
        op: BinaryOp,
        op_span: Span,
        l: (&Expr, &ValueTy),
        r: (&Expr, &ValueTy),
        dim: Dimension,
    ) -> Ty {
        let Some(shape) = self.unify_entrywise(op.symbol(), op_span, l, r) else {
            return Ty::Unknown;
        };
        let rule = if op == BinaryOp::Div {
            TagOp::Quotient
        } else {
            TagOp::Product
        };
        let tags = tag_result(rule, &[l.1.ty.tags(), r.1.ty.tags()]);
        self.tagged_value(dim, shape, tags)
    }

    /// An array type of `shape` claiming `tags` (dropped, never an error,
    /// if the shape cannot carry them).
    pub(crate) fn tagged_value(&mut self, dim: Dimension, shape: Shape, tags: TagSet) -> Ty {
        let ty = TensorType::tagged(&mut self.env.shapes, shape.clone(), tags)
            .unwrap_or_else(|_| TensorType::untagged(shape));
        Ty::value(dim, ty)
    }

    /// The type of `f(args)` when some argument is an array: the arrays
    /// share one shape (numbers broadcast), and the result is an array of
    /// that shape whose entries have the dimension `f` gives for the
    /// arguments' dimensions (`dims`, one per argument, in order). `None`
    /// when no argument is an array.
    pub(crate) fn math_array_type(
        &mut self,
        f: MathFn,
        callee: &Expr,
        args: &[Expr],
        tys: &[Ty],
        dims: &[Dimension],
    ) -> Option<Ty> {
        let arrays: Vec<(&Expr, &ValueTy)> = args
            .iter()
            .zip(tys)
            .filter_map(|(a, t)| match t {
                Ty::Value(v) if !v.is_scalar() => Some((a, v)),
                _ => None,
            })
            .collect();
        let &(first_e, first_v) = arrays.first()?;
        let mut shape = self.env.shapes.apply(first_v.ty.shape());
        let mut tags = tag_result(TagOp::Map1, &[first_v.ty.tags()]);
        if let Some(&(second_e, second_v)) = arrays.get(1) {
            let op_span = callee.span.join(second_e.span);
            let l = (first_e, first_v);
            let r = (second_e, second_v);
            let Some(unified) = self.unify_entrywise(f.name(), op_span, l, r) else {
                return Some(Ty::Unknown);
            };
            shape = unified;
            tags = tag_result(TagOp::Map2, &[first_v.ty.tags(), second_v.ty.tags()]);
        }
        let dim = match f.result_dim(dims) {
            Ok(d) => d,
            Err(e) => {
                self.math_dim_error(f, e, callee, args, dims);
                return Some(Ty::Unknown);
            }
        };
        Some(self.tagged_value(dim, shape, tags))
    }
}
