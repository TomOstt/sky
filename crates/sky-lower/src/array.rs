//! Lowering arrays: literals, indexing and slicing, the builtins `zeros`,
//! `identity` and `len`, and the Cholesky solve the checker records for
//! `A⁻¹ @ b`.
//!
//! As everywhere in lowering, the result type of an operation is the type
//! the checker recorded for the expression. The array operations are
//! specified in `sky_ir::op`.
//!
//! # Literals and tags
//!
//! A literal has no tags of its own: the checker types `[[4, 1], [1, 3]]`
//! as a plain matrix. A *declaration* can promise tags that the checker
//! proves from the entries (`P: PositiveDefinite Matrix[2, 2] = [[4, 1],
//! [1, 3]]`), and uses of `P` are typed with them. The SSA value bound to
//! `P` must carry the same tags, or `P⁻¹ @ b` could not become a Cholesky
//! solve and a function reading `P` would take an untagged parameter. So a
//! pre-pass ([`Lower::collect_literal_bindings`]) finds the literals that
//! are the value of a declaration, and their result type is the
//! declaration's (`binding_types`) when it agrees with the literal's own in
//! dimension and shape. The pre-pass walks the statements that can hold a
//! declaration (bodies of `def`, `if` and `while`); a new kind of compound
//! statement needs an arm there, or a declared literal inside it lowers
//! untagged.
//!
//! # `A⁻¹ @ b`
//!
//! `sky run` factors `A` and solves wherever the checker recorded the
//! expression's span in `Checked::cholesky_solves`, and forms `A⁻¹` and
//! multiplies everywhere else. Lowering makes the same choice: a recorded
//! `A⁻¹ @ b` becomes `solve[cholesky](A, b)` directly, rather than
//! `matmul(inverse(A), b)` for the Cholesky pass in `sky-opt` to rewrite.
//! The two would compute the same thing only if that pass ran, and it is
//! optional; the program's meaning (the last bits of its output, and which
//! error a non-symmetric matrix gives) must not depend on it. Anything the
//! checker did not record stays an `inverse` and a `matmul`, which the pass
//! is free to rewrite whenever it can prove the tag.

use sky_ast::{Expr, ExprKind, IndexItem, Stmt, StmtKind};
use sky_diag::Span;
use sky_ir::{OpKind, SliceAxis, Type, ValueId};
use sky_types::{ShapeDim, Tag, Ty};
use sky_units::Dimension;

use crate::ctx::{Lower, R};
use crate::LowerError;

/// `e` without the parentheses around it.
fn strip_parens(mut e: &Expr) -> &Expr {
    while let ExprKind::Paren(inner) = &e.kind {
        e = inner;
    }
    e
}

impl Lower<'_> {
    /// Records, for every literal that is the value of a declaration with a
    /// type (`x: T = [...]`, anywhere in the file), the span of the name
    /// it declares. See the module docs.
    pub(crate) fn collect_literal_bindings(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            match &s.kind {
                StmtKind::Binding {
                    name,
                    value: Some(value),
                    ..
                } => {
                    let inner = strip_parens(value);
                    if matches!(inner.kind, ExprKind::Array(_)) {
                        self.literal_bindings.insert(inner.span, name.span);
                    }
                }
                StmtKind::Def { body, .. } | StmtKind::While { body, .. } => {
                    self.collect_literal_bindings(&body.stmts);
                }
                StmtKind::If {
                    branches,
                    else_block,
                } => {
                    for (_, b) in branches {
                        self.collect_literal_bindings(&b.stmts);
                    }
                    if let Some(b) = else_block {
                        self.collect_literal_bindings(&b.stmts);
                    }
                }
                _ => {}
            }
        }
    }

    /// The type of a literal: the checker's, with the tags of the
    /// declaration it initializes when there is one.
    fn literal_type(&self, e: &Expr) -> R<Type> {
        let mut ty = self.ty_of(e)?;
        let declared = self
            .literal_bindings
            .get(&e.span)
            .and_then(|name| self.checked.binding_types.get(name))
            .map(|t| self.checked.resolve_ty(t));
        if let (Type::Number(n), Some(Ty::Value(v))) = (&mut ty, declared) {
            if v.dim == n.dim && *v.ty.shape() == n.shape {
                n.tags = v.ty.tags().closure();
            }
        }
        Ok(ty)
    }

    /// `[a, b, c]` and `[[a, b], [c, d]]`: the entries are lowered left to
    /// right, row by row, and `array.literal` arranges them. Like the
    /// interpreter, a literal is a matrix when all its elements are
    /// literals themselves.
    #[inline(never)]
    pub(crate) fn array_literal(&mut self, e: &Expr, elems: &[Expr]) -> R<ValueId> {
        let matrix =
            !elems.is_empty() && elems.iter().all(|x| matches!(x.kind, ExprKind::Array(_)));
        let rows: Vec<&[Expr]> = if matrix {
            elems
                .iter()
                .filter_map(|row| match &row.kind {
                    ExprKind::Array(items) => Some(items.as_slice()),
                    _ => None,
                })
                .collect()
        } else {
            vec![elems]
        };
        let width = rows[0].len();
        if width == 0 || rows.iter().any(|r| r.len() != width) {
            return Err(LowerError::new(
                if width == 0 {
                    "an array literal needs at least one element"
                } else {
                    "matrix rows must have the same length"
                },
                e.span,
            ));
        }
        // A big literal of plain numbers is one `const.array`: its entries
        // are read from the source in one pass (the checker's own
        // `entry_value`, so the same `f64`s), instead of one operation and
        // one operand each. A literal with any other entry, or a number too
        // large to hold, takes the general path below.
        if rows.len() * width >= OpKind::CONST_ARRAY_MIN {
            if let Some(v) = self.const_array(e, &rows, width, matrix)? {
                return Ok(v);
            }
        }
        let mut operands = Vec::with_capacity(rows.len() * width);
        let mut spans = Vec::with_capacity(rows.len() * width);
        for row in rows {
            for x in row {
                operands.push(self.expr(x)?);
                spans.push(x.span);
            }
        }
        let ty = self.literal_type(e)?;
        let cols = matrix.then_some(width as u64);
        Ok(self.op1s(OpKind::ArrayLiteral { cols }, operands, spans, ty, e.span))
    }

    /// The literal as one `const.array`, if every entry is a numeric
    /// constant (`sky_types::entry_value`) and the checker typed the
    /// literal as a number; `None` otherwise.
    fn const_array(
        &mut self,
        e: &Expr,
        rows: &[&[Expr]],
        width: usize,
        matrix: bool,
    ) -> R<Option<ValueId>> {
        let mut bits = Vec::with_capacity(rows.len() * width);
        for row in rows {
            for x in *row {
                match sky_types::entry_value(self.src, x) {
                    Some(v) => bits.push(v.to_bits()),
                    None => return Ok(None),
                }
            }
        }
        let ty = self.literal_type(e)?;
        if !matches!(ty, Type::Number(_)) {
            return Ok(None);
        }
        let cols = matrix.then_some(width as u64);
        let kind = OpKind::ConstArray {
            cols,
            bits: bits.into(),
        };
        Ok(Some(self.op1(kind, Vec::new(), ty, e.span)))
    }

    /// `base[items]`: `index` when every item is a position, `slice` when
    /// one is a range. The base, then each item's expressions, are lowered
    /// left to right, as the interpreter evaluates them.
    #[inline(never)]
    pub(crate) fn index_expr(&mut self, e: &Expr, base: &Expr, items: &[IndexItem]) -> R<ValueId> {
        let mut operands = vec![self.expr(base)?];
        let mut spans = vec![base.span];
        let mut axes = Vec::with_capacity(items.len());
        // The whole of each item (`1:3`), which a slice's errors point at.
        let mut item_spans = Vec::with_capacity(items.len());
        for item in items {
            match item {
                IndexItem::Expr(x) => {
                    operands.push(self.expr(x)?);
                    spans.push(x.span);
                    axes.push(SliceAxis::At);
                    item_spans.push(x.span);
                }
                IndexItem::Slice {
                    start, stop, span, ..
                } => {
                    for bound in [start, stop].into_iter().flatten() {
                        operands.push(self.expr(bound)?);
                        spans.push(bound.span);
                    }
                    axes.push(SliceAxis::Range {
                        start: start.is_some(),
                        stop: stop.is_some(),
                    });
                    item_spans.push(*span);
                }
            }
        }
        let ty = self.ty_of(e)?;
        let kind = if axes.iter().all(|a| matches!(a, SliceAxis::At)) {
            OpKind::Index
        } else {
            // `slice` also records the span of each axis's item.
            spans.extend(item_spans);
            OpKind::Slice { axes }
        };
        Ok(self.op1s(kind, operands, spans, ty, e.span))
    }

    /// The builtins `zeros`, `identity` and `len`, or `None` if `name` is
    /// not one (or has the wrong number of arguments, which the checker
    /// rejected). `zeros` and `identity` take their sizes as operands, so
    /// a size the checker could not read as a constant still works.
    #[inline(never)]
    pub(crate) fn array_builtin(
        &mut self,
        e: &Expr,
        name: &str,
        args: &[Expr],
    ) -> R<Option<ValueId>> {
        let arity_ok = match name {
            "zeros" => matches!(args.len(), 1 | 2),
            "identity" | "len" => args.len() == 1,
            _ => false,
        };
        if !arity_ok {
            return Ok(None);
        }
        let mut vals = Vec::with_capacity(args.len());
        for a in args {
            vals.push(self.expr(a)?);
        }
        let spans: Vec<_> = args.iter().map(|a| a.span).collect();
        if name == "len" {
            return Ok(Some(self.len_of(e, vals[0], spans[0])));
        }
        let kind = if name == "zeros" {
            OpKind::Zeros
        } else {
            OpKind::Identity
        };
        let ty = self.ty_of(e)?;
        Ok(Some(self.op1s(kind, vals, spans, ty, e.span)))
    }

    /// `len(x)` of the lowered `x`: a constant when the first axis's length
    /// is known from `x`'s type (`x` itself is still evaluated, for its
    /// failures and effects), else a `len` operation.
    fn len_of(&mut self, e: &Expr, x: ValueId, xspan: Span) -> ValueId {
        let ty = Type::scalar(Dimension::DIMENSIONLESS);
        if let Type::Number(n) = self.st.b.ty(x) {
            if let Some(ShapeDim::Known(len)) = n.shape.dims().first() {
                return self.const_float(*len as f64, ty, e.span);
            }
        }
        self.op1s(OpKind::Len, vec![x], vec![xspan], ty, e.span)
    }

    /// `A⁻¹ @ b` at a site the checker recorded in
    /// `Checked::cholesky_solves`: `solve[cholesky](A, b)`. `None` if the
    /// expression is not of that form (then it lowers as any `@`).
    #[inline(never)]
    pub(crate) fn cholesky_site(&mut self, e: &Expr, lhs: &Expr, rhs: &Expr) -> R<Option<ValueId>> {
        if !self.checked.cholesky_solves.contains(&e.span) {
            return Ok(None);
        }
        let Some(a_expr) = self.inverted_operand(lhs) else {
            return Ok(None);
        };
        let a = self.expr(a_expr)?;
        let b = self.expr(rhs)?;
        let ty = self.ty_of(e)?;
        let known_pd = matches!(self.st.b.ty(a), Type::Number(n) if n.tags.closure().contains(Tag::PositiveDefinite));
        if known_pd {
            let kind = OpKind::Solve {
                method: sky_ir::SolveMethod::Cholesky,
            };
            let spans = vec![a_expr.span, rhs.span];
            return Ok(Some(self.op1s(kind, vec![a, b], spans, ty, e.span)));
        }
        // The value's type does not say PositiveDefinite although the
        // checker's does: only if the pre-pass missed the declaration (see
        // the module docs). Keep the general form, which is the same up to
        // rounding, rather than emit a solve the verifier would reject.
        let inv_ty = self.ty_of(lhs)?;
        let inv = self.op1s(
            OpKind::Inverse,
            vec![a],
            vec![a_expr.span],
            inv_ty,
            lhs.span,
        );
        let spans = vec![lhs.span, rhs.span];
        Ok(Some(self.op1s(
            OpKind::MatMul,
            vec![inv, b],
            spans,
            ty,
            e.span,
        )))
    }

    /// `A` if `e` is `A⁻¹` or `inv(A)` (the builtin, not a user's own
    /// `inv`), possibly parenthesized: the forms the checker rewrites.
    pub(crate) fn inverted_operand<'x>(&self, e: &'x Expr) -> Option<&'x Expr> {
        match &strip_parens(e).kind {
            ExprKind::Inverse { operand, .. } => Some(operand),
            ExprKind::Call { callee, args } => match (&callee.kind, args.as_slice()) {
                (ExprKind::Name(id), [arg])
                    if self.text(id.span) == "inv"
                        && !self.checked.expr_types.contains_key(&callee.span) =>
                {
                    Some(arg)
                }
                _ => None,
            },
            _ => None,
        }
    }
}
