//! Lowering expressions. Every expression becomes SSA values in the
//! current block; the operation's result type is the checker's type for the
//! expression (`Checked::expr_types`), never recomputed. Where the checker
//! says "unknown" but the operation's result is fixed by its kind (a
//! comparison is a Bool, a string literal a Str, `print` gives Nothing),
//! that fixed type is used.

use sky_ast::{BinaryOp, BoolOp, Expr, ExprKind, Ident, Literal, UnaryOp};
use sky_diag::Span;
use sky_interp::{number_literal, unit_literal_si};
use sky_ir::{BlockId, CmpOp, OpId, OpKind, Type, ValueId};
use sky_syntax::lexer::unescape_str;
use sky_types::{const_exponent, FuncSig, Ty};
use sky_units::lookup_constant;

use crate::ctx::{join_ty, Lower, R};
use crate::LowerError;

fn cmp_of(op: BinaryOp) -> Option<CmpOp> {
    Some(match op {
        BinaryOp::Eq => CmpOp::Eq,
        BinaryOp::Ne => CmpOp::Ne,
        BinaryOp::Lt => CmpOp::Lt,
        BinaryOp::Le => CmpOp::Le,
        BinaryOp::Gt => CmpOp::Gt,
        BinaryOp::Ge => CmpOp::Ge,
        _ => return None,
    })
}

impl Lower<'_> {
    /// Lowers `e` and returns the value it computes.
    pub(crate) fn expr(&mut self, e: &Expr) -> R<ValueId> {
        match &e.kind {
            ExprKind::Literal(lit) => self.literal(e, lit),
            ExprKind::Name(id) => self.name(e, id),
            ExprKind::UnitLiteral { value, unit } => {
                let (Literal::Int(vspan) | Literal::Float(vspan)) = value else {
                    return self.err("a unit needs a number", e.span);
                };
                let (si, _) =
                    unit_literal_si(self.src, *vspan, unit, e.span).map_err(LowerError::from)?;
                let ty = self.ty_of(e)?;
                Ok(self.const_float(si, ty, e.span))
            }
            ExprKind::Paren(inner) => self.expr(inner),
            ExprKind::Unary { op, operand, .. } => self.unary(e, *op, operand),
            ExprKind::Inverse { operand, .. } => {
                let a = self.expr(operand)?;
                let ty = self.ty_of(e)?;
                Ok(self.op1(OpKind::Inverse, vec![a], ty, e.span))
            }
            ExprKind::Binary { op, lhs, rhs, .. } => {
                if *op == BinaryOp::MatMul {
                    if let Some(v) = self.cholesky_site(e, lhs, rhs)? {
                        return Ok(v);
                    }
                }
                let a = self.expr(lhs)?;
                let ty = if op.is_comparison() {
                    Type::Bool
                } else {
                    self.ty_of(e)?
                };
                self.binary_with(*op, a, rhs, ty, e.span)
            }
            ExprKind::BoolOp { op, operands, .. } => self.bool_op(e.span, *op, operands),
            ExprKind::Call { callee, args } => self.call(e, callee, args),
            ExprKind::IfElse { then, cond, else_ } => self.if_else(e, then, cond, else_),
            // `v as unit` is a `show` op: the unit as text, the value in SI.
            // It is a display value, only ever an operand of `print`.
            ExprKind::As { value, unit, .. } => {
                let a = self.expr(value)?;
                let unit = self.text(unit.span()).to_string();
                Ok(self.op1(OpKind::Show { unit }, vec![a], Type::Str, e.span))
            }
            ExprKind::Index { base, items } => self.index_expr(e, base, items),
            ExprKind::Array(elems) => self.array_literal(e, elems),
            ExprKind::Attribute { name, .. } => Err(LowerError::new(
                format!("attribute `{}` is not supported yet", self.text(name.span)),
                e.span,
            )
            .with_note(ARRAYS)),
        }
    }

    pub(crate) fn const_float(&mut self, x: f64, ty: Type, span: Span) -> ValueId {
        self.op1(
            OpKind::ConstFloat { bits: x.to_bits() },
            Vec::new(),
            ty,
            span,
        )
    }

    fn literal(&mut self, e: &Expr, lit: &Literal) -> R<ValueId> {
        match lit {
            Literal::Int(span) | Literal::Float(span) => {
                let x = number_literal(self.src, *span).map_err(LowerError::from)?;
                let ty = self.ty_of(e)?;
                Ok(self.const_float(x, ty, e.span))
            }
            Literal::Str(span) => {
                let s = unescape_str(self.text(*span));
                Ok(self.op1(OpKind::ConstStr(s), Vec::new(), Type::Str, e.span))
            }
            Literal::Bool { value, .. } => Ok(self.const_bool(*value, e.span)),
        }
    }

    /// A name read: the current value of a variable (a function's
    /// parameters and module names it reads are in scope from its start),
    /// else a predefined constant, which is a literal.
    fn name(&mut self, e: &Expr, id: &Ident) -> R<ValueId> {
        let name = self.text(id.span);
        // Also rejects a function used as a value.
        let ty = self.ty_of(e)?;
        if let Some(v) = self.st.scope.get(name) {
            return Ok(*v);
        }
        if let Some(c) = lookup_constant(name) {
            return Ok(self.const_float(c.value, ty, e.span));
        }
        self.err(
            format!(
                "cannot find `{}` in this scope",
                sky_diag::display_capped(name)
            ),
            e.span,
        )
    }

    fn unary(&mut self, e: &Expr, op: UnaryOp, operand: &Expr) -> R<ValueId> {
        match op {
            UnaryOp::Neg => {
                let a = self.expr(operand)?;
                let ty = self.ty_of(e)?;
                Ok(self.op1(OpKind::Neg, vec![a], ty, e.span))
            }
            // `+x` is `x`.
            UnaryOp::Pos => self.expr(operand),
            UnaryOp::Not => {
                let a = self.expr(operand)?;
                Ok(self.op1(OpKind::Not, vec![a], Type::Bool, e.span))
            }
            UnaryOp::Partial | UnaryOp::Nabla => Err(LowerError::new(
                "derivative operators are not supported yet",
                e.span,
            )
            .with_note("neither `sky run` nor SIR can evaluate `∂` or `∇` yet")),
        }
    }

    /// `a op rhs` with `a` already lowered. A constant exponent
    /// (`sky_types::const_exponent`, the checker's own reading) becomes an
    /// attribute of `pow.const` and its expression is not lowered.
    pub(crate) fn binary_with(
        &mut self,
        op: BinaryOp,
        a: ValueId,
        rhs: &Expr,
        ty: Type,
        span: Span,
    ) -> R<ValueId> {
        if op == BinaryOp::Pow {
            if let Some(exponent) = const_exponent(self.src, rhs) {
                return Ok(self.op1(OpKind::PowConst { exponent }, vec![a], ty, span));
            }
        }
        let b = self.expr(rhs)?;
        let kind = match op {
            BinaryOp::Add => OpKind::Add,
            BinaryOp::Sub => OpKind::Sub,
            BinaryOp::Mul => OpKind::Mul,
            BinaryOp::Div => OpKind::Div,
            BinaryOp::Pow => OpKind::Pow,
            BinaryOp::MatMul => OpKind::MatMul,
            other => match cmp_of(other) {
                Some(c) => OpKind::Cmp(c),
                None => return self.err("internal error: unknown binary operator", span),
            },
        };
        Ok(self.op1(kind, vec![a, b], ty, span))
    }

    /// `a and b and c` is a running value `t`, updated one operand at a time:
    /// `t = a; t = if t { b } else { false }; t = if t { c } else { false }`,
    /// and `a or b or c` is `t = if t { true } else { b }` likewise. Every
    /// `if` is a sibling in the current block, so a chain of any length adds
    /// **one** level of region nesting (an operand is evaluated inside an
    /// arm), not one per operand. Short-circuiting stays exact: once `t` has
    /// decided the result it stays decided (`false and x` is `false`, `true
    /// or x` is `true`), and operand `k` is evaluated only in the arm that
    /// runs while `t` has not, so a later operand's effects and errors
    /// happen exactly when the interpreter's would. The last `if` yields the
    /// last operand's own value, as the interpreter returns it.
    fn bool_op(&mut self, span: Span, op: BoolOp, operands: &[Expr]) -> R<ValueId> {
        // `or` stops on true, `and` on false.
        let decisive = op == BoolOp::Or;
        let Some((first, rest)) = operands.split_first() else {
            return self.err("internal error: an empty `and`/`or`", span);
        };
        let mut t = self.expr(first)?;
        for operand in rest {
            let outer = self.st.b.at();
            let (ifop, _) = self.st.b.op(OpKind::If, vec![t], Vec::new(), span);
            let then_b = self.new_arm(ifop);
            let else_b = self.new_arm(ifop);
            // The arm that stops the evaluation yields the decisive value;
            // the other evaluates the next operand.
            let (go, short) = if decisive {
                (else_b, then_b)
            } else {
                (then_b, else_b)
            };
            self.st.b.set_at(short);
            let k = self.const_bool(decisive, span);
            self.st.b.op(OpKind::Yield, vec![k], Vec::new(), span);
            self.st.b.set_at(go);
            let v = self.expr(operand)?;
            // Evaluating an operand may open and close regions of its own;
            // the yield belongs to this arm.
            self.st.b.set_at(go);
            self.st.b.op(OpKind::Yield, vec![v], Vec::new(), span);
            self.st.b.set_at(outer);
            t = self.st.b.add_result(ifop, Type::Bool);
        }
        Ok(t)
    }

    /// A new region with one empty block, for an `if` arm; returns the block.
    pub(crate) fn new_arm(&mut self, op: OpId) -> BlockId {
        let r = self.st.b.new_region(op);
        self.st.b.new_block(r, Vec::new())
    }

    /// `then if cond else else_`: an `if` with one result; only the chosen
    /// arm runs.
    #[inline(never)]
    fn if_else(&mut self, e: &Expr, then: &Expr, cond: &Expr, else_: &Expr) -> R<ValueId> {
        let c = self.expr(cond)?;
        let outer = self.st.b.at();
        let (op, _) = self.st.b.op(OpKind::If, vec![c], Vec::new(), e.span);
        let mut tys = Vec::new();
        for arm in [then, else_] {
            let block = self.new_arm(op);
            self.st.b.set_at(block);
            let v = self.expr(arm)?;
            tys.push(self.st.b.ty(v).clone());
            self.st.b.op(OpKind::Yield, vec![v], Vec::new(), arm.span);
        }
        self.st.b.set_at(outer);
        // The checker joined the arms' types (tags joined); where it could
        // not type the whole (`Unknown`), join what the arms turned out to be.
        let ty = match self.checked.expr_ty(e.span) {
            Some(t @ (Ty::Value(_) | Ty::Bool)) => crate::ctx::map_ty(&t).unwrap_or(Type::Unknown),
            _ => join_ty(&tys[0], &tys[1]),
        };
        Ok(self.st.b.add_result(op, ty))
    }

    /// A call: the builtins `print` and `inv` (unless the program bound the
    /// name itself, in which case the checker typed the callee), or a
    /// function defined with `def`, resolved the way the checker resolved it.
    #[inline(never)]
    fn call(&mut self, e: &Expr, callee: &Expr, args: &[Expr]) -> R<ValueId> {
        let mut target = callee;
        while let ExprKind::Paren(inner) = &target.kind {
            target = inner;
        }
        if let ExprKind::Name(id) = &target.kind {
            match self.checked.expr_types.get(&target.span) {
                Some(Ty::Func(sig)) => {
                    let sig = sig.clone();
                    return self.call_user(e, &sig, args);
                }
                // The checker typed the callee, so the name is bound to
                // something that is not a function (a parameter, say).
                Some(_) => {
                    return Err(LowerError::new(
                        format!(
                            "`{}` is not a function known at compile time",
                            sky_diag::display_capped(self.text(id.span))
                        ),
                        callee.span,
                    )
                    .with_note(
                        "SIR has no function values: only functions defined with `def` can be called",
                    ));
                }
                // Never typed: a builtin, as the checker read it.
                None => match (self.text(id.span), args) {
                    ("print", _) => {
                        self.print_call(args, e.span)?;
                        return Ok(self.op1(
                            OpKind::ConstNothing,
                            Vec::new(),
                            Type::Nothing,
                            e.span,
                        ));
                    }
                    ("inv", [arg]) => {
                        let a = self.expr(arg)?;
                        let ty = self.ty_of(e)?;
                        return Ok(self.op1(OpKind::Inverse, vec![a], ty, e.span));
                    }
                    (name @ ("zeros" | "identity" | "len"), _) => {
                        if let Some(v) = self.array_builtin(e, name, args)? {
                            return Ok(v);
                        }
                    }
                    (name, _) => {
                        if let Some(f) = sky_ir::MathFn::from_name(name) {
                            let mut vals = Vec::with_capacity(args.len());
                            for a in args {
                                vals.push(self.expr(a)?);
                            }
                            let ty = self.ty_of(e)?;
                            return Ok(self.op1(OpKind::Math(f), vals, ty, e.span));
                        }
                    }
                },
            }
        }
        self.err(
            "only functions defined with `def` can be called",
            callee.span,
        )
    }

    /// `print(args...)` as an operation (no result).
    pub(crate) fn print_call(&mut self, args: &[Expr], span: Span) -> R<()> {
        let mut vals = Vec::with_capacity(args.len());
        for a in args {
            vals.push(self.expr(a)?);
        }
        self.st.b.op(OpKind::Print, vals, Vec::new(), span);
        Ok(())
    }

    /// A call to a user function: the arguments, then the module-level
    /// names the callee reads, as their current values.
    fn call_user(&mut self, e: &Expr, sig: &FuncSig, args: &[Expr]) -> R<ValueId> {
        let Some(info) = self.fns.get(&sig.def_span()) else {
            return self.err(
                format!(
                    "`{}` is not available here",
                    sky_diag::display_capped(sig.name())
                ),
                e.span,
            );
        };
        let (name, result, arity, globals) = (
            info.name.clone(),
            info.result.clone(),
            info.params,
            info.globals.clone(),
        );
        if args.len() != arity {
            return self.err(
                format!(
                    "`{}` takes {arity} arguments, found {}",
                    sig.name(),
                    args.len()
                ),
                e.span,
            );
        }
        let mut vals = Vec::with_capacity(args.len() + globals.len());
        for a in args {
            vals.push(self.expr(a)?);
        }
        for g in &globals {
            vals.push(self.module_value(g, e.span)?);
        }
        // The call's own type has the callee's size variables solved for
        // this call; `result` (from the signature) covers what the checker
        // left unknown, including "returns nothing".
        let ty = match self.checked.expr_ty(e.span) {
            Some(t @ (Ty::Value(_) | Ty::Bool)) => crate::ctx::map_ty(&t).unwrap_or(result),
            _ => result,
        };
        Ok(self.op1(OpKind::Call { callee: name }, vals, ty, e.span))
    }

    /// The value module-level name `name` has now, as a callee would read
    /// it: in `<main>` the current binding, in a function the value it was
    /// passed; a predefined constant nobody rebound is a literal.
    pub(crate) fn module_value(&mut self, name: &str, span: Span) -> R<ValueId> {
        let found = if self.st.is_main {
            self.st.scope.get(name)
        } else {
            self.st.module_vals.get(name)
        };
        if let Some(v) = found {
            return Ok(*v);
        }
        if let Some(c) = lookup_constant(name) {
            return Ok(self.const_float(c.value, Type::scalar(c.dim), span));
        }
        self.err(
            format!("`{}` has no value here", sky_diag::display_capped(name)),
            span,
        )
    }
}

/// Why an index or attribute write, or an attribute read, is rejected.
pub(crate) const ARRAYS: &str =
    "`sky run` cannot execute this either: arrays are immutable and attributes are not supported in v0.1";
