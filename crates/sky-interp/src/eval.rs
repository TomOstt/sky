//! The evaluator: one recursive walk over the checked AST.
//!
//! `match`es over `StmtKind`, `ExprKind`, `UnaryOp`, `BinaryOp` and friends
//! are exhaustive on purpose (no `_ =>` arms), so a node another change adds
//! to the AST fails to compile here instead of being silently ignored.

use std::io::Write;

use sky_ast::ast::span_text;
use std::collections::BTreeMap;
use std::sync::Arc;

use sky_ast::{
    AssignTarget, BinaryOp, Block, BoolOp, Expr, ExprKind, Literal, Module, Stmt, StmtKind,
    UnaryOp, UnitExpr,
};
use sky_diag::Span;
use sky_syntax::lexer::unescape_str;
use sky_units::{lookup, Dimension, Rational};

use crate::arith::{
    arith, arity_mismatch, arrays_unsupported, equal, expect_bool, order, overflow, pow,
    reciprocal, recursion_too_deep, sign, ArithOp, StackMeter, MAX_CALL_DEPTH, R,
};
use crate::value::Function;
use crate::{RuntimeEnv, RuntimeError, Value};

/// How a statement finished: normally (with the value of an expression
/// statement, for the REPL), or by leaving its block early.
enum Flow {
    Normal(Option<Value>),
    Return(Value),
    Break,
    Continue,
}

/// Runs `module` (parsed from `src`) against `env`, writing `print` output
/// to `out`. Returns the value of the last statement if it is an expression
/// statement, for the REPL to echo; `sky run` ignores it.
///
/// On error, bindings made by earlier statements of this module remain in
/// `env`; a caller that wants all-or-nothing (the REPL) snapshots and
/// restores it. Run this under `sky_ast::with_deep_stack`.
pub fn run_module<W: Write>(
    module: &Module,
    src: &str,
    env: &mut RuntimeEnv,
    out: &mut W,
) -> Result<Option<Value>, RuntimeError> {
    let mut it = Interp {
        src,
        env,
        out,
        frames: Vec::new(),
        stack: StackMeter::new(),
    };
    let mut last = None;
    for stmt in &module.stmts {
        last = match it.stmt(stmt)? {
            Flow::Normal(v) => v,
            // The checker rejects `return`, `break` and `continue` outside a
            // function or loop, so a checked module never ends up here.
            Flow::Return(_) | Flow::Break | Flow::Continue => None,
        };
    }
    Ok(last)
}

struct Interp<'a> {
    src: &'a str,
    /// Module-level names.
    env: &'a mut RuntimeEnv,
    out: &'a mut dyn Write,
    /// The local variables of each active user-function call, innermost
    /// last. Inside a call, names are read from its frame and then from the
    /// module (Python's rule); assignments go to the frame.
    frames: Vec<BTreeMap<String, Value>>,
    /// Where the run started; see [`crate::arith::STACK_BUDGET`].
    stack: StackMeter,
}

impl<'a> Interp<'a> {
    fn text(&self, span: Span) -> &'a str {
        span_text(self.src, span).unwrap_or("")
    }

    fn lookup(&self, name: &str) -> Option<&Value> {
        self.frames
            .last()
            .and_then(|f| f.get(name))
            .or_else(|| self.env.get(name))
    }

    fn assign(&mut self, name: &str, v: Value) {
        match self.frames.last_mut() {
            Some(frame) => {
                frame.insert(name.to_string(), v);
            }
            None => self.env.set(name, v),
        }
    }

    fn check_stack(&self, span: Span) -> R<()> {
        self.stack.check(span)
    }

    /// Runs a block's statements until one leaves it early.
    fn block(&mut self, b: &Block) -> R<Flow> {
        self.check_stack(b.span)?;
        for s in &b.stmts {
            match self.stmt(s)? {
                Flow::Normal(_) => {}
                other => return Ok(other),
            }
        }
        Ok(Flow::Normal(None))
    }

    fn stmt(&mut self, s: &Stmt) -> R<Flow> {
        match &s.kind {
            StmtKind::Expr(e) => Ok(Flow::Normal(Some(self.eval(e)?))),
            StmtKind::If {
                branches,
                else_block,
            } => {
                for (cond, block) in branches {
                    if self.expect_bool("if", cond)? {
                        return self.block(block);
                    }
                }
                match else_block {
                    Some(block) => self.block(block),
                    None => Ok(Flow::Normal(None)),
                }
            }
            StmtKind::While { cond, body } => {
                while self.expect_bool("while", cond)? {
                    match self.block(body)? {
                        Flow::Break => break,
                        Flow::Normal(_) | Flow::Continue => {}
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                    }
                }
                Ok(Flow::Normal(None))
            }
            StmtKind::Def {
                name,
                params,
                ret: _,
                body,
            } => {
                let name = self.text(name.span);
                let func = Function {
                    name: name.to_string(),
                    params: params
                        .iter()
                        .map(|p| self.text(p.name.span).to_string())
                        .collect(),
                    body: body.clone(),
                };
                self.assign(name, Value::Func(Arc::new(func)));
                Ok(Flow::Normal(None))
            }
            StmtKind::Return(value) => {
                let v = match value {
                    Some(e) => self.eval(e)?,
                    None => Value::None,
                };
                Ok(Flow::Return(v))
            }
            StmtKind::Pass => Ok(Flow::Normal(None)),
            StmtKind::Break => Ok(Flow::Break),
            StmtKind::Continue => Ok(Flow::Continue),
            StmtKind::Assign { target, value } => match target {
                AssignTarget::Name(id) => {
                    let v = self.eval(value)?;
                    let name = self.text(id.span);
                    self.assign(name, v);
                    Ok(Flow::Normal(None))
                }
                // The checker rejects these writes (spec: "Statements and
                // assignment"), so a checked program never gets here.
                AssignTarget::Index { span, .. } | AssignTarget::Attribute { span, .. } => {
                    Err(write_through(*span))
                }
            },
            StmtKind::CompoundAssign {
                target,
                op,
                op_span: _,
                value,
            } => match target {
                AssignTarget::Name(id) => {
                    let name = self.text(id.span);
                    let Some(cur) = self.lookup(name).cloned() else {
                        return Err(RuntimeError::new(
                            format!("cannot find `{name}` in this scope"),
                            id.span,
                        )
                        .with_label("not defined"));
                    };
                    let rhs = self.eval(value)?;
                    let whole = id.span.join(value.span);
                    let v = self.binary_values(op.binary(), cur, rhs, whole, id.span, value)?;
                    self.assign(name, v);
                    Ok(Flow::Normal(None))
                }
                AssignTarget::Index { span, .. } | AssignTarget::Attribute { span, .. } => {
                    Err(write_through(*span))
                }
            },
            // Annotations are read by the checker only; at run time a
            // binding is its value.
            StmtKind::Binding { name, ty: _, value } => match value {
                Some(value) => {
                    let v = self.eval(value)?;
                    let name = self.text(name.span);
                    self.assign(name, v);
                    Ok(Flow::Normal(None))
                }
                // A declaration without a value only makes sense for an
                // array (the annotation is a shape).
                None => Err(arrays_unsupported(s.span)),
            },
        }
    }

    fn eval(&mut self, e: &Expr) -> R<Value> {
        self.check_stack(e.span)?;
        match &e.kind {
            ExprKind::Literal(lit) => match lit {
                Literal::Int(span) | Literal::Float(span) => Ok(Value::scalar(self.number(*span)?)),
                Literal::Str(span) => Ok(Value::Str(unescape_str(self.text(*span)))),
                Literal::Bool { value, .. } => Ok(Value::Bool(*value)),
            },
            ExprKind::Name(id) => {
                let name = self.text(id.span);
                match self.lookup(name) {
                    Some(v) => Ok(v.clone()),
                    None => Err(RuntimeError::new(
                        format!("cannot find `{name}` in this scope"),
                        id.span,
                    )
                    .with_label("not defined")),
                }
            }
            ExprKind::UnitLiteral { value, unit } => {
                let span = match value {
                    Literal::Int(s) | Literal::Float(s) => *s,
                    Literal::Str(s) | Literal::Bool { span: s, .. } => {
                        return Err(RuntimeError::new("a unit needs a number", *s));
                    }
                };
                let x = self.number(span)?;
                // Converted to SI once, here (spec "Unit values").
                let (si, dim) = self.apply_unit(unit, x)?;
                if !si.is_finite() {
                    return Err(overflow(e.span));
                }
                Ok(Value::number(si, dim))
            }
            ExprKind::Unary {
                op,
                op_span: _,
                operand,
            } => match op {
                UnaryOp::Neg | UnaryOp::Pos => {
                    let v = self.eval(operand)?;
                    sign(*op == UnaryOp::Neg, v, operand.span)
                }
                UnaryOp::Not => Ok(Value::Bool(!self.expect_bool("not", operand)?)),
                UnaryOp::Partial | UnaryOp::Nabla => Err(RuntimeError::new(
                    "derivative operators are not supported by the interpreter yet",
                    e.span,
                )),
            },
            ExprKind::Binary {
                op,
                op_span: _,
                lhs,
                rhs,
            } => match op {
                BinaryOp::Add
                | BinaryOp::Sub
                | BinaryOp::Mul
                | BinaryOp::Div
                | BinaryOp::Pow
                | BinaryOp::Eq
                | BinaryOp::Ne
                | BinaryOp::Lt
                | BinaryOp::Le
                | BinaryOp::Gt
                | BinaryOp::Ge => {
                    let l = self.eval(lhs)?;
                    let r = self.eval(rhs)?;
                    self.binary_values(*op, l, r, e.span, lhs.span, rhs)
                }
                BinaryOp::MatMul => Err(arrays_unsupported(e.span)),
            },
            ExprKind::BoolOp { op, operands, .. } => {
                // The operand that settles the result: `false` for `and`,
                // `true` for `or`. Stop at the first one, so the rest are
                // never evaluated (spec "Booleans and conditions").
                let decisive = match op {
                    BoolOp::And => false,
                    BoolOp::Or => true,
                };
                for operand in operands {
                    if self.expect_bool(op.keyword(), operand)? == decisive {
                        return Ok(Value::Bool(decisive));
                    }
                }
                Ok(Value::Bool(!decisive))
            }
            ExprKind::Inverse {
                operand,
                op_span: _,
            } => {
                let v = self.eval(operand)?;
                self.reciprocal("⁻¹", v, e.span, operand.span)
            }
            ExprKind::Call { callee, args } => self.call(e, callee, args),
            ExprKind::Array(_) | ExprKind::Index { .. } => Err(arrays_unsupported(e.span)),
            ExprKind::Attribute { base: _, name } => Err(RuntimeError::new(
                format!(
                    "attribute `{}` is not supported by the interpreter yet",
                    self.text(name.span)
                ),
                e.span,
            )),
            ExprKind::Paren(inner) => self.eval(inner),
            ExprKind::IfElse { then, cond, else_ } => {
                if self.expect_bool("if", cond)? {
                    self.eval(then)
                } else {
                    self.eval(else_)
                }
            }
        }
    }

    /// Evaluates an operand of `and`, `or` or `not`, which must be a Bool.
    fn expect_bool(&mut self, kw: &str, operand: &Expr) -> R<bool> {
        let v = self.eval(operand)?;
        expect_bool(kw, &v, operand.span)
    }

    /// Reads a numeric literal's text (`1_000`, `2.5e-3`) as an `f64`.
    fn number(&self, span: Span) -> R<f64> {
        let text = self.text(span).replace('_', "");
        match text.parse::<f64>() {
            Ok(x) if x.is_finite() => Ok(x),
            Ok(_) => Err(RuntimeError::new("number too large", span)
                .with_label("does not fit in a 64-bit float")
                .with_note("the largest Sky number is about 1.8e308")),
            Err(_) => Err(RuntimeError::new("invalid number", span)),
        }
    }

    /// Applies unit `u` to the value `v`, factor by factor: `36 km/h` is
    /// `36 * 1000 / 3600`, which is exactly 10, where multiplying by a
    /// precomputed `1000/3600` would not be. A single unit is `v * scale`,
    /// as the spec says. Returns the SI value and the dimension.
    fn apply_unit(&self, u: &UnitExpr, v: f64) -> R<(f64, Dimension)> {
        let overflow_dim = || RuntimeError::new("dimension exponent too large", u.span());
        match u {
            UnitExpr::Name(id) => {
                let name = self.text(id.span);
                match lookup(name) {
                    Some(unit) => Ok((v * unit.scale, unit.dim)),
                    None => Err(RuntimeError::new(format!("unknown unit `{name}`"), id.span)),
                }
            }
            UnitExpr::Mul { lhs, rhs, .. } => {
                let (v, d1) = self.apply_unit(lhs, v)?;
                let (v, d2) = self.apply_unit(rhs, v)?;
                Ok((v, d1.checked_mul(d2).map_err(|_| overflow_dim())?))
            }
            UnitExpr::Div { lhs, rhs, .. } => {
                let (v, d1) = self.apply_unit(lhs, v)?;
                let (scale, d2) = self.apply_unit(rhs, 1.0)?;
                Ok((v / scale, d1.checked_div(d2).map_err(|_| overflow_dim())?))
            }
            UnitExpr::Pow { base, num, den, .. } => {
                let (scale, d) = self.apply_unit(base, 1.0)?;
                let exp = i64::try_from(*den)
                    .ok()
                    .and_then(|den| Rational::checked_new(*num, den).ok())
                    .ok_or_else(overflow_dim)?;
                let dim = d.pow_rat(exp).map_err(|_| overflow_dim())?;
                let factor = if *den == 1 {
                    // Integer exponent: multiply or divide, `s^-2` is `/ s^2`.
                    let k = i32::try_from(num.unsigned_abs()).map_err(|_| overflow_dim())?;
                    let p = scale.powi(k);
                    if *num < 0 {
                        return Ok((v / p, dim));
                    }
                    p
                } else {
                    scale.powf(*num as f64 / *den as f64)
                };
                Ok((v * factor, dim))
            }
        }
    }

    fn call(&mut self, call: &Expr, callee: &Expr, args: &[Expr]) -> R<Value> {
        // Builtins are looked up only if the program has not bound the
        // name itself, so a user's own `print` or `inv` wins (the checker
        // applies the same rule).
        if let ExprKind::Name(id) = &callee.kind {
            let name = self.text(id.span);
            if self.lookup(name).is_none() {
                match name {
                    "print" => return self.print(args),
                    "inv" => return self.inv(call, args),
                    _ => {}
                }
            }
        }
        match self.eval(callee)? {
            Value::Func(func) => self.call_user(&func, call, args),
            other => Err(
                RuntimeError::new(format!("cannot call {}", other.kind()), callee.span)
                    .with_label("not a function"),
            ),
        }
    }

    /// Calls a user function: arguments are evaluated in the caller's
    /// scope, then the body runs in a new frame holding the parameters.
    fn call_user(&mut self, func: &Arc<Function>, call: &Expr, args: &[Expr]) -> R<Value> {
        if args.len() != func.params.len() {
            // The checker checks arity; this guards the interpreter anyway.
            return Err(arity_mismatch(
                &func.name,
                func.params.len(),
                args.len(),
                call.span,
            ));
        }
        if self.frames.len() >= MAX_CALL_DEPTH {
            return Err(recursion_too_deep(&func.name, call.span));
        }
        let mut frame = BTreeMap::new();
        for (name, a) in func.params.iter().zip(args) {
            let v = self.eval(a)?;
            frame.insert(name.clone(), v);
        }
        self.frames.push(frame);
        let flow = self.block(&func.body);
        self.frames.pop();
        match flow? {
            Flow::Return(v) => Ok(v),
            Flow::Normal(_) | Flow::Break | Flow::Continue => Ok(Value::None),
        }
    }

    /// `print(a, b, ...)`: the values separated by single spaces, then a
    /// newline. Every argument is evaluated before anything is written, so
    /// a failing argument leaves no partial line.
    fn print(&mut self, args: &[Expr]) -> R<Value> {
        let mut line = String::new();
        for (i, a) in args.iter().enumerate() {
            let v = self.eval(a)?;
            if i > 0 {
                line.push(' ');
            }
            line.push_str(&v.print_text());
        }
        line.push('\n');
        let span = match (args.first(), args.last()) {
            (Some(a), Some(b)) => a.span.join(b.span),
            _ => Span::point(0),
        };
        self.out
            .write_all(line.as_bytes())
            .map_err(|e| RuntimeError::new(format!("cannot write output: {e}"), span))?;
        Ok(Value::None)
    }

    /// `inv(x)`: the function spelling of `x⁻¹`.
    fn inv(&mut self, call: &Expr, args: &[Expr]) -> R<Value> {
        let [arg] = args else {
            return Err(RuntimeError::new(
                format!("`inv` takes 1 argument, found {}", args.len()),
                call.span,
            ));
        };
        let v = self.eval(arg)?;
        self.reciprocal("inv", v, call.span, arg.span)
    }

    /// `1/x` with the inverse dimension.
    fn reciprocal(&self, what: &str, v: Value, whole: Span, operand: Span) -> R<Value> {
        reciprocal(what, &v, whole, operand)
    }

    /// Applies a binary operator to two evaluated operands. `whole` is the
    /// span of the expression, `lspan` the left operand's, and `rexpr` the
    /// right operand (whose source a constant exponent is read from).
    fn binary_values(
        &self,
        op: BinaryOp,
        l: Value,
        r: Value,
        whole: Span,
        lspan: Span,
        rexpr: &Expr,
    ) -> R<Value> {
        let rspan = rexpr.span;
        match op {
            BinaryOp::Add => arith(ArithOp::Add, &l, &r, whole, lspan, rspan),
            BinaryOp::Sub => arith(ArithOp::Sub, &l, &r, whole, lspan, rspan),
            BinaryOp::Mul => arith(ArithOp::Mul, &l, &r, whole, lspan, rspan),
            BinaryOp::Div => arith(ArithOp::Div, &l, &r, whole, lspan, rspan),
            // The result's dimension comes from the constant rational
            // exponent in the source, exactly as the checker computes it.
            BinaryOp::Pow => pow(
                &l,
                &r,
                || sky_types::const_exponent(self.src, rexpr),
                whole,
                lspan,
                rspan,
            ),
            BinaryOp::Eq | BinaryOp::Ne => {
                let eq = equal(&l, &r, whole, lspan, rspan)?;
                Ok(Value::Bool(eq == (op == BinaryOp::Eq)))
            }
            BinaryOp::Lt => self.ordering(op, |a, b| a < b, &l, &r, whole, lspan, rspan),
            BinaryOp::Le => self.ordering(op, |a, b| a <= b, &l, &r, whole, lspan, rspan),
            BinaryOp::Gt => self.ordering(op, |a, b| a > b, &l, &r, whole, lspan, rspan),
            BinaryOp::Ge => self.ordering(op, |a, b| a >= b, &l, &r, whole, lspan, rspan),
            BinaryOp::MatMul => Err(arrays_unsupported(whole)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ordering(
        &self,
        op: BinaryOp,
        holds: fn(f64, f64) -> bool,
        l: &Value,
        r: &Value,
        whole: Span,
        lspan: Span,
        rspan: Span,
    ) -> R<Value> {
        order(op.symbol(), holds, l, r, whole, lspan, rspan).map(Value::Bool)
    }
}

fn write_through(span: Span) -> RuntimeError {
    RuntimeError::new("cannot assign through an index or attribute", span)
        .with_note("arrays are immutable in v0.1")
}
