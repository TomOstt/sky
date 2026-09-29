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

use crate::value::Function;
use crate::{RuntimeEnv, RuntimeError, Value};

type R<T> = Result<T, RuntimeError>;

/// Integer exponents up to this size use `powi` (repeated multiplication,
/// exact whenever the true result is representable); larger or fractional
/// ones use `powf`.
const POWI_MAX: f64 = 64.0;

/// Most user-function calls active at once, like Python's recursion limit:
/// a friendly error for runaway recursion.
pub const MAX_CALL_DEPTH: usize = 1000;

/// How much of the stack the walk may use before it stops with an error.
/// The walk recurses on the Rust stack, and how much each level costs
/// depends on the build (a debug build needs several KiB per nested
/// expression), so a fixed depth count can't be both safe and generous.
/// Instead every `eval` and block measures how far the stack has grown
/// since `run_module` started. The budget leaves 1 MiB of
/// `sky_ast::DEEP_STACK_BYTES` (`run_module` must run under
/// `with_deep_stack`) for the frames between two checks and for printing.
const STACK_BUDGET: usize = sky_ast::DEEP_STACK_BYTES - (1 << 20);

/// The address of a local: where the stack is now. Only ever compared with
/// another such address, never dereferenced.
#[inline(never)]
fn stack_position() -> usize {
    let marker = 0u8;
    std::hint::black_box(&marker) as *const u8 as usize
}

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
        stack_base: stack_position(),
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
    /// [`stack_position`] when the run started; see [`STACK_BUDGET`].
    stack_base: usize,
}

fn arrays_unsupported(span: Span) -> RuntimeError {
    RuntimeError::new("arrays are not supported by the interpreter yet", span)
        .with_label("this needs an array")
        .with_note("the checker accepts array types in v0.1, but `sky run` cannot execute them yet")
}

/// `length`, or `L·M` for a dimension without a name (as the checker says it).
fn describe(d: &Dimension) -> String {
    match d.name() {
        Some(n) => n.to_string(),
        None => d.to_string(),
    }
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

    /// An error if the walk has used up [`STACK_BUDGET`]: deep recursion
    /// combined with deeply nested expressions ends here, cleanly, rather
    /// than in a stack overflow.
    fn check_stack(&self, span: Span) -> R<()> {
        // Direction-agnostic: the stack grows down on every target Sky
        // supports, but `abs_diff` doesn't need to know.
        if self.stack_base.abs_diff(stack_position()) > STACK_BUDGET {
            return Err(RuntimeError::new(
                "the program nests too deeply for the interpreter",
                span,
            )
            .with_label("stopped here")
            .with_note("`sky run` stops before it runs out of stack")
            .with_help("reduce the recursion depth, or use a `while` loop"));
        }
        Ok(())
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
                UnaryOp::Neg | UnaryOp::Pos => match self.eval(operand)? {
                    Value::Number { value, dim } => {
                        let value = if *op == UnaryOp::Neg { -value } else { value };
                        Ok(Value::number(value, dim))
                    }
                    other => Err(RuntimeError::new(
                        format!(
                            "`{}` needs a number, found {}",
                            if *op == UnaryOp::Neg { "-" } else { "+" },
                            other.kind()
                        ),
                        operand.span,
                    )),
                },
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
            ExprKind::Index { .. } => Err(arrays_unsupported(e.span)),
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
        match self.eval(operand)? {
            Value::Bool(b) => Ok(b),
            other => Err(RuntimeError::new(
                format!("`{kw}` needs a Bool operand, found {}", other.kind()),
                operand.span,
            )
            .with_label("not a Bool")
            .with_note("v0.1 has no truthiness")),
        }
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
            return Err(RuntimeError::new(
                format!(
                    "`{}` takes {} argument{}, found {}",
                    func.name,
                    func.params.len(),
                    if func.params.len() == 1 { "" } else { "s" },
                    args.len()
                ),
                call.span,
            ));
        }
        if self.frames.len() >= MAX_CALL_DEPTH {
            return Err(RuntimeError::new(
                format!("recursion is too deep: more than {MAX_CALL_DEPTH} nested calls"),
                call.span,
            )
            .with_label(format!("calling `{}` here", func.name))
            .with_note("`sky run` limits nested calls instead of overflowing the stack")
            .with_help("check the recursion's base case, or use a `while` loop"));
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
        let Value::Number { value, dim } = v else {
            return Err(RuntimeError::new(
                format!("`{what}` needs a number, found {}", v.kind()),
                operand,
            ));
        };
        if value == 0.0 {
            return Err(div_zero(whole, operand));
        }
        let dim = dim
            .inverse()
            .map_err(|_| RuntimeError::new("dimension exponent too large", whole))?;
        Ok(Value::number(finite(1.0 / value, whole)?, dim))
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
            BinaryOp::Add | BinaryOp::Sub => {
                let (verb, prep) = if op == BinaryOp::Add {
                    ("add", "and")
                } else {
                    ("subtract", "from")
                };
                let (Value::Number { value: a, dim: da }, Value::Number { value: b, dim: db }) =
                    (&l, &r)
                else {
                    // `subtract b from a`: name the right operand first.
                    let (x, y) = if op == BinaryOp::Add {
                        (&l, &r)
                    } else {
                        (&r, &l)
                    };
                    return Err(RuntimeError::new(
                        format!("cannot {verb} {} {prep} {}", x.kind(), y.kind()),
                        whole,
                    ));
                };
                if da != db {
                    let message = if op == BinaryOp::Add {
                        format!("cannot add {} and {}", describe(da), describe(db))
                    } else {
                        format!("cannot subtract {} from {}", describe(db), describe(da))
                    };
                    return Err(RuntimeError::new(message, whole)
                        .with_label("mismatched dimensions")
                        .with_secondary(lspan, describe(da))
                        .with_secondary(rspan, describe(db)));
                }
                let x = if op == BinaryOp::Add { a + b } else { a - b };
                Ok(Value::number(finite(x, whole)?, *da))
            }
            BinaryOp::Mul | BinaryOp::Div => {
                let (Value::Number { value: a, dim: da }, Value::Number { value: b, dim: db }) =
                    (&l, &r)
                else {
                    return Err(RuntimeError::new(
                        format!(
                            "cannot {} {} and {}",
                            if op == BinaryOp::Mul {
                                "multiply"
                            } else {
                                "divide"
                            },
                            l.kind(),
                            r.kind()
                        ),
                        whole,
                    ));
                };
                let dim = if op == BinaryOp::Mul {
                    da.checked_mul(*db)
                } else {
                    da.checked_div(*db)
                }
                .map_err(|_| RuntimeError::new("dimension exponent too large", whole))?;
                if op == BinaryOp::Div {
                    if *b == 0.0 {
                        return Err(div_zero(whole, rspan));
                    }
                    return Ok(Value::number(finite(a / b, whole)?, dim));
                }
                Ok(Value::number(finite(a * b, whole)?, dim))
            }
            BinaryOp::Pow => self.pow(l, r, whole, lspan, rexpr),
            BinaryOp::Eq | BinaryOp::Ne => {
                let eq = match (&l, &r) {
                    (Value::Number { value: a, dim: da }, Value::Number { value: b, dim: db }) => {
                        if da != db {
                            return Err(cannot_compare(&l, &r, whole, lspan, rspan));
                        }
                        a == b
                    }
                    (Value::Bool(a), Value::Bool(b)) => a == b,
                    (Value::Str(a), Value::Str(b)) => a == b,
                    _ => return Err(cannot_compare(&l, &r, whole, lspan, rspan)),
                };
                Ok(Value::Bool(eq == (op == BinaryOp::Eq)))
            }
            BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                let (Value::Number { value: a, dim: da }, Value::Number { value: b, dim: db }) =
                    (&l, &r)
                else {
                    return Err(RuntimeError::new(
                        format!(
                            "cannot order {} and {} with `{}`",
                            l.kind(),
                            r.kind(),
                            op.symbol()
                        ),
                        whole,
                    ));
                };
                if da != db {
                    return Err(cannot_compare(&l, &r, whole, lspan, rspan));
                }
                Ok(Value::Bool(match op {
                    BinaryOp::Lt => a < b,
                    BinaryOp::Le => a <= b,
                    BinaryOp::Gt => a > b,
                    BinaryOp::Ge => a >= b,
                    BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Div
                    | BinaryOp::Pow
                    | BinaryOp::MatMul
                    | BinaryOp::Eq
                    | BinaryOp::Ne => false, // not reachable: this arm is the four orderings
                }))
            }
            BinaryOp::MatMul => Err(arrays_unsupported(whole)),
        }
    }

    /// `base ^ exp`. The result's dimension comes from the constant
    /// rational exponent in the source, exactly as the checker computes it.
    fn pow(&self, base: Value, exp: Value, whole: Span, lspan: Span, rexpr: &Expr) -> R<Value> {
        let (Value::Number { value: b, dim: db }, Value::Number { value: x, dim: dx }) =
            (&base, &exp)
        else {
            return Err(RuntimeError::new(
                format!("cannot raise {} to {}", base.kind(), exp.kind()),
                whole,
            ));
        };
        if !dx.is_dimensionless() {
            return Err(
                RuntimeError::new("the exponent must be dimensionless", rexpr.span)
                    .with_label(format!("this is {}", describe(dx))),
            );
        }
        let dim = if db.is_dimensionless() {
            Dimension::DIMENSIONLESS
        } else {
            let Some(r) = sky_types::const_exponent(self.src, rexpr) else {
                return Err(RuntimeError::new(
                    "the exponent must be a constant number when the base has a unit",
                    rexpr.span,
                )
                .with_secondary(lspan, format!("this is {}", describe(db))));
            };
            db.pow_rat(r)
                .map_err(|_| RuntimeError::new("dimension exponent too large", whole))?
        };
        if *b == 0.0 && *x < 0.0 {
            return Err(div_zero(whole, lspan));
        }
        if *b < 0.0 && x.fract() != 0.0 {
            return Err(RuntimeError::new("the result is not a real number", whole)
                .with_label("a negative base with a fractional exponent")
                .with_note("Sky numbers are real; it never produces NaN or a complex number"));
        }
        let value = if x.fract() == 0.0 && x.abs() <= POWI_MAX {
            b.powi(*x as i32)
        } else {
            b.powf(*x)
        };
        Ok(Value::number(finite(value, whole)?, dim))
    }
}

fn write_through(span: Span) -> RuntimeError {
    RuntimeError::new("cannot assign through an index or attribute", span)
        .with_note("arrays are immutable in v0.1")
}

fn overflow(span: Span) -> RuntimeError {
    RuntimeError::new("arithmetic overflow", span)
        .with_label("the result does not fit in a 64-bit float")
        .with_note("Sky numbers never become inf or NaN; the largest is about 1.8e308")
}

/// `x` if it is finite. Every operand is finite, so a non-finite result of
/// `+ - * / ^` is an overflow.
fn finite(x: f64, span: Span) -> R<f64> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(overflow(span))
    }
}

fn div_zero(whole: Span, zero: Span) -> RuntimeError {
    RuntimeError::new("division by zero", whole)
        .with_label("this divides by zero")
        .with_secondary(zero, "this is zero")
        .with_note("Sky reports division by zero instead of returning inf or NaN")
}

fn cannot_compare(l: &Value, r: &Value, whole: Span, lspan: Span, rspan: Span) -> RuntimeError {
    let (a, b) = match (l, r) {
        (Value::Number { dim: da, .. }, Value::Number { dim: db, .. }) => {
            (describe(da), describe(db))
        }
        _ => (l.kind().to_string(), r.kind().to_string()),
    };
    RuntimeError::new(format!("cannot compare {a} and {b}"), whole)
        .with_secondary(lspan, a)
        .with_secondary(rspan, b)
}
