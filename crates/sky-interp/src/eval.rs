//! The evaluator: one recursive walk over the checked AST.
//!
//! `match`es over `StmtKind`, `ExprKind`, `UnaryOp`, `BinaryOp` and friends
//! are exhaustive on purpose (no `_ =>` arms), so a node another change adds
//! to the AST fails to compile here instead of being silently ignored.

use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::sync::Arc;

use sky_ast::ast::span_text;

use sky_ast::{
    AssignTarget, BinaryOp, Block, BoolOp, Expr, ExprKind, IndexItem, Literal, Module, Stmt,
    StmtKind, UnaryOp,
};
use sky_diag::Span;
use sky_syntax::lexer::unescape_str;
use sky_units::MathFn;

use crate::arith::{
    arith, arity_mismatch, equal, expect_bool, order, pow, reciprocal, recursion_too_deep, sign,
    ArithOp, StackMeter, MAX_CALL_DEPTH, R,
};
use crate::array::{self, Ix};
use crate::derivative::DerivativeHook;
use crate::literal::{number_literal, unit_literal_si};
use crate::show::show_value;
use crate::value::Function;
use crate::{RuntimeEnv, RuntimeError, Value};

#[path = "for_loop.rs"]
mod for_loop;

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
    run_module_with(module, src, env, out, &HashSet::new())
}

/// Like [`run_module`], and honours the checker's rewrites: `solves` is
/// `Checked::cholesky_solves`, the span of every `A⁻¹ @ b` the checker
/// decided to rewrite to a Cholesky solve (its `A` is PositiveDefinite). At
/// such a span the interpreter factors `A` and solves for `b` instead of
/// forming `A⁻¹`. The spans must come from checking this same `src`. With an
/// empty set the program still runs, computing the inverse and multiplying,
/// which gives the same answer up to rounding.
pub fn run_module_with<W: Write>(
    module: &Module,
    src: &str,
    env: &mut RuntimeEnv,
    out: &mut W,
    solves: &HashSet<Span>,
) -> Result<Option<Value>, RuntimeError> {
    run(module, src, env, out, solves, None)
}

/// Like [`run_module_with`], and evaluates `derivative(f, x)(args)` with
/// `hook` (see [`DerivativeHook`]); without one, a derivative is a runtime
/// error, because the interpreter has no automatic differentiation of its
/// own.
pub fn run_module_with_derivatives<W: Write>(
    module: &Module,
    src: &str,
    env: &mut RuntimeEnv,
    out: &mut W,
    solves: &HashSet<Span>,
    hook: &mut dyn DerivativeHook,
) -> Result<Option<Value>, RuntimeError> {
    run(module, src, env, out, solves, Some(hook))
}

fn run<'a, W: Write>(
    module: &Module,
    src: &'a str,
    env: &'a mut RuntimeEnv,
    out: &'a mut W,
    solves: &'a HashSet<Span>,
    derivatives: Option<&'a mut dyn DerivativeHook>,
) -> Result<Option<Value>, RuntimeError> {
    let mut it = Interp {
        src,
        env,
        out,
        solves,
        frames: Vec::new(),
        stack: StackMeter::new(),
        derivatives,
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

/// One active call of a user function.
struct Frame {
    /// The call's local variables: the parameters and what the body binds.
    vars: BTreeMap<String, Value>,
    /// The function being run, for its captured functions.
    func: Arc<Function>,
}

pub(crate) struct Interp<'a> {
    src: &'a str,
    /// Module-level names.
    pub(crate) env: &'a mut RuntimeEnv,
    pub(crate) out: &'a mut dyn Write,
    /// The `A⁻¹ @ b` sites to solve by Cholesky; see [`run_module_with`].
    solves: &'a HashSet<Span>,
    /// The active user-function calls, innermost last. Inside a call, names
    /// are read from its frame and then from the module (Python's rule);
    /// assignments go to the frame. *Calls* of a name are the exception:
    /// they are bound where the function was defined (see [`Function`]).
    frames: Vec<Frame>,
    /// Where the run started; see [`crate::arith::STACK_BUDGET`].
    pub(crate) stack: StackMeter,
    /// What evaluates `derivative(f, x)(args)`, if the caller has one.
    pub(crate) derivatives: Option<&'a mut dyn DerivativeHook>,
}

impl<'a> Interp<'a> {
    pub(crate) fn text(&self, span: Span) -> &'a str {
        span_text(self.src, span).unwrap_or("")
    }

    pub(crate) fn lookup(&self, name: &str) -> Option<&Value> {
        self.frames
            .last()
            .and_then(|f| f.vars.get(name))
            .or_else(|| self.env.get(name))
    }

    /// The function a call of `name` runs, if a call binds it lexically:
    /// inside a function body, and `name` is not a local, the function
    /// itself or the one in scope where it was defined. A call binds the
    /// definition the checker typed it against, so a later `def` of the same
    /// name, or `name = 5`, does not change it. `None` at the top level
    /// (which uses the current binding, as the checker does), and for names
    /// the body binds itself (a parameter, a nested `def`).
    pub(crate) fn lexical_function(&self, name: &str) -> Option<Arc<Function>> {
        let frame = self.frames.last()?;
        if frame.vars.contains_key(name) {
            return None;
        }
        if frame.func.name == name {
            return Some(frame.func.clone());
        }
        frame.func.captured.get(name).cloned()
    }

    /// Whether the program binds `name` itself, as a call sees it: then it
    /// wins over a builtin of that name. Inside a function that is the
    /// function's own view (locals, itself, what it captured), not the
    /// module's later state.
    pub(crate) fn is_user_bound(&self, name: &str) -> bool {
        match self.frames.last() {
            Some(f) => {
                f.vars.contains_key(name)
                    || f.func.name == name
                    || f.func.captured.contains_key(name)
            }
            None => self.env.get(name).is_some(),
        }
    }

    /// The functions in scope now, for a `def` to capture: at the top level
    /// the module's, inside a function what it captured plus its own name and
    /// its local functions (a nested `def` sees the definitions its
    /// enclosing function saw, not the module's state at the time it runs).
    fn visible_functions(&self) -> BTreeMap<String, Arc<Function>> {
        let mut out = BTreeMap::new();
        match self.frames.last() {
            Some(f) => {
                out.extend(f.func.captured.iter().map(|(k, v)| (k.clone(), v.clone())));
                out.insert(f.func.name.clone(), f.func.clone());
                for (k, v) in &f.vars {
                    match v {
                        Value::Func(func) => out.insert(k.clone(), func.clone()),
                        _ => out.remove(k),
                    };
                }
            }
            None => {
                for (k, v) in self.env.bindings() {
                    if let Value::Func(func) = v {
                        out.insert(k.to_string(), func.clone());
                    }
                }
            }
        }
        out
    }

    fn assign(&mut self, name: &str, v: Value) {
        match self.frames.last_mut() {
            Some(frame) => {
                frame.vars.insert(name.to_string(), v);
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
            StmtKind::For { var, iter, body } => self.for_stmt(var, iter, body),
            StmtKind::Def {
                name,
                params,
                ret: _,
                body,
            } => {
                let name_span = name.span;
                let name = self.text(name.span);
                let func = Function {
                    name: name.to_string(),
                    def_span: name_span,
                    params: params
                        .iter()
                        .map(|p| self.text(p.name.span).to_string())
                        .collect(),
                    body: body.clone(),
                    captured: self.visible_functions(),
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
                // A declaration without a value is an input: the checker
                // takes its shape, tags and unit on trust, and nothing in
                // v0.1 supplies one.
                None => Err(crate::arith::declared_without_value(
                    self.text(name.span),
                    s.span,
                )),
            },
        }
    }

    pub(crate) fn eval(&mut self, e: &Expr) -> R<Value> {
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
                // Converted to SI once, here (spec "Unit values").
                let (si, dim) = unit_literal_si(self.src, span, unit, e.span)?;
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
                BinaryOp::MatMul => self.matmul(e, lhs, rhs),
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
            ExprKind::Array(elems) => self.array_literal(e, elems),
            ExprKind::Index { base, items } => self.index(e, base, items),
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
            ExprKind::As { value, unit, .. } => {
                let v = self.eval(value)?;
                show_value(self.src, unit, &v, e.span)
            }
        }
    }

    /// Evaluates an operand of `and`, `or` or `not`, which must be a Bool.
    fn expect_bool(&mut self, kw: &str, operand: &Expr) -> R<bool> {
        let v = self.eval(operand)?;
        expect_bool(kw, &v, operand.span)
    }

    /// Reads a numeric literal's text as an `f64` (see [`crate::literal`]).
    fn number(&self, span: Span) -> R<f64> {
        number_literal(self.src, span)
    }

    fn call(&mut self, call: &Expr, callee: &Expr, args: &[Expr]) -> R<Value> {
        if let Some(derivative) = self.derivative_call(call, callee, args) {
            return derivative;
        }
        // Builtins are looked up only if the program has not bound the
        // name itself, so a user's own `print` or `inv` wins (the checker
        // applies the same rule).
        if let ExprKind::Name(id) = &callee.kind {
            let name = self.text(id.span);
            if let Some(func) = self.lexical_function(name) {
                return self.call_user(&func, call, args);
            }
            if !self.is_user_bound(name) {
                match name {
                    "print" => return self.print(args),
                    "inv" => return self.inv(call, args),
                    "zeros" | "identity" | "len" => return self.array_builtin(name, call, args),
                    _ => {
                        if let Some(f) = MathFn::from_name(name) {
                            return self.math(f, call, args);
                        }
                    }
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

    /// `[a, b]` or `[[a, b], [c, d]]`: entries are evaluated left to right,
    /// row by row. A matrix is a literal whose elements are all literals.
    fn array_literal(&mut self, e: &Expr, elems: &[Expr]) -> R<Value> {
        let matrix = elems.iter().all(|x| matches!(x.kind, ExprKind::Array(_)));
        let mut rows = Vec::new();
        if matrix {
            for row in elems {
                let ExprKind::Array(items) = &row.kind else {
                    continue;
                };
                rows.push(self.entries(items)?);
            }
        } else {
            rows.push(self.entries(elems)?);
        }
        array::literal(rows, !matrix, e.span)
    }

    fn entries(&mut self, items: &[Expr]) -> R<Vec<(Value, Span)>> {
        items
            .iter()
            .map(|x| Ok((self.eval(x)?, x.span)))
            .collect::<R<Vec<_>>>()
    }

    /// `base[items]`, reading only.
    fn index(&mut self, e: &Expr, base: &Expr, items: &[IndexItem]) -> R<Value> {
        let b = self.eval(base)?;
        let mut ixs = Vec::with_capacity(items.len());
        for item in items {
            ixs.push(match item {
                IndexItem::Expr(x) => Ix::At {
                    i: array::to_index(&self.eval(x)?, x.span)?,
                    span: x.span,
                },
                IndexItem::Slice {
                    start, stop, span, ..
                } => {
                    let mut bound = |x: &Option<Expr>| match x {
                        Some(x) => array::to_index(&self.eval(x)?, x.span).map(Some),
                        None => Ok(None),
                    };
                    Ix::Slice {
                        start: bound(start)?,
                        stop: bound(stop)?,
                        span: *span,
                    }
                }
            });
        }
        array::index(&b, &ixs, e.span, base.span)
    }

    /// `l @ r`. Where the checker rewrote `A⁻¹ @ b` (`solves` holds the
    /// span of the whole expression) the interpreter factors `A` and solves
    /// for `b`, and never forms `A⁻¹`.
    fn matmul(&mut self, e: &Expr, lhs: &Expr, rhs: &Expr) -> R<Value> {
        if self.solves.contains(&e.span) {
            if let Some(a) = self.inverted_operand(lhs) {
                let av = self.eval(a)?;
                let bv = self.eval(rhs)?;
                return array::cholesky_solve(&av, &bv, e.span, a.span, rhs.span);
            }
        }
        let l = self.eval(lhs)?;
        let r = self.eval(rhs)?;
        array::matmul(&l, &r, e.span, lhs.span, rhs.span)
    }

    /// `A` if `e` is `A⁻¹` or `inv(A)` (the builtin, not a user's own
    /// `inv`), possibly parenthesized: the forms the checker rewrites.
    fn inverted_operand<'x>(&self, e: &'x Expr) -> Option<&'x Expr> {
        let mut e = e;
        while let ExprKind::Paren(inner) = &e.kind {
            e = inner;
        }
        match &e.kind {
            ExprKind::Inverse { operand, .. } => Some(operand),
            ExprKind::Call { callee, args } => match (&callee.kind, args.as_slice()) {
                (ExprKind::Name(id), [arg])
                    if self.text(id.span) == "inv" && !self.is_user_bound("inv") =>
                {
                    Some(arg)
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// `zeros(n)`, `zeros(m, n)`, `identity(n)`, `len(x)`.
    fn array_builtin(&mut self, name: &str, call: &Expr, args: &[Expr]) -> R<Value> {
        let takes = |n: &str| {
            RuntimeError::new(
                format!("`{name}` takes {n}, found {}", args.len()),
                call.span,
            )
        };
        let mut sizes = Vec::with_capacity(args.len());
        for a in args {
            let v = self.eval(a)?;
            if name == "len" {
                let [_] = args else {
                    return Err(takes("1 argument"));
                };
                return array::len(&v, a.span);
            }
            sizes.push(array::size_arg(name, &v, a.span)?);
        }
        match (name, sizes.as_slice()) {
            ("zeros", [_] | [_, _]) => array::zeros(&sizes, call.span),
            ("identity", [n]) => array::identity(*n, call.span),
            ("zeros", _) => Err(takes("1 or 2 arguments")),
            _ => Err(takes("1 argument")),
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
        let mut vars = BTreeMap::new();
        for (name, a) in func.params.iter().zip(args) {
            let v = self.eval(a)?;
            vars.insert(name.clone(), v);
        }
        self.frames.push(Frame {
            vars,
            func: func.clone(),
        });
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

    /// A builtin math function: arguments left to right, then the shared
    /// evaluation in `math`.
    fn math(&mut self, f: MathFn, call: &Expr, args: &[Expr]) -> R<Value> {
        let vals = args
            .iter()
            .map(|a| self.eval(a))
            .collect::<R<Vec<Value>>>()?;
        let spans: Vec<Span> = args.iter().map(|a| a.span).collect();
        crate::math::apply(f, &vals, call.span, &spans)
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
            BinaryOp::MatMul => array::matmul(&l, &r, whole, lspan, rspan),
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
