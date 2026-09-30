//! Contracts: `assert`, `requires` and `ensures` (spec "Contracts").
//!
//! # Typing
//!
//! Every condition is typed like the condition of an `if`: a Bool of shape
//! `()`, with the dimensions of its comparisons checked as usual (`h >= 0 m`
//! is a length against a length; `h >= 0` is an error). A `requires` is
//! checked where it is written, at the top of the body, where only the
//! parameters and the names above the `def` are bound. An `ensures` is
//! checked after the body, against the names the body *started* with (the
//! parameters as passed, not what the body made of them) plus `result`, the
//! returned value, typed as the function's return type: postconditions talk
//! about the call, not about its locals.
//!
//! # Static facts
//!
//! The checker never runs user code, so it knows a contract's truth only
//! where it is arithmetic on numbers the source spells out ([`ConstVal`]):
//!
//! * an `assert`, `requires` or `ensures` whose condition is such a constant
//!   expression and false warns that it always fails;
//! * a call whose arguments are all constants (`fall_time(-5 m)`) of a
//!   function with a `requires` that mentions only its parameters and
//!   constants, and that the arguments make false, warns at the call.
//!
//! Both are warnings, not errors, because the checker does not know whether
//! the line runs. A contract that is *true* is left to the SIR `contracts`
//! pass, which sees every call of the function and says so in `--explain`.
//! The evaluator does the interpreter's own arithmetic (`f64`, the unit
//! table's factors through [`crate::apply_unit`]) and gives up (`None`) on
//! anything else: a name it does not know, a power, a call, a division by
//! zero or an overflow.

use std::sync::Arc;

use sky_ast::ast::span_text;
use sky_ast::{
    AssertKind, BinaryOp, Block, BoolOp, Expr, ExprKind, IndexItem, Literal, Param, Stmt, StmtKind,
    UnaryOp,
};
use sky_diag::{Diagnostic, Span};

use crate::check::{Checker, Scope, Ty};
use crate::constant::{apply_unit, entry_value, parse_number};
use sky_diag::display_capped as quote;

/// The `requires` lines of a function, kept so a call with constant
/// arguments can be checked against them.
pub(crate) struct ContractDef {
    /// The parameter names, in order.
    params: Vec<String>,
    /// The condition of each `requires`.
    requires: Vec<Expr>,
}

/// What [`Checker::begin_contracts`] keeps for the end of the body.
pub(crate) struct EntryScope {
    scope: Scope,
}

/// The leading `requires`/`ensures` statements of a `def` body (the parser
/// only allows them there).
fn head(body: &Block) -> impl Iterator<Item = &Stmt> {
    body.stmts.iter().take_while(|s| {
        matches!(
            s.kind,
            StmtKind::Assert {
                kind: AssertKind::Requires | AssertKind::Ensures,
                ..
            }
        )
    })
}

fn ensures_of(body: &Block) -> Vec<&Expr> {
    head(body)
        .filter_map(|s| match &s.kind {
            StmtKind::Assert {
                kind: AssertKind::Ensures,
                cond,
                ..
            } => Some(cond),
            _ => None,
        })
        .collect()
}

impl Checker<'_> {
    /// `assert cond`, `requires cond`, `ensures cond` where it is written.
    /// An `ensures` is checked at the end of its function instead
    /// ([`Checker::finish_contracts`]).
    pub(crate) fn assert_stmt(&mut self, kind: AssertKind, cond: &Expr) {
        if kind == AssertKind::Ensures {
            return;
        }
        if mentions(self.src, cond, "result") && !self.env.is_bound("result") {
            self.error(
                Diagnostic::error(
                    format!("`result` is not available in `{}`", kind.keyword()),
                    cond.span,
                )
                .with_label("`result` is only the returned value inside `ensures`")
                .with_help("state what the result must satisfy in an `ensures` line"),
            );
            self.type_ignoring_result(cond);
            return;
        }
        let t = self.expr(cond);
        self.condition(kind.keyword(), cond, t);
        self.warn_constant_failure(kind, cond);
    }

    /// Before a `def` body: the parameters as they enter it, if the body
    /// has an `ensures` (its scope is theirs), and the check that no
    /// parameter is called `result`.
    pub(crate) fn begin_contracts(&mut self, params: &[Param], body: &Block) -> Option<EntryScope> {
        if ensures_of(body).is_empty() {
            return None;
        }
        for p in params {
            if self.ident(&p.name) == "result" {
                self.error(
                    Diagnostic::error("`result` is the returned value in `ensures`", p.name.span)
                        .with_label("this parameter has the same name")
                        .with_help("rename the parameter, so `ensures` can refer to both"),
                );
            }
        }
        Some(EntryScope {
            scope: self.env.snapshot(),
        })
    }

    /// After a `def` body: checks its `ensures` lines in the entry scope
    /// with `result` bound to the return type (`None` if the function
    /// returns nothing, and then `result` is an error), and remembers the
    /// `requires` for calls.
    pub(crate) fn finish_contracts(
        &mut self,
        def: Span,
        fname: &str,
        params: &[Param],
        body: &Block,
        entry: Option<EntryScope>,
        result: Option<Ty>,
    ) {
        let requires: Vec<Expr> = head(body)
            .filter_map(|s| match &s.kind {
                StmtKind::Assert {
                    kind: AssertKind::Requires,
                    cond,
                    ..
                } => Some(cond.clone()),
                _ => None,
            })
            .collect();
        if !requires.is_empty() {
            let params = params
                .iter()
                .map(|p| self.ident(&p.name).to_string())
                .collect();
            self.contract_defs
                .insert(def, Arc::new(ContractDef { params, requires }));
        }
        let Some(entry) = entry else { return };
        let after = self.env.snapshot();
        let diverged = self.diverged;
        self.env.restore(entry.scope);
        let has_result = result.is_some();
        if let Some(t) = result {
            self.env.bind("result", t);
        }
        for cond in ensures_of(body) {
            if !has_result && mentions(self.src, cond, "result") {
                self.error(
                    Diagnostic::error(
                        format!(
                            "`{}` returns no value, so there is no `result`",
                            quote(fname)
                        ),
                        cond.span,
                    )
                    .with_label("`result` is the returned value")
                    .with_help("return a value, or state the postcondition about the arguments"),
                );
                self.type_ignoring_result(cond);
                continue;
            }
            let first_new = self.diagnostics.len();
            let t = self.expr(cond);
            // A name the body made up is not in scope here.
            for d in self.diagnostics.iter_mut().skip(first_new) {
                if d.message.starts_with("cannot find") {
                    d.help.push(
                        "an `ensures` sees the parameters as passed and `result`, not the function's locals"
                            .to_string(),
                    );
                }
            }
            self.condition("ensures", cond, t);
            self.warn_constant_failure(AssertKind::Ensures, cond);
        }
        self.env.restore(after);
        self.diverged = diverged;
    }

    /// Types `cond` with `result` unknown, so the rest of it is checked and
    /// every expression has a type (the error about `result` was reported).
    fn type_ignoring_result(&mut self, cond: &Expr) {
        let saved = self.env.snapshot();
        self.env.bind("result", Ty::Unknown);
        self.expr(cond);
        self.env.restore(saved);
    }

    /// A condition that is a constant expression and false always fails.
    fn warn_constant_failure(&mut self, kind: AssertKind, cond: &Expr) {
        if self.const_value(cond, &[]) != Some(ConstVal::Bool(false)) {
            return;
        }
        let word = kind.keyword();
        let text = self.quoted(cond.span);
        let message = match kind {
            AssertKind::Assert => format!("this `assert` always fails: `{text}` is false"),
            AssertKind::Requires => {
                format!("this `requires` can never hold: `{text}` is false")
            }
            AssertKind::Ensures => {
                format!("this `ensures` can never hold: `{text}` is false")
            }
        };
        self.diagnostics.push(
            Diagnostic::warning(message, cond.span)
                .with_label("always false")
                .with_note(format!(
                    "the condition uses only numbers written in the source, so `{word}` fails whenever it runs"
                )),
        );
    }

    /// A call of a function with a `requires` that the constant arguments
    /// make false: warns at the call. Only when every argument is a
    /// constant and the condition is decided by them alone.
    pub(crate) fn warn_call_breaks_contract(
        &mut self,
        def: Span,
        name: &str,
        args: &[Expr],
        call: Span,
    ) {
        let Some(contract) = self.contract_defs.get(&def).cloned() else {
            return;
        };
        if args.len() != contract.params.len() {
            return;
        }
        let mut bound: Vec<(&str, f64)> = Vec::new();
        let mut written: Vec<(&str, Span)> = Vec::new();
        for (p, a) in contract.params.iter().zip(args) {
            if let Some(v) = entry_value(self.src, a) {
                bound.push((p.as_str(), v));
                written.push((p.as_str(), a.span));
            }
        }
        for cond in &contract.requires {
            if self.const_value(cond, &bound) != Some(ConstVal::Bool(false)) {
                continue;
            }
            let shown: Vec<String> = written
                .iter()
                .filter(|(p, _)| mentions(self.src, cond, p))
                .map(|(p, a)| format!("{} = {}", quote(p), self.quoted(*a)))
                .collect();
            let mut d = Diagnostic::warning(
                format!("this call breaks the precondition of `{}`", quote(name)),
                call,
            )
            .with_label("the `requires` is false for these arguments")
            .with_secondary(
                cond.span,
                format!("`{}` is false here", self.quoted(cond.span)),
            );
            if !shown.is_empty() {
                d = d.with_note(format!("with {}", shown.join(", ")));
            }
            d = d.with_note("the call fails at run time if it runs");
            self.diagnostics.push(d);
            return;
        }
    }

    /// The value of `e` if it is arithmetic on numbers the source spells
    /// out, with the names in `bound` standing for their constants. See the
    /// module documentation for what it gives up on.
    fn const_value(&self, e: &Expr, bound: &[(&str, f64)]) -> Option<ConstVal> {
        const_eval(self.src, e, bound, 0)
    }
}

/// A constant's value.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum ConstVal {
    Num(f64),
    Bool(bool),
}

const MAX_CONST_DEPTH: usize = 64;

fn finite(x: f64) -> Option<ConstVal> {
    x.is_finite().then_some(ConstVal::Num(x))
}

fn const_eval(src: &str, e: &Expr, bound: &[(&str, f64)], depth: usize) -> Option<ConstVal> {
    if depth > MAX_CONST_DEPTH {
        return None;
    }
    let go = |x: &Expr| const_eval(src, x, bound, depth + 1);
    match &e.kind {
        ExprKind::Paren(inner) => go(inner),
        ExprKind::Literal(Literal::Int(s) | Literal::Float(s)) => {
            finite(parse_number(span_text(src, *s)?).ok()?)
        }
        ExprKind::Literal(Literal::Bool { value, .. }) => Some(ConstVal::Bool(*value)),
        ExprKind::UnitLiteral {
            value: Literal::Int(s) | Literal::Float(s),
            unit,
        } => {
            let v = parse_number(span_text(src, *s)?).ok()?;
            finite(apply_unit(src, unit, v).ok()?.0)
        }
        ExprKind::Name(id) => {
            let name = span_text(src, id.span)?;
            let (_, v) = bound.iter().find(|(n, _)| *n == name)?;
            finite(*v)
        }
        ExprKind::Unary { op, operand, .. } => match (op, go(operand)?) {
            (UnaryOp::Neg, ConstVal::Num(x)) => finite(-x),
            (UnaryOp::Pos, ConstVal::Num(x)) => finite(x),
            (UnaryOp::Not, ConstVal::Bool(b)) => Some(ConstVal::Bool(!b)),
            _ => None,
        },
        ExprKind::Binary { op, lhs, rhs, .. } => {
            let (ConstVal::Num(a), ConstVal::Num(b)) = (go(lhs)?, go(rhs)?) else {
                return None;
            };
            match op {
                BinaryOp::Add => finite(a + b),
                BinaryOp::Sub => finite(a - b),
                BinaryOp::Mul => finite(a * b),
                BinaryOp::Div if b != 0.0 => finite(a / b),
                BinaryOp::Eq => Some(ConstVal::Bool(a == b)),
                BinaryOp::Ne => Some(ConstVal::Bool(a != b)),
                BinaryOp::Lt => Some(ConstVal::Bool(a < b)),
                BinaryOp::Le => Some(ConstVal::Bool(a <= b)),
                BinaryOp::Gt => Some(ConstVal::Bool(a > b)),
                BinaryOp::Ge => Some(ConstVal::Bool(a >= b)),
                _ => None,
            }
        }
        ExprKind::BoolOp { op, operands, .. } => {
            // Left to right, stopping at the operand that decides: the ones
            // after it never run, whatever they are.
            let decisive = *op == BoolOp::Or;
            for operand in operands {
                match go(operand)? {
                    ConstVal::Bool(b) if b == decisive => return Some(ConstVal::Bool(decisive)),
                    ConstVal::Bool(_) => {}
                    ConstVal::Num(_) => return None,
                }
            }
            Some(ConstVal::Bool(!decisive))
        }
        _ => None,
    }
}

/// Whether `e` reads the name `name`.
fn mentions(src: &str, e: &Expr, name: &str) -> bool {
    let go = |x: &Expr| mentions(src, x, name);
    match &e.kind {
        ExprKind::Name(id) => span_text(src, id.span) == Some(name),
        ExprKind::Literal(_) | ExprKind::UnitLiteral { .. } => false,
        ExprKind::Unary { operand, .. }
        | ExprKind::Inverse { operand, .. }
        | ExprKind::Transpose { operand, .. } => go(operand),
        ExprKind::Paren(inner) => go(inner),
        ExprKind::Binary { lhs, rhs, .. } => go(lhs) || go(rhs),
        ExprKind::BoolOp { operands, .. } | ExprKind::Array(operands) => operands.iter().any(go),
        ExprKind::IfElse { then, cond, else_ } => go(then) || go(cond) || go(else_),
        ExprKind::As { value, .. } => go(value),
        ExprKind::Attribute { base, .. } => go(base),
        ExprKind::Index { base, items } => {
            go(base)
                || items.iter().any(|i| match i {
                    IndexItem::Expr(x) => go(x),
                    IndexItem::Slice { start, stop, .. } => {
                        [start, stop].into_iter().flatten().any(&go)
                    }
                })
        }
        ExprKind::Call { callee, args } => go(callee) || args.iter().any(go),
    }
}
