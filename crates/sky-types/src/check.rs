//! The checker walk: types a parsed [`Module`] and reports diagnostics.
//!
//! This is the first slice that consumes the real AST (`sky-ast`). It types
//! number literals with units, names and their bindings, the arithmetic and
//! comparison operators, `@`, postfix `⁻¹` (and `inv(x)`), `^`, the Bool
//! literals `true`/`false`, and `and`/`or`/`not` (strict Bool operands, no
//! truthiness, per the spec's "Booleans and conditions"), the builtin
//! `print(...)` call, and calls to user functions. Anything else (strings,
//! indexing, attributes, `∂`, `∇`, calls to names that are not functions) is typed as
//! [`Ty::Unknown`] without a diagnostic, and an operator with an unknown
//! operand is unknown too, so one error never cascades.
//!
//! Blocks, control flow, and functions (`if`/`while`/`def`, scoping, calls to
//! user functions) live in [`crate::flow`]; the rules are in the spec's
//! "Blocks and control flow".
//!
//! # Types
//!
//! Every numeric value is a [`ValueTy`]: an element [`Dimension`] plus a
//! [`TensorType`] (shape and property tags). A scalar is the rank-0 case.
//!
//! # Declaring shapes, tags, and units
//!
//! A binding annotation is a [`TypeExpr`]: `A: PositiveDefinite Matrix[N, N]
//! kg` declares a positive-definite square matrix of kilograms. Without a
//! value the name is an input, and its tags are taken on trust (this is how a
//! program gets a positive-definite matrix). With a value, the value must
//! have the declared shape, dimension (if a unit is declared), and tags (the
//! value's closed tag set must include them). Tests and other front ends can
//! also seed names through [`Env`] and [`check_module_with`].
//!
//! # Purity
//!
//! The walk never evaluates user code. The only "values" it looks at are
//! constant exponents like the `2` in `x^2`, read from the source text.
//!
//! # Output size
//!
//! Source text quoted in a diagnostic (and in the `--explain` text, and in
//! shape-variable names shown in types) is capped at
//! [`crate::QUOTE_MAX_CHARS`] characters; spans still cover the full text.
//! At most [`MAX_STORED_REWRITES`] rewrite sites are stored, and later ones
//! are only counted in [`Checked::rewrites_omitted`].
//!
//! # Depth
//!
//! The walk recurses once per nesting level, and the parser caps nesting at
//! [`sky_ast::MAX_DEPTH`]. Callers run it on a thread with a deep stack, like
//! the parser (see the crate's `deep_nesting` test).

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use sky_ast::ast::span_text;
use sky_ast::{
    AssignTarget, BinaryOp, CompoundOp, DimExpr, Expr, ExprKind, Ident, IndexItem, Literal, Module,
    ShapeKind, Stmt, StmtKind, TypeExpr, UnaryOp, UnitExpr,
};
use sky_diag::{Diagnostic, Severity, Span};
use sky_units::{lookup, lookup_constant, suggest_unit, Dimension, Rational, CONSTANTS};

use crate::flow::{FnCtx, FuncSig, LoopFrame};
use crate::provenance::{label_dim, Provenance};
use crate::rewrite::{cholesky_rewrite, MatmulLhs, Rewrite};
use crate::rules::{elementwise, inverse, matmul, TensorType};
use crate::scale::{negated_tags, scaled_tags};
use crate::shape::{Shape, ShapeDim, ShapeError, ShapeUnifier, ShapeVar};
use crate::tags::{Tag, TagSet};
use sky_diag::display_capped as quote;

/// The type of a numeric value: element dimension plus shape and tags.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ValueTy {
    /// The dimension of every element (for a scalar, of the value).
    pub dim: Dimension,
    /// Shape and property tags. Rank 0 is a scalar.
    pub ty: TensorType,
}

impl ValueTy {
    /// A scalar of dimension `dim`.
    pub fn scalar(dim: Dimension) -> Self {
        ValueTy {
            dim,
            ty: TensorType::scalar(),
        }
    }

    /// Whether this is a scalar (rank 0).
    pub fn is_scalar(&self) -> bool {
        self.ty.shape().rank() == 0
    }
}

/// The type the walk assigns to an expression or a name.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Ty {
    /// A number, vector, matrix, or tensor.
    Value(ValueTy),
    /// The result of a comparison.
    Bool,
    /// A function defined with `def`: its signature, for checking calls.
    Func(Arc<FuncSig>),
    /// Not typed in this slice, or the result of an error already reported.
    Unknown,
}

impl Ty {
    /// A scalar of dimension `dim`.
    pub fn scalar(dim: Dimension) -> Self {
        Ty::Value(ValueTy::scalar(dim))
    }

    /// An array (or scalar) with element dimension `dim`.
    pub fn value(dim: Dimension, ty: TensorType) -> Self {
        Ty::Value(ValueTy { dim, ty })
    }
}

impl fmt::Display for ValueTy {
    /// `Matrix[3,3] {Symmetric} [L]`: shape, tags if any, element dimension.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{}]", self.ty, self.dim)
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ty::Value(v) => v.fmt(f),
            Ty::Bool => f.write_str("Bool"),
            Ty::Func(sig) => write!(f, "function `{}`", sig.name()),
            Ty::Unknown => f.write_str("?"),
        }
    }
}

/// Names in scope, plus the shape unifier and the named shape variables.
///
/// Shape variables are named per module: every `N` in the module's
/// annotations (and in seeded types from [`Env::shape_var`]) is the same
/// variable.
#[derive(Clone, Debug, Default)]
pub struct Env {
    names: BTreeMap<String, Ty>,
    /// Names still bound to a predefined constant (see [`Env::with_prelude`]).
    /// A name leaves the set when the program rebinds it, which is when the
    /// checker warns once.
    pub(crate) prelude: BTreeSet<String>,
    pub(crate) shape_vars: BTreeMap<String, ShapeVar>,
    pub(crate) shapes: ShapeUnifier,
    /// Where each name was last bound in the source, for labels. Names
    /// seeded through [`Env::bind`] have none.
    pub(crate) origins: BTreeMap<String, Span>,
    /// Names bound on some paths through the code so far but not all
    /// (after an `if` without `else`, say), mapped to one of the bindings
    /// and whether the paths were those of a loop. Such a name is not in
    /// `names`: using it is an error.
    pub(crate) maybe: BTreeMap<String, (Span, bool)>,
}

/// The name-binding part of an [`Env`] at one point of the walk: what an
/// `if` branch or loop body starts from and ends with, and what merges
/// combine. The shape unifier and shape-variable names are deliberately not
/// part of it (see [`crate::flow`]).
#[derive(Clone, Debug, Default)]
pub(crate) struct Scope {
    pub(crate) names: BTreeMap<String, Ty>,
    pub(crate) origins: BTreeMap<String, Span>,
    pub(crate) maybe: BTreeMap<String, (Span, bool)>,
}

impl Env {
    /// An empty environment.
    pub fn new() -> Self {
        Env::default()
    }

    /// An environment with every physical constant bound as a scalar of its
    /// dimension, under both its Unicode and its ASCII spelling (`ħ` and
    /// `hbar`), so `E = m * c^2` is energy when `m` is a mass. This is what
    /// [`check_module`] (and so `sky check`) starts from; [`Env::new`] and
    /// [`check_module_with`] do not add the prelude on their own.
    ///
    /// A program may rebind a constant; the checker then warns once, because
    /// `c` and `h` are also common variable names.
    pub fn with_prelude() -> Self {
        let mut env = Env::new();
        for c in CONSTANTS {
            for name in [Some(c.name), c.ascii].into_iter().flatten() {
                env.names.insert(name.to_string(), Ty::scalar(c.dim));
                env.prelude.insert(name.to_string());
            }
        }
        env
    }

    /// The shape variable called `name`, created on first use.
    pub fn shape_var(&mut self, name: &str) -> ShapeVar {
        if let Some(v) = self.shape_vars.get(name) {
            return v.clone();
        }
        // The name is for display only (identity is the id), so store it
        // capped like any quoted source text.
        let v = self.shapes.fresh(Arc::<str>::from(quote(name)));
        self.shape_vars.insert(name.to_string(), v.clone());
        v
    }

    /// Binds `name` to `ty`, replacing any earlier binding.
    pub fn bind(&mut self, name: impl Into<String>, ty: Ty) {
        let name = name.into();
        self.prelude.remove(&name);
        self.origins.remove(&name);
        self.maybe.remove(&name);
        self.names.insert(name, ty);
    }

    /// The current bindings, to restore later or merge with others.
    pub(crate) fn snapshot(&self) -> Scope {
        Scope {
            names: self.names.clone(),
            origins: self.origins.clone(),
            maybe: self.maybe.clone(),
        }
    }

    /// Replaces the bindings with `scope`.
    pub(crate) fn restore(&mut self, scope: Scope) {
        self.names = scope.names;
        self.origins = scope.origins;
        self.maybe = scope.maybe;
    }

    /// Whether `name` is bound (not merely maybe-bound).
    pub(crate) fn is_bound(&self, name: &str) -> bool {
        self.names.contains_key(name)
    }

    /// Binds `name` to an array with element dimension `dim`. Fails if
    /// `tags` is non-empty and `shape` is not a square matrix.
    pub fn bind_array(
        &mut self,
        name: impl Into<String>,
        shape: Shape,
        tags: TagSet,
        dim: Dimension,
    ) -> Result<(), ShapeError> {
        let ty = TensorType::tagged(&mut self.shapes, shape, tags)?;
        self.bind(name, Ty::value(dim, ty));
        Ok(())
    }

    /// The type of `name`, with solved shape variables substituted.
    pub fn get(&self, name: &str) -> Option<Ty> {
        self.names.get(name).map(|t| self.resolve(t))
    }

    /// Every binding in name order, with solved shape variables substituted.
    pub fn bindings(&self) -> impl Iterator<Item = (&str, Ty)> + '_ {
        self.names
            .iter()
            .map(|(n, t)| (n.as_str(), self.resolve(t)))
    }

    /// The shape unifier (for callers that inspect solved variables).
    pub fn shapes(&self) -> &ShapeUnifier {
        &self.shapes
    }

    fn resolve(&self, t: &Ty) -> Ty {
        match t {
            Ty::Value(v) => Ty::value(v.dim, self.shapes.apply_type(&v.ty)),
            other => other.clone(),
        }
    }
}

/// A `A⁻¹ @ b` site the checker decided to rewrite to a Cholesky solve.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RewriteSite {
    /// The whole `A⁻¹ @ b` expression.
    pub span: Span,
    /// `A`, the operand of `⁻¹`.
    pub matrix: Span,
    /// `b`, the right operand of `@`.
    pub rhs: Span,
    /// The plan, including the `--explain` text.
    pub rewrite: Rewrite,
}

impl RewriteSite {
    /// A note diagnostic for `sky check --explain`: points at the rewritten
    /// expression and carries the explanation.
    pub fn to_diagnostic(&self, source: &str) -> Diagnostic {
        let text = quote(span_text(source, self.span).unwrap_or("A⁻¹ @ b"));
        Diagnostic::note(
            format!("`{text}` is rewritten to a Cholesky solve"),
            self.span,
        )
        .with_secondary(self.matrix, "positive definite")
        .with_note(self.rewrite.explanation())
    }
}

/// Most [`RewriteSite`]s kept in [`Checked::rewrites`]; later ones are only
/// counted, in [`Checked::rewrites_omitted`]. The cap never stops the walk
/// and never affects diagnostics.
pub const MAX_STORED_REWRITES: usize = 100;

/// The result of checking a module.
#[derive(Clone, Debug)]
pub struct Checked {
    /// Errors and warnings, in source order of the walk.
    pub diagnostics: Vec<Diagnostic>,
    /// Rewrite decisions, in source order: at most [`MAX_STORED_REWRITES`],
    /// the first ones found. This bounds `sky check --explain` output.
    pub rewrites: Vec<RewriteSite>,
    /// Rewrite decisions made after [`MAX_STORED_REWRITES`] were stored.
    /// They are still rewrites; only their explanations are dropped.
    pub rewrites_omitted: usize,
    /// The environment after the last statement.
    pub env: Env,
}

impl Checked {
    /// Whether any diagnostic is an error.
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error)
    }
}

/// Checks `module` (parsed from `source`) in an environment holding only the
/// predefined physical constants ([`Env::with_prelude`]).
pub fn check_module(module: &Module, source: &str) -> Checked {
    check_module_with(module, source, Env::with_prelude())
}

/// Checks `module` (parsed from `source`) starting from `env`, which can
/// seed names with shapes, tags, and element units without source text.
pub fn check_module_with(module: &Module, source: &str, env: Env) -> Checked {
    let mut c = Checker {
        src: source,
        shadow_reported: Vec::new(),
        env,
        diagnostics: Vec::new(),
        rewrites: Vec::new(),
        rewrites_omitted: 0,
        fns: Vec::new(),
        loops: Vec::new(),
        diverged: false,
        prov: Provenance::new(source),
    };
    for stmt in &module.stmts {
        c.stmt(stmt);
    }
    Checked {
        diagnostics: c.diagnostics,
        rewrites: c.rewrites,
        rewrites_omitted: c.rewrites_omitted,
        env: c.env,
    }
}

/// A type annotation once read: what the name is declared to be.
pub(crate) struct Declared {
    /// Shape and closed tags.
    pub(crate) ty: TensorType,
    /// The declared element dimension, or `None` if the annotation names no
    /// unit (then the value's dimension is used, or dimensionless).
    pub(crate) dim: Option<Dimension>,
    /// The tags as written, before closure.
    written: TagSet,
    unit_span: Option<Span>,
}

pub(crate) struct Checker<'s> {
    pub(crate) src: &'s str,
    /// Size-variable names already reported as shadowing a value in the
    /// annotation being read.
    shadow_reported: Vec<String>,
    pub(crate) env: Env,
    diagnostics: Vec<Diagnostic>,
    rewrites: Vec<RewriteSite>,
    rewrites_omitted: usize,
    /// The `def`s being checked, innermost last.
    pub(crate) fns: Vec<FnCtx>,
    /// The `while` loops around the current statement within the current
    /// function (or the module), innermost last.
    pub(crate) loops: Vec<LoopFrame>,
    /// Whether the statement just checked cannot fall through: it ended in
    /// `return`, `break`, or `continue`, or every branch of an `if` did.
    pub(crate) diverged: bool,
    /// Why each binding has its dimension (see [`crate::provenance`]).
    pub(crate) prov: Provenance,
}

/// Human wording for a dimension in error text: its name if it has one
/// (`length`), otherwise its formula (`L·M`).
pub(crate) fn describe(d: &Dimension) -> String {
    match d.name() {
        Some(n) => n.to_string(),
        None => d.to_string(),
    }
}

/// The exact value of a decimal literal's source text, `INT.FRAC`, as a
/// reduced [`Rational`]: `0.5` is 1/2, `1.5` is 3/2, `2.0` is 2. Never goes
/// through `f64`, so `0.1` is exactly 1/10. `None` for an exponent form
/// (`1e-1`), separators, or a numerator or denominator that does not fit in
/// `i64`: the caller reports "must be a constant number".
fn exact_decimal(text: &str) -> Option<Rational> {
    let (int, frac) = text.split_once('.')?;
    // Trailing zeros carry no value (`2.500` is `2.5`), and dropping them
    // keeps a long but harmless literal from overflowing the denominator.
    let frac = frac.trim_end_matches('0');
    let all_numeric = |s: &str| s.bytes().all(|b| b.is_ascii_digit());
    if int.is_empty() || !all_numeric(int) || !all_numeric(frac) {
        return None;
    }
    let num: i64 = format!("{int}{frac}").parse().ok()?;
    let den = 10i64.checked_pow(u32::try_from(frac.len()).ok()?)?;
    Rational::checked_new(num, den).ok()
}

fn rank_noun(rank: usize) -> String {
    match rank {
        0 => "scalar".to_string(),
        1 => "vector".to_string(),
        2 => "matrix".to_string(),
        n => format!("rank-{n} tensor"),
    }
}

pub(crate) fn strip_parens(mut e: &Expr) -> &Expr {
    while let ExprKind::Paren(inner) = &e.kind {
        e = inner;
    }
    e
}

/// A constant exponent: an integer or decimal literal, possibly negated or
/// parenthesized, or a quotient of two such (`(1/2)`). A decimal literal is
/// read from its source text as the exact fraction it spells (`0.5` is 1/2);
/// see [`exact_decimal`].
///
/// The single definition shared by the checker (for dimensions) and the
/// interpreter (for values), so the two can never disagree about which
/// exponents are constant.
pub fn const_exponent(src: &str, e: &Expr) -> Option<Rational> {
    let text = |span: Span| span_text(src, span);
    match &e.kind {
        ExprKind::Literal(Literal::Float(span)) => exact_decimal(text(*span)?),
        ExprKind::Literal(Literal::Int(span)) => {
            text(*span)?.parse::<i64>().ok().map(Rational::integer)
        }
        ExprKind::Paren(inner) => const_exponent(src, inner),
        ExprKind::Unary {
            op: UnaryOp::Neg,
            operand,
            ..
        } => const_exponent(src, operand)?.checked_neg().ok(),
        ExprKind::Unary {
            op: UnaryOp::Pos,
            operand,
            ..
        } => const_exponent(src, operand),
        ExprKind::Binary {
            op: BinaryOp::Div,
            lhs,
            rhs,
            ..
        } => {
            let (a, b) = (const_exponent(src, lhs)?, const_exponent(src, rhs)?);
            if b.is_zero() {
                return None;
            }
            a.checked_div(b).ok()
        }
        _ => None,
    }
}

impl<'s> Checker<'s> {
    pub(crate) fn text(&self, span: Span) -> &'s str {
        span_text(self.src, span).unwrap_or("…")
    }

    pub(crate) fn ident(&self, id: &Ident) -> &'s str {
        self.text(id.span)
    }

    /// Source text for a diagnostic: [`quote`]d (capped and escaped). Use
    /// this, never [`Self::text`], for any text shown to the user.
    pub(crate) fn quoted(&self, span: Span) -> Cow<'s, str> {
        quote(self.text(span))
    }

    pub(crate) fn error(&mut self, d: Diagnostic) {
        self.diagnostics.push(d);
    }

    /// `this is Matrix[3,3] [1]` for secondary labels.
    pub(crate) fn this_is(&self, v: &ValueTy) -> String {
        format!(
            "this is {}",
            ValueTy {
                dim: v.dim,
                ty: self.env.shapes.apply_type(&v.ty),
            }
        )
    }

    /// Whether `t` is a matrix whose two dimensions are known to be equal
    /// (the same number, or the same shape variable).
    fn is_square_matrix(&self, t: &TensorType) -> bool {
        match self.env.shapes.apply(t.shape()).dims() {
            [rows, cols] => rows == cols,
            _ => false,
        }
    }

    /// `e`'s source text, parenthesized unless it is an atom (a name,
    /// literal, unit literal, call, index, attribute, or parenthesized
    /// expression), so it can be spliced into a larger suggested expression
    /// such as `(K⁻¹)⁻¹` or `(-2) * ones(A)`.
    pub(crate) fn operand_text(&self, e: &Expr) -> String {
        let text = self.quoted(e.span);
        match e.kind {
            ExprKind::Binary { .. } | ExprKind::Unary { .. } | ExprKind::Inverse { .. } => {
                format!("({text})")
            }
            _ => text.to_string(),
        }
    }

    /// If `t` is Bool, reports that `what` needs a number (v0.1 has no
    /// truthiness in either direction) and returns true.
    fn reject_bool(&mut self, what: &str, e: &Expr, t: &Ty) -> bool {
        if *t != Ty::Bool {
            return false;
        }
        self.error(
            Diagnostic::error(format!("`{what}` needs a number, found Bool"), e.span)
                .with_label("this is Bool")
                .with_note("v0.1 has no truthiness: a Bool is never a number"),
        );
        true
    }

    /// Checks one operand of `and`, `or`, or `not` (`kw`): it must be a
    /// Bool. Numbers and tensors are errors (no truthiness), on the
    /// operand's span.
    fn bool_operand(&mut self, kw: &str, e: &Expr, t: Ty) {
        self.bool_needed(kw, "operand", e, t);
    }

    /// Checks the condition of `if`, `elif`, `while`, or a conditional
    /// expression (`kw`, without backticks): a Bool of shape `()`, the same
    /// rule and help as the operands of `and`/`or`/`not`.
    pub(crate) fn condition(&mut self, kw: &str, e: &Expr, t: Ty) {
        self.bool_needed(kw, "condition", e, t);
    }

    /// The shared check behind [`Self::bool_operand`] and
    /// [`Self::condition`]; `noun` is "operand" or "condition".
    fn bool_needed(&mut self, kw: &str, noun: &str, e: &Expr, t: Ty) {
        let v = match t {
            Ty::Value(v) => v,
            Ty::Func(sig) => {
                self.error(
                    Diagnostic::error(
                        format!(
                            "`{kw}` needs a Bool {noun}, found function `{}`",
                            quote(sig.name())
                        ),
                        e.span,
                    )
                    .with_label("not a Bool")
                    .with_help("call it to get a value, then compare that value"),
                );
                return;
            }
            _ => return, // Bool, or unknown (already reported).
        };
        let found = ValueTy {
            dim: v.dim,
            ty: self.env.shapes.apply_type(&v.ty),
        };
        let mut d = Diagnostic::error(format!("`{kw}` needs a Bool {noun}, found {found}"), e.span)
            .with_label("not a Bool")
            .with_note("v0.1 has no truthiness");
        if v.is_scalar() {
            let text = self.operand_text(e);
            d = match zero_of(&v.dim) {
                Some(zero) => d.with_help(format!("compare it instead, e.g. `{text} != {zero}`")),
                None => d.with_help("compare it with a value of the same dimension"),
            };
        } else {
            d = d.with_help("a condition must be a single `Bool` (shape `()`)");
        }
        self.error(d);
    }

    /// Binds the name `id` to `ty`. The first time a program rebinds a
    /// predefined constant, warns that it shadows it: allowed (like a Python
    /// builtin), but `c = 3` or `h = 0.1` silently losing the constant is
    /// exactly the surprise this is meant to prevent.
    fn bind_name(&mut self, id: &Ident, ty: Ty) {
        self.warn_if_shadowing(id);
        let name = self.ident(id);
        self.bind_at(name, ty, id.span);
    }

    fn warn_if_shadowing(&mut self, id: &Ident) {
        let name = self.ident(id);
        if !self.env.prelude.contains(name) {
            return;
        }
        let Some(c) = lookup_constant(name) else {
            return;
        };
        let q = self.quoted(id.span);
        let d = Diagnostic::warning(format!("`{q}` shadows {}", c.description), id.span)
            .with_label("shadows a predefined constant")
            .with_note(format!(
                "`{q}` is predefined by Sky: {} ({}, {:e} in SI units{}), defined in the sky-units constants table",
                c.description,
                describe(&c.dim),
                c.value,
                // π is not a measured constant.
                if c.dim.is_dimensionless() { "" } else { ", CODATA 2018" }
            ))
            .with_help(format!(
                "rename your variable to keep using the constant, e.g. `my_{q} = …`"
            ));
        self.diagnostics.push(d);
    }

    /// Binds `name` at `span` (its origin, for labels), as a local of the
    /// function being checked if there is one.
    pub(crate) fn bind_at(&mut self, name: &str, ty: Ty, span: Span) {
        self.env.bind(name, ty);
        self.env.origins.insert(name.to_string(), span);
        if let Some(ctx) = self.fns.last_mut() {
            ctx.locals.insert(name.to_string());
        }
    }

    pub(crate) fn stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Expr(e) => {
                self.expr(e);
            }
            StmtKind::Assign { target, value } => {
                let t = self.expr(value);
                match target {
                    AssignTarget::Name(id) => {
                        self.note_binding(id.span, Some(value), None);
                        self.bind_name(id, t);
                    }
                    _ => self.reject_write(target, "="),
                }
            }
            StmtKind::CompoundAssign {
                target,
                op,
                op_span,
                value,
            } => match target {
                AssignTarget::Name(id) => self.compound_assign(id, *op, *op_span, value),
                _ => {
                    self.expr(value);
                    self.reject_write(target, op.symbol());
                }
            },
            StmtKind::Binding { name, ty, value } => {
                let declared = self.annotation(ty);
                let found = value.as_ref().map(|v| (v, self.expr(v)));
                let declares_unit = declared.as_ref().is_some_and(|d| d.dim.is_some());
                self.note_binding(name.span, value.as_ref(), declares_unit.then_some(ty.span));
                let bound = self.binding_type(ty, declared, found);
                self.bind_name(name, bound);
            }
            StmtKind::If {
                branches,
                else_block,
            } => self.if_stmt(stmt.span, branches, else_block.as_ref()),
            StmtKind::While { cond, body } => self.while_stmt(stmt.span, cond, body),
            StmtKind::Def {
                name,
                params,
                ret,
                body,
            } => self.def_stmt(name, params, ret.as_ref(), body),
            StmtKind::Return(value) => self.return_stmt(stmt.span, value.as_ref()),
            StmtKind::Pass => {}
            StmtKind::Break => self.break_stmt(stmt.span),
            StmtKind::Continue => self.continue_stmt(stmt.span),
        }
    }

    /// The type a binding gives its name. The declaration wins: its shape,
    /// its tags, and (if it names one) its unit. A value must agree with all
    /// three, and only fills in what the declaration leaves open, which is
    /// the unit: without a declared unit the dimension comes from the value,
    /// or is dimensionless when there is no value.
    ///
    /// Tags are a promise the checker must be able to keep: the value's
    /// closed tag set has to include the declared ones. A declaration with
    /// no value is an input, and its tags are taken on trust.
    fn binding_type(
        &mut self,
        ty: &TypeExpr,
        declared: Option<Declared>,
        found: Option<(&Expr, Ty)>,
    ) -> Ty {
        let Some(declared) = declared else {
            return found.map_or(Ty::Unknown, |(_, t)| t);
        };
        let Declared {
            ty: tensor,
            dim,
            written,
            unit_span,
        } = declared;
        let declared_dim = dim.unwrap_or(Dimension::DIMENSIONLESS);
        let Some((value, found)) = found else {
            return Ty::value(declared_dim, tensor);
        };
        let v = match found {
            Ty::Value(v) => v,
            Ty::Bool => {
                self.error(
                    Diagnostic::error("expected a number, found Bool", value.span)
                        .with_label("this is Bool")
                        .with_secondary(ty.span, "expected because of this annotation")
                        .with_note("v0.1 has no truthiness: a Bool is never a number"),
                );
                return Ty::value(declared_dim, tensor);
            }
            Ty::Func(sig) => {
                self.error(
                    Diagnostic::error(
                        format!("expected a number, found function `{}`", quote(sig.name())),
                        value.span,
                    )
                    .with_label("this is a function")
                    .with_secondary(ty.span, "expected because of this annotation"),
                );
                return Ty::value(declared_dim, tensor);
            }
            Ty::Unknown => return Ty::value(declared_dim, tensor),
        };
        if let Err(e) = self.env.shapes.unify(tensor.shape(), v.ty.shape()) {
            let label = self.this_is(&v);
            self.error(
                Diagnostic::error(format!("mismatched shapes: {e}"), value.span)
                    .with_label(label)
                    .with_secondary(ty.span, "expected because of this annotation"),
            );
        }
        if let (Some(want), Some(unit)) = (dim, unit_span) {
            if want != v.dim {
                let d = Diagnostic::error(
                    format!("expected {}, found {}", describe(&want), describe(&v.dim)),
                    value.span,
                )
                .with_label(format!("this is {}", label_dim(&v.dim)))
                .with_secondary(
                    unit,
                    format!("expected {} because of this unit", describe(&want)),
                );
                let d = self.with_derivations(d, &[(value, v.dim)]);
                self.error(d);
            }
        }
        // The first property the user wrote that the value isn't known to
        // have (the closure adds implied ones, which they didn't write).
        let missing = written.iter().find(|t| !v.ty.tags().closure().contains(*t));
        if let Some(tag) = missing {
            let label = self.this_is(&v);
            let subject = self.quoted(value.span);
            let mut d = Diagnostic::error(
                format!("`{subject}` is not known to be {}", tag.name()),
                value.span,
            )
            .with_label(label)
            .with_secondary(ty.span, "expected because of this annotation")
            .with_note(
                "the checker only knows a matrix property if it can prove it: from a declaration, or an operation that preserves it",
            );
            d = match &value.kind {
                ExprKind::Name(_) => {
                    let shape = self.env.shapes.apply(v.ty.shape());
                    d.with_help(format!(
                        "if `{subject}` is an input with this property, declare it that way where it is defined: `{subject}: {} {shape}`",
                        tag.name()
                    ))
                }
                _ => d.with_help(format!(
                    "declare the {} property on an input the value is computed from",
                    tag.name()
                )),
            };
            self.error(d);
        }
        Ty::value(dim.unwrap_or(v.dim), tensor)
    }

    /// Types every expression inside `[...]`, slice bounds included.
    fn index_items(&mut self, items: &[IndexItem]) {
        for item in items {
            match item {
                IndexItem::Expr(e) => {
                    self.expr(e);
                }
                IndexItem::Slice { start, stop, .. } => {
                    for bound in [start, stop].into_iter().flatten() {
                        self.expr(bound);
                    }
                }
            }
        }
    }

    /// Reports a write through an index or attribute target (`A[i] = x`,
    /// `a.b += 1`), which the v0.1 checker rejects (spec: "Statements and
    /// assignment"). The target's own subexpressions are still typed.
    fn reject_write(&mut self, target: &AssignTarget, op: &str) {
        let d = match target {
            AssignTarget::Name(_) => return,
            AssignTarget::Index {
                base, items, span, ..
            } => {
                self.expr(base);
                self.index_items(items);
                Diagnostic::error(
                    format!(
                        "cannot assign to an element of `{}`",
                        self.quoted(base.span)
                    ),
                    *span,
                )
                .with_note("arrays are immutable in v0.1")
            }
            AssignTarget::Attribute { base, name, span } => {
                self.expr(base);
                Diagnostic::error(
                    format!("cannot assign to attribute `{}`", self.quoted(name.span)),
                    *span,
                )
                .with_note("nothing has assignable attributes in v0.1")
            }
        };
        self.error(d.with_label(format!("written with `{op}`")));
    }

    /// `x op= value`: types `x op value` like the binary expression, then
    /// requires the result to keep `x`'s type (dimension, shape, and tags).
    /// `x` keeps its type either way.
    fn compound_assign(&mut self, id: &Ident, op: CompoundOp, op_span: Span, value: &Expr) {
        if self.reject_captured_update(id, op, op_span) {
            self.expr(value);
            return;
        }
        let name_expr = Expr {
            kind: ExprKind::Name(*id),
            span: id.span,
        };
        self.warn_if_shadowing(id);
        // Silent lookup: typing `whole` below reports an undefined name once.
        let declared = self.env.get(self.ident(id)).unwrap_or(Ty::Unknown);
        let whole = Expr {
            kind: ExprKind::Binary {
                op: op.binary(),
                op_span,
                lhs: Box::new(name_expr.clone()),
                rhs: Box::new(value.clone()),
            },
            span: id.span.join(value.span),
        };
        let result = self.expr(&whole);
        let (Ty::Value(dv), Ty::Value(rv)) = (&declared, &result) else {
            return; // unknown or Bool: already reported, or nothing to compare
        };
        let name = self.quoted(id.span);
        let sym = op.symbol();
        if dv.dim != rv.dim {
            let d = Diagnostic::error(
                format!(
                    "`{sym}` would change `{name}` from {} to {}",
                    describe(&dv.dim),
                    describe(&rv.dim)
                ),
                op_span,
            )
            .with_secondary(id.span, format!("this is {}", label_dim(&dv.dim)))
            .with_note(format!(
                "the result of `{sym}` must keep the type of `{name}`"
            ));
            let mut sides = vec![(&name_expr, dv.dim)];
            sides.extend(self.name_dim(value).map(|dim| (value, dim)));
            let d = self.with_derivations(d, &sides).with_help(format!(
                "bind a new name instead: `y = {name} {} …`",
                op.binary().symbol()
            ));
            self.error(d);
            return;
        }
        if let Err(e) = self.env.shapes.unify(dv.ty.shape(), rv.ty.shape()) {
            let label = self.this_is(dv);
            self.error(
                Diagnostic::error(
                    format!("`{sym}` would change the shape of `{name}`: {e}"),
                    op_span,
                )
                .with_secondary(id.span, label)
                .with_note(format!(
                    "the result of `{sym}` must keep the type of `{name}`"
                )),
            );
            return;
        }
        let (had, has) = (dv.ty.tags().closure(), rv.ty.tags().closure());
        if !has.is_subtag_of(had) {
            let lost: Vec<&str> = had
                .iter()
                .filter(|t| !has.contains(*t))
                .map(|t| t.name())
                .collect();
            let label = self.this_is(dv);
            self.error(
                Diagnostic::error(
                    format!(
                        "`{sym}` would drop `{name}`'s {} tag{}",
                        lost.join(", "),
                        if lost.len() == 1 { "" } else { "s" }
                    ),
                    op_span,
                )
                .with_secondary(id.span, label)
                .with_note(format!(
                    "the result of `{sym}` must keep the type of `{name}`"
                )),
            );
        }
    }

    /// Reads a type annotation (`PositiveDefinite Matrix[N, N] kg`): the
    /// shape, the tags (validated here, the parser only knows they are
    /// names), and the element unit. Reports every error it finds and
    /// returns `None` if there was one, so a bad annotation never cascades.
    pub(crate) fn annotation(&mut self, ty: &TypeExpr) -> Option<Declared> {
        self.shadow_reported.clear();
        let mut ok = true;
        let shape = match &ty.shape {
            None => Shape::scalar(),
            Some(ann) => {
                let want = match ann.kind {
                    ShapeKind::Scalar => Some(0),
                    ShapeKind::Vector => Some(1),
                    ShapeKind::Matrix => Some(2),
                    ShapeKind::Tensor => None,
                };
                if want.is_some_and(|n| n != ann.dims.len()) {
                    let (kind, n) = (ann.kind.name(), want.unwrap_or(0));
                    self.error(Diagnostic::error(
                        format!(
                            "`{kind}` takes {n} dimension{}, found {}",
                            if n == 1 { "" } else { "s" },
                            ann.dims.len()
                        ),
                        ann.span,
                    ));
                    ok = false;
                }
                let mut dims = Vec::with_capacity(ann.dims.len());
                for d in &ann.dims {
                    match self.shape_dim(d) {
                        Some(d) => dims.push(d),
                        None => ok = false,
                    }
                }
                Shape::new(dims)
            }
        };
        let mut tags = TagSet::EMPTY;
        for id in &ty.tags {
            match Tag::from_name(self.ident(id)) {
                Some(t) => tags = tags.with(t),
                None => {
                    self.unknown_tag(id, ty);
                    ok = false;
                }
            }
        }
        let dim = match &ty.unit {
            Some(u) => {
                let d = self.annotation_unit(u, ty);
                ok &= d.is_some();
                d
            }
            None => None,
        };
        if !ok {
            return None;
        }
        match TensorType::tagged(&mut self.env.shapes, shape, tags) {
            Ok(t) => Some(Declared {
                ty: t,
                dim,
                written: tags,
                unit_span: ty.unit.as_ref().map(UnitExpr::span),
            }),
            Err(e) => {
                self.error(
                    Diagnostic::error(e.to_string(), ty.span)
                        .with_label("matrix properties apply to square matrices only")
                        .with_help(
                            "declare a square matrix, e.g. `Symmetric Matrix[N, N]` or `Diagonal Matrix[3, 3]`",
                        ),
                );
                None
            }
        }
    }

    /// `PositveDefinite Matrix[3, 3]`: a capitalized name before the shape
    /// that is not a matrix property.
    fn unknown_tag(&mut self, id: &Ident, ty: &TypeExpr) {
        let name = self.ident(id);
        let mut d = Diagnostic::error(
            format!("unknown matrix property `{}`", quote(name)),
            id.span,
        )
        .with_label("not a matrix property");
        let known = || Tag::ALL.map(|t| t.name()).join(", ");
        match suggest_tag(name) {
            Some(t) => d = d.with_help(format!("did you mean `{}`?", t.name())),
            // `x: N m`, `x: Pa s`: the parser reads a capitalized name
            // followed by another name as a matrix property, so a unit
            // written with a space lands here.
            None if lookup(name).is_some() && ty.shape.is_none() => {
                let rest = ty
                    .unit
                    .as_ref()
                    .map_or(Cow::Borrowed("…"), |u| self.quoted(u.span()));
                d = d.with_help(format!(
                    "`{}` is a unit; compound units are written with `*` and no spaces: `{}*{rest}`",
                    quote(name),
                    quote(name),
                ));
            }
            None => d = d.with_help(format!("the matrix properties are {}", known())),
        }
        self.error(d);
    }

    /// The unit of an annotation. A capitalized name that is not a unit but
    /// is (or is close to) a matrix property is a property written after
    /// the shape (`Matrix[3, 3] PositiveDefinite`): the parser can't tell,
    /// having no tag table, so the explanation is here.
    fn annotation_unit(&mut self, u: &UnitExpr, ty: &TypeExpr) -> Option<Dimension> {
        if let UnitExpr::Name(id) = u {
            let name = self.ident(id);
            let exact = Tag::from_name(name);
            if lookup(name).is_none() && (exact.is_some() || suggest_tag(name).is_some()) {
                let tag = exact.or_else(|| suggest_tag(name)).map_or("", Tag::name);
                let mut written: Vec<Cow<'_, str>> =
                    ty.tags.iter().map(|t| self.quoted(t.span)).collect();
                written.push(Cow::Borrowed(tag));
                let words = written.join(" ");
                let (message, help) = match &ty.shape {
                    Some(shape) if shape.kind != ShapeKind::Scalar => (
                        if exact.is_some() {
                            format!(
                                "matrix property `{}` must come before the shape",
                                quote(name)
                            )
                        } else {
                            format!("unknown matrix property `{}`", quote(name))
                        },
                        format!(
                            "put matrix properties before the shape: `{words} {}`",
                            self.quoted(shape.span)
                        ),
                    ),
                    _ => (
                        if exact.is_some() {
                            format!("matrix property `{}` needs a shape", quote(name))
                        } else {
                            format!("unknown matrix property `{}`", quote(name))
                        },
                        format!("write a square matrix type: `{words} Matrix[N, N]`"),
                    ),
                };
                let mut d = Diagnostic::error(message, id.span).with_help(help);
                if let Some(shape) = &ty.shape {
                    d = d.with_secondary(shape.span, "the shape");
                }
                self.error(d);
                return None;
            }
        }
        self.unit(u)
    }

    fn shape_dim(&mut self, dim: &DimExpr) -> Option<ShapeDim> {
        match dim {
            DimExpr::Int(span) => match self.text(*span).parse::<u64>() {
                Ok(n) => Some(ShapeDim::Known(n)),
                Err(_) => {
                    self.error(Diagnostic::error(
                        format!("shape dimension `{}` is too large", self.quoted(*span)),
                        *span,
                    ));
                    None
                }
            },
            DimExpr::Name(id) => {
                let name = self.ident(id);
                // A predefined constant nobody has rebound is not a value the
                // program wrote, so `Matrix[G, G]` still means a size variable.
                if self.env.names.contains_key(name) && !self.env.prelude.contains(name) {
                    if self.shadow_reported.iter().any(|n| n == name) {
                        return None; // one error per name per annotation
                    }
                    self.shadow_reported.push(name.to_string());
                    // `N = 3` then `Matrix[N, N]`: the size variable is not
                    // the value 3, so reusing the name is an error.
                    self.error(
                        Diagnostic::error(
                            format!(
                                "size variable `{}` has the same name as a value in scope",
                                quote(name)
                            ),
                            id.span,
                        )
                        .with_label("size variable")
                        .with_help("size variables are separate from values"),
                    );
                    return None;
                }
                Some(ShapeDim::Var(self.env.shape_var(name)))
            }
        }
    }

    pub(crate) fn expr(&mut self, e: &Expr) -> Ty {
        match &e.kind {
            ExprKind::Literal(lit) => match lit {
                Literal::Int(_) | Literal::Float(_) => Ty::scalar(Dimension::DIMENSIONLESS),
                Literal::Bool { .. } => Ty::Bool,
                Literal::Str(_) => Ty::Unknown,
            },
            ExprKind::BoolOp { op, operands, .. } => {
                // Flat: a long chain is a loop here, not recursion.
                for operand in operands {
                    let t = self.expr(operand);
                    self.bool_operand(op.keyword(), operand, t);
                }
                Ty::Bool
            }
            ExprKind::UnitLiteral { unit, .. } => match self.unit(unit) {
                Some(d) => Ty::scalar(d),
                None => Ty::Unknown,
            },
            ExprKind::Name(id) => {
                let name = self.ident(id);
                match self.env.get(name) {
                    Some(t) => t,
                    None => self.undefined_name(name, id.span),
                }
            }
            ExprKind::Paren(inner) => self.expr(inner),
            ExprKind::Unary {
                op,
                op_span: _,
                operand,
            } => {
                let t = self.expr(operand);
                if *op == UnaryOp::Not {
                    self.bool_operand("not", operand, t);
                    return Ty::Bool;
                }
                if matches!(op, UnaryOp::Neg | UnaryOp::Pos)
                    && self.reject_bool(op_sym(*op), operand, &t)
                {
                    return Ty::Unknown;
                }
                match (op, t) {
                    (UnaryOp::Pos, t @ Ty::Value(_)) => t,
                    (UnaryOp::Neg, Ty::Value(v)) => self.retag(v, negated_tags),
                    _ => Ty::Unknown,
                }
            }
            ExprKind::Inverse { operand, op_span } => {
                let t = self.expr(operand);
                if self.reject_bool("⁻¹", operand, &t) {
                    return Ty::Unknown;
                }
                self.invert(t, *op_span, operand.span)
            }
            ExprKind::Call { callee, args } => {
                if let Some(arg) = self.inv_call(callee, args) {
                    let t = self.expr(arg);
                    return self.invert(t, callee.span, arg.span);
                }
                if self.is_print(callee) {
                    // The builtin: any arguments, no result to use.
                    for a in args {
                        self.expr(a);
                    }
                    return Ty::Unknown;
                }
                if let Some(t) = self.math_call(callee, args, e.span) {
                    return t;
                }
                let callee_ty = self.expr(callee);
                self.call(callee_ty, callee, args, e.span)
            }
            ExprKind::Index { base, items } => {
                self.expr(base);
                self.index_items(items);
                Ty::Unknown
            }
            ExprKind::Attribute { base, .. } => {
                self.expr(base);
                Ty::Unknown
            }
            ExprKind::IfElse { then, cond, else_ } => self.if_else(then, cond, else_),
            ExprKind::Binary {
                op,
                op_span,
                lhs,
                rhs,
            } => match op {
                BinaryOp::MatMul => self.matmul(e.span, *op_span, lhs, rhs),
                BinaryOp::Pow => self.pow(*op_span, lhs, rhs),
                _ => {
                    let l = self.expr(lhs);
                    let r = self.expr(rhs);
                    self.binary(*op, *op_span, (lhs, l), (rhs, r))
                }
            },
        }
    }

    /// `inv(x)` with exactly one argument: the function spelling of `x⁻¹`.
    fn inv_call<'e>(&self, callee: &Expr, args: &'e [Expr]) -> Option<&'e Expr> {
        match (&callee.kind, args) {
            (ExprKind::Name(id), [arg])
                if self.ident(id) == "inv" && self.env.get("inv").is_none() =>
            {
                Some(arg)
            }
            _ => None,
        }
    }

    /// The builtin `print(...)`: callee `print` while the program has not
    /// bound that name itself (a user's own `print` shadows it, as with
    /// `inv`). Without this the callee would be an undefined name.
    fn is_print(&self, callee: &Expr) -> bool {
        matches!(&callee.kind, ExprKind::Name(id)
            if self.ident(id) == "print" && self.env.get("print").is_none())
    }

    fn unit(&mut self, u: &UnitExpr) -> Option<Dimension> {
        let r = match u {
            UnitExpr::Name(id) => {
                let name = self.ident(id);
                return match lookup(name) {
                    Some(unit) => Some(unit.dim),
                    None => {
                        let mut d =
                            Diagnostic::error(format!("unknown unit `{}`", quote(name)), id.span);
                        if lookup_constant(name).is_some() {
                            // `10 c`: a constant is a value, so it multiplies.
                            let q = quote(name);
                            d = d.with_help(format!(
                                "`{q}` is a physical constant, not a unit; multiply by it: `x * {q}`"
                            ));
                        } else if let Some(s) = suggest_unit(name) {
                            d = d.with_help(format!("did you mean `{}`?", quote(&s)));
                        }
                        self.error(d);
                        None
                    }
                };
            }
            UnitExpr::Mul { lhs, rhs, .. } => {
                let (a, b) = (self.unit(lhs), self.unit(rhs));
                a?.checked_mul(b?)
            }
            UnitExpr::Div { lhs, rhs, .. } => {
                let (a, b) = (self.unit(lhs), self.unit(rhs));
                a?.checked_div(b?)
            }
            UnitExpr::Pow {
                base,
                num,
                den,
                exponent_span,
                ..
            } => {
                let b = self.unit(base)?;
                let exp = i64::try_from(*den)
                    .ok()
                    .and_then(|den| Rational::checked_new(*num, den).ok());
                let Some(exp) = exp else {
                    self.overflow(*exponent_span);
                    return None;
                };
                b.pow_rat(exp)
            }
        };
        match r {
            Ok(d) => Some(d),
            Err(_) => {
                self.overflow(u.span());
                None
            }
        }
    }

    fn overflow(&mut self, span: Span) {
        self.error(Diagnostic::error("dimension exponent too large", span));
    }

    /// Keeps the shape and dimension, maps the tags.
    fn retag(&mut self, v: ValueTy, rule: fn(TagSet) -> TagSet) -> Ty {
        let tags = rule(v.ty.tags());
        let shape = v.ty.shape().clone();
        let ty = TensorType::tagged(&mut self.env.shapes, shape.clone(), tags)
            .unwrap_or_else(|_| TensorType::untagged(shape));
        Ty::value(v.dim, ty)
    }

    fn invert(&mut self, t: Ty, op_span: Span, operand: Span) -> Ty {
        let Ty::Value(v) = t else {
            return Ty::Unknown;
        };
        let Ok(dim) = v.dim.inverse() else {
            self.overflow(op_span.join(operand));
            return Ty::Unknown;
        };
        match inverse(&mut self.env.shapes, &v.ty) {
            Ok(ty) => Ty::value(dim, ty),
            Err(e) => {
                let label = self.this_is(&v);
                self.error(
                    Diagnostic::error(e.to_string(), op_span).with_secondary(operand, label),
                );
                Ty::Unknown
            }
        }
    }

    fn matmul(&mut self, span: Span, op_span: Span, lhs: &Expr, rhs: &Expr) -> Ty {
        // For the rewrite, keep the type of `A` in `A⁻¹` (or `inv(A)`).
        let inner = strip_parens(lhs);
        let inverted = match &inner.kind {
            ExprKind::Inverse { operand, op_span } => Some((&**operand, *op_span)),
            ExprKind::Call { callee, args } => {
                self.inv_call(callee, args).map(|a| (a, callee.span))
            }
            _ => None,
        };
        let (a, l) = match inverted {
            Some((operand, inv_span)) => {
                let a = self.expr(operand);
                let l = self.invert(a.clone(), inv_span, operand.span);
                (Some((operand, a)), l)
            }
            None => (None, self.expr(lhs)),
        };
        let r = self.expr(rhs);
        let lb = self.reject_bool("@", lhs, &l);
        if self.reject_bool("@", rhs, &r) || lb {
            return Ty::Unknown;
        }
        let (Ty::Value(lv), Ty::Value(rv)) = (l, r) else {
            return Ty::Unknown;
        };
        let Ok(dim) = lv.dim.checked_mul(rv.dim) else {
            self.overflow(span);
            return Ty::Unknown;
        };
        if let Some((operand, Ty::Value(av))) = a {
            if let Some(rewrite) =
                cholesky_rewrite(MatmulLhs::Inverse(&av.ty), &rv.ty, &self.env.shapes)
            {
                if self.rewrites.len() < MAX_STORED_REWRITES {
                    let rewrite = self.name_operands(rewrite, operand, rhs);
                    self.rewrites.push(RewriteSite {
                        span,
                        matrix: operand.span,
                        rhs: rhs.span,
                        rewrite,
                    });
                } else {
                    self.rewrites_omitted += 1;
                }
            }
        }
        match matmul(&mut self.env.shapes, &lv.ty, &rv.ty) {
            Ok(ty) => Ty::value(dim, ty),
            Err(e) => {
                let (ll, rl) = (self.this_is(&lv), self.this_is(&rv));
                self.error(
                    Diagnostic::error(e.to_string(), op_span)
                        .with_secondary(lhs.span, ll)
                        .with_secondary(rhs.span, rl),
                );
                Ty::Unknown
            }
        }
    }

    /// Replaces the explanation's placeholder names `A` and `b` with the
    /// operands' source text: `K is Matrix[3,3] and PositiveDefinite, f is
    /// Vector[3]`. In formulas a compound operand is parenthesized.
    fn name_operands(&self, rewrite: Rewrite, matrix: &Expr, rhs: &Expr) -> Rewrite {
        let Rewrite::CholeskySolve { a, b, result, .. } = rewrite;
        let (a_text, b_text) = (self.quoted(matrix.span), self.quoted(rhs.span));
        let (a_f, b_f) = (self.operand_text(matrix), self.operand_text(rhs));
        let explanation = format!(
            "rewrite `{a_f}⁻¹ @ {b_f}` to a Cholesky solve: {a_text} is {a} and PositiveDefinite, \
             {b_text} is {b}. Factor {a_f} = L·Lᵀ (L lower-triangular), solve L·y = {b_f}, \
             then Lᵀ·x = y. x = {a_f}⁻¹·{b_f} is {result}, and {a_f}⁻¹ is never formed."
        );
        Rewrite::CholeskySolve {
            a,
            b,
            result,
            explanation,
        }
    }

    fn const_exponent(&self, e: &Expr) -> Option<Rational> {
        const_exponent(self.src, e)
    }

    fn pow(&mut self, op_span: Span, base: &Expr, exp: &Expr) -> Ty {
        let b = self.expr(base);
        let x = self.expr(exp);
        let bb = self.reject_bool("^", base, &b);
        if self.reject_bool("^", exp, &x) || bb {
            return Ty::Unknown;
        }
        let Ty::Value(bv) = b else {
            return Ty::Unknown;
        };
        let power_span = op_span.join(exp.span);
        let constant = self.const_exponent(exp);
        if !bv.is_scalar() {
            let base_text = self.quoted(base.span);
            let rank = bv.ty.shape().rank();
            let label = self.this_is(&bv);
            let d = if rank == 2 && constant == Some(Rational::integer(-1)) {
                Diagnostic::error("`^-1` does not invert a matrix", power_span)
                    .with_label("a power, not the inverse operator")
                    .with_secondary(base.span, label)
                    .with_help(format!("write `{base_text}⁻¹` or `inv({base_text})`"))
            } else {
                let mut d = Diagnostic::error(
                    format!("cannot raise a {} to a power", rank_noun(rank)),
                    power_span,
                )
                .with_secondary(base.span, label)
                .with_note("`^` needs a scalar base in v0.1");
                if rank == 1 && constant == Some(Rational::integer(2)) {
                    d = d.with_help(format!(
                        "for the squared magnitude write `{base_text} @ {base_text}`"
                    ));
                } else if rank == 2 {
                    d = d.with_help("use `@` for matrix products");
                }
                d
            };
            self.error(d);
            return Ty::Unknown;
        }
        match x {
            Ty::Value(xv) if !xv.is_scalar() => {
                let label = self.this_is(&xv);
                self.error(
                    Diagnostic::error("the exponent must be a scalar", exp.span).with_label(label),
                );
                return Ty::Unknown;
            }
            Ty::Value(xv) if !xv.dim.is_dimensionless() => {
                self.error(
                    Diagnostic::error("the exponent must be dimensionless", exp.span)
                        .with_label(format!("this is {}", describe(&xv.dim))),
                );
                return Ty::Unknown;
            }
            _ => {}
        }
        match constant {
            Some(r) => match bv.dim.pow_rat(r) {
                Ok(d) => Ty::scalar(d),
                Err(_) => {
                    self.overflow(base.span.join(exp.span));
                    Ty::Unknown
                }
            },
            None if bv.dim.is_dimensionless() => Ty::scalar(Dimension::DIMENSIONLESS),
            None => {
                self.error(
                    Diagnostic::error(
                        "the exponent must be a constant number when the base has a unit",
                        exp.span,
                    )
                    .with_secondary(base.span, format!("this is {}", describe(&bv.dim))),
                );
                Ty::Unknown
            }
        }
    }

    fn binary(&mut self, op: BinaryOp, op_span: Span, l: (&Expr, Ty), r: (&Expr, Ty)) -> Ty {
        let ((le, lt), (re, rt)) = (l, r);
        match (&lt, &rt) {
            (Ty::Bool, Ty::Bool) if matches!(op, BinaryOp::Eq | BinaryOp::Ne) => return Ty::Bool,
            (Ty::Bool, _) | (_, Ty::Bool) => {
                // `==`/`!=` between a Bool and an unknown operand may be
                // fine; stay silent rather than guess.
                let eq = matches!(op, BinaryOp::Eq | BinaryOp::Ne);
                if !(eq && (lt == Ty::Unknown || rt == Ty::Unknown)) {
                    self.reject_bool(op.symbol(), le, &lt);
                    self.reject_bool(op.symbol(), re, &rt);
                }
                return Ty::Unknown;
            }
            _ => {}
        }
        let (Ty::Value(lv), Ty::Value(rv)) = (lt, rt) else {
            return Ty::Unknown;
        };
        match op {
            BinaryOp::Add | BinaryOp::Sub => self.additive(op, op_span, (le, lv), (re, rv)),
            BinaryOp::Mul | BinaryOp::Div => self.multiplicative(op, op_span, (le, lv), (re, rv)),
            _ if op.is_comparison() => {
                if !lv.is_scalar() || !rv.is_scalar() {
                    let (ll, rl) = (self.this_is(&lv), self.this_is(&rv));
                    self.error(
                        Diagnostic::error(
                            format!("`{}` needs scalar operands", op.symbol()),
                            op_span,
                        )
                        .with_secondary(le.span, ll)
                        .with_secondary(re.span, rl),
                    );
                    return Ty::Unknown;
                }
                if lv.dim != rv.dim {
                    self.dim_mismatch(
                        format!(
                            "cannot compare {} and {}",
                            describe(&lv.dim),
                            describe(&rv.dim)
                        ),
                        op,
                        op_span,
                        (le, &lv),
                        (re, &rv),
                    );
                }
                Ty::Bool
            }
            _ => Ty::Unknown,
        }
    }

    fn dim_mismatch(
        &mut self,
        message: String,
        op: BinaryOp,
        op_span: Span,
        l: (&Expr, &ValueTy),
        r: (&Expr, &ValueTy),
    ) {
        let d = Diagnostic::error(message, op_span)
            .with_label("mismatched dimensions")
            .with_secondary(l.0.span, label_dim(&l.1.dim))
            .with_secondary(r.0.span, label_dim(&r.1.dim))
            .with_note(format!(
                "both operands of `{}` must have the same dimension",
                op.symbol()
            ));
        let help = self.suggest((l.0, &l.1.dim), (r.0, &r.1.dim));
        let d = self.with_derivations(d, &[(l.0, l.1.dim), (r.0, r.1.dim)]);
        self.error(d.with_help(help.unwrap_or_else(|| "convert one side, or fix the unit".into())));
    }

    fn additive(
        &mut self,
        op: BinaryOp,
        op_span: Span,
        l: (&Expr, ValueTy),
        r: (&Expr, ValueTy),
    ) -> Ty {
        let ((le, lv), (re, rv)) = (l, r);
        let sym = op.symbol();
        match (lv.is_scalar(), rv.is_scalar()) {
            (true, true) => {}
            (false, false) => {
                // Shape variables are solved only if the whole operation
                // type-checks. When the dimensions differ the operation fails
                // below (after the shape error, which has priority), so probe
                // on a copy then; otherwise commit.
                let mut probe;
                let shapes = if lv.dim == rv.dim {
                    &mut self.env.shapes
                } else {
                    probe = self.env.shapes.clone();
                    &mut probe
                };
                if let Err(e) = elementwise(shapes, &lv.ty, &rv.ty) {
                    let (ll, rl) = (self.this_is(&lv), self.this_is(&rv));
                    self.error(
                        Diagnostic::error(e.to_string(), op_span)
                            .with_secondary(le.span, ll)
                            .with_secondary(re.span, rl),
                    );
                    return Ty::Unknown;
                }
            }
            (left_scalar, _) => {
                let (arr, arr_e, sc_e) = if left_scalar {
                    (&rv, re, le)
                } else {
                    (&lv, le, re)
                };
                let a = self.quoted(arr_e.span);
                let noun = rank_noun(arr.ty.shape().rank());
                let message = match (op, left_scalar) {
                    (BinaryOp::Add, false) => format!("cannot add a scalar to a {noun}"),
                    (BinaryOp::Add, true) => format!("cannot add a {noun} to a scalar"),
                    (_, false) => format!("cannot subtract a scalar from a {noun}"),
                    (_, true) => format!("cannot subtract a {noun} from a scalar"),
                };
                let c = self.operand_text(sc_e);
                let (ones, identity) = if left_scalar {
                    (format!("{c} * ones({a}) {sym} {a}"), format!("I {sym} {a}"))
                } else {
                    (format!("{a} {sym} {c} * ones({a})"), format!("{a} {sym} I"))
                };
                let sc = if left_scalar { &lv } else { &rv };
                let (ll, rl) = (self.this_is(&lv), self.this_is(&rv));
                let mut d = Diagnostic::error(message, op_span)
                    .with_label("no broadcasting between a scalar and an array")
                    .with_secondary(le.span, ll)
                    .with_secondary(re.span, rl);
                if sc.dim != arr.dim {
                    // Neither rewrite would type-check, so don't offer one.
                    d = d.with_note(format!(
                        "the dimensions differ as well: {} and {}",
                        describe(&sc.dim),
                        describe(&arr.dim)
                    ));
                } else if self.is_square_matrix(&arr.ty) {
                    d = d.with_help(format!(
                        "write `{ones}` for every element, or `{identity}` for the identity"
                    ));
                } else {
                    d = d.with_help(format!("write `{ones}` for every element"));
                }
                self.error(d);
                return Ty::Unknown;
            }
        }
        if lv.dim != rv.dim {
            let message = if op == BinaryOp::Add {
                format!("cannot add {} and {}", describe(&lv.dim), describe(&rv.dim))
            } else {
                format!(
                    "cannot subtract {} from {}",
                    describe(&rv.dim),
                    describe(&lv.dim)
                )
            };
            self.dim_mismatch(message, op, op_span, (le, &lv), (re, &rv));
            return Ty::Unknown;
        }
        // Elementwise results carry no tags (see `rules::elementwise`).
        let shape = self.env.shapes.apply(lv.ty.shape());
        Ty::value(lv.dim, TensorType::untagged(shape))
    }

    fn multiplicative(
        &mut self,
        op: BinaryOp,
        op_span: Span,
        l: (&Expr, ValueTy),
        r: (&Expr, ValueTy),
    ) -> Ty {
        let ((le, lv), (re, rv)) = (l, r);
        let dim = match op {
            BinaryOp::Mul => lv.dim.checked_mul(rv.dim),
            _ => lv.dim.checked_div(rv.dim),
        };
        let Ok(dim) = dim else {
            self.overflow(le.span.join(re.span));
            return Ty::Unknown;
        };
        match (lv.is_scalar(), rv.is_scalar()) {
            (true, true) => Ty::scalar(dim),
            // `c * A`, `A * c`, `A / c`: scale every element.
            (true, false) if op == BinaryOp::Mul => {
                self.retag(ValueTy { dim, ty: rv.ty }, scaled_tags)
            }
            (false, true) => self.retag(ValueTy { dim, ty: lv.ty }, scaled_tags),
            (true, false) => {
                let (a, rank) = (self.quoted(re.span), rv.ty.shape().rank());
                let label = self.this_is(&rv);
                let mut d = Diagnostic::error(
                    format!("cannot divide a scalar by a {}", rank_noun(rank)),
                    op_span,
                )
                .with_secondary(re.span, label);
                if rank == 2 {
                    d = d.with_help(format!("for the matrix inverse write `{a}⁻¹`"));
                }
                self.error(d);
                Ty::Unknown
            }
            (false, false) => {
                let (ll, rl) = (self.this_is(&lv), self.this_is(&rv));
                let verb = if op == BinaryOp::Mul {
                    "multiply"
                } else {
                    "divide"
                };
                let mut d = Diagnostic::error(
                    format!("cannot {verb} two arrays with `{}`", op.symbol()),
                    op_span,
                )
                .with_secondary(le.span, ll)
                .with_secondary(re.span, rl)
                .with_note("v0.1 has no elementwise `*` or `/` between arrays");
                if op == BinaryOp::Mul {
                    d = d.with_help("use `@` for the matrix product");
                }
                self.error(d);
                Ty::Unknown
            }
        }
    }
}

/// The symbol of a unary arithmetic operator, for messages.
fn op_sym(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "-",
        UnaryOp::Pos => "+",
        UnaryOp::Partial => "∂",
        UnaryOp::Nabla => "∇",
        UnaryOp::Not => "not",
    }
}

/// A zero of dimension `d` written as Sky source (`0`, `0 m`, `0 kg*m/s^2`),
/// for help text. Uses a coherent table unit when one has exactly this
/// dimension, otherwise composes the SI base units (m, kg, s, A, K, mol, cd).
/// `None` only if an exponent cannot be negated.
fn zero_of(d: &Dimension) -> Option<String> {
    use sky_units::{BaseDim, UNITS};
    if d.is_dimensionless() {
        return Some("0".to_string());
    }
    if let Some(u) = UNITS.iter().find(|u| u.dim == *d && u.scale == 1.0) {
        return Some(format!("0 {}", u.name));
    }
    let symbol = |b: BaseDim| match b {
        BaseDim::Length => "m",
        BaseDim::Mass => "kg",
        BaseDim::Time => "s",
        BaseDim::Temperature => "K",
        BaseDim::Current => "A",
        BaseDim::Amount => "mol",
        BaseDim::Luminosity => "cd",
    };
    let power = |sym: &str, e: Rational| {
        if e == Rational::ONE {
            sym.to_string()
        } else if e.is_integer() {
            format!("{sym}^{}", e.numer())
        } else {
            format!("{sym}^({}/{})", e.numer(), e.denom())
        }
    };
    let (mut pos, mut neg) = (Vec::new(), Vec::new());
    // SI order for derived units: kg·m·s⁻²·K.
    let order = [
        BaseDim::Mass,
        BaseDim::Length,
        BaseDim::Time,
        BaseDim::Temperature,
        BaseDim::Current,
        BaseDim::Amount,
        BaseDim::Luminosity,
    ];
    for b in order {
        let e = d.exponent(b);
        if e.is_zero() {
            continue;
        }
        let sym = symbol(b);
        if e.numer() > 0 {
            pos.push(power(sym, e));
        } else {
            neg.push((sym, e));
        }
    }
    let unit = if pos.is_empty() {
        neg.iter()
            .map(|&(s, e)| power(s, e))
            .collect::<Vec<_>>()
            .join("*")
    } else {
        let mut u = pos.join("*");
        for (s, e) in neg {
            u.push('/');
            u.push_str(&power(s, e.checked_neg().ok()?));
        }
        u
    };
    Some(format!("0 {unit}"))
}

/// The matrix property `name` is probably a misspelling or abbreviation of: within two
/// edits (and a third of its length, so short unit names like `Pa` never
/// match), ignoring case.
fn suggest_tag(name: &str) -> Option<Tag> {
    let lower: Vec<char> = name.chars().flat_map(char::to_lowercase).collect();
    if lower.len() > 32 {
        return None;
    }
    Tag::ALL
        .into_iter()
        .map(|t| {
            let want: Vec<char> = t.name().chars().flat_map(char::to_lowercase).collect();
            // `Positive` for `PositiveDefinite`: an abbreviation counts as
            // an exact hit, however many letters are missing.
            if lower.len() >= 4 && want.starts_with(&lower) {
                return (0, t);
            }
            (edit_distance(&lower, &want), t)
        })
        .filter(|&(d, _)| d <= 2 && d <= lower.len() / 3)
        .min_by_key(|&(d, _)| d)
        .map(|(_, t)| t)
}

/// Levenshtein distance between two short strings.
fn edit_distance(a: &[char], b: &[char]) -> usize {
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut diag = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let up = row[j + 1];
            row[j + 1] = (diag + usize::from(ca != cb)).min(up + 1).min(row[j] + 1);
            diag = up;
        }
    }
    row[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tags::Tag;

    fn check_src(src: &str, env: Env) -> Checked {
        let lexed = sky_syntax_lex(src);
        let (module, errors) = sky_ast::parse_module(&lexed, src);
        assert!(errors.is_empty(), "{errors:#?}");
        check_module_with(&module, src, env)
    }

    fn sky_syntax_lex(src: &str) -> Vec<sky_ast::Token> {
        sky_syntax::lex(src).expect("lexes").tokens
    }

    fn span_of(src: &str, needle: &str) -> Span {
        let start = src.find(needle).expect("needle") as u32;
        Span::new(start, start + needle.len() as u32)
    }

    fn pd_env() -> Env {
        let mut env = Env::new();
        let n = env.shape_var("N");
        env.bind_array(
            "A",
            Shape::matrix(n.clone(), n),
            TagSet::of(Tag::PositiveDefinite),
            Dimension::DIMENSIONLESS,
        )
        .unwrap();
        env.bind_array("b", Shape::vector(3u64), TagSet::EMPTY, Dimension::LENGTH)
            .unwrap();
        env
    }

    #[test]
    fn decimal_exponents_are_exact_fractions() {
        for (text, num, den) in [
            ("0.5", 1, 2),
            ("1.5", 3, 2),
            ("0.25", 1, 4),
            ("2.0", 2, 1),
            ("2.500", 5, 2),
            ("0.1", 1, 10),
            ("10.0", 10, 1),
            ("0.0", 0, 1),
            ("9223372036854775807.0", i64::MAX, 1),
            ("0.000000000000000000", 0, 1),
        ] {
            assert_eq!(exact_decimal(text), Some(Rational::new(num, den)), "{text}");
        }
        for text in [
            "1e-1",
            "1.5e3",
            "1e5",
            "1_0.5",
            "0.5_5",
            "9223372036854775808.0", // does not fit i64
            "0.0000000000000000001", // denominator 10^19 does not fit i64
            "1.0000000000000000001",
            "5",
            ".5",
            "-0.5",
        ] {
            assert_eq!(exact_decimal(text), None, "{text}");
        }
    }

    #[test]
    fn decimal_exponent_powers_a_unit() {
        let c = check_src("x = (9 m)^0.5\ny = (8 m)^1e-1\n", Env::new());
        let msgs: Vec<_> = c.diagnostics.iter().map(|d| d.message.as_str()).collect();
        assert_eq!(
            msgs,
            ["the exponent must be a constant number when the base has a unit"]
        );
        let Some(Ty::Value(x)) = c.env.get("x") else {
            panic!("x is a value");
        };
        assert_eq!(
            x.dim,
            Dimension::LENGTH.pow_rat(Rational::new(1, 2)).unwrap()
        );
    }

    #[test]
    fn rewrite_sites_carry_spans() {
        let src = "x = A⁻¹ @ b\n";
        let c = check_src(src, pd_env());
        assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
        assert_eq!(c.rewrites.len(), 1);
        let site = &c.rewrites[0];
        assert_eq!(site.span, span_of(src, "A⁻¹ @ b"));
        assert_eq!(site.matrix, span_of(src, "A"));
        assert_eq!(site.rhs, span_of(src, "b"));
        assert!(matches!(site.rewrite, Rewrite::CholeskySolve { .. }));
        // N was solved from b's length; b's unit carries through.
        assert_eq!(c.env.get("x").unwrap().to_string(), "Vector[3] [L]");
    }

    #[test]
    fn parenthesized_inverse_still_rewrites_but_power_does_not() {
        let c = check_src("x = (A⁻¹) @ b\n", pd_env());
        assert_eq!(c.rewrites.len(), 1);
        let src = "x = A^-1 @ b\n";
        let c = check_src(src, pd_env());
        assert!(c.rewrites.is_empty());
        assert_eq!(c.diagnostics.len(), 1);
        let d = &c.diagnostics[0];
        assert_eq!(d.primary.span, span_of(src, "^-1"));
        assert_eq!(d.help, vec!["write `A⁻¹` or `inv(A)`".to_string()]);
    }

    #[test]
    fn rewrite_explanation_uses_operand_text() {
        let mut env = Env::new();
        env.bind_array(
            "K",
            Shape::matrix(3u64, 3u64),
            TagSet::of(Tag::PositiveDefinite),
            Dimension::DIMENSIONLESS,
        )
        .unwrap();
        env.bind_array(
            "f",
            Shape::vector(3u64),
            TagSet::EMPTY,
            Dimension::DIMENSIONLESS,
        )
        .unwrap();
        let c = check_src("u = K⁻¹ @ f\n", env);
        let text = c.rewrites[0].rewrite.explanation();
        assert!(
            text.contains("K is Matrix[3,3] and PositiveDefinite, f is Vector[3]"),
            "{text}"
        );
        assert!(!text.contains(" A ") && !text.contains(" b "), "{text}");
    }

    #[test]
    fn identity_is_suggested_only_for_square_matrices() {
        let mut env = pd_env();
        env.bind_array(
            "v",
            Shape::vector(3u64),
            TagSet::EMPTY,
            Dimension::DIMENSIONLESS,
        )
        .unwrap();
        env.bind_array(
            "M",
            Shape::matrix(3u64, 4u64),
            TagSet::EMPTY,
            Dimension::DIMENSIONLESS,
        )
        .unwrap();
        let c = check_src("x = A + 2\ny = v + 2\nz = M + 2\nw = 2 m + b\n", env);
        let help: Vec<Vec<String>> = c.diagnostics.iter().map(|d| d.help.clone()).collect();
        assert_eq!(
            help,
            vec![
                vec![
                    "write `A + 2 * ones(A)` for every element, or `A + I` for the identity"
                        .to_string()
                ],
                vec!["write `v + 2 * ones(v)` for every element".to_string()],
                vec!["write `M + 2 * ones(M)` for every element".to_string()],
                // b is in metres too: the `ones` form applies, `I` doesn't.
                vec!["write `2 m * ones(b) + b` for every element".to_string()],
            ]
        );
    }

    #[test]
    fn size_variable_rule_leaves_the_unit_n_alone() {
        // `N` as a unit after a number never touches the value or size
        // variable namespaces, whether or not the unit table has `N` yet.
        let src = "N = 3 m\nF = 10 N\nA: Matrix[M, M]\n";
        let c = check_src(src, Env::new());
        assert!(
            c.diagnostics
                .iter()
                .all(|d| !d.message.contains("size variable")),
            "{:#?}",
            c.diagnostics
        );
        match sky_units::lookup("N") {
            Some(newton) => {
                assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
                assert_eq!(c.env.get("F"), Some(Ty::scalar(newton.dim)));
            }
            None => assert_eq!(c.diagnostics[0].message, "unknown unit `N`"),
        }
        // A size variable named like a value is reported once, at the
        // first use in the annotation.
        let src = "N = 3\nA: Matrix[N, N]\n";
        let c = check_src(src, Env::new());
        assert_eq!(c.diagnostics.len(), 1);
        assert_eq!(c.diagnostics[0].primary.span, Span::new(16, 17));
    }

    #[test]
    fn scalar_inverse_flips_the_dimension() {
        let c = check_src("f = (2 s)⁻¹\ng = inv(3 m/s)\n", Env::new());
        assert!(c.diagnostics.is_empty());
        assert_eq!(c.env.get("f"), Some(Ty::scalar(Dimension::FREQUENCY)));
        assert_eq!(
            c.env.get("g"),
            Some(Ty::scalar(Dimension::VELOCITY.inverse().unwrap()))
        );
    }

    #[test]
    fn unit_mismatch_spans() {
        let src = "x = 5 m + 20 K\n";
        let c = check_src(src, Env::new());
        assert_eq!(c.diagnostics.len(), 1);
        let d = &c.diagnostics[0];
        assert_eq!(d.message, "cannot add length and temperature");
        assert_eq!(d.primary.span, span_of(src, "+"));
        let secondary: Vec<_> = d
            .secondary
            .iter()
            .map(|l| (l.span, l.message.as_deref().unwrap_or("")))
            .collect();
        assert_eq!(
            secondary,
            vec![
                (span_of(src, "5 m"), "length"),
                (span_of(src, "20 K"), "temperature"),
            ]
        );
        assert_eq!(c.env.get("x"), Some(Ty::Unknown));
    }

    #[test]
    fn negation_and_scaling_tags() {
        let mut env = Env::new();
        env.bind_array(
            "Q",
            Shape::matrix(2u64, 2u64),
            TagSet::of(Tag::Orthogonal),
            Dimension::DIMENSIONLESS,
        )
        .unwrap();
        let c = check_src("n = -Q\ns = 2 * Q\np = +Q\n", env);
        assert!(c.diagnostics.is_empty());
        let tags = |name: &str| match c.env.get(name) {
            Some(Ty::Value(v)) => v.ty.tags(),
            other => panic!("{other:?}"),
        };
        assert_eq!(tags("n"), TagSet::of(Tag::Orthogonal));
        assert_eq!(tags("s"), TagSet::EMPTY);
        assert_eq!(tags("p"), TagSet::of(Tag::Orthogonal));
    }

    #[test]
    fn errors_do_not_cascade() {
        // One undefined name, one error: every use of the unknown result is
        // silent.
        let c = check_src("x = y + 1 m\nz = x * x @ x\n", Env::new());
        assert_eq!(c.diagnostics.len(), 1, "{:#?}", c.diagnostics);
        assert_eq!(c.diagnostics[0].message, "cannot find `y` in this scope");
    }

    #[test]
    fn binding_value_must_match_the_annotation() {
        let mut env = Env::new();
        env.bind_array(
            "B",
            Shape::matrix(3u64, 5u64),
            TagSet::EMPTY,
            Dimension::MASS,
        )
        .unwrap();
        let src = "A: Matrix[3, 4] = B\n";
        let c = check_src(src, env);
        assert_eq!(c.diagnostics.len(), 1);
        assert_eq!(c.diagnostics[0].primary.span, span_of(src, "B"));
        // The annotation wins for the shape; the value gives the unit.
        assert_eq!(c.env.get("A").unwrap().to_string(), "Matrix[3,4] [M]");
    }

    #[test]
    fn a_declaration_without_a_value_binds_tags_shape_and_unit() {
        let c = check_src("A: PositiveDefinite Matrix[N, N] kg\n", Env::new());
        assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
        assert_eq!(
            c.env.get("A").unwrap().to_string(),
            "Matrix[N,N] {Symmetric, PositiveDefinite} [M]"
        );
    }

    #[test]
    fn a_declared_unit_wins_over_the_value_and_a_mismatch_is_reported() {
        let src = "m: kg = 3 s\n";
        let c = check_src(src, Env::new());
        assert_eq!(c.diagnostics.len(), 1, "{:#?}", c.diagnostics);
        assert_eq!(c.diagnostics[0].message, "expected mass, found time");
        assert_eq!(c.diagnostics[0].primary.span, span_of(src, "3 s"));
        assert_eq!(c.env.get("m").unwrap().to_string(), "Scalar [M]");
    }

    #[test]
    fn tag_suggestions_cover_typos_and_abbreviations_but_not_short_units() {
        for (typo, tag) in [
            ("PositveDefinite", Tag::PositiveDefinite),
            ("Symetric", Tag::Symmetric),
            ("Positive", Tag::PositiveDefinite),
            ("Diagonal", Tag::Diagonal),
        ] {
            assert_eq!(suggest_tag(typo), Some(tag), "{typo}");
        }
        for name in ["N", "Pa", "Hz", "Foo", "Matrix", ""] {
            assert_eq!(suggest_tag(name), None, "{name}");
        }
    }

    #[test]
    fn exponents_need_constants_when_the_base_has_a_unit() {
        let mut env = Env::new();
        env.bind("n", Ty::scalar(Dimension::DIMENSIONLESS));
        let c = check_src("a = 2^n\nb = (3 m)^(1/2)\nc = (3 m)^n\n", env);
        assert_eq!(c.diagnostics.len(), 1, "{:#?}", c.diagnostics);
        assert_eq!(c.env.get("a"), Some(Ty::scalar(Dimension::DIMENSIONLESS)));
        assert_eq!(
            c.env.get("b"),
            Some(Ty::scalar(
                Dimension::LENGTH.pow_rat(Rational::new(1, 2)).unwrap()
            ))
        );
    }

    #[test]
    fn exponent_overflow_is_a_diagnostic() {
        let src = format!("x = (1 m^{})^2\n", i64::MAX);
        let c = check_src(&src, Env::new());
        assert_eq!(c.diagnostics.len(), 1);
        assert_eq!(c.diagnostics[0].message, "dimension exponent too large");
    }
}
