//! Blocks, control flow, and functions: the part of the checker walk that
//! `check.rs` hands `if`, `while`, `def`, `return`, `break`, `continue`, the
//! conditional expression, and calls to user functions to. The rules are in
//! the spec's "Blocks and control flow"; this file is where they are
//! enforced. Nothing here evaluates user code.
//!
//! # Scoping
//!
//! Python's: `if`, `while` and `for` bodies share the enclosing scope, and a
//! `def` opens a new one. After an `if`, the bindings of every branch that can
//! fall through are merged ([`Checker::merge`]): a name bound in all of them
//! keeps its type (dimension and shape must agree, tags are joined with
//! [`TagSet::join`]), a name bound in only some becomes *maybe-defined*
//! (using it is an error), and a name bound with conflicting types is an
//! error at the later binding. A branch that ends in `return`, `break`, or
//! `continue` doesn't fall through, so it takes no part in the merge.
//!
//! # Shape variables
//!
//! The [`ShapeUnifier`] is one monotone constraint set for the whole module,
//! and scopes never snapshot it:
//! * A constraint learned inside a branch stays after it. A size variable
//!   names one size for the module, so `N = 3` learned in one branch is a
//!   fact about `N`, not about that path.
//! * Merges never unify. Two branches that leave a name with different
//!   shapes (after substituting solved variables) conflict; merging must not
//!   invent a constraint that neither branch wrote.
//! * A function's size variables are its own: `N` in a `def`'s annotations
//!   is not the module's `N`. After the body is checked, the variables its
//!   signature still leaves open are *generic*, and each call gets fresh
//!   copies, so `def f(A: Matrix[N, N], b: Vector[N])` can be called with
//!   `3` and then `4`.
//!
//! [`TagSet::join`]: crate::tags::TagSet::join
//! [`ShapeUnifier`]: crate::shape::ShapeUnifier

use std::collections::BTreeSet;
use std::sync::Arc;

use sky_ast::{Block, CompoundOp, Expr, ExprKind, Ident, Literal, Param, TypeExpr};
use sky_diag::{Diagnostic, Span};
use sky_units::Dimension;

use crate::check::{describe, strip_parens, Checker, Scope, Ty, ValueTy};
use crate::derivative::DerivativeOf;
use crate::provenance::label_dim;
use crate::rules::TensorType;
use crate::shape::{Shape, ShapeContext, ShapeDim, ShapeVar};
use sky_diag::display_capped as quote;

/// A function's signature, as the checker knows it: enough to check a call
/// and to type its result.
///
/// Annotated parameters carry the shape their annotation states and, until
/// annotations can state units, are dimensionless; unannotated parameters
/// accept anything (they are unknown inside the body too).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FuncSig {
    pub(crate) name: String,
    /// The function's name in its `def`.
    pub(crate) def_span: Span,
    pub(crate) params: Vec<ParamSig>,
    /// What a call gives: the declared or inferred return type, or
    /// [`Ty::Unknown`] when nothing is known (no `return` with a value, a
    /// return of an unknown value, or a recursive call while the body is
    /// still being checked).
    pub(crate) ret: Ty,
    /// The size variables the signature leaves open; each call replaces
    /// them with fresh ones.
    pub(crate) generic: Vec<ShapeVar>,
    /// Set for the signature of `derivative(f, x)`: which function and
    /// parameter it is the derivative of (see [`crate::derivative`]). `None`
    /// for a `def`.
    pub(crate) derivative: Option<Arc<DerivativeOf>>,
}

impl FuncSig {
    /// For the function `derivative(f, x)` gives, what it differentiates;
    /// `None` for a function defined with `def`.
    pub fn derivative_of(&self) -> Option<&DerivativeOf> {
        self.derivative.as_deref()
    }

    /// The function's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// How many arguments a call takes.
    pub fn arity(&self) -> usize {
        self.params.len()
    }

    /// The type a call gives, with the signature's own size variables.
    pub fn ret(&self) -> &Ty {
        &self.ret
    }

    /// The span of the function's name in its `def`: the key of
    /// [`crate::Checked::funcs`], and what tells two functions with the same
    /// name apart.
    pub fn def_span(&self) -> Span {
        self.def_span
    }

    /// The name of parameter `i`.
    pub fn param_name(&self, i: usize) -> Option<&str> {
        self.params.get(i).map(|p| p.name.as_str())
    }

    /// The annotated type of parameter `i` (with the signature's own size
    /// variables), or `None` for an unannotated parameter, which accepts
    /// anything and is unknown inside the body.
    pub fn param_ty(&self, i: usize) -> Option<&ValueTy> {
        self.params.get(i).and_then(|p| p.ty.as_ref())
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct ParamSig {
    pub(crate) name: String,
    /// The annotated type; `None` for an unannotated parameter or one whose
    /// annotation was rejected (already reported).
    pub(crate) ty: Option<ValueTy>,
    /// The annotation's span, for labels.
    pub(crate) annotation: Option<Span>,
}

/// One `return` in the function being checked.
pub(crate) struct ReturnSite {
    span: Span,
    /// The value's span and type; `None` for a bare `return`.
    value: Option<(Span, Ty)>,
}

/// What the checker tracks about the `def` it is inside.
pub(crate) struct FnCtx {
    name: String,
    /// The shape of `-> T`, and the span of `T`.
    /// The `-> T` annotation: shape and closed tags, its unit if it names
    /// one, and its span.
    declared: Option<(TensorType, Option<Dimension>, Span)>,
    returns: Vec<ReturnSite>,
    /// Names bound in this function so far (parameters and assignments):
    /// what `x += 1` may update.
    pub(crate) locals: BTreeSet<String>,
}

/// How often a loop's body is known to run: it decides which paths leave
/// the loop (see [`Checker::loop_body`]).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Runs {
    /// Maybe never (`while c`, `range(n)` with a variable `n`): the state
    /// before the loop is one way out, so a name first bound in the body
    /// is only maybe-defined after it.
    Maybe,
    /// At least once (a `for` over an array, which is never empty, or over
    /// a `range` with constant bounds): the body's own states are the only
    /// ways out, so what it binds is defined after the loop.
    AtLeastOnce,
    /// Until a `break` (`while true`).
    Forever,
}

/// One loop being checked: the states its `break`s and `continue`s left.
#[derive(Default)]
pub(crate) struct LoopFrame {
    breaks: Vec<Scope>,
    continues: Vec<Scope>,
}

/// Where a merge happens, for its wording.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum MergeKind {
    Branches,
    Loop,
}

fn article(word: &str) -> &'static str {
    if word.starts_with(['a', 'e', 'i', 'o', 'u']) {
        "an"
    } else {
        "a"
    }
}

impl Checker<'_> {
    // ----- type wording -----

    /// `a length`, `` `Vector[3] [L]` ``, `a Bool`: a type as a noun phrase.
    fn ty_noun(&self, t: &Ty) -> String {
        match t {
            Ty::Value(v) if v.is_scalar() => match v.dim.name() {
                Some("dimensionless") => "a dimensionless number".to_string(),
                Some(n) => format!("{} {n}", article(n)),
                None => format!("`{}`", v.dim),
            },
            Ty::Value(v) => format!(
                "`{}`",
                ValueTy {
                    dim: v.dim,
                    ty: self.env.shapes.apply_type(&v.ty),
                }
            ),
            Ty::Bool => "a Bool".to_string(),
            Ty::Func(f) => format!("function `{}`", quote(f.name())),
            Ty::Ode(o) => o.describe(),
            Ty::Unknown => "unknown".to_string(),
        }
    }

    /// The short form for a `this is ...` label: `length`, `Vector[3] [L]`.
    fn ty_label(&self, t: &Ty) -> String {
        match t {
            Ty::Value(v) if v.is_scalar() => describe(&v.dim),
            Ty::Value(v) => ValueTy {
                dim: v.dim,
                ty: self.env.shapes.apply_type(&v.ty),
            }
            .to_string(),
            Ty::Bool => "Bool".to_string(),
            Ty::Func(f) => format!("function `{}`", quote(f.name())),
            Ty::Ode(o) => o.describe(),
            Ty::Unknown => "?".to_string(),
        }
    }

    // ----- joining types -----

    /// The type that fits both `a` and `b`, or `None` if they conflict.
    /// Values must agree in dimension and (solved) shape, and their tags
    /// are joined. An unknown side makes the result unknown, silently.
    /// Never unifies shapes: see the module docs.
    pub(crate) fn join_types(&mut self, a: &Ty, b: &Ty) -> Option<Ty> {
        match (a, b) {
            (Ty::Unknown, _) | (_, Ty::Unknown) => Some(Ty::Unknown),
            (Ty::Bool, Ty::Bool) => Some(Ty::Bool),
            (Ty::Value(x), Ty::Value(y)) => {
                if x.dim != y.dim {
                    return None;
                }
                let shape = self.env.shapes.apply(x.ty.shape());
                if shape != self.env.shapes.apply(y.ty.shape()) {
                    return None;
                }
                let tags = x.ty.tags().join(y.ty.tags());
                let ty = TensorType::tagged(&mut self.env.shapes, shape.clone(), tags)
                    .unwrap_or_else(|_| TensorType::untagged(shape));
                Some(Ty::value(x.dim, ty))
            }
            // Two different functions under one name (a `def` in each branch)
            // can't be told apart at a call: calls to it are unchecked.
            (Ty::Func(f), Ty::Func(g)) => Some(if f == g { a.clone() } else { Ty::Unknown }),
            (Ty::Ode(x), Ty::Ode(y)) => Some(if x == y { a.clone() } else { Ty::Unknown }),
            _ => None,
        }
    }

    // ----- statements -----

    pub(crate) fn block(&mut self, block: &Block) {
        for stmt in &block.stmts {
            self.stmt(stmt);
        }
    }

    /// `if`/`elif`/`else`: every branch starts from the state before the
    /// `if`; the states of the branches that fall through are merged. Without
    /// an `else`, "no branch ran" is one more state to merge.
    #[inline(never)]
    pub(crate) fn if_stmt(
        &mut self,
        span: Span,
        branches: &[(Expr, Block)],
        else_block: Option<&Block>,
    ) {
        let before = self.env.snapshot();
        let mut ends: Vec<Scope> = Vec::new();
        for (i, (cond, block)) in branches.iter().enumerate() {
            self.env.restore(before.clone());
            let t = self.expr(cond);
            self.condition(if i == 0 { "if" } else { "elif" }, cond, t);
            self.diverged = false;
            self.block(block);
            if !self.diverged {
                ends.push(self.env.snapshot());
            }
        }
        self.env.restore(before.clone());
        self.diverged = false;
        match else_block {
            Some(block) => {
                self.block(block);
                if !self.diverged {
                    ends.push(self.env.snapshot());
                }
            }
            None => ends.push(before.clone()),
        }
        self.join_paths(ends, before, span, MergeKind::Branches);
    }

    /// Continues after a statement that had several paths out: the merge of
    /// `paths`, or, if none falls through, the code after it is unreachable
    /// (`fallback` keeps names defined so it doesn't cascade).
    fn join_paths(&mut self, paths: Vec<Scope>, fallback: Scope, span: Span, kind: MergeKind) {
        if paths.is_empty() {
            self.diverged = true;
            self.env.restore(fallback);
            return;
        }
        self.diverged = false;
        let merged = self.merge(paths, span, kind);
        self.env.restore(merged);
    }

    /// `while cond: body`. The body is checked once, from the state before
    /// the loop. The state after it merges every way out: the condition
    /// failing (before the first iteration, or after any: the states at the
    /// end of the body and at each `continue`) and each `break`. A loop
    /// whose condition is the literal `true` can only be left by `break`.
    #[inline(never)]
    pub(crate) fn while_stmt(&mut self, span: Span, cond: &Expr, body: &Block) {
        let before = self.env.snapshot();
        let t = self.expr(cond);
        self.condition("while", cond, t);
        let forever = matches!(
            strip_parens(cond).kind,
            ExprKind::Literal(Literal::Bool { value: true, .. })
        );
        let runs = if forever { Runs::Forever } else { Runs::Maybe };
        self.loop_body(span, before, body, runs);
    }

    /// The body of a loop and the merge of every way out of it, for `while`
    /// and `for` alike (a `for` is a `while` over an index, see
    /// [`crate::for_loop`]). `before` is the state on entering the loop,
    /// which is where the zero-iteration path ends up.
    pub(crate) fn loop_body(&mut self, span: Span, before: Scope, body: &Block, runs: Runs) {
        self.loops.push(LoopFrame::default());
        self.diverged = false;
        self.block(body);
        let end_diverged = self.diverged;
        let frame = self.loops.pop().unwrap_or_default();
        let mut paths = Vec::new();
        if runs != Runs::Forever {
            if runs == Runs::Maybe {
                paths.push(before.clone());
            }
            if !end_diverged {
                paths.push(self.env.snapshot());
            }
            paths.extend(frame.continues);
        }
        paths.extend(frame.breaks);
        self.join_paths(paths, before, span, MergeKind::Loop);
    }

    pub(crate) fn break_stmt(&mut self, span: Span) {
        if self.loops.is_empty() {
            self.error(
                Diagnostic::error("`break` outside a loop", span)
                    .with_label("not inside a loop")
                    .with_help(
                        "`break` only works inside a `while` or `for` loop of the same function",
                    ),
            );
        } else {
            let here = self.env.snapshot();
            if let Some(frame) = self.loops.last_mut() {
                frame.breaks.push(here);
            }
        }
        self.diverged = true;
    }

    pub(crate) fn continue_stmt(&mut self, span: Span) {
        if self.loops.is_empty() {
            self.error(
                Diagnostic::error("`continue` outside a loop", span)
                    .with_label("not inside a loop")
                    .with_help(
                        "`continue` only works inside a `while` or `for` loop of the same function",
                    ),
            );
        } else {
            let here = self.env.snapshot();
            if let Some(frame) = self.loops.last_mut() {
                frame.continues.push(here);
            }
        }
        self.diverged = true;
    }

    // ----- merging -----

    /// Merges the states of several paths that reach one point (at least
    /// one; `span` is the statement, for errors without a better place).
    #[inline(never)]
    fn merge(&mut self, paths: Vec<Scope>, span: Span, kind: MergeKind) -> Scope {
        if paths.len() == 1 {
            return paths.into_iter().next().unwrap_or_default();
        }
        let universe: BTreeSet<String> = paths
            .iter()
            .flat_map(|p| p.names.keys().chain(p.maybe.keys()))
            .cloned()
            .collect();
        let mut out = Scope::default();
        for name in universe {
            let present: Vec<(usize, &Ty)> = paths
                .iter()
                .enumerate()
                .filter_map(|(i, p)| p.names.get(&name).map(|t| (i, t)))
                .collect();
            let origin = |i: usize| paths.get(i).and_then(|p| p.origins.get(&name)).copied();
            let Some(&(first, first_ty)) = present.first() else {
                // Only maybe-bound on every path: still maybe-bound.
                if let Some(m) = paths.iter().find_map(|p| p.maybe.get(&name)) {
                    out.maybe.insert(name, *m);
                }
                continue;
            };
            if present.len() < paths.len() {
                let at = origin(first)
                    .or_else(|| paths.iter().find_map(|p| p.maybe.get(&name).map(|m| m.0)))
                    .unwrap_or(span);
                out.maybe.insert(name, (at, kind == MergeKind::Loop));
                continue;
            }
            let mut acc = first_ty.clone();
            let mut acc_origin = origin(first);
            let mut last_origin = acc_origin;
            for &(i, ty) in &present[1..] {
                match self.join_types(&acc, ty) {
                    Some(t) => acc = t,
                    None => {
                        self.report_conflict(
                            &name,
                            (&acc, acc_origin),
                            (ty, origin(i)),
                            span,
                            kind,
                        );
                        acc = Ty::Unknown;
                    }
                }
                last_origin = origin(i).or(last_origin);
                acc_origin = last_origin;
            }
            if let Some(o) = last_origin {
                out.origins.insert(name.clone(), o);
            }
            out.names.insert(name, acc);
        }
        out
    }

    /// `x` was bound with different types on two paths: an error at the
    /// later binding, with the earlier one as a secondary label.
    #[cold]
    #[inline(never)]
    fn report_conflict(
        &mut self,
        name: &str,
        a: (&Ty, Option<Span>),
        b: (&Ty, Option<Span>),
        span: Span,
        kind: MergeKind,
    ) {
        // The binding that comes later in the source is the primary one.
        let ((late_ty, late_at), (early_ty, early_at)) = match (a.1, b.1) {
            (Some(x), Some(y)) if x.start > y.start => (a, b),
            _ => (b, a),
        };
        let other = match kind {
            MergeKind::Branches => "in the other branch",
            MergeKind::Loop => "on another path through the loop",
        };
        let name = quote(name);
        let mut d = Diagnostic::error(
            format!(
                "`{name}` is {} here but {} {other}",
                self.ty_noun(late_ty),
                self.ty_noun(early_ty)
            ),
            late_at.unwrap_or(span),
        )
        .with_label("this binding has a different type");
        if let Some(early_at) = early_at {
            d = d.with_secondary(
                early_at,
                format!("`{name}` is {} here", self.ty_noun(early_ty)),
            );
        }
        self.error(d.with_help("give both bindings the same type, or use different names"));
    }

    /// A name that is not bound: maybe-defined (bound on some paths only),
    /// or not defined at all. Reports it and gives the name an unknown
    /// value from here on, so it is reported once.
    pub(crate) fn undefined_name(&mut self, name: &str, span: Span) -> Ty {
        let shown = quote(name);
        if let Some((bound, in_loop)) = self.env.maybe.get(name).copied() {
            self.error(
                Diagnostic::error(format!("`{shown}` may be undefined here"), span)
                    .with_label("not assigned on every path")
                    .with_secondary(bound, format!("`{shown}` is assigned here"))
                    .with_help(if in_loop {
                        "assign it before the loop"
                    } else {
                        "assign it before the `if`"
                    }),
            );
            self.env.bind(name, Ty::Unknown);
            return Ty::Unknown;
        }
        let mut d = Diagnostic::error(format!("cannot find `{shown}` in this scope"), span)
            .with_label("not defined");
        if matches!(name, "True" | "False") {
            d = d.with_help("Sky writes booleans as `true`/`false`");
        } else if !self.fns.is_empty() {
            d = d.with_help("a function body sees only names defined above its `def`");
        }
        self.error(d);
        Ty::Unknown
    }

    // ----- the conditional expression -----

    /// `then if cond else else_`: the condition is a Bool, both arms have
    /// the same type, and that is the type of the whole.
    pub(crate) fn if_else(&mut self, then: &Expr, cond: &Expr, else_: &Expr) -> Ty {
        let ct = self.expr(cond);
        self.condition("if", cond, ct);
        let tt = self.expr(then);
        let et = self.expr(else_);
        if let Some(t) = self.join_types(&tt, &et) {
            return t;
        }
        self.error(
            Diagnostic::error(
                format!(
                    "the `else` arm is {}, but the `if` arm is {}",
                    self.ty_noun(&et),
                    self.ty_noun(&tt)
                ),
                else_.span,
            )
            .with_label(format!("this is {}", self.ty_label(&et)))
            .with_secondary(then.span, format!("this is {}", self.ty_label(&tt)))
            .with_note(
                "both arms of a conditional expression must have the same dimension and shape",
            ),
        );
        Ty::Unknown
    }

    // ----- functions -----

    /// A parameter annotation (`v: Vector[3] m/s`): its shape, tags and
    /// unit. Without a unit the parameter is dimensionless, the same
    /// convention as a binding annotation without a value.
    fn param_annotation(&mut self, ty: &TypeExpr) -> Option<ValueTy> {
        let d = self.annotation(ty)?;
        Some(ValueTy {
            dim: d.dim.unwrap_or(Dimension::DIMENSIONLESS),
            ty: d.ty,
        })
    }

    /// `def name(params) -> ret: body`. The signature is read in a scope of
    /// its own size variables; the body is checked now, against the names
    /// bound so far, in a new scope holding the parameters; afterwards the
    /// scope is dropped and `name` is bound to the finished signature.
    #[inline(never)]
    pub(crate) fn def_stmt(
        &mut self,
        name: &Ident,
        params: &[Param],
        ret: Option<&TypeExpr>,
        body: &Block,
    ) {
        let fname = self.ident(name).to_string();
        let module_vars = std::mem::take(&mut self.env.shape_vars);
        let mut psigs: Vec<ParamSig> = Vec::new();
        let mut seen: Vec<(String, Span)> = Vec::new();
        for p in params {
            let pname = self.ident(&p.name).to_string();
            if let Some((_, first)) = seen.iter().find(|(n, _)| *n == pname) {
                let d = Diagnostic::error(
                    format!("duplicate parameter `{}`", quote(&pname)),
                    p.name.span,
                )
                .with_secondary(*first, "first used here");
                self.error(d);
            } else {
                seen.push((pname.clone(), p.name.span));
            }
            let ty = p.ty.as_ref().and_then(|t| self.param_annotation(t));
            psigs.push(ParamSig {
                name: pname,
                ty,
                annotation: p.ty.as_ref().map(|t| t.span),
            });
        }
        let declared = ret.and_then(|r| self.annotation(r).map(|d| (d.ty, d.dim, r.span)));

        let outer = self.env.snapshot();
        let provisional = Arc::new(FuncSig {
            name: fname.clone(),
            def_span: name.span,
            params: psigs.clone(),
            ret: Ty::Unknown,
            generic: self.open_vars(&psigs, &Ty::Unknown),
            derivative: None,
        });
        self.bind_at(&fname, Ty::Func(provisional), name.span);

        self.fns.push(FnCtx {
            name: fname.clone(),
            declared: declared.clone(),
            returns: Vec::new(),
            locals: BTreeSet::new(),
        });
        let saved_loops = std::mem::take(&mut self.loops);
        for (p, sig) in params.iter().zip(&psigs) {
            let ty = sig.ty.clone().map_or(Ty::Unknown, Ty::Value);
            self.bind_at(&sig.name, ty, p.name.span);
            if let (Some(_), Some(ann)) = (&sig.ty, sig.annotation) {
                self.note_param(p.name.span, ann);
            }
        }
        self.diverged = false;
        let entry = self.begin_contracts(params, body);
        self.block(body);
        let ends_diverged = self.diverged;
        let ctx = self.fns.pop();
        self.loops = saved_loops;

        let has_value = ctx
            .as_ref()
            .is_some_and(|c| c.declared.is_some() || c.returns.iter().any(|r| r.value.is_some()));
        let ret_ty = match ctx {
            Some(ctx) => self.finish_returns(name, &ctx, ends_diverged),
            None => Ty::Unknown,
        };
        let result = has_value.then(|| self.env.resolve(&ret_ty));
        self.finish_contracts(name.span, &fname, params, body, entry, result);
        // Signature with what the body taught us about its size variables.
        let params: Vec<ParamSig> = psigs
            .into_iter()
            .map(|mut p| {
                p.ty = p.ty.map(|v| ValueTy {
                    dim: v.dim,
                    ty: self.env.shapes.apply_type(&v.ty),
                });
                p
            })
            .collect();
        let mut excluded = self.open_vars_of_scope(&outer);
        for v in module_vars.values() {
            if let ShapeDim::Var(rv) = self.env.shapes.resolve(v) {
                excluded.push(rv);
            }
        }
        let generic: Vec<ShapeVar> = self
            .open_vars(&params, &ret_ty)
            .into_iter()
            .filter(|v| !excluded.iter().any(|e| e.id() == v.id()))
            .collect();
        self.env.shape_vars = module_vars;
        self.env.restore(outer);
        self.diverged = false;
        let sig = FuncSig {
            name: fname.clone(),
            def_span: name.span,
            params,
            ret: ret_ty,
            generic,
            derivative: None,
        };
        let sig = Arc::new(sig);
        self.funcs.insert(name.span, sig.clone());
        self.bind_at(&fname, Ty::Func(sig), name.span);
    }

    /// The unsolved size variables in the parameters' types and `ret`.
    fn open_vars(&self, params: &[ParamSig], ret: &Ty) -> Vec<ShapeVar> {
        let mut out = Vec::new();
        for p in params {
            if let Some(v) = &p.ty {
                self.collect_vars(&v.ty, &mut out);
            }
        }
        if let Ty::Value(v) = ret {
            self.collect_vars(&v.ty, &mut out);
        }
        out
    }

    /// The unsolved size variables free in the names of `scope`.
    fn open_vars_of_scope(&self, scope: &Scope) -> Vec<ShapeVar> {
        let mut out = Vec::new();
        for t in scope.names.values() {
            if let Ty::Value(v) = t {
                self.collect_vars(&v.ty, &mut out);
            }
        }
        out
    }

    fn collect_vars(&self, t: &TensorType, out: &mut Vec<ShapeVar>) {
        for d in self.env.shapes.apply(t.shape()).dims() {
            if let ShapeDim::Var(v) = d {
                if !out.iter().any(|o| o.id() == v.id()) {
                    out.push(v.clone());
                }
            }
        }
    }

    /// `return`, with or without a value.
    #[inline(never)]
    pub(crate) fn return_stmt(&mut self, span: Span, value: Option<&Expr>) {
        let typed = value.map(|v| (v.span, self.expr(v)));
        self.diverged = true;
        let Some(ctx) = self.fns.last_mut() else {
            self.error(
                Diagnostic::error("`return` outside a function", span)
                    .with_label("not inside a `def`")
                    .with_help("`return` only works in the body of a `def`"),
            );
            return;
        };
        let declared = ctx.declared.clone();
        ctx.returns.push(ReturnSite {
            span,
            value: typed.clone(),
        });
        if let Some((decl, dim, ann)) = declared {
            self.check_declared_return(span, typed, (&decl, dim, ann), value);
        }
    }

    /// One `return` against the declared `-> T`: it needs a value of a
    /// matching shape and, if `T` names a unit, of that dimension. Without a
    /// declared unit the returned values decide the dimension.
    fn check_declared_return(
        &mut self,
        span: Span,
        value: Option<(Span, Ty)>,
        (declared, dim, ann): (&TensorType, Option<Dimension>, Span),
        expr: Option<&Expr>,
    ) {
        let Some((vspan, t)) = value else {
            self.error(
                Diagnostic::error("`return` needs a value", span)
                    .with_label("no value")
                    .with_secondary(ann, "the function is declared to return this")
                    .with_help("return a value, or remove the `->` annotation"),
            );
            return;
        };
        match t {
            Ty::Value(v) => {
                if let Some(want) = dim.filter(|d| *d != v.dim) {
                    let d = Diagnostic::error(
                        format!("expected {}, found {}", describe(&want), describe(&v.dim)),
                        vspan,
                    )
                    .with_label(format!("this is {}", label_dim(&v.dim)))
                    .with_secondary(ann, "declared here");
                    let d = match expr {
                        Some(e) => self.with_derivations(d, &[(e, v.dim)]),
                        None => d,
                    };
                    self.error(d);
                }
                let what = format!(
                    "`{}` on a value of shape {}",
                    self.quoted(ann),
                    self.env.shapes.apply(v.ty.shape())
                );
                let unified =
                    self.with_origin(ann, what, |u| u.unify(declared.shape(), v.ty.shape()));
                if let Err(e) = unified {
                    let e = e.in_context(ShapeContext::Declaration);
                    let label = self.this_is(&v);
                    self.error(
                        self.shape_diagnostic(&e, vspan)
                            .with_label(label)
                            .with_secondary(ann, "declared here"),
                    );
                }
            }
            Ty::Bool | Ty::Func(_) | Ty::Ode(_) => {
                let found = self.ty_label(&t);
                self.error(
                    Diagnostic::error(format!("expected a number, found {found}"), vspan)
                        .with_label(format!("this is {found}"))
                        .with_secondary(ann, "declared here"),
                );
            }
            Ty::Unknown => {}
        }
    }

    /// The type calls to the function give, from its `return`s: the
    /// declared shape with the dimension the returns agree on, or the type
    /// the returns agree on when nothing is declared. Reports returns that
    /// disagree, and a function that can finish without returning a value
    /// when it returns one elsewhere (or is declared to).
    fn finish_returns(&mut self, name: &Ident, ctx: &FnCtx, ends_diverged: bool) -> Ty {
        let fname = quote(&ctx.name);
        let mut first_value: Option<(Span, Ty)> = None;
        let mut first_bare: Option<Span> = None;
        let mut mixed_reported = false;
        for r in &ctx.returns {
            match &r.value {
                None => {
                    if first_bare.is_none() {
                        first_bare = Some(r.span);
                    }
                    if let (Some((vs, vt)), None, false) =
                        (&first_value, &ctx.declared, mixed_reported)
                    {
                        mixed_reported = true;
                        let d = Diagnostic::error("`return` needs a value", r.span)
                            .with_label("no value")
                            .with_secondary(
                                *vs,
                                format!("an earlier `return` gives {}", self.ty_noun(vt)),
                            )
                            .with_help("give every `return` a value, or none");
                        self.error(d);
                    }
                }
                Some((vs, t)) => {
                    let Some((fs, ft)) = first_value.clone() else {
                        if let (Some(bare), None, false) =
                            (first_bare, &ctx.declared, mixed_reported)
                        {
                            mixed_reported = true;
                            let d = Diagnostic::error("this `return` gives a value", *vs)
                                .with_secondary(bare, "an earlier `return` has no value")
                                .with_help("give every `return` a value, or none");
                            self.error(d);
                        }
                        first_value = Some((*vs, t.clone()));
                        continue;
                    };
                    let joined = self.join_returns(&ft, t, ctx.declared.is_some());
                    match joined {
                        Some(j) => first_value = Some((fs, j)),
                        None => {
                            let d = Diagnostic::error(
                                format!(
                                    "this `return` gives {}, but an earlier `return` gives {}",
                                    self.ty_noun(t),
                                    self.ty_noun(&ft)
                                ),
                                *vs,
                            )
                            .with_label(format!("this is {}", self.ty_label(t)))
                            .with_secondary(fs, format!("this is {}", self.ty_label(&ft)))
                            .with_note(format!("`{fname}` must return one type"));
                            self.error(d);
                            first_value = Some((fs, Ty::Unknown));
                        }
                    }
                }
            }
        }
        let returns_value = ctx.declared.is_some() || first_value.is_some();
        if returns_value && !ends_diverged {
            let mut d = Diagnostic::error(
                format!("`{fname}` can finish without returning a value"),
                name.span,
            )
            .with_label("some path reaches the end of this function")
            .with_help("end every path with a `return`");
            if let Some((_, _, ann)) = &ctx.declared {
                d = d.with_secondary(*ann, "declared to return a value here");
            } else if let Some((vs, _)) = &first_value {
                d = d.with_secondary(*vs, "this `return` gives a value");
            }
            self.error(d);
        }
        match (&ctx.declared, first_value) {
            // A declared unit wins (a mismatch was reported at the return).
            (Some((decl, Some(dim), _)), _) => Ty::value(*dim, self.env.shapes.apply_type(decl)),
            (Some((decl, None, _)), Some((_, Ty::Value(v)))) => {
                Ty::value(v.dim, self.env.shapes.apply_type(decl))
            }
            (Some(_), _) => Ty::Unknown,
            (None, Some((_, t))) => match t {
                Ty::Value(v) => Ty::value(v.dim, self.env.shapes.apply_type(&v.ty)),
                other => other,
            },
            (None, None) => Ty::Unknown,
        }
    }

    /// Two returned types of one function. With a declared type the shapes
    /// were already unified against it, so only the dimension must agree.
    fn join_returns(&mut self, a: &Ty, b: &Ty, declared: bool) -> Option<Ty> {
        if declared {
            return match (a, b) {
                (Ty::Value(x), Ty::Value(y)) if x.dim != y.dim => None,
                (Ty::Unknown, _) | (_, Ty::Unknown) => Some(Ty::Unknown),
                _ => Some(a.clone()),
            };
        }
        self.join_types(a, b)
    }

    /// `x op= e` inside a function where `x` belongs to an enclosing scope:
    /// Python would fail with `UnboundLocalError` at run time, and reading
    /// the outer `x` while writing a local one would be a silent surprise.
    pub(crate) fn reject_captured_update(
        &mut self,
        id: &Ident,
        op: CompoundOp,
        op_span: Span,
    ) -> bool {
        let name = self.ident(id);
        let Some(ctx) = self.fns.last() else {
            return false;
        };
        if ctx.locals.contains(name) || !self.env.is_bound(name) {
            return false;
        }
        let (fname, shown) = (quote(&ctx.name).into_owned(), quote(name).into_owned());
        let sym = op.symbol();
        self.error(
            Diagnostic::error(
                format!(
                    "`{shown}` belongs to the code around `{fname}`, so `{sym}` cannot update it"
                ),
                op_span,
            )
            .with_secondary(id.span, format!("`{shown}` is not assigned in `{fname}`"))
            .with_note(
                "assigning inside a function binds a new local; it never changes an outer name",
            )
            .with_help(format!(
                "bind a new name instead: `y = {shown} {} …`",
                op.binary().symbol()
            )),
        );
        true
    }

    // ----- calls -----

    /// A call whose callee has type `callee_ty`: a user function is checked
    /// against its signature; anything else is typed for its arguments
    /// only (and calling a value is an error).
    pub(crate) fn call(&mut self, callee_ty: Ty, callee: &Expr, args: &[Expr], span: Span) -> Ty {
        match callee_ty {
            Ty::Func(sig) => self.call_user(&sig, args, span),
            other => {
                for a in args {
                    self.expr(a);
                }
                if matches!(other, Ty::Value(_) | Ty::Bool) {
                    self.error(
                        Diagnostic::error(
                            format!("cannot call {}", self.ty_noun(&other)),
                            callee.span,
                        )
                        .with_label(format!("this is {}", self.ty_label(&other)))
                        .with_help("only functions defined with `def` can be called"),
                    );
                }
                Ty::Unknown
            }
        }
    }

    /// A call to a user function: arity, then each annotated parameter
    /// against its argument (after replacing the signature's open size
    /// variables with fresh ones), then the return type.
    #[inline(never)]
    fn call_user(&mut self, sig: &FuncSig, args: &[Expr], span: Span) -> Ty {
        let arg_tys: Vec<Ty> = args.iter().map(|a| self.expr(a)).collect();
        let fname = quote(&sig.name).into_owned();
        if args.len() != sig.params.len() {
            self.arity_error(sig, &fname, args, span);
            return Ty::Unknown;
        }
        let fresh: Vec<(u64, ShapeVar)> = sig
            .generic
            .iter()
            .map(|g| (g.id(), self.env.shapes.fresh(Arc::<str>::from(g.name()))))
            .collect();
        for ((param, arg), t) in sig.params.iter().zip(args).zip(&arg_tys) {
            let (Some(pt), Some(ann)) = (&param.ty, param.annotation) else {
                continue;
            };
            self.check_argument(&fname, &param.name, pt, ann, arg, t, &fresh);
        }
        if sig.derivative.is_none() {
            self.warn_call_breaks_contract(sig.def_span, &sig.name, args, span);
        }
        match &sig.ret {
            Ty::Value(v) => {
                let ty = self.substitute(&v.ty, &fresh);
                Ty::value(v.dim, self.env.shapes.apply_type(&ty))
            }
            other => other.clone(),
        }
    }

    #[cold]
    #[inline(never)]
    fn arity_error(&mut self, sig: &FuncSig, fname: &str, args: &[Expr], span: Span) {
        let n = sig.params.len();
        let plural = if n == 1 { "" } else { "s" };
        let (primary, label) = if args.len() > n {
            let extra = args.get(n).map_or(span, |a| a.span);
            (
                args.last().map_or(extra, |l| extra.join(l.span)),
                "extra argument".to_string(),
            )
        } else {
            let missing = sig.params.get(args.len()).map_or("", |p| p.name.as_str());
            (span, format!("missing argument `{}`", quote(missing)))
        };
        self.error(
            Diagnostic::error(
                format!("`{fname}` takes {n} argument{plural}, found {}", args.len()),
                primary,
            )
            .with_label(label)
            .with_secondary(
                sig.def_span,
                // `derivative(f, x)` takes what `f` takes: point at `f`.
                format!(
                    "`{}` is defined here",
                    quote(sig.derivative_of().map_or(sig.name(), |d| d.func().name()))
                ),
            ),
        );
    }

    /// One argument against its annotated parameter.
    #[allow(clippy::too_many_arguments)]
    fn check_argument(
        &mut self,
        fname: &str,
        pname: &str,
        pt: &ValueTy,
        ann: Span,
        arg: &Expr,
        t: &Ty,
        fresh: &[(u64, ShapeVar)],
    ) {
        let pname = quote(pname);
        match t {
            Ty::Value(av) => {
                // The shape first: a vector passed for a scalar is a shape
                // error, whatever the units.
                let expected = self.substitute(&pt.ty, fresh);
                let what = format!("`{pname}` of `{fname}` given `{}`", self.quoted(arg.span));
                let unified =
                    self.with_origin(arg.span, what, |u| u.unify(expected.shape(), av.ty.shape()));
                if let Err(e) = unified {
                    let e = e.in_context(ShapeContext::Argument);
                    let label = self.this_is(av);
                    self.error(
                        self.shape_diagnostic(&e, arg.span)
                            .with_label(label)
                            .with_secondary(ann, "expected because of this annotation"),
                    );
                    return;
                }
                if av.dim != pt.dim {
                    let d = Diagnostic::error(
                        format!(
                            "`{pname}` of `{fname}` expects {}, found {}",
                            self.ty_noun(&Ty::Value(pt.clone())),
                            self.ty_noun(t)
                        ),
                        arg.span,
                    )
                    .with_label(format!("this is {}", self.ty_label(t)))
                    .with_secondary(ann, "expected because of this annotation");
                    // An annotation without a unit is dimensionless: say so,
                    // and how to accept any unit. With a unit, the argument
                    // is what to fix.
                    let d = if pt.dim.is_dimensionless() {
                        d.with_note("an annotation without a unit is dimensionless")
                            .with_help(
                                "write the unit in the annotation (`x: m`), or leave the parameter unannotated to accept any unit",
                            )
                    } else {
                        d.with_help("convert the argument, or fix the unit in the call")
                    };
                    self.error(d);
                }
            }
            Ty::Bool | Ty::Func(_) | Ty::Ode(_) => {
                let found = self.ty_label(t);
                self.error(
                    Diagnostic::error(format!("expected a number, found {found}"), arg.span)
                        .with_label(format!("this is {found}"))
                        .with_secondary(ann, "expected because of this annotation"),
                );
            }
            Ty::Unknown => {}
        }
    }

    /// `t` with each variable in `fresh` (by id) replaced by its copy.
    fn substitute(&mut self, t: &TensorType, fresh: &[(u64, ShapeVar)]) -> TensorType {
        let dims: Vec<ShapeDim> = t
            .shape()
            .dims()
            .iter()
            .map(|d| match d {
                ShapeDim::Var(v) => match fresh.iter().find(|(id, _)| *id == v.id()) {
                    Some((_, f)) => ShapeDim::Var(f.clone()),
                    None => d.clone(),
                },
                other => other.clone(),
            })
            .collect();
        let shape = Shape::new(dims);
        if t.tags().is_empty() {
            return TensorType::untagged(shape);
        }
        TensorType::tagged(&mut self.env.shapes, shape.clone(), t.tags())
            .unwrap_or_else(|_| TensorType::untagged(shape))
    }
}
