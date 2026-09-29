//! Dimensional provenance: why a name has the dimension it has.
//!
//! A physicist debugging `x = v + t` wants the derivation, not just the two
//! dimensions: `v` is a velocity *because* `v = d / t0`, where `d` is a length
//! (`d = 10 km`) and `t0` a time. This module records, for every name binding,
//! its dimension-contributing operands (names, constants, function results)
//! and renders the chain as a note on dimension-mismatch errors. It also makes
//! the two suggestions that are cheap and safe (see [`Checker::suggest`]).
//!
//! # Bounds
//!
//! * A binding stores at most [`MAX_OPERANDS`] operands, found by a walk that
//!   visits at most [`MAX_VISIT`] expression nodes (iterative, so a deeply
//!   nested expression cannot overflow the stack).
//! * Operands hold their own provenance as shared, immutable snapshots that
//!   are cut to [`MAX_DEPTH`] levels when a binding is recorded. A chain of
//!   10,000 assignments therefore costs a constant amount per binding, and
//!   dropping it never recurses deeply.
//! * Messages show at most [`MAX_DEPTH`] levels, each with at most
//!   [`MAX_OPERANDS`] operands, and every quotation of source text goes
//!   through [`display_capped`](sky_diag::display_capped).
//! * The table is keyed by the binding's source span, so it grows with the
//!   number of binding sites in the source, never with the number of
//!   executions of a loop or merges of scopes; scopes and merges already
//!   carry the span of the binding a name came from (`Env::origins`).

use std::collections::HashMap;
use std::sync::Arc;

use sky_ast::{BinaryOp, Expr, ExprKind};
use sky_diag::{Diagnostic, Span};
use sky_units::Dimension;

use crate::check::{describe, strip_parens, Checker, Ty};

/// Most operands kept per binding and shown per level.
pub(crate) const MAX_OPERANDS: usize = 3;
/// Most levels shown in a message: the name, what it came from, and what
/// that came from.
pub(crate) const MAX_DEPTH: u8 = 3;
/// Most expression nodes one operand walk visits.
const MAX_VISIT: usize = 256;

/// How a name's dimension was established.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// From the value it was assigned.
    Value,
    /// From a declared unit (`x: m`).
    Declared,
    /// From a parameter's annotation.
    Param,
}

/// Where one binding's dimension came from.
#[derive(Debug)]
pub(crate) struct Origin {
    /// The binding's source text: `v = d / t0`, `p: m`.
    text: Span,
    kind: Kind,
    operands: Vec<Operand>,
}

/// One thing that contributed a dimension to an expression.
#[derive(Clone, Debug)]
pub(crate) enum Operand {
    /// A name, with the provenance it had when the expression was written.
    Name {
        span: Span,
        dim: Dimension,
        origin: Option<Arc<Origin>>,
        /// Bound to a predefined constant rather than by the program.
        constant: bool,
    },
    /// The result of calling a user function.
    Call {
        span: Span,
        dim: Dimension,
        /// The function's name in its `def`.
        def: Span,
    },
}

/// The provenance table and the line index for "line N" in notes.
#[derive(Debug, Default)]
pub(crate) struct Provenance {
    table: HashMap<(u32, u32), Arc<Origin>>,
    line_starts: Vec<u32>,
}

impl Provenance {
    pub(crate) fn new(src: &str) -> Self {
        let mut line_starts = vec![0];
        for (i, b) in src.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(u32::try_from(i + 1).unwrap_or(u32::MAX));
            }
        }
        Provenance {
            table: HashMap::new(),
            line_starts,
        }
    }

    fn line(&self, at: Span) -> usize {
        self.line_starts.partition_point(|&s| s <= at.start)
    }
}

/// `o` with the operands' own provenance kept `depth` more levels.
fn trunc(o: &Arc<Origin>, depth: u8) -> Arc<Origin> {
    if o.operands.is_empty() {
        return Arc::clone(o);
    }
    if depth == 0 {
        return Arc::new(Origin {
            text: o.text,
            kind: o.kind,
            operands: Vec::new(),
        });
    }
    let operands = o
        .operands
        .iter()
        .map(|op| match op {
            Operand::Name {
                span,
                dim,
                origin,
                constant,
            } => Operand::Name {
                span: *span,
                dim: *dim,
                origin: origin.as_ref().map(|c| trunc(c, depth - 1)),
                constant: *constant,
            },
            call => call.clone(),
        })
        .collect();
    Arc::new(Origin {
        text: o.text,
        kind: o.kind,
        operands,
    })
}

/// `a length`, `an energy`, or `of dimension L·M` for a dimension without a
/// name.
fn noun(d: &Dimension) -> String {
    match d.name() {
        Some(n) => {
            let vowel = n.starts_with(['a', 'e', 'i', 'o', 'u']);
            format!("{} {n}", if vowel { "an" } else { "a" })
        }
        None => format!("of dimension {d}"),
    }
}

/// A dimension for a label: its name, plus its formula unless the name
/// already is one (a base dimension).
pub(crate) fn label_dim(d: &Dimension) -> String {
    let formula = d.to_string();
    match d.name() {
        Some(n) if !d.is_dimensionless() && formula.chars().count() > 1 => {
            format!("{n} ({formula})")
        }
        _ => describe(d),
    }
}

/// `a`, `a and b`, `a, b and c`.
fn join_list(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [one] => one.clone(),
        [init @ .., last] => format!("{} and {last}", init.join(", ")),
    }
}

impl Checker<'_> {
    /// The dimension-contributing operands of `e`: dimensionful names and
    /// calls of user functions, left to right, at most [`MAX_OPERANDS`].
    /// Literals and unit literals are not listed: their dimension is
    /// evident where they are written.
    fn operands_of(&self, e: &Expr) -> Vec<Operand> {
        let mut out: Vec<Operand> = Vec::new();
        let mut stack = vec![e];
        let mut visited = 0;
        while let Some(e) = stack.pop() {
            visited += 1;
            if visited > MAX_VISIT || out.len() >= MAX_OPERANDS {
                break;
            }
            match &e.kind {
                ExprKind::Paren(inner) => stack.push(inner),
                ExprKind::Unary { operand, .. } | ExprKind::Inverse { operand, .. } => {
                    stack.push(operand);
                }
                // An exponent is dimensionless, so only the base matters.
                ExprKind::Binary { op, lhs, rhs, .. } => {
                    if *op != BinaryOp::Pow {
                        stack.push(rhs);
                    }
                    stack.push(lhs);
                }
                ExprKind::IfElse { then, else_, .. } => {
                    stack.push(else_);
                    stack.push(then);
                }
                ExprKind::Name(id) => {
                    let name = self.ident(id);
                    let Some(Ty::Value(v)) = self.env.get(name) else {
                        continue;
                    };
                    let seen = out.iter().any(|o| match o {
                        Operand::Name { span, .. } => self.text(*span) == name,
                        Operand::Call { .. } => false,
                    });
                    if v.dim.is_dimensionless() || seen {
                        continue;
                    }
                    out.push(Operand::Name {
                        span: id.span,
                        dim: v.dim,
                        origin: self.origin_of(name).map(|o| trunc(&o, MAX_DEPTH - 2)),
                        constant: self.env.prelude.contains(name),
                    });
                }
                ExprKind::Call { callee, args } => {
                    let ExprKind::Name(id) = &strip_parens(callee).kind else {
                        continue;
                    };
                    match self.env.get(self.ident(id)) {
                        Some(Ty::Func(sig)) => {
                            if let Ty::Value(v) = sig.ret() {
                                if !v.dim.is_dimensionless() {
                                    out.push(Operand::Call {
                                        span: e.span,
                                        dim: v.dim,
                                        def: sig.def_span,
                                    });
                                }
                            }
                        }
                        // `inv(x)` is `x⁻¹`: the argument decides.
                        None if self.ident(id) == "inv" && args.len() == 1 => {
                            stack.push(&args[0]);
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn origin_of(&self, name: &str) -> Option<Arc<Origin>> {
        let at = self.env.origins.get(name)?;
        self.prov.table.get(&(at.start, at.end)).cloned()
    }

    fn record(&mut self, at: Span, origin: Origin) {
        self.prov.table.insert((at.start, at.end), Arc::new(origin));
    }

    /// Records where the name bound at `id` (by `id = value`, or by a
    /// declaration whose `declared` annotation wins) gets its dimension.
    /// Call before the name is bound, so that a self-reference (`x = x + y`)
    /// sees the old binding.
    pub(crate) fn note_binding(&mut self, id: Span, value: Option<&Expr>, declared: Option<Span>) {
        let (text, kind, operands) = match (declared, value) {
            (Some(ty), _) => (id.join(ty), Kind::Declared, Vec::new()),
            (None, Some(v)) => (id.join(v.span), Kind::Value, self.operands_of(v)),
            (None, None) => return,
        };
        self.record(
            id,
            Origin {
                text,
                kind,
                operands,
            },
        );
    }

    /// Records a parameter, whose dimension comes from its annotation.
    pub(crate) fn note_param(&mut self, name: Span, ann: Span) {
        self.record(
            name,
            Origin {
                text: name.join(ann),
                kind: Kind::Param,
                operands: Vec::new(),
            },
        );
    }

    /// The 1-based line of `at`, from the precomputed line index (shared
    /// by the dimension and shape debuggers).
    pub(crate) fn line_of(&self, at: Span) -> usize {
        self.prov.line(at)
    }

    /// `` `text`, line N `` or the `declared`/`parameter` variants.
    fn binding_ref(&self, o: &Origin) -> String {
        let (text, line) = (self.quoted(o.text), self.line_of(o.text));
        match o.kind {
            Kind::Value => format!("`{text}`, line {line}"),
            Kind::Declared => format!("declared `{text}`, line {line}"),
            Kind::Param => format!("parameter `{text}`, line {line}"),
        }
    }

    /// One operand as `` `d` is a length (`d = 10 km`, line 3) ``, with what
    /// its own binding was made from while `level` allows.
    fn operand_phrase(&self, op: &Operand, level: u8) -> String {
        match op {
            Operand::Name {
                span,
                dim,
                origin,
                constant,
            } => {
                let name = self.quoted(*span);
                let what = noun(dim);
                match origin {
                    Some(o) => {
                        let mut inner = self.binding_ref(o);
                        if level < MAX_DEPTH && !o.operands.is_empty() {
                            inner.push_str(&format!(", {}", self.where_clause(&o.operands, level)));
                        }
                        format!("`{name}` is {what} ({inner})")
                    }
                    None if *constant => {
                        format!("`{name}` is {what} (a predefined constant)")
                    }
                    None => format!("`{name}` is {what}"),
                }
            }
            Operand::Call { span, dim, def } => {
                let call = self.quoted(*span);
                format!(
                    "`{call}` is {} (a function result, `def` on line {})",
                    noun(dim),
                    self.line_of(*def)
                )
            }
        }
    }

    /// `where `d` is ... and `t0` is ...`, for operands one level below
    /// `level`.
    fn where_clause(&self, ops: &[Operand], level: u8) -> String {
        let phrases: Vec<String> = ops
            .iter()
            .take(MAX_OPERANDS)
            .map(|o| self.operand_phrase(o, level + 1))
            .collect();
        format!("where {}", join_list(&phrases))
    }

    /// The derivation note for the expression `e` of dimension `dim`, or
    /// `None` when there is nothing to add: a literal shows its own
    /// dimension, and a name or expression with no dimensionful operand
    /// (or nothing recorded) has no story.
    fn derivation(&self, e: &Expr, dim: &Dimension) -> Option<String> {
        let e = strip_parens(e);
        let shown = self.quoted(e.span);
        let what = noun(dim);
        match &e.kind {
            ExprKind::Literal(_) | ExprKind::UnitLiteral { .. } => None,
            ExprKind::Name(id) => {
                let name = self.ident(id);
                match self.origin_of(name) {
                    Some(o) => {
                        let text = self.quoted(o.text);
                        let because = match o.kind {
                            Kind::Value => format!("`{text}`"),
                            Kind::Declared => format!("it is declared `{text}`"),
                            Kind::Param => format!("it is the parameter `{text}`"),
                        };
                        let mut s = format!(
                            "`{shown}` is {what} because {because} (line {})",
                            self.line_of(o.text)
                        );
                        if !o.operands.is_empty() {
                            s.push_str(&format!(", {}", self.where_clause(&o.operands, 1)));
                        }
                        Some(s)
                    }
                    None if self.env.prelude.contains(name) => {
                        Some(format!("`{shown}` is {what}: a predefined constant"))
                    }
                    None => None,
                }
            }
            _ => {
                let ops = self.operands_of(e);
                if ops.is_empty() {
                    return None;
                }
                Some(format!(
                    "`{shown}` is {what}, {}",
                    self.where_clause(&ops, 1)
                ))
            }
        }
    }

    /// The dimension of `e` if it is a name, for callers that only have the
    /// dimension of a larger expression.
    pub(crate) fn name_dim(&self, e: &Expr) -> Option<Dimension> {
        let ExprKind::Name(id) = &strip_parens(e).kind else {
            return None;
        };
        match self.env.get(self.ident(id)) {
            Some(Ty::Value(v)) => Some(v.dim),
            _ => None,
        }
    }

    /// Adds a derivation note to `d` for each of `sides` (an expression and
    /// its dimension) that has a story.
    pub(crate) fn with_derivations(
        &self,
        mut d: Diagnostic,
        sides: &[(&Expr, Dimension)],
    ) -> Diagnostic {
        for (e, dim) in sides {
            if let Some(note) = self.derivation(e, dim) {
                d = d.with_note(note);
            }
        }
        d
    }

    /// A suggestion for `l op r` whose dimensions differ, or `None`.
    /// Only two cases are cheap and confident:
    ///
    /// 1. Multiplying or dividing one side by a name that occurs in the
    ///    other side (`v * t` beside a lone `v`) makes the dimensions agree, and no other such choice
    ///    exists: "did you mean `v * t0`?".
    /// 2. The sides differ by exactly one power of time or length.
    ///
    /// Anything else (in particular guessing that `+` was meant to be `*`)
    /// would be a guess, so there is no suggestion.
    pub(crate) fn suggest(&self, l: (&Expr, &Dimension), r: (&Expr, &Dimension)) -> Option<String> {
        let mut found: Vec<(String, Dimension)> = Vec::new();
        for ((se, sd), (oe, od)) in [(l, r), (r, l)] {
            // A lone name has no operands of its own to blame.
            if matches!(strip_parens(oe).kind, ExprKind::Name(_)) {
                continue;
            }
            for op in self.operands_of(oe) {
                let Operand::Name { span, dim, .. } = op else {
                    continue;
                };
                let name = self.quoted(span);
                let side = self.operand_text(se);
                if sd.checked_mul(dim).ok() == Some(*od) {
                    found.push((format!("{side} * {name}"), *od));
                }
                if sd.checked_div(dim).ok() == Some(*od) {
                    found.push((format!("{side} / {name}"), *od));
                }
            }
        }
        found.dedup_by(|a, b| a.0 == b.0);
        if let [(text, dim)] = found.as_slice() {
            return Some(format!("did you mean `{text}` ({})?", noun(dim)));
        }
        if found.is_empty() {
            let ratio = l.1.checked_div(*r.1).ok()?;
            let (a, b) = (self.operand_text(l.0), self.operand_text(r.0));
            for (unit, base) in [("time", Dimension::TIME), ("length", Dimension::LENGTH)] {
                let how = if ratio == base {
                    "times"
                } else if base.inverse().ok() == Some(ratio) {
                    "divided by"
                } else {
                    continue;
                };
                return Some(format!(
                    "these differ by a factor of {unit}: `{a}` is `{b}` {how} a {unit}"
                ));
            }
        }
        None
    }
}
