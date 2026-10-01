//! What an `equation` is, read from its source: the state variables in
//! order, and whether the system is second order, `q' = v`, `v' = f(t, q)`.
//!
//! The checker uses this for `--explain`, and the interpreter to lay out the
//! state it hands to `sky-ode` and to declare `Structure::SecondOrder`, so
//! the two cannot disagree. It looks only at names (never evaluates), and
//! assumes nothing about the equation being well typed: on an equation the
//! checker rejected it still answers, just not usefully.

use std::collections::{BTreeMap, BTreeSet};

use sky_ast::ast::span_text;
use sky_ast::{Equation, EquationItem, Expr, ExprKind, IndexItem};
use sky_diag::Span;

use crate::check::strip_parens;

/// The shape of an equation, from its source.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EquationInfo {
    /// The independent variable's name.
    pub time: String,
    /// The state variables, in declaration order.
    pub states: Vec<String>,
    /// If the system is `q' = v`, `v' = f(t, q)` for every state: the pairs
    /// `(q, v)` as indices into [`EquationInfo::states`], in the order of
    /// `q`'s declaration. `None` for any other system.
    pub second_order: Option<Vec<(usize, usize)>>,
}

impl EquationInfo {
    /// The position variable `q` and its velocity `v` of each pair.
    pub fn pair_names(&self) -> Vec<(&str, &str)> {
        self.second_order
            .iter()
            .flatten()
            .filter_map(|&(q, v)| {
                Some((self.states.get(q)?.as_str(), self.states.get(v)?.as_str()))
            })
            .collect()
    }
}

/// Reads `eq` (parsed from `src`).
pub fn analyze_equation(src: &str, eq: &Equation) -> EquationInfo {
    let text = |span| span_text(src, span).unwrap_or("").to_string();
    let mut states: Vec<String> = Vec::new();
    let mut derivs: BTreeMap<String, &Expr> = BTreeMap::new();
    let mut locals: BTreeMap<String, &Expr> = BTreeMap::new();
    for item in &eq.items {
        match item {
            EquationItem::State { name, .. } => {
                let n = text(name.span);
                if !states.contains(&n) {
                    states.push(n);
                }
            }
            EquationItem::Derivative { name, value, .. } => {
                derivs.entry(text(name.span)).or_insert(value);
            }
            EquationItem::Local { name, value, .. } => {
                locals.entry(text(name.span)).or_insert(value);
            }
        }
    }
    let second_order = second_order_pairs(src, &states, &derivs, &locals);
    EquationInfo {
        time: text(eq.time.span),
        states,
        second_order,
    }
}

fn second_order_pairs(
    src: &str,
    states: &[String],
    derivs: &BTreeMap<String, &Expr>,
    locals: &BTreeMap<String, &Expr>,
) -> Option<Vec<(usize, usize)>> {
    let index = |name: &str| states.iter().position(|s| s == name);
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for (qi, q) in states.iter().enumerate() {
        let rhs = strip_parens(derivs.get(q)?);
        // `q' = v` exactly: a state, and not `q` itself.
        let ExprKind::Name(id) = &rhs.kind else {
            continue;
        };
        let Some(vi) = span_text(src, id.span).and_then(index) else {
            continue;
        };
        if vi != qi {
            pairs.push((qi, vi));
        }
    }
    // Every state is in exactly one pair, once.
    if pairs.is_empty() || pairs.len() * 2 != states.len() {
        return None;
    }
    let mut seen = BTreeSet::new();
    for &(q, v) in &pairs {
        if !seen.insert(q) || !seen.insert(v) {
            return None;
        }
    }
    // The acceleration must not depend on any velocity.
    let velocities: BTreeSet<&str> = pairs
        .iter()
        .filter_map(|&(_, v)| states.get(v).map(String::as_str))
        .collect();
    for &(_, v) in &pairs {
        let accel = derivs.get(states.get(v)?)?;
        let deps = dependencies(src, accel, locals);
        if deps.iter().any(|d| velocities.contains(d.as_str())) {
            return None;
        }
    }
    Some(pairs)
}

/// Every name `e` depends on, through the local names it uses.
pub fn dependencies(src: &str, e: &Expr, locals: &BTreeMap<String, &Expr>) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut expanded: BTreeSet<String> = BTreeSet::new();
    let mut work: Vec<&Expr> = vec![e];
    while let Some(e) = work.pop() {
        for name in free_names(src, e) {
            if let Some(local) = locals.get(&name) {
                if expanded.insert(name.clone()) {
                    work.push(local);
                }
            }
            out.insert(name);
        }
    }
    out
}

/// The names written in `e`, without looking through definitions.
pub fn free_names(src: &str, e: &Expr) -> BTreeSet<String> {
    name_uses(src, e).into_iter().map(|(n, _)| n).collect()
}

/// Every name written in `e` and where, and every `±` in it (as `None`
/// with the span of the `±` expression).
pub(crate) fn name_uses(src: &str, e: &Expr) -> Vec<(String, Span)> {
    uses(src, e)
        .into_iter()
        .filter_map(|(n, s)| Some((n?, s)))
        .collect()
}

/// The spans of the `±` expressions in `e`.
pub(crate) fn plus_minus_uses(src: &str, e: &Expr) -> Vec<Span> {
    uses(src, e)
        .into_iter()
        .filter_map(|(n, s)| n.is_none().then_some(s))
        .collect()
}

fn uses(src: &str, e: &Expr) -> Vec<(Option<String>, Span)> {
    let mut out = Vec::new();
    let mut stack: Vec<&Expr> = vec![e];
    while let Some(e) = stack.pop() {
        match &e.kind {
            ExprKind::Name(id) => {
                if let Some(t) = span_text(src, id.span) {
                    out.push((Some(t.to_string()), id.span));
                }
            }
            ExprKind::Literal(_) | ExprKind::UnitLiteral { .. } => {}
            ExprKind::Unary { operand, .. }
            | ExprKind::Inverse { operand, .. }
            | ExprKind::Transpose { operand, .. } => stack.push(operand),
            ExprKind::Binary { lhs, rhs, .. } => {
                stack.push(lhs);
                stack.push(rhs);
            }
            ExprKind::BoolOp { operands, .. } => stack.extend(operands),
            ExprKind::Array(elems) => stack.extend(elems),
            ExprKind::Call { callee, args } => {
                stack.push(callee);
                stack.extend(args);
            }
            ExprKind::Index { base, items } => {
                stack.push(base);
                for item in items {
                    match item {
                        IndexItem::Expr(x) => stack.push(x),
                        IndexItem::Slice { start, stop, .. } => {
                            stack.extend(start);
                            stack.extend(stop);
                        }
                    }
                }
            }
            ExprKind::Attribute { base, .. } => stack.push(base),
            ExprKind::Paren(inner) => stack.push(inner),
            ExprKind::IfElse { then, cond, else_ } => {
                stack.push(then);
                stack.push(cond);
                stack.push(else_);
            }
            ExprKind::As { value, .. } => stack.push(value),
            ExprKind::PlusMinus { value, sigma, .. } => {
                out.push((None, e.span));
                stack.push(value);
                stack.push(sigma);
            }
        }
    }
    // In source order, so "first use" means the first one written.
    out.sort_by_key(|(_, s)| (s.start, s.end));
    out
}
