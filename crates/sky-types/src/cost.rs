//! `cost` lines: the performance contract of a function (spec
//! "Performance contracts").
//!
//! The checker only checks the declaration. `cost flops <= 2 * N^3` is
//! well formed when its bound is a polynomial in sizes (numbers, `+ - *`,
//! `^` with a whole exponent, `/` by a number: see [`crate::poly`]) and every
//! size it names is a size in the function's parameter annotations
//! (`Matrix[N, K]` names `N` and `K`). Whether the function meets the
//! contract is `sky check`'s question, answered from SIR by the static cost
//! model (`sky_ir::cost`); nothing here evaluates anything.

use std::collections::BTreeSet;

use sky_ast::{Block, DimExpr, Param, StmtKind};
use sky_diag::{display_capped as quote, Diagnostic};

use crate::check::Checker;
use crate::poly::poly_from_expr;

/// The names of the sizes a `def`'s parameter annotations use, in order of
/// first appearance.
pub fn param_sizes(src: &str, params: &[Param]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for p in params {
        let Some(shape) = p.ty.as_ref().and_then(|t| t.shape.as_ref()) else {
            continue;
        };
        for d in &shape.dims {
            if let DimExpr::Name(id) = d {
                let name = sky_ast::ast::span_text(src, id.span).unwrap_or("?");
                if !out.iter().any(|n| n == name) {
                    out.push(name.to_string());
                }
            }
        }
    }
    out
}

impl Checker<'_> {
    /// The `cost` lines at the start of a `def` body: each bound must be a
    /// polynomial in the function's sizes.
    pub(crate) fn check_cost_lines(&mut self, fname: &str, params: &[Param], body: &Block) {
        let mut sizes: Option<Vec<String>> = None;
        for s in &body.stmts {
            match &s.kind {
                StmtKind::Cost { bound, .. } => {
                    let mut names = Vec::new();
                    let poly = poly_from_expr(self.src, bound, &mut names);
                    if let Err(e) = poly {
                        let mut d =
                            Diagnostic::error(e.message, e.span).with_label("in a cost bound");
                        if let Some(h) = e.help {
                            d = d.with_help(h);
                        }
                        self.error(d);
                        continue;
                    }
                    let sizes = sizes.get_or_insert_with(|| param_sizes(self.src, params));
                    let mut reported: BTreeSet<&str> = BTreeSet::new();
                    for (name, span) in &names {
                        if sizes.contains(name) || !reported.insert(name.as_str()) {
                            continue;
                        }
                        let shown = quote(name);
                        let mut d = Diagnostic::error(
                            format!("`{shown}` is not a size of `{}`", quote(fname)),
                            *span,
                        )
                        .with_label("not a size in the parameters");
                        d = if sizes.is_empty() {
                            d.with_help(
                                "a cost is a function of the sizes of the arrays: name them in the parameters, e.g. `A: Matrix[N, N]`",
                            )
                        } else {
                            let list: Vec<String> =
                                sizes.iter().map(|n| format!("`{}`", quote(n))).collect();
                            d.with_help(format!(
                                "the sizes of `{}` are {}",
                                quote(fname),
                                list.join(", ")
                            ))
                        };
                        if self.env.is_bound(name) {
                            d = d.with_note(
                                "a cost depends on the shapes of the arrays, not on the values of numbers",
                            );
                        }
                        self.error(d);
                    }
                }
                StmtKind::Assert {
                    kind: sky_ast::AssertKind::Requires | sky_ast::AssertKind::Ensures,
                    ..
                } => {}
                _ => break,
            }
        }
    }
}
