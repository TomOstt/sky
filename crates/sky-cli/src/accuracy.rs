//! `sky check` and error budgets: what the roundoff analysis
//! (`sky_ir::roundoff`) says about each `accuracy x within tol` statement,
//! and, with `--explain`, about each printed value.
//!
//! Three outcomes for a budget:
//!
//! * **proven**: the analysis bounds the error below the budget. A note, only
//!   under `--explain`: "`x` is accurate to 3.3e-16 relative, within the
//!   budget 1e-12".
//! * **unproven**: it can neither prove nor refute the budget. A warning that
//!   names the operations that dominate the error (a catastrophic
//!   cancellation, an ill-conditioned function, a loop of unknown length).
//!   When the checker's cancellation warning already reports the same
//!   expression, the warning says so.
//! * **violated**: the analysis proves the error exceeds the budget (it
//!   encloses the exact value, and the `f64` is too far from it). An error:
//!   `sky check` exits 1.
//!
//! The analysis runs on the module as lowering produces it, before any
//! optimization pass: the accuracy of the program as written. A program
//! that cannot be lowered yet gets a warning that its budgets were not
//! checked, never silence.

use sky_ast::ast::span_text;
use sky_ast::{Stmt, StmtKind};
use sky_diag::{display_capped, Diagnostic, Severity, Span};
use sky_ir::roundoff::{self, Blame, BlameKind, Goal, Options, Tolerance, Verdict};
use sky_ir::{OpKind, ValueDef};

/// The most printed-value notes `--explain` shows.
const MAX_PRINT_NOTES: usize = 25;

/// Whether `stmts` (or a block inside them) has an `accuracy` statement.
pub fn has_accuracy(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match &s.kind {
        StmtKind::Accuracy { .. } => true,
        StmtKind::If {
            branches,
            else_block,
        } => {
            branches.iter().any(|(_, b)| has_accuracy(&b.stmts))
                || else_block.as_ref().is_some_and(|b| has_accuracy(&b.stmts))
        }
        StmtKind::While { body, .. } | StmtKind::For { body, .. } | StmtKind::Def { body, .. } => {
            has_accuracy(&body.stmts)
        }
        _ => false,
    })
}

/// Below this a bound says only that the enclosure of the exact value is
/// the limit, not the rounding: 27 digits.
const NEGLIGIBLE: f64 = 1e-25;

/// A bound as a short number: `3.3e-16`.
fn bound(x: f64) -> String {
    if x < NEGLIGIBLE {
        "under 1e-25".to_string()
    } else {
        format!("{x:.1e}")
    }
}

/// The diagnostics for the budgets of `module` (checked without errors),
/// and with `explain`, the printed values' accuracy. `existing` are the
/// checker's diagnostics, to link to its cancellation warnings.
pub fn diagnostics(
    module: &sky_ast::Module,
    src: &str,
    checked: &sky_types::Checked,
    existing: &[Diagnostic],
    explain: bool,
) -> Vec<Diagnostic> {
    let budgets = has_accuracy(&module.stmts);
    if !budgets && !explain {
        return Vec::new();
    }
    let (ir, goals) = match sky_lower::lower_with_accuracy(module, src, checked) {
        Ok(x) => x,
        Err(e) => {
            if !budgets {
                return Vec::new();
            }
            let mut d = Diagnostic::warning(
                format!("the accuracy budgets were not checked: {}", e.message),
                e.span,
            )
            .with_note("the program checks, but lowering to SIR does not support it yet, and the roundoff analysis runs on SIR");
            if let Some(n) = e.note {
                d = d.with_note(n);
            }
            return vec![d];
        }
    };
    if sky_ir::verify(&ir).is_err() {
        return Vec::new();
    }
    let analysis = roundoff::analyze(&ir, &Options::default());
    let mut out = Vec::new();
    for goal in &goals {
        if let Some(d) = report(&analysis, &ir, goal, src, existing, explain) {
            out.push(d);
        }
    }
    if explain {
        out.extend(printed_values(&analysis, &ir, src));
    }
    out
}

/// The budget as written (`1e-12`, `2 mm`), from the statement's text.
fn budget_text(src: &str, goal: &Goal) -> String {
    let fallback = || match goal.tolerance {
        Tolerance::Relative(t) | Tolerance::Absolute(t) => format!("{t:e}"),
    };
    let Some(text) = span_text(src, goal.span) else {
        return fallback();
    };
    text.trim_start_matches("accuracy")
        .trim_start()
        .strip_prefix(goal.name.as_str())
        .map(str::trim_start)
        .and_then(|t| t.strip_prefix("within"))
        .map(|t| display_capped(t.trim()).into_owned())
        .filter(|t| !t.is_empty())
        .unwrap_or_else(fallback)
}

fn kind_of(t: Tolerance) -> &'static str {
    match t {
        Tolerance::Relative(_) => "relative",
        Tolerance::Absolute(_) => "absolute",
    }
}

fn report(
    analysis: &roundoff::Analysis,
    ir: &sky_ir::Module,
    goal: &Goal,
    src: &str,
    existing: &[Diagnostic],
    explain: bool,
) -> Option<Diagnostic> {
    let name = display_capped(&goal.name);
    let budget = budget_text(src, goal);
    let kind = kind_of(goal.tolerance);
    let si = match goal.tolerance {
        Tolerance::Absolute(_) => " (SI units)",
        Tolerance::Relative(_) => "",
    };
    match analysis.check(ir, goal) {
        Verdict::Proven { abs, rel } => {
            if !explain {
                return None;
            }
            let (got, exact) = match goal.tolerance {
                Tolerance::Relative(_) => (rel, rel == 0.0),
                Tolerance::Absolute(_) => (abs, abs == 0.0),
            };
            let message = if exact {
                format!("`{name}` is computed exactly, within the budget {budget}")
            } else {
                format!(
                    "`{name}` is accurate to {} {kind}{si}, within the budget {budget}",
                    bound(got)
                )
            };
            let mut d = Diagnostic::note(message, goal.span)
                .with_note("proven by the roundoff analysis: every operation's rounding error, propagated through the program, against the exact real result");
            if !exact && matches!(goal.tolerance, Tolerance::Relative(_)) && abs.is_finite() {
                d = d.with_note(format!("the absolute error is at most {}{si}", bound(abs)));
            }
            Some(d)
        }
        Verdict::Violated {
            lower_abs,
            lower_rel,
            computed,
            exact,
            blame,
        } => {
            let least = match goal.tolerance {
                Tolerance::Relative(_) => bound(lower_rel),
                Tolerance::Absolute(_) => format!("{}{si}", bound(lower_abs)),
            };
            let mut d = Diagnostic::error(
                format!("`{name}` does not meet its accuracy budget of {budget}"),
                goal.span,
            )
            .with_label(format!("the {kind} error is at least {least}"));
            if let (Some(c), Some(x)) = (computed, exact) {
                d = d.with_note(format!(
                    "`sky run` computes {}, but the exact value is about {x:.4e}",
                    sky_ir::exact::range::num_text(c)
                ));
            }
            d = d.with_note(
                "proven: the analysis encloses the exact value to about 27 digits, and the f64 result is outside the budget around it",
            );
            Some(with_blame(d, &blame, src, existing))
        }
        Verdict::Unproven { abs, rel, blame } => {
            let (shown, other) = match goal.tolerance {
                Tolerance::Relative(_) => (rel, abs),
                Tolerance::Absolute(_) => (abs, rel),
            };
            let mut d = Diagnostic::warning(
                format!("cannot prove that `{name}` is accurate to within {budget} ({kind})"),
                goal.span,
            )
            .with_label("the budget is declared here");
            d = d.with_note(if shown.is_finite() {
                format!(
                    "the analysis proves a {kind} error of at most {}{si}, which is more than the budget",
                    bound(shown)
                )
            } else if other.is_finite() {
                match goal.tolerance {
                    Tolerance::Relative(_) => format!(
                        "the analysis proves an absolute error of at most {}{si}, but no relative bound: the value may be close to zero, or subtract nearly equal numbers",
                        bound(other)
                    ),
                    Tolerance::Absolute(_) => format!(
                        "the analysis proves a relative error of at most {}, but no absolute bound",
                        bound(other)
                    ),
                }
            } else {
                "the analysis could not bound the error at all".to_string()
            });
            Some(with_blame(d, &blame, src, existing))
        }
        Verdict::Unknown { reason } => Some(
            Diagnostic::warning(
                format!("cannot check the accuracy budget of `{name}`"),
                goal.span,
            )
            .with_note(reason.to_string()),
        ),
        Verdict::Unreached => explain.then(|| {
            Diagnostic::note(
                format!("the accuracy budget of `{name}` is in code that never runs"),
                goal.span,
            )
        }),
    }
}

/// Adds the operations that dominate the error, as labels on their source.
fn with_blame(
    mut d: Diagnostic,
    blame: &[Blame],
    src: &str,
    existing: &[Diagnostic],
) -> Diagnostic {
    for b in blame {
        let text = span_text(src, b.span)
            .map(|t| display_capped(t).into_owned())
            .unwrap_or_default();
        if b.span == d.primary.span {
            d = d.with_note(format!("`{text}` {}", b.describe()));
        } else {
            d = d.with_secondary(b.span, b.describe());
        }
        // The checker's cancellation warning, if it caught the same
        // expression, already carries the stable rewrite.
        let linked = matches!(
            b.kind,
            BlameKind::Cancellation { .. } | BlameKind::Conditioning { .. }
        ) && existing.iter().any(|w| {
            w.severity == Severity::Warning
                && w.primary.span == b.span
                && w.notes.iter().any(|n| n.contains("nearly equal numbers"))
        });
        if linked {
            d = d.with_note(format!(
                "`{text}` is the cancellation reported by the warning above; its help gives a form without the loss"
            ));
        } else if matches!(b.kind, BlameKind::Cancellation { .. }) {
            d = d.with_help(
                "rewrite the expression so that it does not subtract nearly equal numbers",
            );
        }
    }
    d
}

/// `--explain`: how accurate is each printed number?
fn printed_values(
    analysis: &roundoff::Analysis,
    ir: &sky_ir::Module,
    src: &str,
) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    let mut omitted = 0usize;
    let mut last = Span::point(0);
    for f in &ir.functions {
        for op in &f.ops {
            if op.kind != OpKind::Print {
                continue;
            }
            for &v in &op.operands {
                let Some(n) = analysis.num(&f.name, v) else {
                    continue;
                };
                let shown = if n.rel.is_finite() { n.rel } else { n.abs };
                if n.is_exact() || !n.is_bounded() || shown < NEGLIGIBLE {
                    continue;
                }
                let value = f.value(v);
                let what = match (&value.name, value.def) {
                    (Some(name), _) => display_capped(name).into_owned(),
                    (None, ValueDef::OpResult { op: def, .. }) => {
                        match span_text(src, f.op(def).span) {
                            Some(t) => display_capped(t).into_owned(),
                            None => continue,
                        }
                    }
                    _ => continue,
                };
                if out.len() >= MAX_PRINT_NOTES {
                    omitted += 1;
                    last = op.span;
                    continue;
                }
                if n.lower_rel > 0.01 {
                    out.push(
                        Diagnostic::note(
                            format!(
                                "printed value of `{what}` has a relative error of at least {}",
                                bound(n.lower_rel)
                            ),
                            op.span,
                        )
                        .with_note("proven by the roundoff analysis: the exact value is enclosed to about 27 digits, and the printed f64 is far from it"),
                    );
                    continue;
                }
                let how = if n.rel.is_finite() {
                    format!("{} relative", bound(n.rel))
                } else {
                    format!("{} absolute (in SI units)", bound(n.abs))
                };
                out.push(
                    Diagnostic::note(
                        format!("printed value of `{what}` is accurate to within {how}"),
                        op.span,
                    )
                    .with_note("a proven bound on the rounding error against the exact real result (roundoff analysis)"),
                );
            }
        }
    }
    if omitted > 0 {
        out.push(Diagnostic::note(
            format!(
                "{omitted} more printed value{} not shown",
                if omitted == 1 { "" } else { "s" }
            ),
            last,
        ));
    }
    out
}
