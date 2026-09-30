//! `sky check` and performance contracts: what the static cost model
//! (`sky_ir::cost`) says about each `cost` line of a function, and, with
//! `--explain`, a cost summary of every function.
//!
//! Three outcomes for a contract, as for an accuracy budget:
//!
//! * **proven**: the model bounds the cost within the contract for every
//!   size from 1 up. A note, only under `--explain`: "`gram` costs 2·N·M²
//!   flops, within `2 * N * M^2`".
//! * **refuted**: some size makes even the cheapest run cost more than the
//!   bound (or, for `==`, a different amount). An error, with the size, the
//!   dominant term, and the operation that costs the most there.
//! * **unproven**: the model cannot decide (a loop it cannot bound, a
//!   recursion, a value of unknown shape). A warning naming why.
//!
//! The cost is that of the module after the optimization pipeline without
//! `specialize` (`sky_opt::default_passes_without_specialize`). A contract
//! states the cost of the function for every size and argument, so it is
//! checked on the function itself: specialization makes clones for the
//! facts of particular calls, and drops a generic function no call reaches
//! any more, which would leave a contract unchecked. A program that cannot
//! be lowered yet gets a warning that its contracts were not checked, never
//! silence.

use std::collections::{BTreeMap, BTreeSet};

use sky_ast::ast::span_text;
use sky_ast::{CostMetric, CostOp, Expr, Ident, Module, Stmt, StmtKind};
use sky_diag::{display_capped, Diagnostic, Span};
use sky_ir::cost::{self, Amount, At, FunctionCost, Poly, Reason, ReasonKind, Site};
use sky_types::poly::poly_from_expr;
use sky_types::Checked;
use sky_units::Rational;

/// The most sites of a function `--explain` names as its hot spots.
const HOT_SPOTS: usize = 3;
/// The sizes tried when looking for a size that refutes a contract.
const WITNESS_SIZES: [i64; 9] = [1, 2, 3, 5, 10, 100, 1000, 10_000, 1_000_000];
/// The most sizes searched jointly; with more, all sizes are set alike.
const WITNESS_GRID_VARS: usize = 4;

/// A `cost` line.
struct Line<'a> {
    span: Span,
    metric: CostMetric,
    op: CostOp,
    bound: &'a Expr,
}

/// A `def` and its `cost` lines.
struct Def<'a> {
    span: Span,
    name: Ident,
    lines: Vec<Line<'a>>,
}

fn collect<'a>(stmts: &'a [Stmt], out: &mut Vec<Def<'a>>) {
    for s in stmts {
        match &s.kind {
            StmtKind::Def { name, body, .. } => {
                let lines = body
                    .stmts
                    .iter()
                    .filter_map(|b| match &b.kind {
                        StmtKind::Cost {
                            metric, op, bound, ..
                        } => Some(Line {
                            span: b.span,
                            metric: *metric,
                            op: *op,
                            bound,
                        }),
                        _ => None,
                    })
                    .collect();
                out.push(Def {
                    span: s.span,
                    name: *name,
                    lines,
                });
                collect(&body.stmts, out);
            }
            StmtKind::If {
                branches,
                else_block,
            } => {
                for (_, b) in branches {
                    collect(&b.stmts, out);
                }
                if let Some(b) = else_block {
                    collect(&b.stmts, out);
                }
            }
            StmtKind::While { body, .. } | StmtKind::For { body, .. } => {
                collect(&body.stmts, out);
            }
            _ => {}
        }
    }
}

/// The 1-based line of a byte offset.
fn line_of(src: &str, span: Span) -> usize {
    let end = (span.start as usize).min(src.len());
    src.as_bytes()[..end]
        .iter()
        .filter(|b| **b == b'\n')
        .count()
        + 1
}

fn quote(s: &str) -> String {
    display_capped(s).into_owned()
}

/// The diagnostics for the `cost` lines of `module` (checked without
/// errors), and with `explain`, the cost summary of each function.
pub fn diagnostics(
    module: &Module,
    src: &str,
    checked: &Checked,
    explain: bool,
) -> Vec<Diagnostic> {
    let mut defs = Vec::new();
    collect(&module.stmts, &mut defs);
    let contracts = defs.iter().any(|d| !d.lines.is_empty());
    if !contracts && !explain {
        return Vec::new();
    }
    let ir = match lower(module, src, checked) {
        Ok(ir) => ir,
        Err((message, span, note)) => {
            if !contracts {
                return Vec::new();
            }
            let mut d = Diagnostic::warning(
                format!("the cost contracts were not checked: {message}"),
                span,
            )
            .with_note("the program checks, but lowering to SIR does not support it yet, and the cost model runs on SIR");
            if let Some(n) = note {
                d = d.with_note(n);
            }
            return vec![d];
        }
    };
    let analysis = cost::analyze(&ir);
    let mut out = Vec::new();
    for def in &defs {
        // The function lowered from this `def` (a later `def` of the same
        // name is `name#2`); derivative functions made from it share its span.
        let source_name = src_name(src, def.name);
        let Some(func) = ir.functions.iter().find(|f| {
            f.span == def.span
                && (f.name == source_name || f.name.starts_with(&format!("{source_name}#")))
        }) else {
            continue;
        };
        let Some(fc) = analysis.get(&func.name) else {
            continue;
        };
        let name = quote(&src_name(src, def.name));
        for line in &def.lines {
            out.extend(check_line(src, &name, fc, line, explain));
        }
        if explain {
            out.extend(summary(src, &name, def.name.span, fc));
        }
    }
    out
}

fn src_name(src: &str, id: Ident) -> String {
    span_text(src, id.span).unwrap_or("?").to_string()
}

/// Lowers and optimizes, as `sky check --explain` shows the program, but
/// without specialization (see the module documentation).
fn lower(
    module: &Module,
    src: &str,
    checked: &Checked,
) -> Result<sky_ir::Module, (String, Span, Option<String>)> {
    let mut ir = sky_lower::lower(module, src, checked).map_err(|e| (e.message, e.span, e.note))?;
    sky_opt::default_passes_without_specialize(sky_ir::PassManager::new())
        .verify_each(true)
        .run(&mut ir)
        .map_err(|e| (format!("internal error: {e}"), module.span, None))?;
    Ok(ir)
}

// ----- wording -----

fn words(metric: CostMetric, p: &Poly) -> &'static str {
    match metric {
        CostMetric::Flops => {
            if p.as_constant() == Some(Rational::ONE) {
                "flop"
            } else {
                "flops"
            }
        }
        CostMetric::BytesRead => "bytes read",
        CostMetric::BytesWritten => "bytes written",
        CostMetric::Bytes => "bytes of memory traffic",
        CostMetric::Allocations => {
            if p.as_constant() == Some(Rational::ONE) {
                "allocation"
            } else {
                "allocations"
            }
        }
        CostMetric::AllocatedBytes => "bytes allocated",
    }
}

fn reason_text(src: &str, current: &str, r: &Reason) -> String {
    let line = line_of(src, r.span);
    let what = match &r.kind {
        ReasonKind::Loop => format!("loop at line {line}"),
        ReasonKind::Recursion { callee } => {
            format!("recursive call to `{}` at line {line}", quote(callee))
        }
        ReasonKind::UnknownShape => {
            format!("an operation at line {line} on a value of unknown shape")
        }
        ReasonKind::UnknownSize => format!("an array of unknown size at line {line}"),
        ReasonKind::FreeSize { name } => format!(
            "the call at line {line} does not fix the size `{}` of its callee",
            quote(name)
        ),
        ReasonKind::TooLarge => format!("the cost at line {line} is too large to represent"),
    };
    if r.function != current && !r.function.is_empty() && r.function != sky_ir::Module::MAIN {
        format!("{what}, in `{}`", quote(&r.function))
    } else {
        what
    }
}

fn reasons_text(src: &str, current: &str, a: &Amount) -> String {
    let mut parts: Vec<String> = a
        .unbounded
        .iter()
        .take(2)
        .map(|r| reason_text(src, current, r))
        .collect();
    if a.unbounded.len() > 2 {
        parts.push(format!("{} more", a.unbounded.len() - 2));
    }
    parts.join("; ")
}

/// An amount as `12·N`, `at most 12·N`, or `unbounded (loop at line 7)`,
/// followed by the measure's words.
fn amount_phrase(src: &str, current: &str, metric: CostMetric, a: &Amount) -> String {
    if a.is_unbounded() {
        let w = words(metric, &Poly::zero());
        return format!("{w} unbounded ({})", reasons_text(src, current, a));
    }
    match a.exact_poly() {
        Some(p) => format!("{p} {}", words(metric, p)),
        None => format!("at most {} {}", a.hi, words(metric, &a.hi)),
    }
}

/// The measure `a` of allocations with the floats they hold.
fn allocation_phrase(src: &str, current: &str, count: &Amount, bytes: &Amount) -> Option<String> {
    if count.is_zero() && bytes.is_zero() {
        return None;
    }
    if count.is_unbounded() || bytes.is_unbounded() {
        let a = if count.is_unbounded() { count } else { bytes };
        return Some(format!(
            "allocations unbounded ({})",
            reasons_text(src, current, a)
        ));
    }
    let floats = |p: &Poly| p.checked_scale(Rational::new(1, 8));
    if let (Some(c), Some(b)) = (count.exact_poly(), bytes.exact_poly()) {
        if let Some(n) = c.as_constant().filter(|n| n.is_integer() && n.numer() > 0) {
            // "2 allocations of N floats" when they are all of one size.
            let each = b
                .checked_scale(Rational::new(1, n.numer()))
                .and_then(|e| floats(&e));
            if let Some(each) = each.filter(|e| e.terms().len() == 1 && single_unit(e)) {
                let noun = if n == Rational::ONE {
                    "allocation"
                } else {
                    "allocations"
                };
                return Some(format!("{n} {noun} of {each} floats"));
            }
        }
        let noun = words(CostMetric::Allocations, c);
        let total = floats(b).map_or_else(|| b.to_string(), |f| f.to_string());
        return Some(format!("{c} {noun} ({total} floats in all)"));
    }
    let noun = words(CostMetric::Allocations, &count.hi);
    let total = floats(&bytes.hi).map_or_else(|| bytes.hi.to_string(), |f| f.to_string());
    Some(format!(
        "at most {} {noun} ({total} floats in all)",
        count.hi
    ))
}

/// Whether a one-term polynomial has coefficient 1 (`N`, `N·M`, and `1`).
fn single_unit(p: &Poly) -> bool {
    p.terms().first().is_some_and(|(_, c)| *c == Rational::ONE)
}

// ----- the summary -----

fn summary(src: &str, name: &str, at: Span, fc: &FunctionCost) -> Option<Diagnostic> {
    let c = &fc.cost;
    if c.is_zero() {
        return None;
    }
    let head = format!("cost of `{name}`");
    // Everything the function costs has no bound, for the same reason.
    let measures = [&c.flops, &c.allocations, &c.bytes_read, &c.bytes_written];
    let nonzero: Vec<&&Amount> = measures.iter().filter(|a| !a.is_zero()).collect();
    if let Some(first) = nonzero
        .first()
        .filter(|_| nonzero.iter().all(|a| a.is_unbounded()))
    {
        return Some(
            Diagnostic::note(
                format!("{head}: unbounded ({})", reasons_text(src, name, first)),
                at,
            )
            .with_note("the static cost model has no bound for it"),
        );
    }
    let mut parts = Vec::new();
    if !c.flops.is_zero() {
        parts.push(amount_phrase(src, name, CostMetric::Flops, &c.flops));
    }
    parts.extend(allocation_phrase(
        src,
        name,
        &c.allocations,
        &c.allocated_bytes,
    ));
    let mut d = Diagnostic::note(
        format!(
            "{head}: {}",
            if parts.is_empty() {
                "no flops".into()
            } else {
                parts.join(", ")
            }
        ),
        at,
    );
    let mut traffic = Vec::new();
    if !c.bytes_read.is_zero() {
        traffic.push(amount_phrase(
            src,
            name,
            CostMetric::BytesRead,
            &c.bytes_read,
        ));
    }
    if !c.bytes_written.is_zero() {
        traffic.push(amount_phrase(
            src,
            name,
            CostMetric::BytesWritten,
            &c.bytes_written,
        ));
    }
    if !traffic.is_empty() {
        d = d.with_note(format!("memory traffic: {}", traffic.join(", ")));
    }
    for site in hot_spots(fc) {
        let text = span_text(src, site.span)
            .map(|t| format!("`{}`", quote(t)))
            .unwrap_or_else(|| site.what.clone());
        let mut what = Vec::new();
        if !site.cost.flops.is_zero() {
            what.push(amount_phrase(
                src,
                name,
                CostMetric::Flops,
                &site.cost.flops,
            ));
        }
        if what.is_empty() {
            let bytes = site.cost.bytes_read.add(
                &site.cost.bytes_written,
                At {
                    function: &fc.name,
                    span: site.span,
                },
            );
            what.push(amount_phrase(src, name, CostMetric::Bytes, &bytes));
        }
        d = d.with_secondary(
            site.span,
            format!("hot spot: {text} costs {}", what.join(", ")),
        );
    }
    Some(d)
}

/// The costliest sites of a function (at most [`HOT_SPOTS`]), when there is
/// more than one thing to tell apart.
fn hot_spots(fc: &FunctionCost) -> Vec<&Site> {
    let depends_on_sizes = [&fc.cost.flops, &fc.cost.bytes_read, &fc.cost.bytes_written]
        .iter()
        .any(|a| !a.hi.vars().is_empty());
    if fc.sites.len() < 2 || !depends_on_sizes {
        return Vec::new();
    }
    let weight = |s: &Site| {
        let flops = s.cost.flops.hi.weight_at(1000.0);
        let bytes =
            s.cost.bytes_read.hi.weight_at(1000.0) + s.cost.bytes_written.hi.weight_at(1000.0);
        (flops, bytes)
    };
    let bounded = |s: &Site| {
        let c = &s.cost;
        [&c.flops, &c.bytes_read, &c.bytes_written]
            .iter()
            .all(|a| !a.is_unbounded())
    };
    let mut ranked: Vec<&Site> = fc.sites.iter().filter(|s| bounded(s)).collect();
    ranked.sort_by(|a, b| {
        let (wa, wb) = (weight(a), weight(b));
        wb.0.total_cmp(&wa.0).then(wb.1.total_cmp(&wa.1))
    });
    ranked.truncate(HOT_SPOTS);
    let mut top: Vec<&Site> = ranked;
    top.sort_by_key(|s| s.span.start);
    top
}

// ----- contracts -----

fn point_text(point: &BTreeMap<String, Rational>) -> String {
    let parts: Vec<String> = point
        .iter()
        .map(|(k, v)| format!("{} = {v}", quote(k)))
        .collect();
    match parts.as_slice() {
        [] => String::new(),
        [one] => one.clone(),
        [init @ .., last] => format!("{} and {last}", init.join(", ")),
    }
}

/// The smallest sizes (by their sum) at which `violated` holds.
fn witness(
    vars: &BTreeSet<String>,
    violated: impl Fn(&BTreeMap<String, Rational>) -> bool,
) -> Option<BTreeMap<String, Rational>> {
    let names: Vec<&String> = vars.iter().collect();
    let k = names.len();
    let mut points: Vec<BTreeMap<String, Rational>> = Vec::new();
    if k <= WITNESS_GRID_VARS {
        let mut index = vec![0usize; k];
        loop {
            points.push(
                names
                    .iter()
                    .zip(&index)
                    .map(|(n, i)| ((*n).clone(), Rational::integer(WITNESS_SIZES[*i])))
                    .collect(),
            );
            // Odometer.
            let mut pos = 0;
            while pos < k {
                index[pos] += 1;
                if index[pos] < WITNESS_SIZES.len() {
                    break;
                }
                index[pos] = 0;
                pos += 1;
            }
            if pos == k {
                break;
            }
        }
    } else {
        for &s in &WITNESS_SIZES {
            points.push(
                names
                    .iter()
                    .map(|n| ((*n).clone(), Rational::integer(s)))
                    .collect(),
            );
        }
    }
    let sum = |p: &BTreeMap<String, Rational>| p.values().map(|v| v.numer()).sum::<i64>();
    points.sort_by_key(sum);
    points.into_iter().find(|p| violated(p))
}

fn site_value(site: &Site, metric: CostMetric, point: &BTreeMap<String, Rational>) -> Option<f64> {
    let get = |a: &Amount| {
        a.lo.eval(point)
            .map(|r| r.numer() as f64 / r.denom() as f64)
    };
    let c = &site.cost;
    match metric {
        CostMetric::Flops => get(&c.flops),
        CostMetric::BytesRead => get(&c.bytes_read),
        CostMetric::BytesWritten => get(&c.bytes_written),
        CostMetric::Bytes => Some(get(&c.bytes_read)? + get(&c.bytes_written)?),
        CostMetric::Allocations => get(&c.allocations),
        CostMetric::AllocatedBytes => get(&c.allocated_bytes),
    }
}

fn check_line(
    src: &str,
    name: &str,
    fc: &FunctionCost,
    line: &Line<'_>,
    explain: bool,
) -> Option<Diagnostic> {
    let mut names = Vec::new();
    let bound = poly_from_expr(src, line.bound, &mut names).ok()?;
    let at = At {
        function: &fc.name,
        span: line.span,
    };
    let amount = fc.cost.metric(line.metric, at);
    let bound_text = span_text(src, line.bound.span)
        .map(quote)
        .unwrap_or_else(|| bound.to_source());
    let contract = span_text(src, line.span)
        .map(|t| quote(t.trim()))
        .unwrap_or_default();
    let metric_words = |p: &Poly| words(line.metric, p);

    let vars: BTreeSet<String> = amount
        .lo
        .vars()
        .into_iter()
        .chain(amount.hi.vars())
        .chain(bound.vars())
        .collect();
    let eval = |p: &Poly, pt: &BTreeMap<String, Rational>| p.eval(pt);

    // Proven.
    let proven = if amount.is_unbounded() {
        false
    } else {
        match line.op {
            CostOp::Le => amount.hi.le_for_all_sizes(&bound),
            CostOp::Eq => amount.exact_poly() == Some(&bound),
        }
    };
    if proven {
        if !explain {
            return None;
        }
        let message = match (line.op, amount.exact_poly()) {
            (CostOp::Le, Some(p)) => format!(
                "`{name}` costs {p} {}, within `{bound_text}`",
                metric_words(p)
            ),
            (CostOp::Le, None) => format!(
                "`{name}` costs at most {} {}, within `{bound_text}`",
                amount.hi,
                metric_words(&amount.hi)
            ),
            (CostOp::Eq, _) => format!(
                "`{name}` costs exactly {bound} {}, as `{contract}` states",
                metric_words(&bound)
            ),
        };
        return Some(
            Diagnostic::note(message, line.span)
                .with_note("proven from the static cost model, for every size from 1 up")
                .with_note(
                    "the cost is that of the optimized program (spec \"Performance contracts\")",
                ),
        );
    }

    // Refuted: a size at which even the cheapest run is out of the bound.
    let too_much = |pt: &BTreeMap<String, Rational>| matches!((eval(&amount.lo, pt), eval(&bound, pt)), (Some(l), Some(b)) if l > b);
    let too_little = |pt: &BTreeMap<String, Rational>| {
        line.op == CostOp::Eq
            && !amount.is_unbounded()
            && matches!((eval(&amount.hi, pt), eval(&bound, pt)), (Some(h), Some(b)) if h < b)
    };
    if let Some(pt) = witness(&vars, |pt| too_much(pt) || too_little(pt)) {
        let over = too_much(&pt);
        let shown = |p: &Poly| eval(p, &pt).map_or_else(|| "?".to_string(), |r| r.to_string());
        let (have, exact) = if over {
            (&amount.lo, amount.exact_poly().is_some())
        } else {
            (&amount.hi, amount.exact_poly().is_some())
        };
        let qualifier = match (exact, over) {
            (true, _) => "",
            (false, true) => "at least ",
            (false, false) => "at most ",
        };
        let message = match line.op {
            CostOp::Le => format!(
                "`{name}` costs {qualifier}{have} {}, which is more than `{bound_text}`",
                metric_words(have)
            ),
            CostOp::Eq => format!(
                "`{name}` costs {qualifier}{have} {}, not {bound} as `{contract}` states",
                metric_words(have)
            ),
        };
        let mut d = Diagnostic::error(message, line.span).with_label(if over {
            "the cost is more than this"
        } else {
            "the cost is less than this"
        });
        if !pt.is_empty() {
            let here = eval(have, &pt).map_or_else(Poly::zero, Poly::constant);
            d = d.with_note(format!(
                "for example with {}: it costs {} {}, and the contract says {}",
                point_text(&pt),
                shown(have),
                metric_words(&here),
                shown(&bound)
            ));
        }
        if over {
            // The dominant term of what the bound does not cover.
            let uncovered = have.checked_sub(&bound).and_then(|excess| {
                excess
                    .terms()
                    .into_iter()
                    .find(|(_, c)| c.numer() > 0)
                    .map(|(m, c)| dominant_text(m, c))
            });
            if let Some(term) = uncovered.filter(|_| !bound.terms().is_empty() || have.degree() > 0)
            {
                d = d.with_note(format!(
                    "the dominant term of the excess is {term}, which the bound does not cover"
                ));
            }
            if let Some(site) = fc
                .sites
                .iter()
                .filter_map(|s| site_value(s, line.metric, &pt).map(|v| (s, v)))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(s, _)| s)
            {
                let text = span_text(src, site.span)
                    .map(|t| format!("`{}`", quote(t)))
                    .unwrap_or_else(|| site.what.clone());
                let contribution = site_poly(site, line.metric, at);
                d = d.with_secondary(
                    site.span,
                    format!(
                        "{text} costs {contribution} {}",
                        metric_words(&contribution)
                    ),
                );
            }
        }
        d = d.with_note("proven false by the static cost model: a cost contract holds for every size from 1 up, lower-order terms included");
        if exact && line.op == CostOp::Le {
            d = d.with_help(format!(
                "the cost is exactly {have}: write `cost {} <= {}`",
                line.metric.keyword(),
                have.to_source()
            ));
        }
        return Some(d);
    }

    // Unproven.
    let mut d = Diagnostic::warning(
        format!("cannot prove that `{name}` meets `{contract}`",),
        line.span,
    )
    .with_label("this contract is not proven");
    if amount.is_unbounded() {
        d = d.with_note(format!(
            "the cost has no upper bound: {}",
            reasons_text(src, &fc.name, &amount)
        ));
        for r in amount.unbounded.iter().take(4) {
            if r.function == fc.name {
                d = d.with_secondary(r.span, "this has no known bound on its cost");
            }
        }
        d = d.with_note("the model bounds a counted loop (`for i in range(10)`, `range(len(v))`); a loop with another condition, a `break`, or a recursion has no bound");
    } else {
        d = d.with_note(format!(
            "the model says it costs at most {} {}, which it cannot show is within `{bound_text}` for every size from 1 up",
            amount.hi,
            metric_words(&amount.hi)
        ));
        d = d.with_note("the comparison of polynomials is sound but not complete: a contract may hold and still not be proven");
    }
    Some(d)
}

fn dominant_text(m: &sky_types::poly::Monomial, c: Rational) -> String {
    let p = Poly::constant(c);
    let mono = m.powers().fold(Poly::integer(1), |acc, (v, e)| {
        acc.checked_mul(&Poly::var(v).checked_pow(e).unwrap_or_default())
            .unwrap_or_default()
    });
    p.checked_mul(&mono)
        .map_or_else(|| "?".to_string(), |t| t.to_string())
}

/// What a site contributes to a metric, as a polynomial (its lower bound).
fn site_poly(site: &Site, metric: CostMetric, at: At<'_>) -> Poly {
    site.cost.metric(metric, at).lo
}
