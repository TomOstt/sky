//! Moving between SIR blocks and the e-graph.
//!
//! # What is rewritten
//!
//! A block is cut into *segments*: maximal runs of operations that are
//! either *expression ops* (constants and the arithmetic and math ops, with
//! only `Number` operands and results of a known type) or other pure ops
//! (comparisons, Bool and string constants). Everything else is a
//! *barrier*: effectful ops (`print`, `call`), control flow (`if`, `while`),
//! terminators, ops that can fail but are not expression ops (`solve`), and
//! any op that touches an `Unknown`-typed value (its dynamic checks could
//! fail, so it may not be moved or dropped). A barrier ends the segment on
//! each side of it, so **no operation moves across an effectful or failing
//! op, and none is moved between blocks**.
//!
//! Inside a segment the expression ops form a DAG whose inputs are values
//! defined before the segment or by non-expression ops: those are opaque
//! *leaves*. The DAG's *roots* are the expression ops whose result is used
//! by anything outside the segment's expression ops (or by nothing: a dead
//! `div` still fails at run time, so it stays a root).
//!
//! # Writing back
//!
//! After saturation the cheapest form of each root is extracted, in program
//! order, and emitted where the root was. Operations that stay keep their
//! own span; a node a rule creates gets the join of the spans it merges.
//! Every use of an old root value is redirected to the new value. Ops that
//! are no longer needed are removed from the block (they stay in the
//! function's arena, unreferenced, as after any pass that deletes ops).
//!
//! # Failure behaviour under `Reproducible`
//!
//! The rules only drop operations that cannot fail (see `rules`). On top of
//! that, the write-back checks that the operations that can fail are
//! emitted in the same relative order as in the original block, and that
//! each of them corresponds to an original one. If not, the segment is left
//! exactly as it was. Together this means the first failing operation, and
//! so the error message and span, is unchanged.

use std::collections::{HashMap, HashSet};

use egg::{EGraph, Extractor, Id, Language, RecExpr, Runner, SimpleScheduler, StopReason};
use sky_diag::Span;
use sky_ir::exact::{self, Analysis};
use sky_ir::{
    BlockId, Effect, Function, NumberType, OpData, OpId, OpKind, Repr, Type, ValueData, ValueDef,
    ValueId,
};
use sky_units::Rational;

use crate::analysis::{add_node_with_facts, render_with, Graph, LeafInfo, NodeFacts, SkyAnalysis};
use crate::cost::{explain_choice, program_cost, term_cost, CostModel};
use crate::lang::{Lit, LitVal, Op, SkyNode};
use crate::rules::{self, RuleClass};
use crate::{Rewrite, RewriteConfig, RewriteContract, SegmentReport, Stop};

/// The most e-classes (or expression nodes) of a root that an `EG-COST-001`
/// explanation prices; a larger term is left unexplained.
const EXPLAIN_LIMIT: usize = 4096;

/// Rewrites every block of `func`. Returns the explanations of the rewrites
/// that were applied, and one report per segment that was tried.
pub(crate) fn rewrite_function(
    func: &mut Function,
    cfg: &RewriteConfig,
) -> (Vec<Rewrite>, Vec<SegmentReport>) {
    // egg times its rebuilds and its runner with `quanta`, whose first use
    // calibrates the CPU counter against the OS clock: about 200 ms, paid by
    // every `sky` process that optimizes, before any work. Without a time
    // limit (the default, so the result does not depend on the machine's
    // speed) egg never needs the time, so it gets a mock clock, which needs
    // no calibration. A configured limit keeps the real clock.
    if cfg.time_limit.is_none() {
        let (clock, _mock) = quanta::Clock::mock();
        return quanta::with_clock(&clock, || rewrite_function_timed(func, cfg));
    }
    rewrite_function_timed(func, cfg)
}

/// [`rewrite_function`] under whatever clock is current.
fn rewrite_function_timed(
    func: &mut Function,
    cfg: &RewriteConfig,
) -> (Vec<Rewrite>, Vec<SegmentReport>) {
    let mut rewrites = Vec::new();
    let mut reports = Vec::new();
    let uses = use_counts(func);
    // Facts about the operations as the program wrote them. A rewrite adds
    // values but never changes what an existing value is, so the table
    // stays valid for every segment.
    let facts = exact::analyze(func);
    for block in reachable_blocks(func) {
        // A `fused` body is opaque: its `neg` means `0 - x` (see `OpKind::Fused`).
        let owner = func.block(block).parent.and_then(|r| func.region(r).parent);
        if owner.is_some_and(|o| func.op(o).kind == OpKind::Fused) {
            continue;
        }
        // Last segment first: committing one changes the positions of the
        // ops after it in the block, never those before it, and the
        // segments hold positions.
        let mut per_segment = Vec::new();
        for seg in segments(func, block).into_iter().rev() {
            per_segment.push(rewrite_segment(func, block, &seg, &uses, &facts, cfg));
        }
        for (r, report) in per_segment.into_iter().rev() {
            rewrites.extend(r);
            reports.extend(report);
        }
    }
    (rewrites, reports)
}

/// The blocks of `func` that its body reaches, in arena order. The arena also
/// holds blocks no region owns any more: `fold` inlines the taken arm of an
/// `if` by moving its operations into the parent block, and leaves the arm's
/// block behind still listing them. Rewriting such a block would rewrite the
/// live operations through a list nobody reads, and leave the values it made
/// undefined.
fn reachable_blocks(func: &Function) -> Vec<BlockId> {
    let mut seen = vec![false; func.blocks.len()];
    let mut stack = vec![func.body];
    while let Some(r) = stack.pop() {
        for &b in &func.region(r).blocks {
            if std::mem::replace(&mut seen[b.index()], true) {
                continue;
            }
            for &o in &func.block(b).ops {
                stack.extend(func.op(o).regions.iter().copied());
            }
        }
    }
    (0..func.blocks.len())
        .filter(|&i| seen[i])
        .map(|i| BlockId(i as u32))
        .collect()
}

fn use_counts(func: &Function) -> Vec<u32> {
    let mut uses = vec![0u32; func.values.len()];
    for block in reachable_blocks(func) {
        for op in &func.block(block).ops {
            for v in &func.op(*op).operands {
                if let Some(n) = uses.get_mut(v.index()) {
                    *n += 1;
                }
            }
        }
    }
    uses
}

// ---------------------------------------------------------------------
// Segments
// ---------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    Expr,
    Pure,
    Barrier,
}

/// One maximal run of expression and pure ops.
struct Segment {
    /// Positions in the block's op list, with each op's class.
    ops: Vec<(usize, OpId, Class)>,
}

fn number_type(func: &Function, v: ValueId) -> Option<&NumberType> {
    match func.values.get(v.index())?.ty {
        Type::Number(ref n) if matches!(n.repr, Repr::Unchosen | Repr::F64) => Some(n),
        _ => None,
    }
}

fn is_scalar(t: &NumberType) -> bool {
    t.shape.rank() == 0
}

fn classify(func: &Function, id: OpId) -> Class {
    let op = func.op(id);
    let props = op.kind.properties();
    let unknown = op.operands.iter().chain(&op.results).any(
        |v| !matches!(func.values.get(v.index()).map(|d| &d.ty), Some(t) if *t != Type::Unknown),
    );
    if unknown || props.terminator || !op.regions.is_empty() {
        return Class::Barrier;
    }
    if is_expr_op(func, op) {
        return Class::Expr;
    }
    if props.effects.is_pure() {
        Class::Pure
    } else {
        Class::Barrier
    }
}

fn is_expr_op(func: &Function, op: &OpData) -> bool {
    let arity = match &op.kind {
        OpKind::ConstFloat { .. } | OpKind::ConstRational(_) => 0,
        OpKind::Neg | OpKind::PowConst { .. } | OpKind::Inverse | OpKind::Transpose => 1,
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div | OpKind::Pow | OpKind::MatMul => 2,
        OpKind::Math(f) => f.arity(),
        _ => return false,
    };
    if op.operands.len() != arity || op.results.len() != 1 {
        return false;
    }
    if let OpKind::ConstFloat { bits } = &op.kind {
        if !f64::from_bits(*bits).is_finite() {
            return false;
        }
    }
    let Some(result) = number_type(func, op.results[0]) else {
        return false;
    };
    let operands_ok = op.operands.iter().all(|v| number_type(func, *v).is_some());
    let scalar_only = matches!(
        op.kind,
        OpKind::ConstFloat { .. } | OpKind::ConstRational(_) | OpKind::Math(_)
    );
    let scalars = !scalar_only
        || (is_scalar(result)
            && op
                .operands
                .iter()
                .all(|v| number_type(func, *v).is_some_and(is_scalar)));
    operands_ok && scalars
}

fn segments(func: &Function, block: BlockId) -> Vec<Segment> {
    let mut out = Vec::new();
    let mut cur: Vec<(usize, OpId, Class)> = Vec::new();
    let flush = |cur: &mut Vec<(usize, OpId, Class)>, out: &mut Vec<Segment>| {
        if cur.iter().any(|(_, _, c)| *c == Class::Expr) {
            out.push(Segment {
                ops: std::mem::take(cur),
            });
        } else {
            cur.clear();
        }
    };
    for (pos, id) in func.block(block).ops.iter().enumerate() {
        match classify(func, *id) {
            Class::Barrier => flush(&mut cur, &mut out),
            c => cur.push((pos, *id, c)),
        }
    }
    flush(&mut cur, &mut out);
    out
}

// ---------------------------------------------------------------------
// One segment
// ---------------------------------------------------------------------

/// Something an emitted operation reads.
#[derive(Clone, Copy, Debug)]
enum Src {
    Value(ValueId),
    New(usize),
}

struct Staged {
    kind: OpKind,
    operands: Vec<Src>,
    ty: NumberType,
    span: Span,
    /// The source spans of the operands, when the extracted node is one of
    /// the program's own (same operation on the same operand classes): where
    /// the original read them. Empty when unknown (a node a rule made).
    operand_spans: Vec<Span>,
    class: Id,
    /// The block position of the root this op is emitted before.
    place: usize,
}

fn stop_of(reason: Option<StopReason>) -> Stop {
    match reason {
        Some(StopReason::Saturated) => Stop::Saturated,
        Some(StopReason::IterationLimit(_)) => Stop::IterationLimit,
        Some(StopReason::NodeLimit(_)) => Stop::NodeLimit,
        Some(StopReason::TimeLimit(_)) => Stop::TimeLimit,
        _ => Stop::Other,
    }
}

fn rewrite_segment(
    func: &mut Function,
    block: BlockId,
    seg: &Segment,
    uses: &[u32],
    facts: &Analysis,
    cfg: &RewriteConfig,
) -> (Vec<Rewrite>, Option<SegmentReport>) {
    let exprs: Vec<(usize, OpId)> = seg
        .ops
        .iter()
        .filter(|(_, _, c)| *c == Class::Expr)
        .map(|(p, o, _)| (*p, *o))
        .collect();
    let mut report = SegmentReport {
        expr_ops: exprs.len(),
        iterations: 0,
        egraph_nodes: 0,
        stop: None,
        skipped: None,
        applied: 0,
    };
    if exprs.len() > cfg.max_segment_ops {
        report.skipped = Some("segment too large");
        return (Vec::new(), Some(report));
    }

    // ---- build the RecExpr ----
    let mut expr: RecExpr<SkyNode> = RecExpr::default();
    // Per node of `expr`: the IR type and span it came from.
    let mut meta: Vec<(Option<NumberType>, Option<Span>, Option<NodeFacts>)> = Vec::new();
    // Per node of `expr`: which fallible original op (by order) made it.
    let mut fallible_of: Vec<Option<usize>> = Vec::new();
    let mut push = |expr: &mut RecExpr<SkyNode>,
                    node: SkyNode,
                    ty: Option<NumberType>,
                    span: Option<Span>,
                    node_facts: Option<NodeFacts>,
                    fallible: Option<usize>| {
        meta.push((ty, span, node_facts));
        fallible_of.push(fallible);
        expr.add(node)
    };
    let mut rec_of_value: HashMap<ValueId, Id> = HashMap::new();
    let mut leaves: Vec<LeafInfo> = Vec::new();
    let mut leaf_values: Vec<ValueId> = Vec::new();
    let mut leaf_rec: HashMap<ValueId, Id> = HashMap::new();
    let mut internal: HashMap<ValueId, u32> = HashMap::new();
    let mut fallible_count = 0usize;
    // Per node of `expr` made from an operation: the spans that operation
    // recorded for its operands.
    let mut operand_spans_of: HashMap<Id, Vec<Span>> = HashMap::new();

    for (_, id) in &exprs {
        let op = func.op(*id).clone();
        let mut args: Vec<Id> = Vec::new();
        for v in op.operands.iter() {
            // The exponent of a `PowConst` is the attribute, not an operand.
            *internal.entry(*v).or_insert(0) += 1;
            if let Some(r) = rec_of_value.get(v) {
                args.push(*r);
                continue;
            }
            if let Some(r) = leaf_rec.get(v) {
                args.push(*r);
                continue;
            }
            let ty = number_type(func, *v).cloned();
            let span = match func.value(*v).def {
                ValueDef::OpResult { op, .. } => Some(func.op(op).span),
                ValueDef::BlockArg { .. } => None,
            };
            let idx = leaves.len() as u32;
            leaves.push(LeafInfo {
                ty: ty.clone().expect("expression operands are numbers"),
                span,
                name: func.value(*v).name.clone().unwrap_or_else(|| v.to_string()),
            });
            leaf_values.push(*v);
            let r = push(
                &mut expr,
                SkyNode::leaf(Op::Leaf(idx)),
                ty,
                span,
                None,
                None,
            );
            leaf_rec.insert(*v, r);
            args.push(r);
        }
        let result_ty = number_type(func, op.results[0]).cloned();
        let node_op = match &op.kind {
            OpKind::ConstFloat { bits } => {
                let dim = result_ty
                    .as_ref()
                    .map(|t| t.dim)
                    .unwrap_or(sky_units::Dimension::DIMENSIONLESS);
                Op::Lit(Lit::new(LitVal::Float(*bits), dim))
            }
            OpKind::ConstRational(q) => {
                let dim = result_ty
                    .as_ref()
                    .map(|t| t.dim)
                    .unwrap_or(sky_units::Dimension::DIMENSIONLESS);
                Op::Lit(Lit::new(LitVal::Rat(*q), dim))
            }
            OpKind::Add => Op::Add,
            OpKind::Sub => Op::Sub,
            OpKind::Mul => Op::Mul,
            OpKind::Div => Op::Div,
            OpKind::Neg => Op::Neg,
            OpKind::Inverse => Op::Inv,
            OpKind::MatMul => Op::MatMul,
            OpKind::Transpose => Op::Transpose,
            OpKind::Pow => Op::Pow,
            OpKind::Math(f) => Op::math(*f),
            OpKind::PowConst { exponent } => {
                let lit = Lit::new(LitVal::Rat(*exponent), sky_units::Dimension::DIMENSIONLESS);
                let e = push(
                    &mut expr,
                    SkyNode::leaf(Op::Lit(lit)),
                    Some(NumberType::scalar(sky_units::Dimension::DIMENSIONLESS)),
                    Some(op.span),
                    None,
                    None,
                );
                args.push(e);
                Op::PowC
            }
            _ => unreachable!("classified as an expression op"),
        };
        let fallible = if op.kind.properties().effects.contains(Effect::Fail) {
            fallible_count += 1;
            Some(fallible_count - 1)
        } else {
            None
        };
        // The facts describe the program as written. `Fast` rewrites may
        // change a value in its last bits, which could move a value across
        // the edge of a range, so only `Reproducible` (bit-identical
        // rewrites) may rely on them.
        let node_facts = facts
            .known(op.results[0])
            .filter(|_| cfg.contract == RewriteContract::Reproducible)
            .map(|k| NodeFacts {
                positive: k.is_positive(),
                nonzero: k.is_nonzero(),
                infallible: facts.proof(*id).is_some(),
            });
        let r = push(
            &mut expr,
            SkyNode::new(node_op, args),
            result_ty,
            Some(op.span),
            node_facts,
            fallible,
        );
        if !op.operand_spans.is_empty() && op.operand_spans.len() == op.operands.len() {
            operand_spans_of.insert(r, op.operand_spans.clone());
        }
        rec_of_value.insert(op.results[0], r);
    }

    // Roots: results used outside the segment's expression ops, or unused.
    let mut roots: Vec<(usize, OpId)> = Vec::new();
    for (pos, id) in &exprs {
        let v = func.op(*id).results[0];
        let total = uses.get(v.index()).copied().unwrap_or(0);
        let inside = internal.get(&v).copied().unwrap_or(0);
        if total == 0 || total > inside {
            roots.push((*pos, *id));
        }
    }
    let root_recs: Vec<Id> = roots
        .iter()
        .map(|(_, id)| rec_of_value[&func.op(*id).results[0]])
        .collect();
    push(
        &mut expr,
        SkyNode::new(Op::Roots, root_recs.clone()),
        None,
        None,
        None,
        None,
    );

    // ---- saturate ----
    let mut eg: Graph = EGraph::new(SkyAnalysis::new(cfg.contract, leaves));
    let mut eg_of: Vec<Id> = Vec::with_capacity(expr.as_ref().len());
    let mut fallible_classes: HashMap<usize, Id> = HashMap::new();
    for (i, node) in expr.as_ref().iter().enumerate() {
        let mapped = node.clone().map_children(|c| eg_of[usize::from(c)]);
        let (ty, span, node_facts) = meta[i].clone();
        let id = add_node_with_facts(&mut eg, mapped, ty, span, node_facts);
        eg_of.push(id);
        if let Some(k) = fallible_of[i] {
            fallible_classes.insert(k, id);
        }
    }
    let roots_id = *eg_of.last().expect("the roots node was added");
    eg.rebuild();

    let rules = rules::rewrites(cfg.contract, cfg.only_rules.as_deref());
    let mut runner =
        Runner::<SkyNode, SkyAnalysis, ()>::new(SkyAnalysis::new(cfg.contract, Vec::new()))
            .with_egraph(eg)
            .with_scheduler(SimpleScheduler)
            .with_iter_limit(cfg.iter_limit)
            .with_node_limit(cfg.node_limit)
            .with_time_limit(
                cfg.time_limit
                    .unwrap_or(std::time::Duration::from_secs(3600)),
            )
            .run(&rules);
    report.iterations = runner.iterations.len();
    report.egraph_nodes = runner.egraph.total_size();
    report.stop = Some(stop_of(runner.stop_reason.take()));
    let eg = &runner.egraph;

    // ---- extract and stage ----
    let extractor = Extractor::new(eg, CostModel::new(eg));
    let root_class = eg.find(roots_id);
    let root_children: Vec<Id> = extractor.find_best_node(root_class).args.clone();

    // Original positions of the fallible ops, by class.
    let mut original_order: HashMap<Id, usize> = HashMap::new();
    for (k, id) in &fallible_classes {
        let c = eg.find(*id);
        let e = original_order.entry(c).or_insert(*k);
        *e = (*e).min(*k);
    }
    // Spans of the nodes that came from the program, by canonical node.
    let mut node_spans: HashMap<SkyNode, (u64, Span)> = HashMap::new();
    for (n, (seq, span)) in &eg.analysis.node_spans {
        let canon = n.clone().map_children(|c| eg.find(c));
        let e = node_spans.entry(canon).or_insert((*seq, *span));
        if *seq < e.0 {
            *e = (*seq, *span);
        }
    }

    // Likewise the operand spans of the program's own nodes, by canonical
    // node (the first one added wins, as for the span).
    let mut node_operand_spans: HashMap<SkyNode, Vec<Span>> = HashMap::new();
    for (i, node) in expr.as_ref().iter().enumerate() {
        if let Some(spans) = operand_spans_of.get(&Id::from(i)) {
            let canon = node
                .clone()
                .map_children(|c| eg.find(eg_of[usize::from(c)]));
            node_operand_spans
                .entry(canon)
                .or_insert_with(|| spans.clone());
        }
    }

    let mut staged: Vec<Staged> = Vec::new();
    let mut memo: HashMap<Id, Src> = HashMap::new();
    let mut root_srcs: Vec<(OpId, Src)> = Vec::new();
    let mut ok = true;
    for ((pos, root_op), class) in roots.iter().zip(&root_children) {
        let root_span = func.op(*root_op).span;
        match resolve(
            eg,
            &extractor,
            &node_spans,
            &node_operand_spans,
            &leaf_values,
            &mut staged,
            &mut memo,
            eg.find(*class),
            *pos,
            root_span,
        ) {
            Some(src) => root_srcs.push((*root_op, src)),
            None => {
                ok = false;
                break;
            }
        }
    }
    if !ok {
        report.skipped = Some("could not write the extracted form back");
        return (Vec::new(), Some(report));
    }

    // Which firings are in the output: those whose right-hand side was
    // extracted, and the rules whose right-hand sides those then rewrote.
    let canon = |n: &SkyNode| n.clone().map_children(|c| eg.find(c));
    let firings = &eg.analysis.firings;
    let mut used: HashSet<Id> = memo.keys().copied().collect();
    used.extend(root_children.iter().map(|c| eg.find(*c)));
    let mut by_rhs: HashMap<SkyNode, Vec<usize>> = HashMap::new();
    for (i, f) in firings.iter().enumerate() {
        for n in &f.rhs_nodes {
            by_rhs.entry(canon(n)).or_default().push(i);
        }
    }
    let mut hit = vec![false; firings.len()];
    let mut work: Vec<usize> = Vec::new();
    for (i, f) in firings.iter().enumerate() {
        let r = eg.find(f.root);
        if used.contains(&r) {
            let best = extractor.find_best_node(r);
            if f.rhs_nodes.iter().any(|n| canon(n) == *best) {
                hit[i] = true;
                work.push(i);
            }
        }
    }
    while let Some(g) = work.pop() {
        for n in &firings[g].root_nodes {
            for f in by_rhs.get(&canon(n)).into_iter().flatten() {
                if !hit[*f] {
                    hit[*f] = true;
                    work.push(*f);
                }
            }
        }
    }
    let applied: Vec<&crate::analysis::Firing> = firings
        .iter()
        .zip(&hit)
        .filter(|(_, h)| **h)
        .map(|(f, _)| f)
        .collect();
    if applied.is_empty() {
        return (Vec::new(), Some(report));
    }

    // ---- why the extracted form of an array root is the cheaper one ----
    let mut cost_notes: Vec<Rewrite> = Vec::new();
    for (k, ((_, root_op), class)) in roots.iter().zip(&root_children).enumerate() {
        let class = eg.find(*class);
        let is_array = eg[class]
            .data
            .ty
            .as_ref()
            .is_some_and(|t| t.shape.rank() > 0);
        if !is_array {
            continue;
        }
        let rec = root_recs[k];
        let best = |c: Id| extractor.find_best_node(eg.find(c)).clone();
        let program = |i: Id| expr[i].clone();
        let (before, after) = (
            render_with(eg, rec, &program),
            render_with(eg, class, &best),
        );
        if before == after {
            continue;
        }
        let (Some(old), Some(new)) = (
            program_cost(eg, &expr, &eg_of, usize::from(rec), EXPLAIN_LIMIT),
            term_cost(eg, class, &best, EXPLAIN_LIMIT),
        ) else {
            continue;
        };
        if new.weighted() >= old.weighted() {
            continue;
        }
        let because = explain_choice(&after, &before, &new, &old);
        cost_notes.push(Rewrite {
            pass: "egraph",
            rule: crate::COST_RULE,
            span: func.op(*root_op).span,
            before,
            after,
            because,
        });
    }

    // ---- checks that keep behaviour ----
    let position_of: HashMap<ValueId, usize> = seg
        .ops
        .iter()
        .flat_map(|(p, o, _)| func.op(*o).results.iter().map(move |v| (*v, *p)))
        .collect();
    // The order of failing operations is kept under `Reproducible`, and under
    // `Relaxed` unless a relaxed rule (which re-associates or distributes
    // operations that can overflow) is in the output.
    let relaxed_used = applied
        .iter()
        .any(|f| rules::class_of(f.rule) == Some(RuleClass::Relaxed));
    let check_order = cfg.contract == RewriteContract::Reproducible
        || (cfg.contract == RewriteContract::Relaxed && !relaxed_used);
    let mut last_order: Option<usize> = None;
    for s in &staged {
        for src in &s.operands {
            if let Src::Value(v) = src {
                if position_of.get(v).is_some_and(|p| *p >= s.place) {
                    report.skipped = Some("a value would be used before it is defined");
                    return (Vec::new(), Some(report));
                }
            }
        }
        if check_order && s.kind.properties().effects.contains(Effect::Fail) {
            match original_order.get(&s.class) {
                Some(k) if last_order.is_none_or(|l| *k > l) => last_order = Some(*k),
                _ => {
                    report.skipped = Some("the order of operations that can fail would change");
                    return (Vec::new(), Some(report));
                }
            }
        }
    }

    // ---- commit ----
    let mut new_ids: Vec<(OpId, ValueId)> = Vec::with_capacity(staged.len());
    for s in &staged {
        let op_id = OpId(func.ops.len() as u32);
        let value = ValueId(func.values.len() as u32);
        let operands = s
            .operands
            .iter()
            .map(|src| match src {
                Src::Value(v) => *v,
                Src::New(i) => new_ids[*i].1,
            })
            .collect();
        func.ops.push(OpData {
            kind: s.kind.clone(),
            operands,
            results: vec![value],
            regions: Vec::new(),
            span: s.span,
            // An extracted node that is one of the program's own keeps where
            // its operands were read; one a rule made has no such place
            // (the e-class merges expressions that read their operands in
            // different places): unknown, and a diagnostic points at the
            // operation's own span.
            operand_spans: s.operand_spans.clone(),
            parent: block,
        });
        func.values.push(ValueData {
            ty: Type::Number(s.ty.clone()),
            def: ValueDef::OpResult {
                op: op_id,
                index: 0,
            },
            name: None,
        });
        new_ids.push((op_id, value));
    }
    let resolved = |src: Src| match src {
        Src::Value(v) => v,
        Src::New(i) => new_ids[i].1,
    };

    let mut replace: HashMap<ValueId, ValueId> = HashMap::new();
    for (root_op, src) in &root_srcs {
        let old = func.op(*root_op).results[0];
        let new = resolved(*src);
        if new != old {
            replace.insert(old, new);
            if func.values[new.index()].name.is_none() {
                func.values[new.index()].name = func.values[old.index()].name.clone();
            }
        }
    }
    let root_ops: HashMap<OpId, ()> = roots.iter().map(|(_, o)| (*o, ())).collect();
    let expr_ops: HashSet<OpId> = exprs.iter().map(|(_, o)| *o).collect();
    let old_list = func.block(block).ops.clone();
    let mut new_list: Vec<OpId> = Vec::with_capacity(old_list.len());
    for (pos, id) in old_list.iter().enumerate() {
        if root_ops.contains_key(id) {
            for (i, s) in staged.iter().enumerate() {
                if s.place == pos {
                    new_list.push(new_ids[i].0);
                }
            }
        }
        if !expr_ops.contains(id) {
            new_list.push(*id);
        }
    }
    func.blocks[block.index()].ops = new_list;
    if !replace.is_empty() {
        for op in &mut func.ops {
            for v in &mut op.operands {
                if let Some(n) = replace.get(v) {
                    *v = *n;
                }
            }
        }
    }

    let note = match cfg.contract {
        RewriteContract::Reproducible => "",
        RewriteContract::Relaxed => " (contract: relaxed)",
        RewriteContract::Fast => " (contract: fast)",
    };
    let mut rewrites = Vec::new();
    for f in applied {
        rewrites.push(Rewrite {
            pass: "egraph",
            rule: f.rule,
            span: f.span.unwrap_or(func.span),
            before: f.before.clone(),
            after: f.after.clone(),
            because: format!("{}{note}", f.because),
        });
    }
    rewrites.extend(cost_notes);
    report.applied = rewrites.len();
    (rewrites, Some(report))
}

/// Stages the extracted form of `class`, children first, and returns where
/// its value lives. Iterative, so a deep expression cannot overflow the
/// stack.
#[allow(clippy::too_many_arguments)]
fn resolve(
    eg: &Graph,
    extractor: &Extractor<'_, CostModel<'_>, SkyNode, SkyAnalysis>,
    node_spans: &HashMap<SkyNode, (u64, Span)>,
    node_operand_spans: &HashMap<SkyNode, Vec<Span>>,
    leaf_values: &[ValueId],
    staged: &mut Vec<Staged>,
    memo: &mut HashMap<Id, Src>,
    class: Id,
    place: usize,
    fallback_span: Span,
) -> Option<Src> {
    let mut stack: Vec<(Id, bool)> = vec![(class, false)];
    while let Some((c, expanded)) = stack.pop() {
        let c = eg.find(c);
        if memo.contains_key(&c) {
            continue;
        }
        let node = extractor.find_best_node(c).clone();
        let value_children: &[Id] = if node.op == Op::PowC {
            &node.args[..1]
        } else {
            &node.args
        };
        if !expanded {
            stack.push((c, true));
            for k in value_children.iter().rev() {
                if !memo.contains_key(&eg.find(*k)) {
                    stack.push((*k, false));
                }
            }
            continue;
        }
        let ty = eg[c].data.ty.clone()?;
        let span = node_spans
            .get(&node)
            .map(|(_, s)| *s)
            .or(eg[c].data.span)
            .unwrap_or(fallback_span);
        let kind = match node.op {
            Op::Leaf(i) => {
                memo.insert(c, Src::Value(*leaf_values.get(i as usize)?));
                continue;
            }
            Op::Lit(l) => match l.val {
                LitVal::Float(bits) => OpKind::ConstFloat { bits },
                LitVal::Rat(q) => OpKind::ConstRational(q),
            },
            Op::Add => OpKind::Add,
            Op::Sub => OpKind::Sub,
            Op::Mul => OpKind::Mul,
            Op::Div => OpKind::Div,
            Op::Neg => OpKind::Neg,
            Op::Pow => OpKind::Pow,
            Op::Inv => OpKind::Inverse,
            Op::MatMul => OpKind::MatMul,
            Op::Transpose => OpKind::Transpose,
            Op::Math(_) => OpKind::Math(node.op.math_fn()?),
            Op::PowC => {
                let exponent: Rational = eg[node.args[1]].data.rat?;
                OpKind::PowConst { exponent }
            }
            Op::Roots => return None,
        };
        let operands: Vec<Src> = value_children
            .iter()
            .map(|k| memo.get(&eg.find(*k)).copied())
            .collect::<Option<_>>()?;
        let operand_spans = node_operand_spans
            .get(&node)
            .filter(|s| s.len() == operands.len())
            .cloned()
            .unwrap_or_default();
        staged.push(Staged {
            kind,
            operands,
            ty,
            span,
            operand_spans,
            class: c,
            place,
        });
        memo.insert(c, Src::New(staged.len() - 1));
    }
    memo.get(&eg.find(class)).copied()
}
