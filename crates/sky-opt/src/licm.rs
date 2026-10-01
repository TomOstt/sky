//! Loop-invariant code motion.
//!
//! An operation in the body of a `while` (its `before` or `after` region) is
//! **invariant** when none of its operands is defined inside the loop: not a
//! region argument, not the result of an operation in either region. It
//! computes the same value on every iteration, so it can be computed once
//! before the loop. What makes that hard is not the value but the *effects*:
//! Sky reports the first failure in program order, and output before the
//! failure is part of the behaviour. This pass moves an operation only where
//! that order is provably kept.
//!
//! # What may move
//!
//! Candidates are top-level operations of the two regions (not those inside
//! an `if` arm, which may not run at all) that have results and no `io`
//! effect, and no regions but a `fused` kernel's. Not `call`s: a call can
//! fail by exhausting the call depth, whatever the callee does. Then:
//!
//! 1. **Operations that cannot fail move always.** That is a kind that never
//!    fails (`transpose`, `neg`, a constant, an array literal, `sin`) or an
//!    operation `sky_ir::exact` proves cannot fail with these operands
//!    (`x * 0.5`, `v[1]` of a `Vector[3]`, `sqrt(2)`); the test is the one
//!    `dce` uses to remove an unused operation. They go to the **pre-header**
//!    (right before the loop). If the loop runs zero times the operation was
//!    computed for nothing, which costs time and nothing else.
//! 2. **An operation that can fail moves to the pre-header only if the loop
//!    reaches it before anything else can be observed**, because then its
//!    failure comes at the same place in the program order, and no output
//!    or other failure can come first. On the *first iteration* (the only
//!    one that matters: the operation is invariant, so if it fails it fails
//!    on the first iteration), the operations that run before it must be
//!    **clean**: no `io`, no `call`, no nested `while`, and each either
//!    cannot fail at all or is shown by a range analysis of the first
//!    iteration not to fail there (the loop's initial values are often
//!    constants: `i + 1` with `i = 0` cannot overflow). The operation is
//!    reached first in two cases:
//!    * it is in the `before` region, which always runs at least once, after
//!      only clean operations. The pre-header computes it, in the same
//!      order among the moved operations.
//!    * it is in the `after` region, and the loop's first test is known to
//!      be true (the first-iteration evaluation of the condition decides it:
//!      `while i < 10` with `i = 0`), so the `after` region is reached, after
//!      only clean operations.
//! 3. **Otherwise, under a guard.** An operation that can fail in the `after`
//!    region, when the loop may run zero times, must not be computed then:
//!    a loop that never runs must not start failing. It moves into a guarded
//!    pre-header, the classic loop rotation:
//!
//!    ```text
//!    %c0, %f... = <the before region, on the initial values>   // clean
//!    %r... = if %c0 {
//!        <the hoisted operations>
//!        %w... = while <inits> { ... }                         // the same loop
//!        yield %w...
//!    } else {
//!        yield %f...                     // zero trips: the forwarded values
//!    }
//!    ```
//!
//!    The copy of the `before` region runs on the loop's initial values,
//!    which is exactly what the loop's first test runs (the loop runs it once
//!    more, on the same values, with the same result), so the guard adds no
//!    behaviour: only `before` operations that are clean are copied, and only
//!    when they are straight-line code. If the test is true the hoisted
//!    operations run where the loop would reach them, after the same clean
//!    operations; if it is false the loop's results are the forwarded values,
//!    as they would have been. Loops nested more than 200 deep are not
//!    guarded (the nesting is bounded).
//!
//! Anything else stays. In particular an operation after a `print`, after
//! an operation that can fail and depends on the loop, or in a loop whose
//! `before` region cannot be copied and whose trip count is unknown, is not
//! moved if it can fail.
//!
//! Loops are processed innermost first, so an operation invariant in two
//! loops moves out of both: the inner loop's pre-header is in the outer
//! loop's body, where the outer pass sees it.
//!
//! # Explanations
//!
//! `LICM-001`: the operation is computed once before the loop. `LICM-002`:
//! the same, under the guard. Constants move too, without a note.

use std::collections::{HashMap, HashSet};

use sky_ir::exact::{self, range, Analysis, Range};
use sky_ir::{
    BlockData, BlockId, CmpOp, Effect, Function, Module, OpData, OpId, OpKind, RegionData,
    RegionId, Repr, Type, ValueData, ValueDef, ValueId,
};

use crate::dce::removable;
use crate::rewrite::{Related, Rewrite};
use crate::util::{op_text, replace_uses};

pub const LICM_RULE: &str = "LICM-001";
pub const LICM_GUARDED_RULE: &str = "LICM-002";

/// Loops nested deeper than this are not guarded: a guard adds one level of
/// nesting, and the verifier bounds the depth.
const MAX_GUARD_DEPTH: usize = 200;

/// Moves loop-invariant operations out of every `while`. See the module
/// documentation.
pub fn licm(m: &mut Module) -> Vec<Related> {
    let mut out = Vec::new();
    for f in &mut m.functions {
        let facts = exact::analyze(f);
        let body = f.body;
        Licm {
            f,
            facts: &facts,
            out: &mut out,
        }
        .region(body, 0);
    }
    out
}

/// What the first iteration of a loop is known to compute.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Abs {
    Num(Range),
    Bool(bool),
    Unknown,
}

/// Where an operation moves.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dest {
    /// Right before the loop.
    Pre,
    /// Into the guarded pre-header.
    Guarded,
}

struct Licm<'a> {
    f: &'a mut Function,
    facts: &'a Analysis,
    out: &'a mut Vec<Related>,
}

impl Licm<'_> {
    fn region(&mut self, r: RegionId, depth: usize) {
        for b in self.f.region(r).blocks.clone() {
            for id in self.f.block(b).ops.clone() {
                if !matches!(self.f.op(id).kind, OpKind::Fused) {
                    for inner in self.f.op(id).regions.clone() {
                        self.region(inner, depth + 1);
                    }
                }
                if matches!(self.f.op(id).kind, OpKind::While) {
                    self.hoist(id, depth);
                }
            }
        }
    }

    /// Hoists what can be hoisted out of the loop `w`.
    fn hoist(&mut self, w: OpId, depth: usize) {
        let (Some(&rb), Some(&ra)) = (self.f.op(w).regions.first(), self.f.op(w).regions.get(1))
        else {
            return;
        };
        let (&[bb], &[ab]) = (&self.f.region(rb).blocks[..], &self.f.region(ra).blocks[..]) else {
            return;
        };
        let mut inside: HashSet<ValueId> = HashSet::new();
        self.mark_region(rb, &mut inside);
        self.mark_region(ra, &mut inside);

        let inits = self.f.op(w).operands.clone();
        let before_args = self.f.block(bb).args.clone();
        let after_args = self.f.block(ab).args.clone();
        let mut env: HashMap<ValueId, Abs> = HashMap::new();
        for (a, i) in before_args.iter().zip(&inits) {
            let v = self.abs_of(*i, &env, &inside);
            env.insert(*a, v);
        }

        let mut moves: Vec<(OpId, Dest)> = Vec::new();
        // Results of the operations that move to the guarded pre-header.
        let mut guarded_values: HashSet<ValueId> = HashSet::new();
        // Nothing observable, and nothing that can fail, has happened yet on
        // the first iteration (see the module documentation).
        let mut clean = true;
        let mut proven_true = false;
        let mut guard_possible = false;
        // Whether the `before` operations that stay can be copied.
        let mut copyable = true;

        for (in_after, block) in [(false, bb), (true, ab)] {
            if in_after {
                // The `before` region has been walked: what does its test say
                // on the first iteration, and can it be copied?
                let term = self
                    .f
                    .block(bb)
                    .ops
                    .last()
                    .map(|t| self.f.op(*t).clone())
                    .filter(|t| matches!(t.kind, OpKind::Condition));
                let Some(term) = term else { return };
                let test = term
                    .operands
                    .first()
                    .map_or(Abs::Unknown, |c| self.abs_of(*c, &env, &inside));
                proven_true = test == Abs::Bool(true);
                guard_possible = clean
                    && copyable
                    && depth < MAX_GUARD_DEPTH
                    && matches!(
                        term.operands.first().map(|c| self.f.ty(*c)),
                        Some(Type::Bool)
                    );
                for (a, fwd) in after_args.iter().zip(term.operands.iter().skip(1)) {
                    let v = self.abs_of(*fwd, &env, &inside);
                    env.insert(*a, v);
                }
            }
            let ops = self.f.block(block).ops.clone();
            let body_len = ops.len().saturating_sub(1);
            for &id in &ops[..body_len] {
                let (results, may_fail) = self.eval(id, &env, &inside);
                for (r, a) in self.f.op(id).results.clone().into_iter().zip(results) {
                    env.insert(r, a);
                }
                let dest = self.candidate(id, &inside).and_then(|fallible| {
                    let depends_on_guarded = self
                        .f
                        .op(id)
                        .operands
                        .iter()
                        .any(|v| guarded_values.contains(v));
                    if !fallible {
                        return Some(if depends_on_guarded {
                            Dest::Guarded
                        } else {
                            Dest::Pre
                        });
                    }
                    if !clean {
                        return None;
                    }
                    if !in_after {
                        Some(Dest::Pre)
                    } else if depends_on_guarded {
                        Some(Dest::Guarded)
                    } else if proven_true {
                        Some(Dest::Pre)
                    } else if guard_possible {
                        Some(Dest::Guarded)
                    } else {
                        None
                    }
                });
                match dest {
                    Some(d) => {
                        for r in &self.f.op(id).results {
                            inside.remove(r);
                            if d == Dest::Guarded {
                                guarded_values.insert(*r);
                            }
                        }
                        moves.push((id, d));
                    }
                    None => {
                        // It stays. Anything after it is reached only after
                        // it ran, so it must be clean for them to move
                        // where it could fail or print.
                        if may_fail {
                            clean = false;
                        }
                        if !in_after && !self.f.op(id).regions.is_empty() {
                            copyable = false;
                        }
                    }
                }
            }
        }
        if moves.is_empty() {
            return;
        }
        self.explain_moves(w, bb, &moves, proven_true);

        let parent = self.f.op(w).parent;
        let mut at = self.position(parent, w);
        for &(id, d) in &moves {
            if d == Dest::Pre {
                self.take(id);
                self.f.blocks[parent.index()].ops.insert(at, id);
                self.f.ops[id.index()].parent = parent;
                at += 1;
            }
        }
        let guarded: Vec<OpId> = moves
            .iter()
            .filter(|(_, d)| *d == Dest::Guarded)
            .map(|(o, _)| *o)
            .collect();
        if !guarded.is_empty() {
            self.guard(w, bb, &guarded);
        }
    }

    /// `Some(fallible)` if the operation may move at all: it is invariant
    /// and of a kind that can leave a loop.
    fn candidate(&self, id: OpId, inside: &HashSet<ValueId>) -> Option<bool> {
        let op = self.f.op(id);
        let props = op.kind.properties();
        let movable_kind = !props.terminator
            && !props.effects.contains(Effect::Io)
            && !props.effects.contains(Effect::Call)
            && !matches!(
                op.kind,
                OpKind::While
                    | OpKind::If
                    | OpKind::Input { .. }
                    | OpKind::LiteralError { .. }
                    | OpKind::Assert { .. }
            );
        if !movable_kind || op.results.is_empty() {
            return None;
        }
        if !(op.regions.is_empty() || matches!(op.kind, OpKind::Fused)) {
            return None;
        }
        if op.operands.iter().any(|v| inside.contains(v)) {
            return None;
        }
        Some(!removable(self.f, id, self.facts))
    }

    fn mark_region(&self, r: RegionId, inside: &mut HashSet<ValueId>) {
        for &b in &self.f.region(r).blocks {
            for a in &self.f.block(b).args {
                inside.insert(*a);
            }
            for &o in &self.f.block(b).ops {
                for res in &self.f.op(o).results {
                    inside.insert(*res);
                }
                for &inner in &self.f.op(o).regions {
                    self.mark_region(inner, inside);
                }
            }
        }
    }

    fn position(&self, block: BlockId, op: OpId) -> usize {
        self.f
            .block(block)
            .ops
            .iter()
            .position(|o| *o == op)
            .expect("the operation is in its block")
    }

    /// Detaches `id` from the block it is in.
    fn take(&mut self, id: OpId) {
        let block = self.f.op(id).parent;
        self.f.blocks[block.index()].ops.retain(|o| *o != id);
    }

    // ---------------------------------------------------- first iteration

    /// What is known about `v` on the first iteration.
    fn abs_of(&self, v: ValueId, env: &HashMap<ValueId, Abs>, inside: &HashSet<ValueId>) -> Abs {
        if let Some(a) = env.get(&v) {
            return *a;
        }
        if inside.contains(&v) {
            return Abs::Unknown;
        }
        if let ValueDef::OpResult { op, .. } = self.f.value(v).def {
            if let OpKind::ConstBool(b) = self.f.op(op).kind {
                return Abs::Bool(b);
            }
        }
        match self.facts.known(v) {
            Some(k) => Abs::Num(k.range),
            None => Abs::Unknown,
        }
    }

    /// What the first iteration computes for `id`'s results, and whether it
    /// may fail there (conservatively, from the static rules when the
    /// operation is not evaluated here).
    fn eval(
        &self,
        id: OpId,
        env: &HashMap<ValueId, Abs>,
        inside: &HashSet<ValueId>,
    ) -> (Vec<Abs>, bool) {
        let op = self.f.op(id);
        let unknown = || {
            (
                vec![Abs::Unknown; op.results.len()],
                !removable(self.f, id, self.facts),
            )
        };
        let scalar = |v: &ValueId| is_scalar_f64(self.f.ty(*v));
        let arg = |i: usize| op.operands.get(i).map(|v| self.abs_of(*v, env, inside));
        match &op.kind {
            OpKind::ConstFloat { bits } => {
                (vec![Abs::Num(Range::point(f64::from_bits(*bits)))], false)
            }
            OpKind::ConstRational(q) => (
                vec![Abs::Num(Range::point(q.numer() as f64 / q.denom() as f64))],
                false,
            ),
            OpKind::ConstBool(b) => (vec![Abs::Bool(*b)], false),
            OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div
                if op.operands.iter().all(scalar) && op.results.iter().all(scalar) =>
            {
                let (Some(Abs::Num(a)), Some(Abs::Num(b))) = (arg(0), arg(1)) else {
                    return unknown();
                };
                let step = match op.kind {
                    OpKind::Add => range::add(a, b),
                    OpKind::Sub => range::sub(a, b),
                    OpKind::Mul => range::mul(a, b),
                    _ => range::div(a, b),
                };
                (vec![Abs::Num(step.range)], step.may_fail)
            }
            OpKind::Neg if op.operands.iter().all(scalar) && op.results.iter().all(scalar) => {
                match arg(0) {
                    Some(Abs::Num(a)) => (vec![Abs::Num(range::neg(a).range)], false),
                    _ => unknown(),
                }
            }
            OpKind::Not => match arg(0) {
                Some(Abs::Bool(b)) => (vec![Abs::Bool(!b)], false),
                _ => unknown(),
            },
            OpKind::Cmp(c) => {
                let same_dim = match (op.operands.first(), op.operands.get(1)) {
                    (Some(a), Some(b)) => match (self.f.ty(*a), self.f.ty(*b)) {
                        (Type::Number(x), Type::Number(y)) => {
                            x.dim == y.dim
                                && is_scalar_f64(self.f.ty(*a))
                                && is_scalar_f64(self.f.ty(*b))
                        }
                        _ => false,
                    },
                    _ => false,
                };
                match (same_dim, arg(0), arg(1)) {
                    (true, Some(Abs::Num(a)), Some(Abs::Num(b))) => {
                        let r = decide(*c, a, b).map_or(Abs::Unknown, Abs::Bool);
                        (vec![r], false)
                    }
                    _ => unknown(),
                }
            }
            _ => unknown(),
        }
    }

    // ---------------------------------------------------------- explaining

    fn explain_moves(&mut self, w: OpId, bb: BlockId, moves: &[(OpId, Dest)], proven_true: bool) {
        let loop_span = self.f.op(w).span;
        for &(id, dest) in moves {
            let op = self.f.op(id);
            if matches!(
                op.kind,
                OpKind::ConstFloat { .. }
                    | OpKind::ConstRational(_)
                    | OpKind::ConstBool(_)
                    | OpKind::ConstStr(_)
                    | OpKind::ConstArray { .. }
                    | OpKind::ConstNothing
            ) {
                continue;
            }
            let mnemonic = op.kind.mnemonic();
            // The operands the user named; the temporaries have no name to
            // show.
            let named: Vec<String> = op
                .operands
                .iter()
                .filter_map(|v| self.f.value(*v).name.as_ref().map(|n| format!("`{n}`")))
                .collect();
            let mut because = vec![match named.len() {
                0 => format!(
                    "its operands are defined outside the loop, so `{mnemonic}` computes the \
                     same value on every iteration"
                ),
                1 => format!(
                    "{} is defined outside the loop, so `{mnemonic}` computes the same value \
                     on every iteration",
                    named[0]
                ),
                _ => format!(
                    "{} are defined outside the loop, so `{mnemonic}` computes the same \
                     value on every iteration",
                    named.join(", ")
                ),
            }];
            let fallible = !removable(self.f, id, self.facts);
            if !fallible {
                because.push(match self.facts.why(self.f, id) {
                    Some(why) => format!(
                        "`{mnemonic}` can fail in general, but not here: {why}, so computing it \
                         early cannot change what the program does"
                    ),
                    None => format!(
                        "`{mnemonic}` cannot print or fail, so computing it before the loop \
                         (even if the loop never runs) cannot change what the program does"
                    ),
                });
            }
            let rule = match dest {
                Dest::Pre if fallible => {
                    because.push(format!(
                        "`{mnemonic}` can fail, so it moves only to a place the loop reaches \
                         first: on the first iteration nothing before it can print or fail{}, so \
                         if it fails, the error is the same and comes at the same point",
                        if proven_true && op.parent != bb {
                            ", and the loop's first test is known to be true"
                        } else {
                            ""
                        }
                    ));
                    LICM_RULE
                }
                Dest::Pre => LICM_RULE,
                Dest::Guarded => {
                    because.push(format!(
                        "the loop may run zero times, and a loop that never runs must not \
                         start failing, so `{mnemonic}` is computed only if the loop's first \
                         test is true: the test (which cannot print or fail) is computed \
                         before the loop and again by the loop"
                    ));
                    LICM_GUARDED_RULE
                }
            };
            let text = op_text(self.f, id);
            self.out.push(Related {
                rewrite: Rewrite {
                    pass: "licm",
                    rule,
                    span: op.span,
                    before: text.clone(),
                    after: format!("{text}, before the loop"),
                    because,
                },
                related: loop_span,
            });
        }
    }

    // -------------------------------------------------------------- guard

    /// Wraps the loop `w` in the guard described in the module
    /// documentation, with `guarded` (in order) computed inside it.
    fn guard(&mut self, w: OpId, bb: BlockId, guarded: &[OpId]) {
        let parent = self.f.op(w).parent;
        let inits = self.f.op(w).operands.clone();
        let old_results = self.f.op(w).results.clone();
        let loop_span = self.f.op(w).span;
        let ops = self.f.block(bb).ops.clone();
        let (&term, body) = ops
            .split_last()
            .expect("a before block ends in `condition`");
        let term = self.f.op(term).clone();

        // The `before` region on the initial values, in front of the loop.
        let mut map: HashMap<ValueId, ValueId> = self
            .f
            .block(bb)
            .args
            .iter()
            .copied()
            .zip(inits.iter().copied())
            .collect();
        let mut at = self.position(parent, w);
        for &id in body {
            let src = self.f.op(id).clone();
            let new_id = OpId(self.f.ops.len() as u32);
            let mut results = Vec::new();
            for (i, r) in src.results.iter().enumerate() {
                let v = ValueId(self.f.values.len() as u32);
                self.f.values.push(ValueData {
                    ty: self.f.ty(*r).clone(),
                    def: ValueDef::OpResult {
                        op: new_id,
                        index: i as u32,
                    },
                    name: None,
                });
                map.insert(*r, v);
                results.push(v);
            }
            self.f.ops.push(OpData {
                kind: src.kind.clone(),
                operands: src
                    .operands
                    .iter()
                    .map(|v| map.get(v).copied().unwrap_or(*v))
                    .collect(),
                results,
                regions: Vec::new(),
                span: src.span,
                operand_spans: src.operand_spans.clone(),
                parent,
            });
            self.f.blocks[parent.index()].ops.insert(at, new_id);
            at += 1;
        }
        let image = |v: &ValueId| map.get(v).copied().unwrap_or(*v);
        let test = image(&term.operands[0]);
        let forwarded: Vec<ValueId> = term.operands[1..].iter().map(image).collect();

        // The `if`, and what its results stand for.
        let iff = OpId(self.f.ops.len() as u32);
        let mut new_results = Vec::new();
        for (i, r) in old_results.iter().enumerate() {
            let v = ValueId(self.f.values.len() as u32);
            let ty = self.f.ty(*r).clone();
            let name = self.f.values[r.index()].name.take();
            self.f.values.push(ValueData {
                ty,
                def: ValueDef::OpResult {
                    op: iff,
                    index: i as u32,
                },
                name,
            });
            new_results.push(v);
        }
        for (old, new) in old_results.iter().zip(&new_results) {
            replace_uses(self.f, *old, *new);
        }
        self.f.ops.push(OpData {
            kind: OpKind::If,
            operands: vec![test],
            results: new_results,
            regions: Vec::new(),
            span: loop_span,
            operand_spans: Vec::new(),
            parent,
        });

        let arm = |f: &mut Function, ops: Vec<OpId>, yielded: Vec<ValueId>| -> RegionId {
            let region = RegionId(f.regions.len() as u32);
            let block = BlockId(f.blocks.len() as u32);
            let y = OpId(f.ops.len() as u32);
            f.ops.push(OpData {
                kind: OpKind::Yield,
                operands: yielded,
                results: Vec::new(),
                regions: Vec::new(),
                span: loop_span,
                operand_spans: Vec::new(),
                parent: block,
            });
            for &o in &ops {
                f.ops[o.index()].parent = block;
            }
            let mut all = ops;
            all.push(y);
            f.blocks.push(BlockData {
                args: Vec::new(),
                ops: all,
                parent: Some(region),
            });
            f.regions.push(RegionData {
                blocks: vec![block],
                parent: Some(iff),
            });
            region
        };
        // `then`: the hoisted operations, then the loop itself.
        for &g in guarded {
            self.take(g);
        }
        let idx = self.position(parent, w);
        self.f.blocks[parent.index()].ops.remove(idx);
        let mut then_ops: Vec<OpId> = guarded.to_vec();
        then_ops.push(w);
        let then_region = arm(self.f, then_ops, old_results);
        let else_region = arm(self.f, Vec::new(), forwarded);
        self.f.ops[iff.index()].regions = vec![then_region, else_region];
        self.f.blocks[parent.index()].ops.insert(idx, iff);
    }
}

fn is_scalar_f64(t: &Type) -> bool {
    matches!(
        t,
        Type::Number(n) if n.shape.rank() == 0 && matches!(n.repr, Repr::Unchosen | Repr::F64)
    )
}

/// The truth of `a <op> b` for every pair of values in the two ranges, if it
/// is the same for all of them.
fn decide(c: CmpOp, a: Range, b: Range) -> Option<bool> {
    match c {
        CmpOp::Lt => {
            if a.hi < b.lo {
                Some(true)
            } else if a.lo >= b.hi {
                Some(false)
            } else {
                None
            }
        }
        CmpOp::Le => {
            if a.hi <= b.lo {
                Some(true)
            } else if a.lo > b.hi {
                Some(false)
            } else {
                None
            }
        }
        CmpOp::Gt => decide(CmpOp::Lt, b, a),
        CmpOp::Ge => decide(CmpOp::Le, b, a),
        CmpOp::Eq => {
            if a.hi < b.lo || b.hi < a.lo {
                Some(false)
            } else if a.lo == a.hi && b.lo == b.hi && a.lo == b.lo {
                Some(true)
            } else {
                None
            }
        }
        CmpOp::Ne => decide(CmpOp::Eq, a, b).map(|t| !t),
    }
}
