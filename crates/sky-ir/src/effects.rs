//! Effect inference.
//!
//! A function's [`Function::effects`] is *derived* from its body: the union
//! of the effects of every operation in it, including the operations in
//! nested regions (`if` and `while` have no effects of their own; their
//! regions do), with two rules for calls:
//!
//! * a `call` has the callee's effects. The call operation itself is
//!   [`Effect::Call`] (see [`crate::op::OpKind::properties`]), and that mark
//!   stays in the caller's set, because a call can also fail by exhausting
//!   the call depth, which no callee body says;
//! * recursion (direct or mutual) makes the effect sets a system of
//!   equations, so they are computed to a fixed point: every function starts
//!   with the effects of its own operations, and a caller is revisited
//!   whenever a callee's set grows. Sets only grow and there are three
//!   effects, so this stops after at most three growths per function.
//!
//! A call to a function that is not in the module (a malformed module: the
//! verifier reports it) adds nothing beyond `Call`.
//!
//! Three entry points:
//!
//! * [`infer`] computes the sets without touching the module. The verifier
//!   uses it: a function whose *declared* effects miss an inferred one is an
//!   error, because passes reorder and remove code based on the declaration.
//! * [`infer_effects`] writes the inferred sets back. Lowering calls it, so
//!   there is one implementation of "what does this function do".
//! * [`InferEffects`] is that as a [`Pass`]: it can also *shrink* a set
//!   (folding away the only division removes `fail`), so a pipeline may run
//!   it after transforms to keep the declarations tight.
//!
//! All of this is total and iterative: it works on any `Module`, however
//! malformed (out-of-range ids, regions that contain themselves), never
//! panics and never recurses over region nesting.

use std::collections::HashMap;

use crate::ir::{EffectSet, Function, Module};
use crate::op::OpKind;
use crate::pass::{Changed, Pass, PassCx};

/// The effects of a function's own operations, and the callees it names.
/// Operations that are in the arena but not reachable from the body (a
/// pass removed them from their block) do not count.
fn local(f: &Function) -> (EffectSet, Vec<&str>) {
    let mut effects = EffectSet::PURE;
    let mut callees = Vec::new();
    let mut seen_region = vec![false; f.regions.len()];
    let mut seen_op = vec![false; f.ops.len()];
    let mut todo = vec![f.body];
    while let Some(r) = todo.pop() {
        let Some(region) = f.regions.get(r.index()) else {
            continue;
        };
        if std::mem::replace(&mut seen_region[r.index()], true) {
            continue;
        }
        for b in &region.blocks {
            let Some(block) = f.blocks.get(b.index()) else {
                continue;
            };
            for o in &block.ops {
                let Some(op) = f.ops.get(o.index()) else {
                    continue;
                };
                if std::mem::replace(&mut seen_op[o.index()], true) {
                    continue;
                }
                effects = effects.union(op.kind.properties().effects);
                if let OpKind::Call { callee } = &op.kind {
                    callees.push(callee.as_str());
                }
                todo.extend(op.regions.iter().copied());
            }
        }
    }
    (effects, callees)
}

/// The inferred effects of every function of `m`, in the order of
/// `m.functions`. See the module documentation for the rules.
pub fn infer(m: &Module) -> Vec<EffectSet> {
    let n = m.functions.len();
    let mut by_name: HashMap<&str, usize> = HashMap::new();
    for (i, f) in m.functions.iter().enumerate() {
        by_name.entry(f.name.as_str()).or_insert(i);
    }
    let mut sets = Vec::with_capacity(n);
    // callers[j]: the functions that call function j.
    let mut callers: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut callees_of: Vec<Vec<usize>> = Vec::with_capacity(n);
    for (i, f) in m.functions.iter().enumerate() {
        let (effects, names) = local(f);
        sets.push(effects);
        let mut targets: Vec<usize> = names
            .iter()
            .filter_map(|name| by_name.get(name).copied())
            .collect();
        targets.sort_unstable();
        targets.dedup();
        for &t in &targets {
            callers[t].push(i);
        }
        callees_of.push(targets);
    }
    // Propagate callee effects to callers until nothing grows.
    let mut work: Vec<usize> = (0..n).collect();
    let mut queued = vec![true; n];
    while let Some(i) = work.pop() {
        queued[i] = false;
        let merged = callees_of[i]
            .iter()
            .fold(sets[i], |acc, &c| acc.union(sets[c]));
        if merged != sets[i] {
            sets[i] = merged;
            for &caller in &callers[i] {
                if !std::mem::replace(&mut queued[caller], true) {
                    work.push(caller);
                }
            }
        }
    }
    sets
}

/// Sets every function's `effects` to the inferred set. Returns whether any
/// changed.
pub fn infer_effects(m: &mut Module) -> Changed {
    let inferred = infer(m);
    let mut changed = Changed::No;
    for (f, e) in m.functions.iter_mut().zip(inferred) {
        if f.effects != e {
            f.effects = e;
            changed = Changed::Yes;
        }
    }
    changed
}

/// [`infer_effects`] as a pass. It explains nothing: effects are
/// bookkeeping, not a change of the program.
#[derive(Clone, Copy, Default, Debug)]
pub struct InferEffects;

impl Pass for InferEffects {
    fn name(&self) -> &'static str {
        "effects"
    }

    fn run(&self, m: &mut Module, _cx: &mut PassCx) -> Changed {
        infer_effects(m)
    }
}
