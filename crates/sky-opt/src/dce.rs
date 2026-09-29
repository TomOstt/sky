//! Dead-code elimination.
//!
//! An operation is removed when all of these hold:
//!
//! * its kind is pure by [`sky_ir::OpKind::properties`] (no `Io`, `Fail`
//!   or `Call` effect) and is not a terminator;
//! * none of its results is used;
//! * for an `if`, every operation in both regions is removable by the same
//!   rules (so nothing inside can fail or print), and the condition is
//!   typed `Bool`;
//! * for the operations that check their operands dynamically even though
//!   they are "pure" (`neg`, `not`, `cmp.*`), the operand types are known
//!   well enough that they cannot fail. An `Unknown`-typed operand, or a
//!   comparison of two dimensions or two strings with `<`, can fail in the
//!   interpreter, so those stay.
//!
//! Operations that can fail (`add`, `div`, ...) are never removed, even
//! unused: the run-time error they cause is behaviour. `while` is never
//! removed (it may not terminate). The pass repeats to a fixed point, so a
//! removed operation frees its operands.
//!
//! # Explanations
//!
//! Lowering creates many temporaries; dropping those is not news. A
//! [`Rewrite`] is recorded only when user-written code disappears, which is
//! the case when a removed operation (or, for an `if`, anything inside it)
//! defines a value that has a source `name`.

use sky_ir::{BlockId, CmpOp, Function, Module, OpId, OpKind, RegionId, Type};

use crate::rewrite::Rewrite;
use crate::util::{op_text, reachable_ops};

/// Removes dead pure operations from every function.
pub fn dce(m: &mut Module) -> Vec<Rewrite> {
    let mut out = Vec::new();
    for f in &mut m.functions {
        loop {
            let used = used_values(f);
            let body = f.body;
            if !sweep_region(f, body, &used, &mut out) {
                break;
            }
        }
    }
    out
}

fn used_values(f: &Function) -> Vec<bool> {
    let mut used = vec![false; f.values.len()];
    for o in reachable_ops(f) {
        for v in &f.op(o).operands {
            used[v.index()] = true;
        }
    }
    used
}

fn sweep_region(f: &mut Function, r: RegionId, used: &[bool], out: &mut Vec<Rewrite>) -> bool {
    let mut changed = false;
    for b in f.region(r).blocks.clone() {
        changed |= sweep_block(f, b, used, out);
    }
    changed
}

fn sweep_block(f: &mut Function, b: BlockId, used: &[bool], out: &mut Vec<Rewrite>) -> bool {
    let mut changed = false;
    for id in f.block(b).ops.clone().into_iter().rev() {
        if f.op(id).results.iter().all(|r| !used[r.index()]) && removable(f, id) {
            if let Some(rw) = explain(f, id) {
                out.push(rw);
            }
            f.blocks[b.index()].ops.retain(|o| *o != id);
            changed = true;
        } else {
            for r in f.op(id).regions.clone() {
                changed |= sweep_region(f, r, used, out);
            }
        }
    }
    changed
}

/// Whether removing `id` (if its results are unused) is safe.
fn removable(f: &Function, id: OpId) -> bool {
    let op = f.op(id);
    let props = op.kind.properties();
    if props.terminator || !props.effects.is_pure() {
        return false;
    }
    let known = |v: &sky_ir::ValueId| !matches!(f.ty(*v), Type::Unknown);
    match &op.kind {
        OpKind::While => false,
        // A `fused` fails, or not, as its body does.
        OpKind::Fused => {
            op.operands.iter().all(known) && op.regions.iter().all(|r| region_is_removable(f, *r))
        }
        OpKind::If => {
            matches!(op.operands.first().map(|v| f.ty(*v)), Some(Type::Bool))
                && op.regions.iter().all(|r| region_is_removable(f, *r))
        }
        OpKind::Neg => op.operands.iter().all(|v| f.ty(*v).as_number().is_some()),
        OpKind::Not => op.operands.iter().all(|v| matches!(f.ty(*v), Type::Bool)),
        OpKind::Cmp(c) => {
            // `==` and `!=` accept numbers of one dimension, Bools and
            // strings; ordering needs numbers of one dimension.
            let tys: Vec<&Type> = op.operands.iter().map(|v| f.ty(*v)).collect();
            let [a, b] = tys[..] else { return false };
            match (a, b) {
                (Type::Number(x), Type::Number(y)) => x.dim == y.dim && x.shape.rank() == 0,
                (Type::Bool, Type::Bool) | (Type::Str, Type::Str) => {
                    matches!(c, CmpOp::Eq | CmpOp::Ne)
                }
                _ => false,
            }
        }
        _ => op.operands.iter().all(known),
    }
}

/// Every operation of a region could be removed if its results were
/// unused: nothing in it prints, fails, calls or loops.
fn region_is_removable(f: &Function, r: RegionId) -> bool {
    f.region(r).blocks.iter().all(|b| {
        f.block(*b)
            .ops
            .iter()
            .all(|o| f.op(*o).kind.properties().terminator || removable(f, *o))
    })
}

/// A note when user-written code (a named value) is what goes away.
fn explain(f: &Function, id: OpId) -> Option<Rewrite> {
    let op = f.op(id);
    let mut named: Vec<String> = Vec::new();
    collect_named(f, id, &mut named);
    if named.is_empty() {
        return None;
    }
    let what = if matches!(op.kind, OpKind::If) {
        format!("the `if` that computes {}", named.join(", "))
    } else {
        named.join(", ")
    };
    Some(Rewrite {
        pass: "dce",
        rule: "DCE-001",
        span: op.span,
        before: op_text(f, id),
        after: String::new(),
        because: vec![
            format!("{what} is never used"),
            format!(
                "`{}` has no side effects (it cannot print, fail, call or loop), so removing \
                 it cannot change what the program does",
                op.kind.mnemonic()
            ),
        ],
    })
}

fn collect_named(f: &Function, id: OpId, out: &mut Vec<String>) {
    let op = f.op(id);
    for r in &op.results {
        if let Some(n) = &f.value(*r).name {
            out.push(format!("`{n}`"));
        }
    }
    for region in &op.regions {
        for b in &f.region(*region).blocks {
            for o in &f.block(*b).ops {
                collect_named(f, *o, out);
            }
        }
    }
}
