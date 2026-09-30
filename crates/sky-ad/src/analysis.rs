//! Activity analysis: which values depend on the parameter we differentiate
//! against. Only those get tangents; the rest are copied unchanged, so a
//! loop counter or a flag costs nothing.
//!
//! It is a monotone fixpoint over the whole function (loops feed values
//! back), and it over-approximates: a value flagged active whose tangent
//! turns out to be structurally zero (say `x ^ 0`) is given an explicit
//! zero, so the flags only decide *structure* (which `if` results and
//! `while` slots carry a tangent), never correctness.

use sky_ir::{BlockId, Function, OpKind, ValueId};

use crate::can_carry_tangent;

/// `active[v]` for every value of `f`, with parameter `wrt` as the seed.
pub(crate) fn analyze(f: &Function, wrt: usize) -> Vec<bool> {
    let mut active = vec![false; f.values.len()];
    if let Some(p) = f.param_values().get(wrt) {
        active[p.index()] = true;
    }
    loop {
        let mut changed = false;
        block(f, f.entry(), &mut active, &mut changed);
        if !changed {
            return active;
        }
    }
}

fn set(f: &Function, active: &mut [bool], v: ValueId, changed: &mut bool) {
    if can_carry_tangent(f.ty(v)) && !active[v.index()] {
        active[v.index()] = true;
        *changed = true;
    }
}

/// The operands of the terminator that ends `b`.
pub(crate) fn terminator_operands(f: &Function, b: BlockId) -> &[ValueId] {
    f.block(b)
        .ops
        .last()
        .map(|o| f.op(*o).operands.as_slice())
        .unwrap_or(&[])
}

fn block(f: &Function, b: BlockId, active: &mut [bool], changed: &mut bool) {
    for &id in &f.block(b).ops {
        let op = f.op(id);
        match &op.kind {
            OpKind::If => {
                let mut yields = Vec::new();
                for r in &op.regions {
                    for &blk in &f.region(*r).blocks {
                        block(f, blk, active, changed);
                        yields.push(terminator_operands(f, blk));
                    }
                }
                for (i, r) in op.results.iter().enumerate() {
                    if yields
                        .iter()
                        .any(|y| y.get(i).is_some_and(|v| active[v.index()]))
                    {
                        set(f, active, *r, changed);
                    }
                }
            }
            OpKind::While => {
                let [before, after] = op.regions[..] else {
                    continue;
                };
                let (Some(&before), Some(&after)) = (
                    f.region(before).blocks.first(),
                    f.region(after).blocks.first(),
                ) else {
                    continue;
                };
                block(f, before, active, changed);
                block(f, after, active, changed);
                // One flag per loop-carried slot: its initial value, the
                // block arguments, the forwarded value, the yielded value
                // and the result all carry a tangent or none does.
                let fwd = terminator_operands(f, before);
                let yielded = terminator_operands(f, after);
                for (j, init) in op.operands.iter().enumerate() {
                    let owned = [
                        f.block(before).args.get(j),
                        f.block(after).args.get(j),
                        op.results.get(j),
                    ];
                    let mut slot = vec![init];
                    slot.extend(owned.iter().flatten());
                    slot.extend(fwd.get(j + 1));
                    slot.extend(yielded.get(j));
                    if slot.iter().any(|v| active[v.index()]) {
                        for v in owned.into_iter().flatten() {
                            set(f, active, *v, changed);
                        }
                    }
                }
            }
            OpKind::Fused => {
                // The body's arguments are its operands' entries, and the
                // result is active if the yielded value is.
                let Some(&body) = op.regions.first().and_then(|r| f.region(*r).blocks.first())
                else {
                    continue;
                };
                for (arg, operand) in f.block(body).args.iter().zip(&op.operands) {
                    if active[operand.index()] {
                        set(f, active, *arg, changed);
                    }
                }
                block(f, body, active, changed);
                let yielded = terminator_operands(f, body);
                if yielded.first().is_some_and(|v| active[v.index()]) {
                    for r in &op.results {
                        set(f, active, *r, changed);
                    }
                }
            }
            OpKind::Yield | OpKind::Condition | OpKind::Return => {}
            _ => {
                if op.operands.iter().any(|v| active[v.index()]) {
                    for r in &op.results {
                        set(f, active, *r, changed);
                    }
                }
            }
        }
    }
}
