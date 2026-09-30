//! Unrolling `while` loops whose trip count is a small constant.
//!
//! # When a loop is unrolled
//!
//! A loop is unrolled when running it *symbolically* shows that its
//! condition is a known constant on every check: true on the first
//! `MAX_TRIPS` (8) checks at most, then false. "Known" means [`fold`]
//! (constant folding, exact, bit-identical to the interpreter) decides it:
//! the loop's initial values and the arithmetic on them are constants, and
//! the bound it tests against is a constant too (`len(a)` on a `Vector[3]`
//! is one in a specialization). Nothing is guessed: the trial copy of the
//! function is the loop, run by inlining, and the loop is replaced by the
//! trial only if the trial ended.
//!
//! # Why it cannot change behaviour
//!
//! The unrolled code is the loop's own operations, in the order the loop
//! runs them: the `before` region once more than the `after` region (the
//! last check ends the loop), each copy reading the values the previous copy
//! yielded. Every operation keeps its kind, span and operand spans, so a
//! failure inside an iteration is the same failure, with the same
//! diagnostic, after the same output. Nothing is reordered, dropped or
//! merged (the only operations folding removes are those that computed a
//! constant that provably cannot fail). The loop's results are the values
//! the final check forwarded.
//!
//! # Limits
//!
//! At most [`MAX_TRIPS`] iterations per loop and [`MAX_UNROLL_OPS`]
//! operations added per function, so the code grows by a bounded factor.
//! A loop nested in another is tried first (the outer loop's copies then
//! each hold the already unrolled inner loop), and an inner loop whose trip
//! count depends on the outer iteration is tried again in every copy of the
//! outer body, where it may have become constant.

use std::collections::HashSet;

use sky_diag::Span;
use sky_ir::{Function, OpId, OpKind, ValueId};

use super::copy::Copier;
use crate::fold::fold_function;
use crate::util::{defining_op, detach, in_fused_body, reachable_ops, replace_uses};
use crate::FoldMode;

/// The most iterations of one loop that are unrolled.
pub const MAX_TRIPS: usize = 8;

/// The most operations unrolling adds to one function (copies of loop
/// bodies, nested operations included).
pub const MAX_UNROLL_OPS: usize = 256;

/// A loop that was unrolled.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Unrolled {
    /// The loop's source span.
    pub(crate) span: Span,
    /// How many times its body ran.
    pub(crate) trips: usize,
}

/// Unrolls the loops of `f` that qualify, spending at most `budget`
/// operations. Returns what was unrolled.
pub(crate) fn unroll_function(f: &mut Function, budget: &mut usize) -> Vec<Unrolled> {
    let mut done = Vec::new();
    let mut tried: HashSet<OpId> = HashSet::new();
    loop {
        // Innermost first: reachable order lists a loop before its body.
        let next = reachable_ops(f).into_iter().rev().find(|o| {
            matches!(f.op(*o).kind, OpKind::While) && !tried.contains(o) && !in_fused_body(f, *o)
        });
        let Some(w) = next else { return done };
        tried.insert(w);
        if let Some((trial, trips, added)) = try_unroll(f, w, *budget) {
            done.push(Unrolled {
                span: f.op(w).span,
                trips,
            });
            *f = trial;
            *budget -= added;
        }
    }
}

/// `f` with loop `w` unrolled, the number of iterations, and how many
/// operations it added; `None` if the trip count is not a small constant.
fn try_unroll(f: &Function, w: OpId, budget: usize) -> Option<(Function, usize, usize)> {
    let mut t = f.clone();
    let wd = t.op(w).clone();
    let [before_r, after_r] = wd.regions[..] else {
        return None;
    };
    let (&[before_b], &[after_b]) = (
        &t.region(before_r).blocks[..],
        &t.region(after_r).blocks[..],
    ) else {
        return None;
    };
    let (before_args, after_args) = (
        t.block(before_b).args.clone(),
        t.block(after_b).args.clone(),
    );
    let before_end = *t.block(before_b).ops.last()?;
    let after_end = *t.block(after_b).ops.last()?;
    // The shapes the interpreter relies on.
    if !matches!(t.op(before_end).kind, OpKind::Condition)
        || !matches!(t.op(after_end).kind, OpKind::Yield)
        || before_args.len() != wd.operands.len()
        || t.op(before_end).operands.len() != 1 + wd.results.len()
        || after_args.len() != wd.results.len()
        || t.op(after_end).operands.len() != wd.operands.len()
    {
        return None;
    }
    let mut cur: Vec<ValueId> = wd.operands.clone();
    let mut added = 0;
    for trip in 0..=MAX_TRIPS {
        // The condition check, as a copy of the `before` region.
        let (marker, copied) = inline(&mut t, w, before_b, &before_args, &cur)?;
        added += copied;
        if added > budget {
            return None;
        }
        // Decide it. The folder also folds what the check computes, and
        // updates the operands of the marker, which stands for `condition`.
        fold_function(&mut t, FoldMode::BitIdentical, &mut Vec::new());
        let &flag = t.op(marker).operands.first()?;
        let forwarded: Vec<ValueId> = t.op(marker).operands[1..].to_vec();
        detach(&mut t, marker);
        match const_bool(&t, flag) {
            Some(false) => {
                for (r, y) in wd.results.iter().zip(&forwarded) {
                    replace_uses(&mut t, *r, *y);
                    if t.values[y.index()].name.is_none() {
                        t.values[y.index()].name = t.values[r.index()].name.clone();
                    }
                }
                detach(&mut t, w);
                return Some((t, trip, added));
            }
            Some(true) if trip < MAX_TRIPS => {}
            _ => return None,
        }
        // The body, as a copy of the `after` region.
        let (marker, copied) = inline(&mut t, w, after_b, &after_args, &forwarded)?;
        added += copied;
        if added > budget {
            return None;
        }
        cur = t.op(marker).operands.clone();
        detach(&mut t, marker);
    }
    None
}

/// Copies the operations of `block` (a region of the loop `w`) in front of
/// `w`, with `args` bound to `vals`; the last one, the terminator, stands for
/// the region's exit. Returns the copy of the terminator and how many
/// operations were copied.
///
/// The operations are read *now*, not once at the start: folding the trial
/// (to decide the condition) also folds inside the loop's regions, where it
/// may inline an `if` whose condition is constant, so the block's list of
/// operations changes.
fn inline(
    t: &mut Function,
    w: OpId,
    block: sky_ir::BlockId,
    args: &[ValueId],
    vals: &[ValueId],
) -> Option<(OpId, usize)> {
    let ops = t.block(block).ops.clone();
    let (&end, body) = ops.split_last()?;
    if !t.op(end).kind.properties().terminator {
        return None;
    }
    // Folding may have moved the loop (an `if` around it was inlined).
    let parent = t.op(w).parent;
    let start = t.block(parent).ops.len();
    let mut c = Copier::new(t);
    for (a, v) in args.iter().zip(vals) {
        c.bind(*a, *v);
    }
    for &o in body {
        c.copy_op(o, parent);
    }
    let marker = c.copy_op(end, parent);
    let copied = c.copied;
    // The copies were appended after the block's terminator: move them to
    // just before the loop.
    let ops = &mut t.blocks[parent.index()].ops;
    let new: Vec<OpId> = ops.drain(start..).collect();
    let at = ops
        .iter()
        .position(|o| *o == w)
        .expect("a loop is in its parent block");
    ops.splice(at..at, new);
    Some((marker, copied))
}

fn const_bool(f: &Function, v: ValueId) -> Option<bool> {
    match f.op(defining_op(f, v)?).kind {
        OpKind::ConstBool(b) => Some(b),
        _ => None,
    }
}
