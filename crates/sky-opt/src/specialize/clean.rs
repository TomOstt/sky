//! Simplifying a specialized body, once its parameters are known.
//!
//! Every step keeps the function's behaviour exactly, and each says why:
//!
//! * **`len` of a static shape** becomes the constant. Lowering already does
//!   this where the type has the length; a specialization is what gives a
//!   `Vector[N]` parameter one. (The operand is still computed; only its
//!   length is read from the type.)
//! * **Decided comparisons.** When a parameter is known to be positive or
//!   negative, a comparison whose two sides' enclosures (`sky_ir::exact`,
//!   which contain every `f64` the interpreter can produce) do not overlap
//!   has the same result for every run, so it becomes that Bool. A
//!   comparison never fails for typed scalars of one dimension, so no error
//!   is lost.
//! * **`v[k]` of a literal.** `[a, b, c][1]` is `b`: the entry is already a
//!   value, and the index is proven in bounds.
//! * **Constant folding** ([`crate::fold`]: exact, bit-identical, never folds
//!   an operation that would fail) and **`dce`** ([`crate::dce`]: never
//!   removes an operation that can fail, unless the exactness analysis
//!   proves it cannot).
//! * **Unrolling** loops with a small constant trip count ([`super::unroll`]).
//!
//! The steps feed each other (a folded bound decides a loop, an unrolled
//! loop makes indices constant), so they repeat, at most [`MAX_ROUNDS`]
//! times.

use std::collections::{HashMap, HashSet};

use sky_ir::exact::{self, Proof, Range};
use sky_ir::{Function, OpKind, Type, ValueId};
use sky_types::ShapeDim;
use sky_units::Rational;

use super::unroll::{unroll_function, Unrolled};
use crate::contracts::{compare, Truth};
use crate::dce::dce_function;
use crate::fold::fold_function;
use crate::rewrite::Rewrite;
use crate::util::{detach, in_fused_body, reachable_ops, replace_uses};
use crate::FoldMode;

/// The most times the steps repeat.
pub(crate) const MAX_ROUNDS: usize = 6;

/// What the simplification did.
#[derive(Default, Debug)]
pub(crate) struct Cleanup {
    /// `len(x)` replaced by a constant.
    pub(crate) lengths: usize,
    /// Comparisons decided by a sign fact.
    pub(crate) comparisons: usize,
    /// `[..][k]` replaced by the entry.
    pub(crate) entries: usize,
    /// Operations the folder replaced by constants.
    pub(crate) folded: usize,
    /// `if`s the folder replaced by the arm that runs.
    pub(crate) branches: usize,
    /// Loops unrolled.
    pub(crate) loops: Vec<Unrolled>,
    /// What dead-code removal explained: user-written code that went away
    /// (`spare = x / 2`, never used). Kept, in `dce`'s rules and words, so
    /// that specializing never makes a removal of the user's code silent;
    /// `explain` files them under the pass that ran them, `specialize`.
    pub(crate) removed: Vec<Rewrite>,
}

/// Simplifies `f` in place. `ranges` are what the specialization knows
/// about parameters (by value); `unroll_budget` is how many operations
/// unrolling may still add.
pub(crate) fn cleanup(
    f: &mut Function,
    ranges: &HashMap<ValueId, Range>,
    unroll_budget: &mut usize,
) -> Cleanup {
    let mut done = Cleanup::default();
    for _ in 0..MAX_ROUNDS {
        let mut changed = false;
        changed |= fold_lengths(f, &mut done);
        changed |= decide_comparisons(f, ranges, &mut done);
        changed |= forward_entries(f, &mut done);
        let mut rewrites = Vec::new();
        fold_function(f, FoldMode::BitIdentical, &mut rewrites);
        done.branches += rewrites.iter().filter(|r| r.rule == "FOLD-IF-001").count();
        done.folded += rewrites.iter().filter(|r| r.rule != "FOLD-IF-001").count();
        changed |= !rewrites.is_empty();
        let unrolled = unroll_function(f, unroll_budget);
        changed |= !unrolled.is_empty();
        done.loops.extend(unrolled);
        let before = reachable_ops(f).len();
        dce_function(f, &mut done.removed);
        changed |= reachable_ops(f).len() != before;
        if !changed {
            break;
        }
    }
    done
}

/// The run-time checks `f` still has, given what is known about the
/// parameters: the operations that can fail and are not proven not to,
/// counted once per source operation (a loop body unrolled three times has
/// the checks of one copy: they are the same checks, at the same spans).
pub(crate) fn checks(f: &Function, ranges: &HashMap<ValueId, Range>) -> usize {
    let an = analysis(f, ranges);
    reachable_ops(f)
        .into_iter()
        .filter(|o| {
            let op = f.op(*o);
            op.kind.properties().effects.contains(sky_ir::Effect::Fail)
                && an.proof(*o).is_none()
                && !in_fused_body(f, *o)
        })
        .map(|o| {
            let op = f.op(o);
            (op.kind.mnemonic(), op.span)
        })
        .collect::<HashSet<_>>()
        .len()
}

/// The exactness analysis of `f` with the parameter ranges of `ranges`.
fn analysis(f: &Function, ranges: &HashMap<ValueId, Range>) -> exact::Analysis {
    let params: Vec<Option<Range>> = f
        .param_values()
        .iter()
        .map(|v| ranges.get(v).copied())
        .collect();
    exact::analyze_with_params(f, &params)
}

fn fold_lengths(f: &mut Function, done: &mut Cleanup) -> bool {
    let mut changed = false;
    for o in reachable_ops(f) {
        let op = f.op(o);
        let (OpKind::Len, [base], [res]) = (&op.kind, &op.operands[..], &op.results[..]) else {
            continue;
        };
        let (Type::Number(array), Type::Number(out)) = (f.ty(*base), f.ty(*res)) else {
            continue;
        };
        let Some(ShapeDim::Known(n)) = array.shape.dims().first() else {
            continue;
        };
        if out.shape.rank() != 0 || !out.dim.is_dimensionless() {
            continue;
        }
        let Ok(n) = i64::try_from(*n) else { continue };
        let data = &mut f.ops[o.index()];
        data.kind = OpKind::ConstRational(Rational::integer(n));
        data.operands.clear();
        data.operand_spans.clear();
        done.lengths += 1;
        changed = true;
    }
    changed
}

fn decide_comparisons(
    f: &mut Function,
    ranges: &HashMap<ValueId, Range>,
    done: &mut Cleanup,
) -> bool {
    if ranges.is_empty() {
        return false;
    }
    let (seeded, plain) = (analysis(f, ranges), exact::analyze(f));
    let mut changed = false;
    for o in reachable_ops(f) {
        let op = f.op(o);
        let (OpKind::Cmp(c), [a, b], [res]) = (&op.kind, &op.operands[..], &op.results[..]) else {
            continue;
        };
        if !matches!(f.ty(*res), Type::Bool) || in_fused_body(f, o) {
            continue;
        }
        let range = |an: &exact::Analysis, v: &ValueId| an.known(*v).map(|k| k.range);
        let (Some(x), Some(y)) = (range(&seeded, a), range(&seeded, b)) else {
            continue;
        };
        let already = match (range(&plain, a), range(&plain, b)) {
            (Some(p), Some(q)) => compare(*c, p, q) != Truth::Unknown,
            _ => false,
        };
        let truth = compare(*c, x, y);
        if already || truth == Truth::Unknown {
            continue;
        }
        let data = &mut f.ops[o.index()];
        data.kind = OpKind::ConstBool(truth == Truth::True);
        data.operands.clear();
        data.operand_spans.clear();
        done.comparisons += 1;
        changed = true;
    }
    changed
}

fn forward_entries(f: &mut Function, done: &mut Cleanup) -> bool {
    let an = exact::analyze(f);
    let mut changed = false;
    for o in reachable_ops(f) {
        let op = f.op(o);
        let (OpKind::Index, [base, at], [res]) = (&op.kind, &op.operands[..], &op.results[..])
        else {
            continue;
        };
        if an.proof(o) != Some(Proof::InBounds) || in_fused_body(f, o) {
            continue;
        }
        let Some(def) = crate::util::defining_op(f, *base) else {
            continue;
        };
        let lit = f.op(def);
        if !matches!(lit.kind, OpKind::ArrayLiteral { cols: None }) {
            continue;
        }
        let Some(i) = an.known(*at).and_then(|k| k.range.as_point()) else {
            continue;
        };
        let Some(&entry) = lit.operands.get(i as usize) else {
            continue;
        };
        if f.ty(entry) != f.ty(*res) {
            continue;
        }
        let res = *res;
        replace_uses(f, res, entry);
        if f.values[entry.index()].name.is_none() {
            f.values[entry.index()].name = f.values[res.index()].name.clone();
        }
        detach(f, o);
        done.entries += 1;
        changed = true;
    }
    changed
}
