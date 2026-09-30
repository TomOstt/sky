//! Which operations dominate the error of a value?
//!
//! When a budget cannot be proven, the useful answer is *where the
//! accuracy goes*. The report is advisory (no claim rests on it), and is
//! built from the analysis's own table by a backward pass over the values
//! the goal depends on:
//!
//! * **sensitivity**: how much a unit of absolute error in each value
//!   becomes in the goal (the product of the operations' Lipschitz
//!   constants, [`super::transfer::OpNote::lip`], along the paths);
//! * **origins** of unboundedness: an operation whose result is unbounded
//!   though its operands are bounded (a loop of unknown trip count, a
//!   branch the rounding could flip, a function the analysis cannot bound);
//! * **cancellation**: an addition or subtraction whose terms may cancel
//!   and whose result is much less accurate, relatively, than its operands;
//! * **conditioning**: any other operation that magnifies the relative
//!   error of its operands (`log(1 + x)` near 1, a division by a number
//!   near 0);
//! * **rounding**: otherwise, the operations whose own rounding, times
//!   their sensitivity, is the largest share of the error.

use std::collections::HashMap;

use sky_diag::Span;

use super::num::{mag_max, Num};
use super::{Analysis, Fact};
use crate::ir::{Function, Module, OpId, RegionId, ValueId};
use crate::op::OpKind;

/// One operation that dominates the error of a goal.
#[derive(Clone, PartialEq, Debug)]
pub struct Blame {
    /// The function the operation is in.
    pub function: String,
    pub op: OpId,
    /// Where the operation came from in the source.
    pub span: Span,
    /// The operation's name (`sub`, `log`, `while`, ...).
    pub mnemonic: &'static str,
    pub kind: BlameKind,
}

#[derive(Clone, PartialEq, Debug)]
pub enum BlameKind {
    /// Terms of opposite (or unknown) sign that may nearly cancel: the
    /// rounding errors already in them become the leading digits of the
    /// result. The sizes are the largest magnitudes of the operands and of
    /// the result, when known.
    Cancellation {
        operands: Option<f64>,
        result: Option<f64>,
    },
    /// The operation magnifies relative error: `rel_in` in its operands
    /// becomes `rel_out` in the result.
    Conditioning { rel_in: f64, rel_out: f64 },
    /// The operation's own rounding is this share of the total (0 to 1).
    Rounding { share: f64 },
    /// The error of the operation's result cannot be bounded although its
    /// operands' can.
    Unbounded,
}

impl Blame {
    /// The explanation, to follow the operation's source text: "`x - y`
    /// ...".
    pub fn describe(&self) -> String {
        match &self.kind {
            BlameKind::Cancellation { operands, result } => {
                let terms = if self.mnemonic == "sub" {
                    "subtracts"
                } else {
                    "adds"
                };
                match (operands, result) {
                    (Some(a), Some(r)) => format!(
                        "{terms} numbers of size up to {} and gets a result of size at most {}: the rounding errors already in the terms become the leading digits of the result (catastrophic cancellation)",
                        size(*a),
                        size(*r)
                    ),
                    _ => format!(
                        "{terms} numbers that may nearly cancel: the rounding errors already in the terms become the leading digits of the result (catastrophic cancellation)"
                    ),
                }
            }
            BlameKind::Conditioning { rel_in, rel_out } => format!(
                "magnifies relative error: {} in its operands becomes {} in the result",
                bound(*rel_in),
                bound(*rel_out)
            ),
            BlameKind::Rounding { share } => format!(
                "its own rounding is about {:.0}% of the error",
                share * 100.0
            ),
            BlameKind::Unbounded => match self.mnemonic {
                "while" => "is a loop whose number of iterations the analysis does not know"
                    .to_string(),
                "if" => "is a branch the rounding error can flip: the exact computation may take the other arm"
                    .to_string(),
                "call" => "is a call the analysis cannot follow".to_string(),
                _ => format!("has no bound on its error here (`{}` is not modelled for these operands)", self.mnemonic),
            },
        }
    }
}

fn size(x: f64) -> String {
    crate::exact::range::num_text(x)
}

fn bound(x: f64) -> String {
    if x.is_finite() {
        format!("{x:.1e}")
    } else {
        "no bound".to_string()
    }
}

/// The size of a value's range, when it says something (not "any finite
/// number").
fn size_of(n: &Num) -> Option<f64> {
    let m = mag_max(n.range);
    (m < 1e300).then_some(m)
}

fn post_order(f: &Function, region: RegionId, out: &mut Vec<OpId>) {
    for &b in &f.region(region).blocks {
        for &o in &f.block(b).ops {
            for &r in &f.op(o).regions {
                post_order(f, r, out);
            }
            out.push(o);
        }
    }
}

/// The terms of `x ± y` keep one sign, so the sum does not cancel.
pub(crate) fn same_sign_terms(x: &Num, y: &Num, sub: bool) -> bool {
    let sx = x.sign();
    let sy = if sub { y.sign().flip() } else { y.sign() };
    (sx.non_negative() && sy.non_negative()) || (sx.non_positive() && sy.non_positive())
}

fn bounded(n: &Num) -> bool {
    n.abs.is_finite() || n.rel.is_finite()
}

/// The operations that dominate the error of `goal`, most important first
/// (at most three).
pub(crate) fn blame(a: &Analysis, m: &Module, function: &str, goal: ValueId) -> Vec<Blame> {
    let Some(f) = m.function(function) else {
        return Vec::new();
    };
    let mut order = Vec::new();
    post_order(f, f.body, &mut order);
    let num = |v: ValueId| match a.fact(function, v) {
        Fact::Known(n) => Some(*n),
        _ => None,
    };
    let mut sens: HashMap<ValueId, f64> = HashMap::new();
    sens.insert(goal, 1.0);
    let add = |sens: &mut HashMap<ValueId, f64>, v: ValueId, s: f64| {
        let e = sens.entry(v).or_insert(0.0);
        *e = super::add_up(*e, s);
    };

    let mut origins = Vec::new();
    let mut cancellations: Vec<(f64, Blame)> = Vec::new();
    let mut conditioning: Vec<(f64, Blame)> = Vec::new();
    let mut rounding: Vec<(f64, OpId)> = Vec::new();
    let mut total = 0.0;

    let blame_of = |op: OpId, kind: BlameKind| {
        let data = f.op(op);
        Blame {
            function: function.to_string(),
            op,
            span: data.span,
            mnemonic: data.kind.mnemonic(),
            kind,
        }
    };

    for &id in order.iter().rev() {
        let op = f.op(id);
        let s = op
            .results
            .iter()
            .filter_map(|r| sens.get(r))
            .fold(0.0, |m: f64, &x| m.max(x));
        if s == 0.0 {
            continue;
        }
        match &op.kind {
            OpKind::If => {
                // The results are what the arms yield.
                let mut yields_bounded = true;
                for &r in &op.regions {
                    let Some(&b) = f.region(r).blocks.first() else {
                        continue;
                    };
                    let Some(&last) = f.block(b).ops.last() else {
                        continue;
                    };
                    let y = f.op(last);
                    for (i, &res) in op.results.iter().enumerate() {
                        if let (Some(&v), Some(&sv)) = (y.operands.get(i), sens.get(&res)) {
                            add(&mut sens, v, sv);
                            yields_bounded &= num(v).is_none_or(|n| bounded(&n));
                        }
                    }
                }
                let unbounded_result = op
                    .results
                    .iter()
                    .any(|&r| num(r).is_some_and(|n| !bounded(&n)));
                if unbounded_result && yields_bounded {
                    origins.push(blame_of(id, BlameKind::Unbounded));
                }
            }
            OpKind::While | OpKind::Call { .. } => {
                let unbounded_result = op
                    .results
                    .iter()
                    .any(|&r| num(r).is_some_and(|n| !bounded(&n)));
                let operands_bounded = op
                    .operands
                    .iter()
                    .all(|&v| num(v).is_none_or(|n| bounded(&n)));
                if unbounded_result && operands_bounded {
                    origins.push(blame_of(id, BlameKind::Unbounded));
                }
                for &v in &op.operands {
                    add(&mut sens, v, s);
                }
            }
            OpKind::Yield
            | OpKind::Condition
            | OpKind::Return
            | OpKind::Print
            | OpKind::ConstFloat { .. }
            | OpKind::ConstRational(_) => {}
            kind => {
                let Some(result) = op.results.first().and_then(|&r| num(r)) else {
                    continue;
                };
                let operands: Vec<Option<Num>> = op.operands.iter().map(|&v| num(v)).collect();
                let operands_bounded = operands.iter().flatten().all(bounded);
                let Some(note) = a.note(function, id) else {
                    if !bounded(&result) && operands_bounded {
                        origins.push(blame_of(id, BlameKind::Unbounded));
                    }
                    for &v in &op.operands {
                        add(&mut sens, v, s);
                    }
                    continue;
                };
                if !bounded(&result) && operands_bounded {
                    origins.push(blame_of(id, BlameKind::Unbounded));
                }
                let contribution = super::mul_up(s, note.round_abs);
                if contribution > 0.0 && contribution.is_finite() {
                    rounding.push((contribution, id));
                    total = super::add_up(total, contribution);
                }
                for (k, &v) in op.operands.iter().enumerate().take(2) {
                    add(&mut sens, v, super::mul_up(s, note.lip[k]));
                }
                let amplified = result.rel > 16.0 * note.rel_in.max(1e-15);
                if amplified && note.rel_in.is_finite() {
                    match (kind, operands.as_slice()) {
                        (OpKind::Add | OpKind::Sub, [Some(x), Some(y)])
                            if !same_sign_terms(x, y, matches!(kind, OpKind::Sub)) =>
                        {
                            let operand_size = size_of(x).zip(size_of(y)).map(|(p, q)| p.max(q));
                            cancellations.push((
                                result.rel,
                                blame_of(
                                    id,
                                    BlameKind::Cancellation {
                                        operands: operand_size,
                                        result: size_of(&result),
                                    },
                                ),
                            ));
                        }
                        (OpKind::Add | OpKind::Sub, _) => {}
                        _ => conditioning.push((
                            result.rel,
                            blame_of(
                                id,
                                BlameKind::Conditioning {
                                    rel_in: note.rel_in,
                                    rel_out: result.rel,
                                },
                            ),
                        )),
                    }
                }
            }
        }
    }

    let by_worst = |v: &mut Vec<(f64, Blame)>| {
        v.sort_by(|p, q| q.0.partial_cmp(&p.0).unwrap_or(std::cmp::Ordering::Equal));
    };
    by_worst(&mut cancellations);
    by_worst(&mut conditioning);
    // The origin closest to the inputs first: the last one found walking
    // backwards from the goal.
    origins.reverse();
    let mut out: Vec<Blame> = origins.into_iter().take(2).collect();
    out.extend(cancellations.into_iter().map(|(_, b)| b).take(2));
    if out.is_empty() {
        out.extend(conditioning.into_iter().map(|(_, b)| b).take(2));
    }
    if out.is_empty() {
        rounding.sort_by(|p, q| q.0.partial_cmp(&p.0).unwrap_or(std::cmp::Ordering::Equal));
        for (c, id) in rounding.into_iter().take(3) {
            let share = if total > 0.0 { c / total } else { 0.0 };
            if share >= 0.05 {
                out.push(blame_of(id, BlameKind::Rounding { share }));
            }
        }
    }
    out.truncate(3);
    out
}
