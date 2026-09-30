//! Static discharge of contracts: an `assert[kind]` whose condition is
//! provably true is removed (spec "Contracts").
//!
//! # What is proved
//!
//! The condition is a Bool value; [`truth`] evaluates it in three values
//! (true, false, unknown) from what the exactness analysis knows
//! ([`sky_ir::exact`]): a comparison of two scalars is decided by their
//! **enclosures** (`a >= b` holds when the least value `a` can have is at
//! least the greatest `b` can have), `not` and `if` (which is how `and` and
//! `or` are lowered) follow, and a constant Bool is itself. Nothing else is
//! decided: a condition on an array, a string or a value of unknown type
//! keeps its check. The enclosures are sound (they contain every `f64` the
//! interpreter can compute), so a condition decided true is true.
//!
//! # Where the facts come from
//!
//! * Constants: `assert 1 < 2`, or `requires x >= 0 m` where `x` is `3 m`.
//! * Calls. A function's parameters are unknown to the analysis of its own
//!   body, but the module is closed: every call of a function is a `call`
//!   operation in the module (and `<main>`, the only entry, is called by
//!   nothing). So the enclosure of parameter `i` is the hull, over every
//!   call of the function, of the enclosure of the argument, as computed in
//!   the caller **with unknown parameters** (never with facts this pass
//!   derived, so there is no circularity and recursion needs no fixed
//!   point: `f(n - 1)` inside `f` is "any finite number", and a `requires
//!   n >= 0` of `f` then keeps its check). A function that is never called
//!   gets no facts. A parameter of unknown type, or an array, gets none.
//!
//!   That is what makes `fall_time(20 m)` in a program whose only call it
//!   is discharge `requires h >= 0 m`, and, through the sqrt, `ensures
//!   result > 0 s` as well. Add a second call with a negative height and
//!   the hull includes it: the check stays.
//!
//! # Soundness
//!
//! Removing the `assert` removes only the failure it would cause. The
//! operations that compute its condition stay (until `dce` removes what it
//! can prove infallible) and fail as before, so a condition that fails to
//! *compute* still fails. Which errors happen, and in which order, is
//! unchanged for every run in which the condition would have held, and
//! that is every run: the condition is true for every value the analysis
//! allows.

use std::collections::HashMap;

use sky_ir::exact::{self, Analysis, Range};
use sky_ir::{AssertKind, CmpOp, Function, Module, OpId, OpKind, ValueDef, ValueId};

use crate::rewrite::Rewrite;
use crate::util::{defining_op, detach, reachable_ops};

/// The rule id of an `assert` removed, by contract kind: `CONTRACT-001`
/// (`assert`), `CONTRACT-002` (`requires`), `CONTRACT-003` (`ensures`).
pub fn contract_rule(kind: AssertKind) -> &'static str {
    match kind {
        AssertKind::Assert => "CONTRACT-001",
        AssertKind::Requires => "CONTRACT-002",
        AssertKind::Ensures => "CONTRACT-003",
    }
}

/// The headline `sky check --explain` gives an explanation of this pass
/// ("this `requires` always holds"), if `rule` is one of its rules. `text`
/// is the condition as written.
pub fn contract_headline(rule: &str, text: &str) -> Option<String> {
    let kind = [
        AssertKind::Assert,
        AssertKind::Requires,
        AssertKind::Ensures,
    ]
    .into_iter()
    .find(|k| contract_rule(*k) == rule)?;
    Some(format!(
        "[{rule}] this `{}` always holds, so its check is removed: `{}`",
        kind.keyword(),
        sky_diag::display_capped(text)
    ))
}

/// The comparison as the source writes it.
fn symbol(c: CmpOp) -> &'static str {
    match c {
        CmpOp::Eq => "==",
        CmpOp::Ne => "!=",
        CmpOp::Lt => "<",
        CmpOp::Le => "<=",
        CmpOp::Gt => ">",
        CmpOp::Ge => ">=",
    }
}

/// Three-valued truth.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Truth {
    True,
    False,
    Unknown,
}

impl Truth {
    fn not(self) -> Truth {
        match self {
            Truth::True => Truth::False,
            Truth::False => Truth::True,
            Truth::Unknown => Truth::Unknown,
        }
    }
}

fn decided(b: bool) -> Truth {
    if b {
        Truth::True
    } else {
        Truth::False
    }
}

/// `a op b` for every pair of values in the two enclosures, if it is the
/// same for all of them.
pub(crate) fn compare(op: CmpOp, a: Range, b: Range) -> Truth {
    let (lt_always, ge_always) = (a.hi < b.lo, a.lo >= b.hi);
    let (gt_always, le_always) = (a.lo > b.hi, a.hi <= b.lo);
    match op {
        CmpOp::Lt if lt_always => Truth::True,
        CmpOp::Lt if ge_always => Truth::False,
        CmpOp::Ge if ge_always => Truth::True,
        CmpOp::Ge if lt_always => Truth::False,
        CmpOp::Gt if gt_always => Truth::True,
        CmpOp::Gt if le_always => Truth::False,
        CmpOp::Le if le_always => Truth::True,
        CmpOp::Le if gt_always => Truth::False,
        CmpOp::Eq | CmpOp::Ne => {
            let apart = lt_always || gt_always;
            let same = match (a.as_point(), b.as_point()) {
                (Some(x), Some(y)) => Some(x == y),
                _ => None,
            };
            let eq = match (same, apart) {
                (Some(true), _) => Truth::True,
                (_, true) => Truth::False,
                _ => Truth::Unknown,
            };
            if op == CmpOp::Eq {
                eq
            } else {
                eq.not()
            }
        }
        _ => Truth::Unknown,
    }
}

/// How deep [`truth`] follows a condition. An `and`/`or` chain is a run of
/// sibling `if`s, each using the last, so a long chain is a deep recursion.
const MAX_DEPTH: usize = 64;

fn truth(f: &Function, facts: &Analysis, v: ValueId, depth: usize) -> Truth {
    if depth > MAX_DEPTH {
        return Truth::Unknown;
    }
    let ValueDef::OpResult { op, index } = f.value(v).def else {
        return Truth::Unknown;
    };
    let op = f.op(op);
    let known = |i: usize| {
        op.operands
            .get(i)
            .and_then(|v| facts.known(*v))
            .map(|k| k.range)
    };
    match &op.kind {
        OpKind::ConstBool(b) => decided(*b),
        OpKind::Cmp(c) => match (known(0), known(1)) {
            (Some(a), Some(b)) => compare(*c, a, b),
            _ => Truth::Unknown,
        },
        OpKind::Not => match op.operands.first() {
            Some(a) => truth(f, facts, *a, depth + 1).not(),
            None => Truth::Unknown,
        },
        OpKind::If => {
            let Some(c) = op.operands.first() else {
                return Truth::Unknown;
            };
            let arm = |r: usize| -> Truth {
                match yielded(f, op, r, index as usize) {
                    Some(y) => truth(f, facts, y, depth + 1),
                    None => Truth::Unknown,
                }
            };
            match truth(f, facts, *c, depth + 1) {
                Truth::True => arm(0),
                Truth::False => arm(1),
                Truth::Unknown => {
                    let (t, e) = (arm(0), arm(1));
                    if t == e {
                        t
                    } else {
                        Truth::Unknown
                    }
                }
            }
        }
        _ => Truth::Unknown,
    }
}

/// What region `r` of `op` (an `if`) yields as result `index`.
fn yielded(f: &Function, op: &sky_ir::OpData, r: usize, index: usize) -> Option<ValueId> {
    let region = f.region(*op.regions.get(r)?);
    let block = *region.blocks.first()?;
    let last = f.op(*f.block(block).ops.last()?);
    matches!(last.kind, OpKind::Yield)
        .then(|| last.operands.get(index).copied())
        .flatten()
}

/// The enclosure of each parameter of each called function: the hull over
/// its calls of the argument's enclosure in the caller (analysed with
/// unknown parameters). `None` for a parameter some call passes something
/// unknown for.
fn call_facts(m: &Module) -> HashMap<String, Vec<Option<Range>>> {
    let mut out: HashMap<String, Vec<Option<Range>>> = HashMap::new();
    for f in &m.functions {
        let facts = exact::analyze(f);
        for o in reachable_ops(f) {
            let op = f.op(o);
            let OpKind::Call { callee } = &op.kind else {
                continue;
            };
            let args: Vec<Option<Range>> = op
                .operands
                .iter()
                .map(|v| facts.known(*v).map(|k| k.range))
                .collect();
            match out.get_mut(callee) {
                None => {
                    out.insert(callee.clone(), args);
                }
                Some(seen) => {
                    for (s, a) in seen.iter_mut().zip(&args) {
                        *s = match (*s, *a) {
                            (Some(x), Some(y)) => Some(x.hull(y)),
                            _ => None,
                        };
                    }
                    // A call with fewer arguments (not in verified IR)
                    // leaves the rest unknown.
                    for s in seen.iter_mut().skip(args.len()) {
                        *s = None;
                    }
                }
            }
        }
    }
    out
}

/// Removes every `assert` the analysis proves true, from every function.
pub fn contracts(m: &mut Module) -> Vec<Rewrite> {
    let calls = call_facts(m);
    let mut out = Vec::new();
    for f in &mut m.functions {
        let params = calls.get(&f.name).map(Vec::as_slice).unwrap_or(&[]);
        let facts = exact::analyze_with_params(f, params);
        let proved: Vec<OpId> = reachable_ops(f)
            .into_iter()
            .filter(|o| match &f.op(*o).kind {
                OpKind::Assert { .. } => f
                    .op(*o)
                    .operands
                    .first()
                    .is_some_and(|c| truth(f, &facts, *c, 0) == Truth::True),
                _ => false,
            })
            .collect();
        for o in proved {
            out.push(explain(f, o, &facts, params));
            detach(f, o);
        }
    }
    out
}

fn explain(f: &Function, id: OpId, facts: &Analysis, params: &[Option<Range>]) -> Rewrite {
    let op = f.op(id);
    let OpKind::Assert {
        kind, text, func, ..
    } = &op.kind
    else {
        unreachable!("only asserts are explained")
    };
    let mut because = Vec::new();
    // What is known about the parameters, from the calls.
    let calls_of = func.clone().unwrap_or_else(|| f.name.clone());
    for (v, r) in f.param_values().iter().zip(params) {
        let (Some(name), Some(r)) = (&f.value(*v).name, r) else {
            continue;
        };
        if !r.is_unbounded() && because.len() < 4 {
            because.push(format!(
                "in every call of `{calls_of}` in this program, `{name}` is {r}"
            ));
        }
    }
    // A specialization of the function (`f[h=20]`, `sky_opt::specialize`)
    // has its parameters' values baked in; say so.
    if let Some(known) = crate::specialize::facts_in_name(&f.name) {
        because.push(format!(
            "`{}` is `{calls_of}` specialized for {known} (its constants and shapes are known)",
            f.name
        ));
    }
    if let Some(cond) = op.operands.first() {
        because.push(condition_reason(f, facts, *cond));
    }
    Rewrite {
        pass: "contracts",
        rule: contract_rule(*kind),
        span: op.span,
        before: text.clone(),
        after: String::new(),
        because,
    }
}

/// Why the condition cannot be false: the comparison it is, with the
/// enclosures of its sides, or (for anything longer) that it holds for all of
/// them.
fn condition_reason(f: &Function, facts: &Analysis, cond: ValueId) -> String {
    if let Some(def) = defining_op(f, cond) {
        let op = f.op(def);
        if let OpKind::Cmp(c) = &op.kind {
            if let [a, b] = op.operands[..] {
                let side = |v: ValueId| match facts.known(v) {
                    Some(k) => k.range.to_string(),
                    None => "unknown".to_string(),
                };
                return format!(
                    "the left side of the `{}` is {} and the right side is {}, so it cannot be false",
                    symbol(*c),
                    side(a),
                    side(b)
                );
            }
        }
    }
    "the condition is true for every value the analysis allows, so it cannot be false".to_string()
}
