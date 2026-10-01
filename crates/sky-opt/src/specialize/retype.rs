//! Typing a function again, once its parameters have types.
//!
//! Lowering types a value from what the checker knew. An unannotated
//! parameter is [`Type::Unknown`], and so is everything computed from one:
//! every operation on it is a run-time check (its operands are numbers of one
//! dimension? a Bool?), and the native engine runs it on boxed values. A
//! specialization that gives a parameter a type ([`super::facts::Kind::Type`])
//! calls [`infer`] on the clone, which types every value again from the
//! parameters on, with the rules the verifier enforces
//! ([`sky_ir::infer::result_type`]), and [`apply`] writes the types down.
//!
//! What is retyped is what was `Unknown`. A value the checker typed keeps
//! its type (it cannot have become less precise, and passes downstream read
//! its tags). What comes out is checked by the verifier before the clone is
//! kept ([`super::Specializer::plan`]): a clone in which an operation has
//! operands its rules reject (`n < 2` with `n` a Bool, `n + 1 m` with `n`
//! dimensionless) is not made, and the generic function, which fails at run
//! time exactly there, stays for that call.
//!
//! # Merges and the return flags
//!
//! An `if` result has the join of what its arms yield ([`sky_ir::infer::join`]),
//! a loop's carried value the join of its initial and next values, computed
//! to a fixed point. That is not enough for a function with an early
//! `return`, which lowering compiles to flags (`sky_lower`, "`break`,
//! `continue` and early `return`"):
//!
//! ```text
//! %5, %6 = if %4 { yield true, %n } else { yield false, nothing }   // if n < 2: return n
//! %8 = not %5
//! %9, %10 = if %8 { ...; yield true, %17 } else { yield %5, %6 }    // the rest, unless returned
//! return %10
//! ```
//!
//! `%6` is the returned value when the flag `%5` is true and a stand-in
//! (`nothing`) when it is not, so its type is the join of a number and
//! `nothing`: unknown, and so is everything after it. But the stand-in is
//! never read: its only use is in a region that runs only when `%5` is true.
//! [`dead_slots`] finds the yielded operands like that: **a value `v` an `if`
//! yields in slot `k`, next to a flag in slot `j` that is a different
//! constant in each arm, is dead in one arm when every use of the result `k`
//! is in a region where the result `j` is known to be the other arm's
//! constant**, or is itself a dead slot. (Known by the conditions of the `if`s
//! around the use: inside `if %8` where `%8 = not %5`, `%5` is false; inside
//! its `else`, true.) The result then has the type of the other arm alone, and
//! [`apply`] writes a stand-in of that type (`0`, `false`, `""`) into the
//! dead arm, so that the IR verifies and every engine sees a typed slot. The
//! dead slot is never read, so no behaviour changes.
//!
//! # Recursion
//!
//! A call's result type is the callee's declared result, or, for clones being
//! specialized together ([`super::Specializer::settle`]), the type the
//! iteration has for them so far. It starts at [`Ty::Bot`] (no value yet)
//! and only grows, to a fixed point: `fib` returns `n` (a number) or the sum
//! of two `fib` results, which is a number if they are.

use std::collections::{HashMap, HashSet};

use sky_diag::Span;
use sky_ir::infer::{join, result_type};
use sky_ir::{Function, OpData, OpId, OpKind, RegionId, Type, ValueDef, ValueId};

use crate::util::{detach, reachable_ops, replace_uses};

/// What is known of a value during inference: nothing yet (it is defined
/// only by recursion that has not returned), or a type.
#[derive(Clone, PartialEq, Debug)]
pub(crate) enum Ty {
    Bot,
    T(Type),
}

impl Ty {
    /// The least type that is either.
    pub(crate) fn join(&self, other: &Ty) -> Ty {
        match (self, other) {
            (Ty::Bot, x) | (x, Ty::Bot) => x.clone(),
            (Ty::T(a), Ty::T(b)) => Ty::T(join(a, b)),
        }
    }

    /// The type to write: a value nothing defines is never read.
    pub(crate) fn into_type(self) -> Type {
        match self {
            Ty::Bot => Type::Unknown,
            Ty::T(t) => t,
        }
    }
}

/// What inference reads besides the function.
pub(crate) struct Env<'a> {
    /// The module's functions, for the declared results of callees.
    pub(crate) funcs: &'a [Function],
    /// The results the iteration assumes for some callees (by name).
    pub(crate) results: &'a HashMap<String, Ty>,
}

impl Env<'_> {
    fn call_result(&self, callee: &str, i: usize) -> Ty {
        if let Some(t) = self.results.get(callee) {
            return if i == 0 {
                t.clone()
            } else {
                Ty::T(Type::Unknown)
            };
        }
        match self
            .funcs
            .iter()
            .find(|g| g.name == callee)
            .and_then(|g| g.results.get(i))
        {
            Some(t) if !has_size_vars(t) => Ty::T(t.clone()),
            _ => Ty::T(Type::Unknown),
        }
    }
}

/// Whether a type has a symbolic size (the callee's, not this function's).
fn has_size_vars(t: &Type) -> bool {
    matches!(t, Type::Number(n) if n.shape.dims().iter().any(|d| matches!(d, sky_types::ShapeDim::Var(_))))
}

/// A stand-in of type `t` can be written: a scalar number, a Bool or a
/// string.
fn standin_type(t: &Ty) -> Option<Type> {
    match t {
        Ty::T(t @ Type::Number(n))
            if n.shape.rank() == 0 && n.tags.is_empty() && n.repr == sky_ir::Repr::Unchosen =>
        {
            Some(t.clone())
        }
        Ty::T(t @ (Type::Bool | Type::Str)) => Some(t.clone()),
        _ => None,
    }
}

/// The result of [`infer`].
pub(crate) struct Inference {
    tys: Vec<Ty>,
    seen: Vec<bool>,
    /// The type of what the function returns.
    pub(crate) result: Ty,
    /// Dead slots to fill: (`yield`, slot) and the type of the stand-in.
    placeholders: HashMap<(OpId, usize), Type>,
}

/// Types every value of `f` again from its parameters (see the module
/// documentation). Pure: `f` is not changed.
pub(crate) fn infer(f: &Function, env: &Env) -> Inference {
    let dead = dead_slots(f);
    let mut w = Walker {
        f,
        env,
        dead: &dead,
        tys: vec![Ty::Bot; f.values.len()],
        seen: vec![false; f.values.len()],
        ret: Ty::Bot,
        placeholders: HashMap::new(),
    };
    for &p in f.param_values() {
        w.set(p, Ty::T(f.ty(p).clone()));
    }
    w.region(f.body);
    Inference {
        tys: w.tys,
        seen: w.seen,
        result: w.ret,
        placeholders: w.placeholders,
    }
}

struct Walker<'a> {
    f: &'a Function,
    env: &'a Env<'a>,
    dead: &'a HashSet<(OpId, usize)>,
    tys: Vec<Ty>,
    seen: Vec<bool>,
    ret: Ty,
    placeholders: HashMap<(OpId, usize), Type>,
}

/// The most times a loop's carried types are recomputed before they are all
/// made unknown.
const MAX_LOOP_ROUNDS: usize = 12;

impl Walker<'_> {
    fn set(&mut self, v: ValueId, t: Ty) {
        self.tys[v.index()] = t;
        self.seen[v.index()] = true;
    }

    /// The type of a value: inferred if it has been visited, else declared.
    fn get(&self, v: ValueId) -> Ty {
        if self.seen[v.index()] {
            self.tys[v.index()].clone()
        } else {
            Ty::T(self.f.ty(v).clone())
        }
    }

    /// Gives `v` the inferred type if it was declared unknown, and keeps the
    /// declared type otherwise.
    fn define(&mut self, v: ValueId, inferred: Ty) {
        let declared = self.f.ty(v);
        let t = if matches!(declared, Type::Unknown) {
            inferred
        } else {
            Ty::T(declared.clone())
        };
        self.set(v, t);
    }

    /// Walks a region's block; returns its last operation.
    fn region(&mut self, r: RegionId) -> Option<OpId> {
        let f = self.f;
        let mut last = None;
        for &b in &f.region(r).blocks {
            for &o in &f.block(b).ops {
                self.op(o);
                last = Some(o);
            }
        }
        last
    }

    fn op(&mut self, o: OpId) {
        let f = self.f;
        let op = f.op(o);
        match &op.kind {
            OpKind::If => self.if_op(o, op),
            OpKind::While => self.while_op(op),
            OpKind::Return => {
                let t = match op.operands[..] {
                    [v] => self.get(v),
                    _ => Ty::T(Type::Unknown),
                };
                self.ret = self.ret.join(&t);
            }
            OpKind::Yield | OpKind::Condition => {}
            OpKind::Call { callee } => {
                for (i, &r) in op.results.iter().enumerate() {
                    let t = self.env.call_result(callee, i);
                    self.define(r, t);
                }
            }
            // Typed exactly, or with a region that is already typed.
            OpKind::ConstFloat { .. }
            | OpKind::ConstRational(_)
            | OpKind::ConstBool(_)
            | OpKind::ConstStr(_)
            | OpKind::ConstArray { .. }
            | OpKind::ConstNothing
            | OpKind::LiteralError { .. }
            | OpKind::Input { .. }
            | OpKind::Fused => {
                for &r in &op.results {
                    self.set(r, Ty::T(f.ty(r).clone()));
                }
            }
            _ => {
                let tys: Vec<Ty> = op.operands.iter().map(|v| self.get(*v)).collect();
                let inferred = if tys.contains(&Ty::Bot) {
                    Ty::Bot
                } else {
                    let ts: Vec<&Type> = tys
                        .iter()
                        .filter_map(|t| match t {
                            Ty::T(t) => Some(t),
                            Ty::Bot => None,
                        })
                        .collect();
                    Ty::T(result_type(&op.kind, &ts).unwrap_or(Type::Unknown))
                };
                for (i, &r) in op.results.iter().enumerate() {
                    if i == 0 {
                        self.define(r, inferred.clone());
                    } else {
                        self.set(r, Ty::T(f.ty(r).clone()));
                    }
                }
            }
        }
    }

    fn if_op(&mut self, _o: OpId, op: &OpData) {
        let f = self.f;
        let mut yields: Vec<Option<OpId>> = Vec::new();
        for &r in &op.regions {
            yields.push(self.region(r));
        }
        let arms: Vec<(OpId, &OpData)> = yields
            .iter()
            .filter_map(|y| y.map(|y| (y, f.op(y))))
            .filter(|(_, d)| matches!(d.kind, OpKind::Yield))
            .collect();
        for (k, &r) in op.results.iter().enumerate() {
            for (y, _) in &arms {
                self.placeholders.remove(&(*y, k));
            }
            if arms.len() != op.regions.len() || arms.iter().any(|(_, d)| d.operands.len() <= k) {
                self.set(r, Ty::T(f.ty(r).clone()));
                continue;
            }
            let slot = |w: &Self, d: &OpData| w.get(d.operands[k]);
            let mut all = Ty::Bot;
            let mut live = Ty::Bot;
            let mut dead_arms = Vec::new();
            for (y, d) in &arms {
                let t = slot(self, d);
                all = all.join(&t);
                if self.dead.contains(&(*y, k)) {
                    dead_arms.push((*y, t));
                } else {
                    live = live.join(&t);
                }
            }
            let declared = f.ty(r);
            if !matches!(declared, Type::Unknown) {
                self.set(r, Ty::T(declared.clone()));
                continue;
            }
            match standin_type(&live) {
                Some(t) if !dead_arms.is_empty() => {
                    for (y, slot_ty) in dead_arms {
                        if slot_ty != Ty::T(t.clone()) {
                            self.placeholders.insert((y, k), t.clone());
                        }
                    }
                    self.set(r, Ty::T(t));
                }
                _ => self.set(r, all),
            }
        }
    }

    fn while_op(&mut self, op: &OpData) {
        let f = self.f;
        let ([before, after], n) = (&op.regions[..], op.operands.len()) else {
            return self.leave(op);
        };
        let (Some(&bb), Some(&ab)) = (
            f.region(*before).blocks.first(),
            f.region(*after).blocks.first(),
        ) else {
            return self.leave(op);
        };
        let (bargs, aargs) = (&f.block(bb).args, &f.block(ab).args);
        if bargs.len() != n || aargs.len() != n || op.results.len() != n {
            return self.leave(op);
        }
        let mut carried: Vec<Ty> = op.operands.iter().map(|v| self.get(*v)).collect();
        let mut rounds = 0;
        let forwarded = loop {
            for (a, t) in bargs.iter().zip(&carried) {
                self.define(*a, t.clone());
            }
            let cond = self.region(*before);
            let fwd: Vec<Ty> = match cond.map(|c| f.op(c)) {
                Some(c) if matches!(c.kind, OpKind::Condition) && c.operands.len() == n + 1 => {
                    c.operands[1..].iter().map(|v| self.get(*v)).collect()
                }
                _ => vec![Ty::T(Type::Unknown); n],
            };
            for (a, t) in aargs.iter().zip(&fwd) {
                self.define(*a, t.clone());
            }
            let next: Vec<Ty> = match self.region(*after).map(|y| f.op(y)) {
                Some(y) if matches!(y.kind, OpKind::Yield) && y.operands.len() == n => (0..n)
                    .map(|k| carried[k].join(&self.get(y.operands[k])))
                    .collect(),
                _ => vec![Ty::T(Type::Unknown); n],
            };
            if next == carried || rounds > MAX_LOOP_ROUNDS {
                break fwd;
            }
            rounds += 1;
            carried = if rounds > MAX_LOOP_ROUNDS {
                vec![Ty::T(Type::Unknown); n]
            } else {
                next
            };
        };
        for (r, t) in op.results.iter().zip(&forwarded) {
            self.define(*r, t.clone());
        }
    }

    /// An operation this walk does not look into: its values keep their
    /// declared types.
    fn leave(&mut self, op: &OpData) {
        for &r in &op.results {
            self.set(r, Ty::T(self.f.ty(r).clone()));
        }
    }
}

// ----- dead slots -----

/// A value with the `not`s stripped, and whether an odd number were.
fn strip_not(f: &Function, mut v: ValueId) -> (ValueId, bool) {
    let mut negated = false;
    while let ValueDef::OpResult { op, .. } = f.value(v).def {
        let d = f.op(op);
        match (&d.kind, &d.operands[..]) {
            (OpKind::Not, [x]) => {
                negated = !negated;
                v = *x;
            }
            _ => break,
        }
    }
    (v, negated)
}

/// The operations around `op`, innermost first, with the index of the
/// region `op` is in.
fn enclosing(f: &Function, op: OpId) -> Vec<(OpId, usize)> {
    let mut out = Vec::new();
    let mut block = f.op(op).parent;
    while let Some(region) = f.block(block).parent {
        let Some(owner) = f.region(region).parent else {
            break;
        };
        let idx = f.op(owner).regions.iter().position(|r| *r == region);
        if let Some(idx) = idx {
            out.push((owner, idx));
        }
        block = f.op(owner).parent;
    }
    out
}

/// The Bool `w` is at `op`, if a constant says so, or an `if` around `op`
/// tests it (or its negation).
fn known_at(f: &Function, op: OpId, w: ValueId) -> Option<bool> {
    let (root, w_negated) = strip_not(f, w);
    if let ValueDef::OpResult { op: d, .. } = f.value(root).def {
        if let OpKind::ConstBool(b) = f.op(d).kind {
            return Some(b != w_negated);
        }
    }
    for (owner, idx) in enclosing(f, op) {
        let o = f.op(owner);
        if !matches!(o.kind, OpKind::If) || o.operands.len() != 1 {
            continue;
        }
        let (c_root, c_negated) = strip_not(f, o.operands[0]);
        if c_root != root {
            continue;
        }
        // In the `then` arm the condition is true, in the `else` arm false.
        let cond = idx == 0;
        let root_value = cond != c_negated;
        return Some(root_value != w_negated);
    }
    None
}

/// The (`yield`, slot) pairs whose value is never read (see the module
/// documentation).
pub(crate) fn dead_slots(f: &Function) -> HashSet<(OpId, usize)> {
    let ops = reachable_ops(f);
    let mut uses: HashMap<ValueId, Vec<(OpId, usize)>> = HashMap::new();
    for &o in &ops {
        for (i, v) in f.op(o).operands.iter().enumerate() {
            uses.entry(*v).or_default().push((o, i));
        }
    }
    let ifs: Vec<(OpId, [OpId; 2])> = ops
        .iter()
        .filter_map(|&o| {
            let d = f.op(o);
            if !matches!(d.kind, OpKind::If) || d.regions.len() != 2 || d.results.len() < 2 {
                return None;
            }
            let mut ys = [o; 2];
            for (i, &r) in d.regions.iter().enumerate() {
                let b = *f.region(r).blocks.first()?;
                let y = *f.block(b).ops.last()?;
                let yd = f.op(y);
                if !matches!(yd.kind, OpKind::Yield) || yd.operands.len() != d.results.len() {
                    return None;
                }
                ys[i] = y;
            }
            Some((o, ys))
        })
        .collect();
    let mut dead: HashSet<(OpId, usize)> = HashSet::new();
    loop {
        let mut changed = false;
        for (o, ys) in &ifs {
            let d = f.op(*o);
            for j in 0..d.results.len() {
                let flags = [0, 1].map(|i| known_at(f, ys[i], f.op(ys[i]).operands[j]));
                let (Some(a), Some(b)) = (flags[0], flags[1]) else {
                    continue;
                };
                if a == b {
                    continue;
                }
                for k in 0..d.results.len() {
                    if k == j || dead.contains(&(ys[0], k)) || dead.contains(&(ys[1], k)) {
                        continue;
                    }
                    let Some(us) = uses.get(&d.results[k]) else {
                        continue;
                    };
                    // The value is valid in arm `valid` (where the flag is
                    // `guard`); the other arm's is dead if every use is
                    // guarded, or goes to a slot that is dead itself.
                    for (valid, guard) in [(0usize, a), (1usize, b)] {
                        let guarded = us.iter().all(|&(u, slot)| {
                            known_at(f, u, d.results[j]) == Some(guard) || dead.contains(&(u, slot))
                        });
                        if guarded {
                            dead.insert((ys[1 - valid], k));
                            changed = true;
                            break;
                        }
                    }
                }
            }
        }
        if !changed {
            return dead;
        }
    }
}

// ----- writing the types down -----

/// Writes the types of `inf` into `f`: the value types, the stand-ins of the
/// dead slots, and, with `result`, the function's result type. The run-time
/// type checks that are now satisfied (`expect.bool` of a Bool,
/// `expect.number` of a number) are removed.
pub(crate) fn apply(f: &mut Function, inf: &Inference, result: Option<Type>) {
    for (i, v) in f.values.iter_mut().enumerate() {
        if inf.seen[i] {
            v.ty = inf.tys[i].clone().into_type();
        }
    }
    let mut fill: Vec<(&(OpId, usize), &Type)> = inf.placeholders.iter().collect();
    fill.sort_by_key(|(k, _)| **k);
    for (&(y, slot), ty) in fill {
        let kind = match ty {
            Type::Number(_) => OpKind::ConstFloat { bits: 0 },
            Type::Bool => OpKind::ConstBool(false),
            Type::Str => OpKind::ConstStr(String::new()),
            _ => continue,
        };
        let block = f.op(y).parent;
        let span = f.op(y).span;
        let op = OpId(f.ops.len() as u32);
        let v = ValueId(f.values.len() as u32);
        f.values.push(sky_ir::ValueData {
            ty: ty.clone(),
            def: ValueDef::OpResult { op, index: 0 },
            name: None,
        });
        f.ops.push(OpData {
            kind,
            operands: Vec::new(),
            results: vec![v],
            regions: Vec::new(),
            span,
            operand_spans: Vec::new(),
            parent: block,
        });
        let ops = &mut f.blocks[block.index()].ops;
        let at = ops.len().saturating_sub(1);
        ops.insert(at, op);
        f.ops[y.index()].operands[slot] = v;
    }
    for o in reachable_ops(f) {
        let d = f.op(o);
        let satisfied = match (&d.kind, &d.operands[..]) {
            (OpKind::ExpectBool { .. }, [x]) => matches!(f.ty(*x), Type::Bool),
            (OpKind::ExpectNumber { .. }, [x]) => matches!(f.ty(*x), Type::Number(_)),
            _ => false,
        };
        if satisfied {
            let (x, r) = (d.operands[0], d.results[0]);
            replace_uses(f, r, x);
            detach(f, o);
        }
    }
    if let Some(t) = result {
        if f.results.len() == 1 {
            f.results[0] = t;
        }
    }
}

// ----- counting -----

/// The run-time type checks of `f`: operations that look at an operand of
/// unknown type to see what it is (a number of this dimension? a Bool?),
/// counted once per source operation.
pub(crate) fn dynamic_checks(f: &Function) -> usize {
    reachable_ops(f)
        .into_iter()
        .filter(|&o| {
            let d = f.op(o);
            let looks = match d.kind {
                OpKind::If
                | OpKind::While
                | OpKind::Yield
                | OpKind::Condition
                | OpKind::Return
                | OpKind::Print
                | OpKind::Call { .. }
                | OpKind::ConstFloat { .. }
                | OpKind::ConstRational(_)
                | OpKind::ConstBool(_)
                | OpKind::ConstStr(_)
                | OpKind::ConstArray { .. }
                | OpKind::ConstNothing
                | OpKind::Input { .. }
                | OpKind::LiteralError { .. }
                | OpKind::Fused => false,
                OpKind::Assert { .. } => d.operands.first().is_some_and(|v| unknown(f, *v)),
                _ => d.operands.iter().any(|v| unknown(f, *v)),
            };
            looks
        })
        .map(|o| {
            let d = f.op(o);
            (d.kind.mnemonic(), d.span)
        })
        .collect::<HashSet<(&'static str, Span)>>()
        .len()
}

fn unknown(f: &Function, v: ValueId) -> bool {
    matches!(f.ty(v), Type::Unknown)
}

/// The calls of `f` that pass a value of known type where the callee's
/// parameter is unknown: what a specialization of the callee could use.
pub(crate) fn typed_arguments(f: &Function, funcs: &[Function]) -> usize {
    reachable_ops(f)
        .into_iter()
        .filter(|&o| {
            let d = f.op(o);
            let OpKind::Call { callee } = &d.kind else {
                return false;
            };
            let Some(g) = funcs.iter().find(|g| g.name == *callee) else {
                return false;
            };
            d.operands.len() == g.params.len()
                && d.operands
                    .iter()
                    .zip(&g.params)
                    .any(|(a, p)| matches!(p, Type::Unknown) && !unknown(f, *a))
        })
        .count()
}
