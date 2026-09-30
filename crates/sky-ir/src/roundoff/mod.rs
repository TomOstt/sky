//! Roundoff analysis: how far is each scalar from the exact real number the
//! program denotes? (`docs/design/ir.md`, "Error budgets and roundoff".)
//!
//! SIR means what `sky run` computes: `f64`, one rounding per operation.
//! [`crate::exact`] answers "which `f64`s can this value be" (a range). This
//! module answers the other question a numerical program needs: **"how far
//! is that `f64` from the value the mathematics gives?"** For each scalar it
//! computes a [`Num`]: the range, and bounds on the accumulated rounding
//! error against the exact real computation of the same program (every
//! operation exact; inputs and literals are the exact reals they hold).
//!
//! It is an analysis and a contract, not a change of meaning: nothing here
//! rewrites the program or chooses a precision. Choosing one is Phase 4
//! (code generation); the bounds are what will drive that choice.
//!
//! # Method
//!
//! The analysis is an abstract interpretation of the program, one
//! evaluation per operation ([`transfer`] has each operation's rule):
//!
//! * The **absolute** channel bounds `|computed - exact|`, adding each
//!   operation's rounding error to the incoming errors scaled by the
//!   operation's derivative over the range (a Lipschitz constant).
//! * The **relative** channel bounds `|computed - exact| / |exact|`, adding
//!   the operation's rounding `u = 2^-53` to the incoming relative errors
//!   scaled by the operation's condition number. Addition of two numbers of
//!   one sign, multiplication, division, `sqrt` and the powers keep it
//!   small; a subtraction of nearly equal numbers cannot, and is where the
//!   relative bound is lost (**catastrophic cancellation**).
//! * Each channel is derived from the other when the range allows
//!   ([`Num::tighten`]).
//! * When all the inputs are constants, an enclosure of the exact value in
//!   double-double arithmetic ([`hp`]) bounds the error from below as well,
//!   to prove a budget violated, and from above, tightly.
//!
//! # Soundness
//!
//! Every bound is an inequality that holds for all sizes of the incoming
//! error, not a first-order expansion; bounds are computed with every
//! `f64` step rounded up. The trusted facts are those [`transfer`] lists
//! (correctly rounded `+ - * / sqrt`; math functions within
//! [`LIBM_ULPS`] ulps). Executions that fail (overflow, a domain error) are
//! not analysed. The property tests in `sky-opt` compare the bounds with a
//! double-double reference evaluator.
//!
//! # Control flow
//!
//! * `if`: when the condition's value is known *and* the exact computation
//!   must take the same branch (the operands' error cannot flip the
//!   comparison), only that arm is analysed. When the condition is not
//!   known but robust, both arms are and the results joined. When the
//!   rounding error could flip the comparison, the exact computation may
//!   take the other arm, the results differ by a jump, and the results are
//!   **unbounded**.
//! * `while` and `for`: analysed by unrolling, as long as the condition is
//!   known at every iteration (a counted loop over constants; up to
//!   [`Options::max_loop_iterations`]). A loop whose trip count is not known
//!   is **unbounded**, honestly: every value it defines and every result
//!   has no error bound.
//! * Calls: the callee is analysed in the context of each call (its
//!   arguments' ranges and errors), to a depth of
//!   [`Options::max_call_depth`]. A function the program never calls is
//!   analysed for arbitrary finite, exact inputs.
//! * Arrays, strings and anything else that is not a scalar number are not
//!   modelled: a scalar read out of one is unbounded.
//!
//! An analysis that exceeds [`Options::max_steps`] gives up: every value is
//! then unbounded.

mod blame;
mod hp;
mod num;
mod transfer;

use std::collections::HashMap;

use sky_diag::Span;

pub use blame::{Blame, BlameKind};
pub use num::Num;
pub use transfer::{libm_ulps, LIBM_ULPS, U};

use crate::exact::range::{next_down, next_up, Range};
use crate::ir::{Function, Module, OpId, RegionId, Repr, Type, ValueId};
use crate::op::{CmpOp, OpKind};
use transfer::{transfer, OpNote};

pub(crate) const INF: f64 = f64::INFINITY;

fn fix(x: f64) -> f64 {
    if x.is_nan() {
        INF
    } else {
        x
    }
}

/// `a + b` rounded up (for non-negative `a`, `b`); exact when one is zero.
pub(crate) fn add_up(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        b
    } else if b == 0.0 {
        a
    } else {
        fix(next_up(a + b))
    }
}

/// `a * b` rounded up (non-negative); zero if either is zero.
pub(crate) fn mul_up(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else {
        fix(next_up(a * b))
    }
}

/// `a / b` rounded up (non-negative `a`, positive `b`).
pub(crate) fn div_up(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        0.0
    } else if b <= 0.0 {
        INF
    } else {
        fix(next_up(a / b))
    }
}

/// `1 - r`, rounded down.
pub(crate) fn one_minus(r: f64) -> f64 {
    next_down(1.0 - r)
}

/// `x` made a little larger, to cover the error of the host math library
/// that computed it (a few ulps; the pad is 1e-13 relative).
pub(crate) fn pad(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        fix(next_up(x * (1.0 + 1e-13)))
    }
}

/// Limits of an analysis.
#[derive(Clone, Debug)]
pub struct Options {
    /// Use the double-double enclosure of the exact value where all inputs
    /// are constants ([`hp`]): tighter upper bounds and the lower bounds
    /// that prove a violation. Off, only the first-order bounds remain
    /// (what the soundness tests exercise).
    pub high_precision: bool,
    /// Most operations evaluated, over the whole analysis, before giving up.
    pub max_steps: usize,
    /// Most iterations of one loop that are unrolled.
    pub max_loop_iterations: usize,
    /// Deepest chain of calls that is followed.
    pub max_call_depth: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            high_precision: true,
            max_steps: 2_000_000,
            max_loop_iterations: 100_000,
            max_call_depth: 32,
        }
    }
}

/// The accuracy a value must have.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Tolerance {
    /// `|computed - exact| <= t * |exact|`.
    Relative(f64),
    /// `|computed - exact| <= t`, in the value's SI unit.
    Absolute(f64),
}

/// One `accuracy` declaration, as lowering records it: the value of a name
/// at a point of the program, and the budget it must meet. The `value` is
/// in the module as lowering produced it, before any pass.
#[derive(Clone, Debug, PartialEq)]
pub struct Goal {
    /// The function the value is in (`<main>` for top-level code).
    pub function: String,
    pub value: ValueId,
    /// The source name.
    pub name: String,
    pub tolerance: Tolerance,
    /// The whole declaration.
    pub span: Span,
}

/// What the analysis found about a [`Goal`].
#[derive(Clone, PartialEq, Debug)]
pub enum Verdict {
    /// The budget provably holds: the error is at most `abs` (absolute)
    /// and `rel` (relative), whichever the tolerance uses.
    Proven { abs: f64, rel: f64 },
    /// The budget provably fails: the error is at least `lower_abs`
    /// (absolute) and `lower_rel` (relative).
    Violated { lower_abs: f64, lower_rel: f64 },
    /// The analysis can neither prove nor refute it. `abs` and `rel` are
    /// the bounds it did prove (infinite: none); `blame` names the
    /// operations that dominate the error.
    Unproven {
        abs: f64,
        rel: f64,
        blame: Vec<Blame>,
    },
    /// The value is not a scalar number the analysis follows (an array, an
    /// untyped value) or the analysis gave up.
    Unknown { reason: &'static str },
    /// No analysed execution computes the value (it is in code that never
    /// runs).
    Unreached,
}

/// What is known about one value, as a query result.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Fact<'a> {
    Known(&'a Num),
    Unknown(&'static str),
    Unreached,
}

/// A boolean the analysis follows: its value, and whether the *exact*
/// computation must agree with it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct Flag {
    pub known: Option<bool>,
    pub robust: bool,
}

const UNKNOWN_FLAG: Flag = Flag {
    known: None,
    robust: false,
};

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Av {
    Num(Num),
    Flag(Flag),
    /// Anything else: an array, a string, a value of unknown type.
    Other,
}

fn join_av(a: Av, b: Av) -> Av {
    match (a, b) {
        (Av::Num(x), Av::Num(y)) => Av::Num(x.join(&y)),
        (Av::Flag(x), Av::Flag(y)) => Av::Flag(Flag {
            known: if x.known == y.known { x.known } else { None },
            robust: x.robust && y.robust,
        }),
        _ => Av::Other,
    }
}

fn unbounded(a: Av) -> Av {
    match a {
        Av::Num(n) => Av::Num(n.unbounded()),
        Av::Flag(f) => Av::Flag(Flag { robust: false, ..f }),
        Av::Other => Av::Other,
    }
}

fn is_scalar_number(t: &Type) -> bool {
    matches!(t, Type::Number(n) if n.shape.rank() == 0 && matches!(n.repr, Repr::Unchosen | Repr::F64))
}

/// What nothing is known about, by declared type.
fn unknown_of(f: &Function, v: ValueId) -> Av {
    match f.ty(v) {
        t if is_scalar_number(t) => Av::Num(Num::unknown()),
        Type::Bool => Av::Flag(UNKNOWN_FLAG),
        _ => Av::Other,
    }
}

/// An input of declared type: any finite number, exactly.
fn input_of(f: &Function, v: ValueId) -> Av {
    match f.ty(v) {
        t if is_scalar_number(t) => Av::Num(Num::exact(Range::FINITE)),
        Type::Bool => Av::Flag(Flag {
            known: None,
            robust: true,
        }),
        _ => Av::Other,
    }
}

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
            if a.lo == a.hi && b.lo == b.hi && a.lo == b.lo {
                Some(true)
            } else if a.hi < b.lo || b.hi < a.lo {
                Some(false)
            } else {
                None
            }
        }
        CmpOp::Ne => decide(CmpOp::Eq, a, b).map(|v| !v),
    }
}

fn compare(c: CmpOp, a: &Av, b: &Av) -> Flag {
    match (a, b) {
        (Av::Num(x), Av::Num(y)) => {
            let known = decide(c, x.range, y.range);
            let robust = if x.abs == 0.0 && y.abs == 0.0 {
                true
            } else {
                match known {
                    // The exact operands lie in the hulls: the comparison
                    // must come out the same for all of them.
                    Some(v) => decide(c, x.exact_hull(), y.exact_hull()) == Some(v),
                    None => false,
                }
            };
            Flag { known, robust }
        }
        (Av::Flag(x), Av::Flag(y)) if matches!(c, CmpOp::Eq | CmpOp::Ne) => {
            let known = match (x.known, y.known) {
                (Some(p), Some(q)) => Some((p == q) == (c == CmpOp::Eq)),
                _ => None,
            };
            Flag {
                known,
                robust: x.robust && y.robust,
            }
        }
        _ => UNKNOWN_FLAG,
    }
}

fn flag_of(a: &Av) -> Flag {
    match a {
        Av::Flag(f) => *f,
        _ => UNKNOWN_FLAG,
    }
}

/// The analysis of one function: what each value can be, over every
/// execution of it that the analysis followed.
#[derive(Clone, Debug)]
struct FuncSummary {
    vals: Vec<Option<Av>>,
    notes: Vec<Option<OpNote>>,
    reached: bool,
    gave_up: bool,
}

impl FuncSummary {
    fn new(f: &Function) -> FuncSummary {
        FuncSummary {
            vals: vec![None; f.values.len()],
            notes: vec![None; f.ops.len()],
            reached: false,
            gave_up: false,
        }
    }
}

/// The result of [`analyze`]: a table of what is known about every scalar
/// value of every function.
#[derive(Clone, Debug)]
pub struct Analysis {
    funcs: Vec<FuncSummary>,
    index: HashMap<String, usize>,
}

enum Exit {
    Yield(Vec<Av>),
    Condition(Vec<Av>),
    Return(Vec<Av>),
    /// The block ended without a terminator, or the analysis stopped.
    Stop,
}

struct Act {
    cur: Vec<Option<Av>>,
}

impl Act {
    fn get(&self, v: ValueId) -> Av {
        self.cur
            .get(v.index())
            .copied()
            .flatten()
            .unwrap_or(Av::Other)
    }
}

struct Eval<'m> {
    m: &'m Module,
    opts: &'m Options,
    sum: Vec<FuncSummary>,
    index: HashMap<String, usize>,
    steps: usize,
    depth: usize,
    gave_up: bool,
}

impl<'m> Eval<'m> {
    fn new(m: &'m Module, opts: &'m Options) -> Eval<'m> {
        Eval {
            m,
            opts,
            sum: m.functions.iter().map(FuncSummary::new).collect(),
            index: m
                .functions
                .iter()
                .enumerate()
                .map(|(i, f)| (f.name.clone(), i))
                .collect(),
            steps: 0,
            depth: 0,
            gave_up: false,
        }
    }

    fn finish(mut self) -> Analysis {
        if self.gave_up {
            for s in &mut self.sum {
                s.vals.iter_mut().for_each(|v| *v = None);
                s.gave_up = true;
            }
        }
        Analysis {
            funcs: self.sum,
            index: self.index,
        }
    }

    fn record(&mut self, fi: usize, act: &mut Act, v: ValueId, av: Av) {
        act.cur[v.index()] = Some(av);
        let slot = &mut self.sum[fi].vals[v.index()];
        *slot = Some(match slot.take() {
            None => av,
            Some(old) => join_av(old, av),
        });
    }

    /// Sets the results of `op` (missing ones are unknown).
    fn define(&mut self, fi: usize, act: &mut Act, op: OpId, results: Vec<Av>) {
        let f = &self.m.functions[fi];
        let data = f.op(op);
        for (i, &r) in data.results.iter().enumerate() {
            let av = results.get(i).copied().unwrap_or_else(|| unknown_of(f, r));
            self.record(fi, act, r, av);
        }
    }

    fn note(&mut self, fi: usize, op: OpId, n: OpNote) {
        let slot = &mut self.sum[fi].notes[op.index()];
        *slot = Some(match slot.take() {
            None => n,
            Some(old) => old.join(n),
        });
    }

    /// Every value defined in `region` (and in the regions inside it) is
    /// unknown from now on: the analysis cannot follow it.
    fn poison_region(&mut self, fi: usize, region: RegionId) {
        let f = &self.m.functions[fi];
        for &b in &f.region(region).blocks {
            let block = f.block(b);
            for &a in &block.args {
                self.poison_value(fi, a);
            }
            for &o in &block.ops {
                let op = f.op(o);
                for &r in &op.results {
                    self.poison_value(fi, r);
                }
                for &sub in &op.regions {
                    self.poison_region(fi, sub);
                }
            }
        }
    }

    fn poison_value(&mut self, fi: usize, v: ValueId) {
        let f = &self.m.functions[fi];
        let top = unbounded(unknown_of(f, v));
        let slot = &mut self.sum[fi].vals[v.index()];
        *slot = Some(match slot.take() {
            None => top,
            Some(old) => join_av(old, top),
        });
    }

    fn eval_function(&mut self, fi: usize, args: Vec<Av>) -> Vec<Av> {
        self.sum[fi].reached = true;
        let m = self.m;
        let f = &m.functions[fi];
        let mut act = Act {
            cur: vec![None; f.values.len()],
        };
        let entry = f.entry();
        for (i, &p) in f.block(entry).args.iter().enumerate() {
            let av = args.get(i).copied().unwrap_or_else(|| unknown_of(f, p));
            self.record(fi, &mut act, p, av);
        }
        match self.run_block(fi, &mut act, entry) {
            Exit::Return(values) => values,
            _ => Vec::new(),
        }
    }

    fn eval_region(&mut self, fi: usize, act: &mut Act, region: RegionId, args: Vec<Av>) -> Exit {
        let m = self.m;
        let f = &m.functions[fi];
        let Some(&b) = f.region(region).blocks.first() else {
            return Exit::Stop;
        };
        for (i, &p) in f.block(b).args.iter().enumerate() {
            let av = args.get(i).copied().unwrap_or_else(|| unknown_of(f, p));
            self.record(fi, act, p, av);
        }
        self.run_block(fi, act, b)
    }

    fn run_block(&mut self, fi: usize, act: &mut Act, b: crate::ir::BlockId) -> Exit {
        let m = self.m;
        let f = &m.functions[fi];
        for &id in &f.block(b).ops {
            self.steps += 1;
            if self.steps > self.opts.max_steps {
                self.gave_up = true;
            }
            if self.gave_up {
                return Exit::Stop;
            }
            let op = f.op(id);
            match &op.kind {
                OpKind::Yield => {
                    return Exit::Yield(op.operands.iter().map(|&v| act.get(v)).collect())
                }
                OpKind::Condition => {
                    return Exit::Condition(op.operands.iter().map(|&v| act.get(v)).collect())
                }
                OpKind::Return => {
                    return Exit::Return(op.operands.iter().map(|&v| act.get(v)).collect())
                }
                OpKind::If => self.eval_if(fi, act, id),
                OpKind::While => self.eval_while(fi, act, id),
                OpKind::Call { callee } => self.eval_call(fi, act, id, callee),
                _ => self.eval_simple(fi, act, id),
            }
        }
        Exit::Stop
    }

    fn eval_if(&mut self, fi: usize, act: &mut Act, id: OpId) {
        let m = self.m;
        let op = m.functions[fi].op(id);
        let cond = op
            .operands
            .first()
            .map_or(UNKNOWN_FLAG, |&c| flag_of(&act.get(c)));
        let [then, els] = op.regions[..] else {
            return self.define(fi, act, id, Vec::new());
        };
        let results = match (cond.known, cond.robust) {
            (Some(b), true) => {
                match self.eval_region(fi, act, if b { then } else { els }, vec![]) {
                    Exit::Yield(v) => v,
                    _ => Vec::new(),
                }
            }
            _ => {
                let a = self.eval_region(fi, act, then, vec![]);
                let b = self.eval_region(fi, act, els, vec![]);
                match (a, b) {
                    (Exit::Yield(a), Exit::Yield(b)) => a
                        .into_iter()
                        .zip(b)
                        .map(|(x, y)| {
                            let j = join_av(x, y);
                            // The exact computation may take the other arm.
                            if cond.robust {
                                j
                            } else {
                                unbounded(j)
                            }
                        })
                        .collect(),
                    _ => Vec::new(),
                }
            }
        };
        self.define(fi, act, id, results);
    }

    fn eval_while(&mut self, fi: usize, act: &mut Act, id: OpId) {
        let m = self.m;
        let op = m.functions[fi].op(id);
        let [before, after] = op.regions[..] else {
            return self.define(fi, act, id, Vec::new());
        };
        let mut carried: Vec<Av> = op.operands.iter().map(|&v| act.get(v)).collect();
        let mut iterations = 0usize;
        loop {
            if self.gave_up {
                return;
            }
            let Exit::Condition(mut forwarded) = self.eval_region(fi, act, before, carried.clone())
            else {
                break;
            };
            if forwarded.is_empty() {
                break;
            }
            let flag = flag_of(&forwarded.remove(0));
            match (flag.known, flag.robust) {
                (Some(false), true) => return self.define(fi, act, id, forwarded),
                (Some(true), true) => {
                    iterations += 1;
                    if iterations > self.opts.max_loop_iterations {
                        break;
                    }
                    match self.eval_region(fi, act, after, forwarded) {
                        Exit::Yield(next) => carried = next,
                        _ => break,
                    }
                }
                // The trip count is not known: nothing inside is bounded.
                _ => break,
            }
        }
        self.poison_region(fi, before);
        self.poison_region(fi, after);
        let f = &m.functions[fi];
        let results = op
            .results
            .iter()
            .map(|&r| unbounded(unknown_of(f, r)))
            .collect();
        self.define(fi, act, id, results);
    }

    fn eval_call(&mut self, fi: usize, act: &mut Act, id: OpId, callee: &str) {
        let m = self.m;
        let op = m.functions[fi].op(id);
        let args: Vec<Av> = op.operands.iter().map(|&v| act.get(v)).collect();
        match self.index.get(callee).copied() {
            Some(ci) if self.depth < self.opts.max_call_depth => {
                self.depth += 1;
                let results = self.eval_function(ci, args);
                self.depth -= 1;
                self.define(fi, act, id, results);
            }
            found => {
                // The callee runs in a context the analysis does not follow:
                // nothing it computes can be trusted.
                if let Some(ci) = found {
                    let body = m.functions[ci].body;
                    self.poison_region(ci, body);
                }
                let f = &m.functions[fi];
                let results = op
                    .results
                    .iter()
                    .map(|&r| unbounded(unknown_of(f, r)))
                    .collect();
                self.define(fi, act, id, results);
            }
        }
    }

    fn eval_simple(&mut self, fi: usize, act: &mut Act, id: OpId) {
        let m = self.m;
        let f = &m.functions[fi];
        let op = f.op(id);
        let args: Vec<Av> = op.operands.iter().map(|&v| act.get(v)).collect();
        let arg = |i: usize| args.get(i).copied().unwrap_or(Av::Other);
        let result_av = |unknown: bool| -> Vec<Av> {
            op.results
                .iter()
                .map(|&r| {
                    if unknown {
                        unbounded(unknown_of(f, r))
                    } else {
                        unknown_of(f, r)
                    }
                })
                .collect()
        };
        let results: Vec<Av> = match &op.kind {
            OpKind::ConstBool(b) => vec![Av::Flag(Flag {
                known: Some(*b),
                robust: true,
            })],
            OpKind::ConstStr(_) | OpKind::ConstNothing | OpKind::Print | OpKind::Show { .. } => {
                op.results.iter().map(|_| Av::Other).collect()
            }
            OpKind::Cmp(c) => vec![Av::Flag(compare(*c, &arg(0), &arg(1)))],
            OpKind::Not => {
                let a = flag_of(&arg(0));
                vec![Av::Flag(Flag {
                    known: a.known.map(|b| !b),
                    robust: a.robust,
                })]
            }
            OpKind::ExpectBool { .. } => vec![Av::Flag(flag_of(&arg(0)))],
            OpKind::Input { .. } => op.results.iter().map(|&r| input_of(f, r)).collect(),
            OpKind::Len => op
                .results
                .iter()
                .map(|_| Av::Num(Num::exact(Range::new(0.0, f64::MAX))))
                .collect(),
            _ => {
                let nums: Option<Vec<Num>> = args
                    .iter()
                    .map(|a| match a {
                        Av::Num(n) => Some(*n),
                        _ => None,
                    })
                    .collect();
                let out = nums.and_then(|mut ns| {
                    if !self.opts.high_precision {
                        ns.iter_mut().for_each(|n| n.hp = None);
                    }
                    let refs: Vec<&Num> = ns.iter().collect();
                    let same = op.operands.len() == 2 && op.operands[0] == op.operands[1];
                    transfer(&op.kind, &refs, same)
                });
                match out {
                    Some(mut out) if op.results.len() == 1 => {
                        if !self.opts.high_precision {
                            out.num.hp = None;
                            out.num.lower_abs = 0.0;
                            out.num.lower_rel = 0.0;
                        }
                        self.note(fi, id, out.note);
                        vec![Av::Num(out.num)]
                    }
                    _ => result_av(true),
                }
            }
        };
        self.define(fi, act, id, results);
    }
}

/// The arguments of `f` when nothing is known about its caller: every
/// parameter an exact, arbitrary finite value of its declared type.
fn generic_args(f: &Function) -> Vec<Av> {
    f.param_values().iter().map(|&p| input_of(f, p)).collect()
}

/// Analyses a module: `<main>` first, calling into the functions it calls,
/// then every function it never calls, for arbitrary inputs.
pub fn analyze(m: &Module, opts: &Options) -> Analysis {
    let mut ev = Eval::new(m, opts);
    if let Some(&mi) = ev.index.get(Module::MAIN) {
        let args = generic_args(&m.functions[mi]);
        ev.eval_function(mi, args);
    }
    let mut out = ev.finish();
    for fi in 0..m.functions.len() {
        if out.funcs[fi].reached {
            continue;
        }
        let mut ev = Eval::new(m, opts);
        let args = generic_args(&m.functions[fi]);
        ev.eval_function(fi, args);
        let alone = ev.finish();
        out.funcs[fi] = alone.funcs.into_iter().nth(fi).unwrap_or_else(|| {
            let mut s = FuncSummary::new(&m.functions[fi]);
            s.gave_up = true;
            s
        });
    }
    out
}

/// Analyses the function `name` alone, called with arguments that are exact
/// values in `args` (missing ones: any finite number). For tests and tools
/// that ask about one function.
pub fn analyze_function(m: &Module, name: &str, args: &[Range], opts: &Options) -> Analysis {
    let mut ev = Eval::new(m, opts);
    if let Some(&fi) = ev.index.get(name) {
        let f = &m.functions[fi];
        let params: Vec<Av> = f
            .param_values()
            .iter()
            .enumerate()
            .map(|(i, &p)| match (args.get(i), input_of(f, p)) {
                (Some(&r), Av::Num(_)) => Av::Num(Num::exact(r)),
                (_, av) => av,
            })
            .collect();
        ev.eval_function(fi, params);
    }
    ev.finish()
}

impl Analysis {
    fn func(&self, name: &str) -> Option<&FuncSummary> {
        self.index.get(name).map(|&i| &self.funcs[i])
    }

    /// What is known about `v` of the function `function`.
    pub fn fact(&self, function: &str, v: ValueId) -> Fact<'_> {
        let Some(s) = self.func(function) else {
            return Fact::Unknown("no such function");
        };
        if s.gave_up {
            return Fact::Unknown("the analysis exceeded its budget");
        }
        match s.vals.get(v.index()) {
            None | Some(None) => Fact::Unreached,
            Some(Some(Av::Num(n))) => Fact::Known(n),
            Some(Some(_)) => Fact::Unknown("the value is not a scalar number the analysis follows"),
        }
    }

    /// The [`Num`] of `v`, if known.
    pub fn num(&self, function: &str, v: ValueId) -> Option<&Num> {
        match self.fact(function, v) {
            Fact::Known(n) => Some(n),
            _ => None,
        }
    }

    pub(crate) fn note(&self, function: &str, op: OpId) -> Option<OpNote> {
        self.func(function)?
            .notes
            .get(op.index())
            .copied()
            .flatten()
    }

    /// Whether `function` was analysed in some context (rather than only
    /// for arbitrary inputs).
    pub fn reached_from_main(&self, function: &str) -> bool {
        self.func(function).is_some_and(|s| s.reached)
    }

    /// Checks a [`Goal`].
    pub fn check(&self, m: &Module, goal: &Goal) -> Verdict {
        let n = match self.fact(&goal.function, goal.value) {
            Fact::Known(n) => *n,
            Fact::Unknown(reason) => return Verdict::Unknown { reason },
            Fact::Unreached => return Verdict::Unreached,
        };
        let (holds, fails) = match goal.tolerance {
            Tolerance::Relative(t) => (n.rel <= t, n.lower_rel > t),
            Tolerance::Absolute(t) => (n.abs <= t, n.lower_abs > t),
        };
        if holds {
            Verdict::Proven {
                abs: n.abs,
                rel: n.rel,
            }
        } else if fails {
            Verdict::Violated {
                lower_abs: n.lower_abs,
                lower_rel: n.lower_rel,
            }
        } else {
            Verdict::Unproven {
                abs: n.abs,
                rel: n.rel,
                blame: blame::blame(self, m, &goal.function, goal.value),
            }
        }
    }
}
