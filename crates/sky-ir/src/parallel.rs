//! Automatic parallelism: what may run concurrently without changing any
//! observable behaviour (roadmap Phase 4, "automatic parallelism from the
//! dependency graph").
//!
//! This is an analysis. It changes nothing in the module: the interpreter
//! (`sky-ir-interp`) asks it which operations to run concurrently, and
//! `sky check --explain` asks it for the notes ([`explain`]). Three kinds of
//! work qualify.
//!
//! # 1. Elementwise kernels
//!
//! A [`OpKind::Fused`] kernel, and an elementwise `add`/`sub`/`mul`/`div`/`neg`
//! on arrays, computes every element independently of the others, has no
//! `print` and no call, and may be split into chunks. What makes it
//! observable is only *failure*, so the rule for failure is the whole
//! design:
//!
//! * **Cannot fail** (the exactness analysis proves every operation of the
//!   body in range, [`crate::exact`]): the chunks are independent, no
//!   resolution is needed.
//! * **May fail**: the chunks run to the end and the failure is chosen
//!   afterwards, so that it is the one the sequential run reports. For a
//!   `fused` kernel that is the failing operation that comes *first in body
//!   order* (`fused`'s documented rule), at the lowest element that fails it;
//!   an error names its operation, so chunks report `(operation index,
//!   error)` and the merge takes the smallest index, and among equal indices
//!   the lowest chunk. For an elementwise array operation the only failure
//!   is one error for the whole operation (an overflow, whichever entry
//!   caused it), checked after the entries are computed, exactly as the
//!   sequential loop does. Nothing is printed by a kernel, so nothing can
//!   come out in a different order.
//!
//! # 2. Independent pure calls
//!
//! A *window* is a run of operations in one block: it starts at a call to a
//! function that has no `io` (transitively, [`crate::effects`]), is not
//! recursive, and is expensive enough ([`Config::call_cost`]), and it
//! continues over operations that have no `io`, in themselves or in a region
//! or callee (an operation that prints keeps its place after a call that
//! might fail: had the call failed, the print would not have happened).
//!
//! Every operation of the window has one of three [`Role`]s:
//!
//! * a call that does not read the result of a started call or of a deferred
//!   operation, and is expensive enough, is **started** on another thread;
//! * an operation that reads such a result (`total + h(i)`, or a call whose
//!   argument is one) is **deferred**: it runs after the started calls
//!   are done;
//! * every other operation runs **in place**, in program order, while the
//!   started calls run.
//!
//! A window with at least two started calls is a [`CallGroup`].
//!
//! Errors are resolved in program order. When the started calls are done, the
//! window is replayed in program order: a started call yields its result or
//! its error, a deferred operation runs, and the first failure is the
//! error. That is the sequential run's error, because everything that ran
//! early was pure and its only observable trace is a failure, which the replay
//! reaches in order. An operation that fails *in place* stops the window at
//! once (no call after it is started), and the replay of what precedes it
//! runs first: its failure, if any, is earlier. A callee runs on a worker
//! thread with the same call depth as the caller and its output goes nowhere
//! (it has no `io`), so `recursion is too deep` arises in both runs at the
//! same call.
//!
//! The *stack* is the one resource that a worker changes (a worker starts
//! with a new, empty stack, and `sky run` stops with "the program nests too
//! deeply" at a byte budget). A callee is started elsewhere only if its
//! whole call tree is a bounded, shallow ([`MAX_STACK_WEIGHT`]) chain of
//! calls with no recursion, and the interpreter starts it only while the
//! caller has used at most half the budget, so a run that would have hit the
//! limit hits it in both.
//!
//! # Not parallelized
//!
//! Loops: the values a `while` carries from one iteration to the next make
//! its iterations a chain, so they stay in order (the work *inside* an
//! iteration is what the two kinds above cover). Everything that prints:
//! output order is behaviour.
//!
//! # Cost
//!
//! [`Config`] holds the thresholds, which the benchmarks in `sky-bench`
//! measured (`docs/design/ir.md`): a kernel with fewer elements, or a call
//! that costs less than [`Config::call_cost`] estimated operations, is
//! cheaper to run in place than to start elsewhere.

use std::collections::{HashMap, HashSet};

use sky_diag::Span;

use crate::exact;
use crate::ir::{
    BlockId, Effect, EffectSet, Function, Module, OpData, OpId, RegionId, Type, ValueId,
};
use crate::op::OpKind;
use crate::pass::RuleId;

/// `[PAR-001]`: a kernel runs on several threads.
pub const KERNEL_RULE: RuleId = "PAR-001";
/// `[PAR-002]`: independent pure calls run concurrently.
pub const CALLS_RULE: RuleId = "PAR-002";
/// `[PAR-003]`: something that could be parallel is not, and why.
pub const NOT_PARALLEL_RULE: RuleId = "PAR-003";

/// The most nested calls plus regions a started callee may reach; see the
/// module documentation ("the stack").
pub const MAX_STACK_WEIGHT: u32 = 64;

/// How many iterations a `while` is assumed to run, for a callee's cost.
const ASSUMED_TRIPS: u64 = 100;

/// The longest window of operations scanned from one call, so that analysing
/// a huge block stays linear.
const MAX_WINDOW: usize = 512;

/// The cost of an array operation whose size is not known statically.
const UNKNOWN_ELEMENTS: u64 = 256;

/// The most threads `--threads` accepts.
pub const MAX_THREADS: usize = 1024;

/// The kernel name `sky tune` stores the `fused_work` threshold under (size
/// class `all`, parameter `work`).
pub const FUSED_WORK_KERNEL: &str = "par_fused_work";
/// The kernel name for the `array_elements` threshold (size class `all`,
/// parameter `elements`).
pub const ARRAY_ELEMENTS_KERNEL: &str = "par_array_elements";

/// The `fused_work` and `array_elements` thresholds tuned on this machine,
/// else the defaults of [`Config::with_threads`]: read once per process and
/// clamped, so a stale or edited database cannot switch parallelism off for
/// good or on for a one-element array.
pub fn tuned() -> (usize, usize) {
    static TUNED: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
    *TUNED.get_or_init(|| {
        let d = Config::with_threads(1);
        let fused = sky_tune::lookup(FUSED_WORK_KERNEL, "all").map_or(d.fused_work as i64, |p| {
            p.get_clamped("work", d.fused_work as i64, 0, 1 << 30)
        });
        let array = sky_tune::lookup(ARRAY_ELEMENTS_KERNEL, "all")
            .map_or(d.array_elements as i64, |p| {
                p.get_clamped("elements", d.array_elements as i64, 2, 1 << 30)
            });
        (fused as usize, array as usize)
    })
}

/// How to run: the number of threads and the thresholds under which
/// running elsewhere costs more than it saves.
///
/// The defaults are measured: `cargo run --release -p sky-bench -- par`
/// prints the break-even points they come from.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Config {
    /// Worker threads. 1 means everything runs sequentially, on the calling
    /// thread, exactly as before this analysis existed.
    pub threads: usize,
    /// The least work a `fused` kernel is split for, in element-operations:
    /// the elements times the operations of its body.
    pub fused_work: usize,
    /// The fewest elements an elementwise array operation (`add`, `mul`,
    /// ...) is split for. Far higher than a kernel's threshold per
    /// element-operation: one operation per element is memory-bound.
    pub array_elements: usize,
    /// The least estimated cost (interpreted operations, loops counted as
    /// [`ASSUMED_TRIPS`] iterations) of a callee worth starting elsewhere.
    pub call_cost: u64,
    /// The fewest elements one chunk of a split kernel has.
    pub chunk_min: usize,
}

impl Config {
    /// The measured default thresholds, for `threads` threads.
    pub const fn with_threads(threads: usize) -> Config {
        Config {
            threads,
            fused_work: 4096,
            array_elements: 262_144,
            call_cost: 400,
            chunk_min: 512,
        }
    }

    /// One thread: nothing runs in parallel.
    pub const fn sequential() -> Config {
        Config::with_threads(1)
    }

    /// The most parallel a run can be: every threshold at its minimum, so
    /// that small programs (the fuzzer's) exercise every parallel path.
    /// For tests.
    pub const fn eager(threads: usize) -> Config {
        Config {
            threads,
            fused_work: 0,
            array_elements: 1,
            call_cost: 0,
            chunk_min: 1,
        }
    }

    /// [`Config::with_threads`], with the two size thresholds `sky tune` found
    /// best on this machine where it has measured them (see [`tuned`]).
    /// Tuning changes when work is split, never the output.
    pub fn tuned(threads: usize) -> Config {
        let (fused_work, array_elements) = tuned();
        Config {
            fused_work,
            array_elements,
            ..Config::with_threads(threads)
        }
    }

    /// The default number of threads: `SKY_THREADS` if it is a valid
    /// number, else the machine's parallelism.
    pub fn default_threads() -> usize {
        if let Some(n) = std::env::var("SKY_THREADS")
            .ok()
            .and_then(|v| parse_threads(&v).ok())
        {
            return n;
        }
        std::thread::available_parallelism().map_or(1, usize::from)
    }

    /// The configuration `sky` uses without `--threads`: [`Config::default_threads`]
    /// and the thresholds tuned on this machine (else the measured defaults).
    pub fn from_env() -> Config {
        Config::tuned(Config::default_threads())
    }

    /// Whether anything can run in parallel at all.
    pub fn is_parallel(&self) -> bool {
        self.threads > 1
    }

    /// How many chunks a kernel of `elements` elements is split into: none
    /// smaller than [`Config::chunk_min`], at most four per thread (so that
    /// a slow chunk does not idle the others), and a function of the two
    /// only, never of timing.
    pub fn chunks(&self, elements: usize) -> usize {
        let by_size = elements.div_ceil(self.chunk_min.max(1)).max(1);
        by_size.min(self.threads.saturating_mul(4)).max(1)
    }
}

/// Parses the value of `--threads=` or `SKY_THREADS`: a whole number from 1
/// to [`MAX_THREADS`].
pub fn parse_threads(text: &str) -> Result<usize, String> {
    match text.trim().parse::<usize>() {
        Ok(n) if (1..=MAX_THREADS).contains(&n) => Ok(n),
        _ => Err(format!(
            "`{text}` is not a thread count (use a whole number from 1 to {MAX_THREADS}; 1 is sequential)"
        )),
    }
}

/// A note for `--explain`: where, which rule, what, and why.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Note {
    pub rule: RuleId,
    pub span: Span,
    /// The one-line statement, without the `[rule]` tag.
    pub headline: String,
    /// The facts it rests on, one per entry.
    pub because: Vec<String>,
}

/// What an operation of a window does. See the module documentation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    /// A call started on another thread.
    Start,
    /// Runs in place, in order, while the started calls run.
    InPlace,
    /// Reads a result that is not there yet: runs after the started calls.
    Deferred,
}

/// Calls of one block that run concurrently. See the module documentation.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CallGroup {
    /// The block the operations are in.
    pub block: BlockId,
    /// The operations of the window, in order: the first and the last are
    /// calls that are started elsewhere.
    pub ops: Vec<OpId>,
    /// `roles[i]` is what `ops[i]` does.
    pub roles: Vec<Role>,
}

/// What the analysis found in one function.
#[derive(Clone, Default, Debug)]
pub struct FnPlan {
    pub groups: Vec<CallGroup>,
    /// Indexed by operation: the group that starts there, if any.
    group_at: Vec<u32>,
}

impl FnPlan {
    /// The group whose window starts at `op`.
    pub fn group_starting_at(&self, op: OpId) -> Option<&CallGroup> {
        match self.group_at.get(op.index()) {
            Some(&g) if g != u32::MAX => self.groups.get(g as usize),
            _ => None,
        }
    }
}

/// The analysis of a module: for each function, its call groups.
#[derive(Clone, Debug)]
pub struct Plan {
    functions: Vec<FnPlan>,
    index: HashMap<String, usize>,
}

impl Plan {
    /// A plan with nothing parallel.
    pub fn empty() -> Plan {
        Plan {
            functions: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// The plan of the function called `name`.
    pub fn function(&self, name: &str) -> Option<&FnPlan> {
        self.functions.get(*self.index.get(name)?)
    }

    /// Whether any call group exists.
    pub fn has_groups(&self) -> bool {
        self.functions.iter().any(|f| !f.groups.is_empty())
    }
}

/// Analyses `m` for `cfg`. Total: any module, however malformed, gives a
/// plan (which the interpreter's own checks then run or reject).
pub fn analyze(m: &Module, cfg: &Config) -> Plan {
    if !cfg.is_parallel() {
        return Plan::empty();
    }
    run(m, cfg, None)
}

/// The `--explain` notes for `m` under `cfg`, in function order and, within
/// a function, program order.
pub fn explain(m: &Module, cfg: &Config) -> Vec<Note> {
    let mut notes = Vec::new();
    run(m, cfg, Some(&mut notes));
    notes
}

// ---------------------------------------------------------------------------
// The call graph: purity, stack weight and cost of every function.
// ---------------------------------------------------------------------------

struct Graph<'m> {
    m: &'m Module,
    by_name: HashMap<&'m str, usize>,
    effects: Vec<EffectSet>,
    /// The deepest chain of nested calls and regions the function can
    /// reach, saturated at `MAX_STACK_WEIGHT + 1` (which also is what
    /// recursion gives).
    weight: Vec<u32>,
    cost: Vec<Option<u64>>,
}

/// The operations reachable from a region, nested regions included, once
/// each and in a stable order, with the depth of region nesting.
fn region_ops(f: &Function, root: RegionId) -> (Vec<OpId>, u32) {
    let mut ops = Vec::new();
    let mut seen_region = vec![false; f.regions.len()];
    let mut seen_op = vec![false; f.ops.len()];
    let mut depth = 0;
    let mut todo = vec![(root, 0u32)];
    while let Some((r, d)) = todo.pop() {
        let Some(region) = f.regions.get(r.index()) else {
            continue;
        };
        if std::mem::replace(&mut seen_region[r.index()], true) {
            continue;
        }
        depth = depth.max(d);
        for b in &region.blocks {
            let Some(block) = f.blocks.get(b.index()) else {
                continue;
            };
            for &o in &block.ops {
                let Some(op) = f.ops.get(o.index()) else {
                    continue;
                };
                if std::mem::replace(&mut seen_op[o.index()], true) {
                    continue;
                }
                ops.push(o);
                for &nested in &op.regions {
                    todo.push((nested, d + 1));
                }
            }
        }
    }
    (ops, depth)
}

impl<'m> Graph<'m> {
    fn new(m: &'m Module) -> Graph<'m> {
        let n = m.functions.len();
        let mut by_name = HashMap::new();
        for (i, f) in m.functions.iter().enumerate() {
            by_name.entry(f.name.as_str()).or_insert(i);
        }
        let mut callees = Vec::with_capacity(n);
        let mut own = Vec::with_capacity(n);
        for f in &m.functions {
            let (ops, depth) = region_ops(f, f.body);
            own.push(depth + 1);
            let mut targets: Vec<usize> = ops
                .iter()
                .filter_map(|&o| match &f.ops[o.index()].kind {
                    OpKind::Call { callee } => by_name.get(callee.as_str()).copied(),
                    _ => None,
                })
                .collect();
            targets.sort_unstable();
            targets.dedup();
            callees.push(targets);
        }
        // weight[f] = own[f] + the heaviest callee, saturating: a fixed
        // point of an increasing map with a ceiling, so it stops, and a
        // function on a cycle reaches the ceiling.
        let ceiling = MAX_STACK_WEIGHT + 1;
        let mut weight: Vec<u32> = own.iter().map(|&w| w.min(ceiling)).collect();
        loop {
            let mut changed = false;
            for f in 0..n {
                let heaviest = callees[f].iter().map(|&c| weight[c]).max().unwrap_or(0);
                let next = own[f].saturating_add(heaviest).min(ceiling);
                if next > weight[f] {
                    weight[f] = next;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        Graph {
            m,
            by_name,
            effects: crate::effects::infer(m),
            weight,
            cost: vec![None; n],
        }
    }

    fn callee(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }

    /// The estimated cost of running function `i` once, in interpreted
    /// operations. Only meaningful for a function with a bounded weight
    /// (others give `u64::MAX`, and are never started elsewhere).
    fn cost_of(&mut self, i: usize) -> u64 {
        if self.weight[i] > MAX_STACK_WEIGHT {
            return u64::MAX;
        }
        if let Some(c) = self.cost[i] {
            return c;
        }
        // The chain of calls is at most MAX_STACK_WEIGHT deep, so this
        // recursion is too.
        let m = self.m;
        let f = &m.functions[i];
        let c = self.region_cost(f, f.body);
        self.cost[i] = Some(c);
        c
    }

    fn region_cost(&mut self, f: &Function, region: RegionId) -> u64 {
        let Some(r) = f.regions.get(region.index()) else {
            return 0;
        };
        let mut total = 0u64;
        for b in &r.blocks {
            let Some(block) = f.blocks.get(b.index()) else {
                continue;
            };
            for &o in &block.ops {
                let Some(op) = f.ops.get(o.index()) else {
                    continue;
                };
                total = total.saturating_add(self.op_cost(f, op));
            }
        }
        total
    }

    fn op_cost(&mut self, f: &Function, op: &OpData) -> u64 {
        match &op.kind {
            OpKind::While => {
                let mut body = 0u64;
                for &r in &op.regions {
                    body = body.saturating_add(self.region_cost(f, r));
                }
                body.saturating_mul(ASSUMED_TRIPS).saturating_add(1)
            }
            OpKind::If => {
                let mut arms = 0u64;
                for &r in &op.regions {
                    arms = arms.max(self.region_cost(f, r));
                }
                arms.saturating_add(1)
            }
            OpKind::Call { callee } => match self.callee(callee) {
                Some(c) => self.cost_of(c).saturating_add(2),
                None => 2,
            },
            OpKind::Fused => {
                let n = op
                    .results
                    .first()
                    .and_then(|&r| elements(f, r))
                    .unwrap_or(UNKNOWN_ELEMENTS);
                let body: u64 = op
                    .regions
                    .first()
                    .map_or(1, |&r| self.region_cost(f, r).max(1));
                n.saturating_mul(body)
            }
            OpKind::MatMul => {
                let n = op
                    .results
                    .first()
                    .and_then(|&r| elements(f, r))
                    .unwrap_or(UNKNOWN_ELEMENTS);
                let inner = op
                    .operands
                    .first()
                    .and_then(|&a| last_dim(f, a))
                    .unwrap_or(16);
                n.saturating_mul(inner)
            }
            OpKind::Add
            | OpKind::Sub
            | OpKind::Mul
            | OpKind::Div
            | OpKind::Neg
            | OpKind::Inverse
            | OpKind::Solve { .. } => match op.results.first().and_then(|&r| shaped(f, r)) {
                Some(n) => n.unwrap_or(UNKNOWN_ELEMENTS),
                None => 1,
            },
            _ => 1,
        }
    }
}

/// The number of elements of an array value with a known shape; `None` for
/// a scalar, an unknown type or a symbolic size.
fn elements(f: &Function, v: ValueId) -> Option<u64> {
    shaped(f, v).flatten()
}

/// `None`: not an array. `Some(None)`: an array of symbolic size.
/// `Some(Some(n))`: an array of `n` elements.
fn shaped(f: &Function, v: ValueId) -> Option<Option<u64>> {
    let Type::Number(n) = f.values.get(v.index())?.ty.clone() else {
        return None;
    };
    if n.shape.rank() == 0 {
        return None;
    }
    let mut total = 1u64;
    for d in n.shape.dims() {
        match d {
            sky_types::ShapeDim::Known(k) => total = total.saturating_mul(*k),
            sky_types::ShapeDim::Var(_) => return Some(None),
        }
    }
    Some(Some(total))
}

fn last_dim(f: &Function, v: ValueId) -> Option<u64> {
    let Type::Number(n) = &f.values.get(v.index())?.ty else {
        return None;
    };
    match n.shape.dims().last()? {
        sky_types::ShapeDim::Known(k) => Some(*k),
        sky_types::ShapeDim::Var(_) => None,
    }
}

// ---------------------------------------------------------------------------
// Windows of calls.
// ---------------------------------------------------------------------------

/// Every operation nested in `op` (through its regions), once.
fn nested_ops(f: &Function, op: &OpData) -> Vec<OpId> {
    let mut all = Vec::new();
    for &r in &op.regions {
        all.extend(region_ops(f, r).0);
    }
    all
}

/// Whether `op`, or anything in its regions, or a callee, prints (or is not
/// a well-formed call, which is treated as if it did).
fn has_io(g: &Graph<'_>, f: &Function, op: &OpData) -> bool {
    let one = |o: &OpData| match &o.kind {
        OpKind::Print => true,
        OpKind::Call { callee } => match g.callee(callee) {
            Some(c) => g.effects[c].contains(Effect::Io),
            None => true,
        },
        _ => false,
    };
    one(op)
        || nested_ops(f, op)
            .iter()
            .any(|&o| f.ops.get(o.index()).is_some_and(one))
}

/// Whether `op` reads any of `values`, directly or inside a region.
fn reads_any(f: &Function, op: &OpData, values: &HashSet<ValueId>) -> bool {
    if values.is_empty() {
        return false;
    }
    if op.operands.iter().any(|v| values.contains(v)) {
        return true;
    }
    nested_ops(f, op).iter().any(|&o| {
        f.ops
            .get(o.index())
            .is_some_and(|n| n.operands.iter().any(|v| values.contains(v)))
    })
}

/// How a call is treated.
#[derive(Clone, PartialEq, Eq, Debug)]
enum CallClass {
    /// Started on another thread.
    Start,
    /// Expensive enough, if it were allowed to.
    Cheap(u64),
    /// Its callee prints.
    Prints,
    /// Its callee is recursive, or a chain deeper than the stack allows.
    Deep,
    /// Not a call to a function of the module with the right arity.
    Unknown,
}

fn classify(g: &mut Graph<'_>, op: &OpData, cfg: &Config) -> Option<(usize, CallClass)> {
    let OpKind::Call { callee } = &op.kind else {
        return None;
    };
    let Some(c) = g.callee(callee) else {
        return Some((usize::MAX, CallClass::Unknown));
    };
    if g.m.functions[c].params.len() != op.operands.len()
        || g.m.functions[c].results.len() != op.results.len()
    {
        return Some((c, CallClass::Unknown));
    }
    if g.effects[c].contains(Effect::Io) {
        return Some((c, CallClass::Prints));
    }
    if g.weight[c] > MAX_STACK_WEIGHT {
        return Some((c, CallClass::Deep));
    }
    let cost = g.cost_of(c);
    if cost < cfg.call_cost {
        return Some((c, CallClass::Cheap(cost)));
    }
    Some((c, CallClass::Start))
}

/// A window of calls found from the call at `ops[from]`.
struct Window {
    /// Positions in the block of the calls in the window that do not wait
    /// for another (`from` first), their classes and callees.
    calls: Vec<(usize, CallClass, usize)>,
    /// The role of each operation from `from` to the last call.
    roles: Vec<Role>,
    /// The position of the operation that prints and ended the window.
    stopped_by_io: Option<usize>,
}

/// Scans forward from the call at `ops[from]`. A call whose class satisfies
/// `counts` (and that waits for no other) is a call of the window.
fn window(
    g: &mut Graph<'_>,
    f: &Function,
    ops: &[OpId],
    from: usize,
    cfg: &Config,
    counts: impl Fn(&CallClass) -> bool,
) -> Window {
    // Results that are not there yet: of started calls and deferred operations.
    let mut pending: HashSet<ValueId> = HashSet::new();
    let mut calls = Vec::new();
    let mut roles = Vec::new();
    let mut stopped_by_io = None;
    for (j, &id) in ops.iter().enumerate().skip(from).take(MAX_WINDOW) {
        let Some(op) = f.ops.get(id.index()) else {
            break;
        };
        if op.kind.properties().terminator {
            break;
        }
        if has_io(g, f, op) {
            // The first operation of a window may itself be a call that
            // prints; one that follows ends the window.
            if j > from || !matches!(op.kind, OpKind::Call { .. }) {
                stopped_by_io = Some(j);
                break;
            }
        }
        if reads_any(f, op, &pending) {
            roles.push(Role::Deferred);
            pending.extend(op.results.iter().copied());
            continue;
        }
        match classify(g, op, cfg) {
            Some((_, CallClass::Unknown)) => break,
            Some((c, class)) if counts(&class) => {
                calls.push((j, class, c));
                roles.push(Role::Start);
                pending.extend(op.results.iter().copied());
            }
            _ => roles.push(Role::InPlace),
        }
    }
    if let Some(&(last, _, _)) = calls.last() {
        roles.truncate(last - from + 1);
    }
    Window {
        calls,
        roles,
        stopped_by_io,
    }
}

// ---------------------------------------------------------------------------
// The analysis proper.
// ---------------------------------------------------------------------------

fn run(m: &Module, cfg: &Config, mut notes: Option<&mut Vec<Note>>) -> Plan {
    let mut g = Graph::new(m);
    let mut functions = Vec::with_capacity(m.functions.len());
    let mut index = HashMap::new();
    for (fi, f) in m.functions.iter().enumerate() {
        index.entry(f.name.clone()).or_insert(fi);
        let mut plan = FnPlan::default();
        if cfg.is_parallel() {
            for b in blocks_in_order(f) {
                find_groups(&mut g, f, b, cfg, &mut plan, notes.as_deref_mut());
            }
        }
        if let Some(notes) = notes.as_deref_mut() {
            kernel_notes(f, cfg, notes);
        }
        if !plan.groups.is_empty() {
            plan.group_at = vec![u32::MAX; f.ops.len()];
            for (gi, group) in plan.groups.iter().enumerate() {
                if let Some(first) = group.ops.first() {
                    plan.group_at[first.index()] = gi as u32;
                }
            }
        }
        functions.push(plan);
    }
    Plan { functions, index }
}

/// The blocks reachable from the body, outermost first, then nested ones in
/// the order their operations appear.
fn blocks_in_order(f: &Function) -> Vec<BlockId> {
    let mut out = Vec::new();
    let mut seen_region = vec![false; f.regions.len()];
    let mut todo = vec![f.body];
    while let Some(r) = todo.pop() {
        let Some(region) = f.regions.get(r.index()) else {
            continue;
        };
        if std::mem::replace(&mut seen_region[r.index()], true) {
            continue;
        }
        let mut nested = Vec::new();
        for &b in &region.blocks {
            out.push(b);
            if let Some(block) = f.blocks.get(b.index()) {
                for &o in &block.ops {
                    if let Some(op) = f.ops.get(o.index()) {
                        nested.extend(op.regions.iter().copied());
                    }
                }
            }
        }
        // Reversed: `pop` takes the first one first.
        todo.extend(nested.into_iter().rev());
    }
    out
}

fn callee_name(g: &Graph<'_>, c: usize) -> String {
    g.m.functions
        .get(c)
        .map_or_else(|| "?".to_string(), |f| f.name.clone())
}

fn find_groups(
    g: &mut Graph<'_>,
    f: &Function,
    block: BlockId,
    cfg: &Config,
    plan: &mut FnPlan,
    mut notes: Option<&mut Vec<Note>>,
) {
    let Some(b) = f.blocks.get(block.index()) else {
        return;
    };
    let ops = &b.ops;
    let mut i = 0;
    while i < ops.len() {
        let Some(op) = f.ops.get(ops[i].index()) else {
            i += 1;
            continue;
        };
        let Some((c, class)) = classify(g, op, cfg) else {
            i += 1;
            continue;
        };
        if class == CallClass::Start {
            let w = window(g, f, ops, i, cfg, |c| *c == CallClass::Start);
            if w.calls.len() >= 2 {
                let last = w.calls.last().map_or(i, |c| c.0);
                if let Some(notes) = notes.as_deref_mut() {
                    notes.push(group_note(g, f, ops, &w, cfg));
                }
                plan.groups.push(CallGroup {
                    block,
                    ops: ops[i..=last].to_vec(),
                    roles: w.roles,
                });
                i = last + 1;
                continue;
            }
        }
        // Not started with a partner: say why, when another call follows.
        if let Some(notes) = notes.as_deref_mut() {
            if let Some((n, next)) = why_not_calls(g, f, ops, i, c, &class, cfg) {
                notes.push(n);
                i = next;
                continue;
            }
        }
        i += 1;
    }
}

fn span_of(f: &Function, id: OpId) -> Span {
    f.ops.get(id.index()).map_or(Span::point(0), |o| o.span)
}

fn group_note(g: &Graph<'_>, f: &Function, ops: &[OpId], w: &Window, cfg: &Config) -> Note {
    let first = span_of(f, ops[w.calls[0].0]);
    let span = w
        .calls
        .iter()
        .fold(first, |s, c| s.join(span_of(f, ops[c.0])));
    let mut names: Vec<String> = Vec::new();
    for c in &w.calls {
        let n = format!("`{}`", callee_name(g, c.2));
        if !names.contains(&n) {
            names.push(n);
        }
    }
    // A callee always can fail (nested calls can exhaust the call depth),
    // so the resolution rule is always stated.
    let mut because = vec![
        format!(
            "{} {} no `print` (directly or in what {}), and no call uses another's result, so their order cannot show in the output",
            names.join(", "),
            if names.len() == 1 { "has" } else { "have" },
            if names.len() == 1 { "it calls" } else { "they call" },
        ),
        format!(
            "each is expensive enough to start elsewhere: at least {} estimated operations (the threshold is {})",
            w.calls.iter().map(|c| callee_cost(g, c.2)).min().unwrap_or(0),
            cfg.call_cost
        ),
    ];
    because.push(
        "any of them may fail: every call is awaited and the error of the earliest one in program order is reported, as in a sequential run"
            .to_string(),
    );
    Note {
        rule: CALLS_RULE,
        span,
        headline: format!(
            "these {} calls are independent (pure): run concurrently on up to {} threads",
            w.calls.len(),
            cfg.threads
        ),
        because,
    }
}

fn callee_cost(g: &Graph<'_>, c: usize) -> u64 {
    g.cost.get(c).copied().flatten().unwrap_or(0)
}

/// Why the call at `ops[i]` is not run concurrently with a call after it, if
/// that is something a reader might expect, and where the scan goes on. Cheap
/// calls are not worth a note: a program full of small calls would drown in
/// them.
fn why_not_calls(
    g: &mut Graph<'_>,
    f: &Function,
    ops: &[OpId],
    i: usize,
    c: usize,
    class: &CallClass,
    cfg: &Config,
) -> Option<(Note, usize)> {
    let name = |g: &Graph<'_>, c: usize| callee_name(g, c);
    let span0 = span_of(f, ops[i]);
    // The first later call in the block that would be started, and where.
    let later_start = |g: &mut Graph<'_>| {
        (i + 1..ops.len()).find(|&j| {
            f.ops
                .get(ops[j].index())
                .and_then(|o| classify(g, o, cfg))
                .is_some_and(|(_, cl)| cl == CallClass::Start)
        })
    };
    match class {
        CallClass::Prints => {
            later_start(g)?;
            let n = name(g, c);
            Some((
                Note {
                    rule: NOT_PARALLEL_RULE,
                    span: span0,
                    headline: format!(
                        "the call to `{n}` is not run concurrently with the calls after it"
                    ),
                    because: vec![format!(
                        "`{n}` prints, and output order is behaviour: it stays in program order"
                    )],
                },
                i + 1,
            ))
        }
        CallClass::Deep => {
            let w = window(g, f, ops, i, cfg, |_| true);
            if w.calls.len() < 2 {
                return None;
            }
            let last = w.calls.last().map_or(i, |x| x.0);
            let span = w
                .calls
                .iter()
                .fold(span0, |s, x| s.join(span_of(f, ops[x.0])));
            let n = name(g, c);
            Some((
                Note {
                    rule: NOT_PARALLEL_RULE,
                    span,
                    headline: format!(
                        "these {} independent calls are not run concurrently",
                        w.calls.len()
                    ),
                    because: vec![format!(
                        "`{n}` is recursive (or a chain of calls deeper than {MAX_STACK_WEIGHT}): a worker starts on a new stack, and whether a run stops with \"the program nests too deeply\" must not depend on the thread count"
                    )],
                },
                last + 1,
            ))
        }
        CallClass::Start => {
            // A call and, after an operation that prints, another: the
            // print keeps them apart.
            let w = window(g, f, ops, i, cfg, |cl| *cl == CallClass::Start);
            let p = w.stopped_by_io?;
            if w.calls.len() != 1 {
                return None;
            }
            let after = (p + 1..ops.len()).find(|&j| {
                f.ops
                    .get(ops[j].index())
                    .and_then(|o| classify(g, o, cfg))
                    .is_some_and(|(_, cl)| cl == CallClass::Start)
            })?;
            let span = span0.join(span_of(f, ops[after]));
            Some((
                Note {
                    rule: NOT_PARALLEL_RULE,
                    span,
                    headline: "these calls are not run concurrently".to_string(),
                    because: vec![
                        "a `print` (or a call that prints) sits between them, and output order must not change: had the first call failed, the print would not have happened".to_string(),
                    ],
                },
                i + 1,
            ))
        }
        CallClass::Cheap(_) | CallClass::Unknown => None,
    }
}

// ---------------------------------------------------------------------------
// Kernels.
// ---------------------------------------------------------------------------

fn kernel_notes(f: &Function, cfg: &Config, notes: &mut Vec<Note>) {
    let mut analysis: Option<exact::Analysis> = None;
    for id in region_ops(f, f.body).0 {
        let Some(op) = f.ops.get(id.index()) else {
            continue;
        };
        let (what, threshold, may_fail) = match &op.kind {
            OpKind::Fused => {
                let a = analysis.get_or_insert_with(|| exact::analyze(f));
                let failing = fused_failing(f, op, a);
                // The threshold is in element-operations.
                let ops = fused_body_len(f, op);
                (
                    "elementwise kernel".to_string(),
                    cfg.fused_work.div_ceil(ops.max(1)),
                    Some(failing),
                )
            }
            OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div | OpKind::Neg
                if op.results.first().is_some_and(|&r| shaped(f, r).is_some()) =>
            {
                (
                    format!("elementwise `{}`", op.kind.mnemonic()),
                    cfg.array_elements,
                    None,
                )
            }
            _ => continue,
        };
        let n = op.results.first().and_then(|&r| elements(f, r));
        let fail_text = match (&may_fail, &op.kind) {
            (Some(names), _) if names.is_empty() => {
                "no operation in it can fail (proved by range analysis), so the chunks need no error handling".to_string()
            }
            (Some(names), _) => format!(
                "{} of its operations can fail ({}): if chunks fail, the failing operation that comes first in program order is reported, at its lowest failing element, which is the error a sequential run gives",
                names.len(),
                names.join(", ")
            ),
            (None, _) => "its only failure is an overflow (or a zero divisor) of the whole operation, found after the elements are computed and reported the same way whichever thread found it".to_string(),
        };
        let can_fail = may_fail.as_ref().is_none_or(|n| !n.is_empty());
        let span = op.span;
        let t = cfg.threads;
        let small = |n: u64| n < threshold as u64;
        let note = if !cfg.is_parallel() {
            Note {
                rule: NOT_PARALLEL_RULE,
                span,
                headline: format!("this {what} is not parallelized: threads = 1"),
                because: vec![
                    "`--threads` or SKY_THREADS asks for sequential execution".to_string()
                ],
            }
        } else if let Some(n) = n.filter(|&n| small(n)) {
            Note {
                rule: NOT_PARALLEL_RULE,
                span,
                headline: format!(
                    "this {what} over {n} elements is not parallelized{}",
                    if can_fail { ": it may fail and is small" } else { ": it is small" }
                ),
                because: vec![format!(
                    "{n} elements is under the threshold of {threshold}: splitting it across threads costs more than it saves"
                )],
            }
        } else if let Some(n) = n {
            Note {
                rule: KERNEL_RULE,
                span,
                headline: format!("this {what} over {} elements runs on {t} threads", short_count(n)),
                because: vec![
                    "every element is independent, so the array is split into contiguous chunks (at most four per thread); the result is bit-identical to the sequential one whatever the split".to_string(),
                    "it does not print or call, so no output can change order".to_string(),
                    fail_text,
                ],
            }
        } else {
            Note {
                rule: KERNEL_RULE,
                span,
                headline: format!(
                    "this {what} runs on {t} threads once it has at least {threshold} elements"
                ),
                because: vec![
                    "its size is known only when the program runs; below the threshold splitting costs more than it saves".to_string(),
                    "every element is independent, so the result is bit-identical to the sequential one whatever the split".to_string(),
                    fail_text,
                ],
            }
        };
        notes.push(note);
    }
}

/// How many operations the body of a `fused` has, without its `yield`.
fn fused_body_len(f: &Function, op: &OpData) -> usize {
    op.regions
        .first()
        .map(|&r| region_ops(f, r).0.len().saturating_sub(1))
        .unwrap_or(0)
}

/// The mnemonics of the operations of a `fused` body that can fail and are
/// not proved unable to.
fn fused_failing(f: &Function, op: &OpData, a: &exact::Analysis) -> Vec<String> {
    let mut names = Vec::new();
    for &r in &op.regions {
        let (ops, _) = region_ops(f, r);
        for id in ops {
            let Some(o) = f.ops.get(id.index()) else {
                continue;
            };
            if o.kind.properties().effects.contains(Effect::Fail) && a.proof(id).is_none() {
                let name = format!("`{}`", o.kind.mnemonic());
                if !names.contains(&name) {
                    names.push(name);
                }
            }
        }
    }
    names
}

/// `1000000` as `1e6`, small counts as they are.
fn short_count(n: u64) -> String {
    if n >= 1_000_000 && n % 1_000_000 == 0 && (n / 1_000_000).to_string().len() == 1 {
        format!("{}e6", n / 1_000_000)
    } else if n >= 1_000 && n % 1_000 == 0 && n / 1_000 < 10 {
        format!("{}e3", n / 1_000)
    } else {
        n.to_string()
    }
}
