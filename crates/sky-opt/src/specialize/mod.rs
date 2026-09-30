//! Specialization: a call whose arguments have facts the callee's signature
//! lacks gets a clone of the callee that knows them.
//!
//! ```text
//! def dot(a: Vector[N], b: Vector[N]) -> Scalar: ... while i < len(a): ...
//! dot([1, 2, 3], [4, 5, 6])      # a call with N = 3
//!   -->  call @"dot[N=3]"        # a clone where len(a) is 3 and the loop runs 3 times
//! ```
//!
//! This is ROADMAP Phase 4's "runtime specialisation on shapes and values"
//! done at compile time for what a call site proves, plus a hook
//! ([`specialize_for`], [`SpecializationCache`]) for an engine that learns
//! the values while it runs. See `docs/design/ir.md`, "Specialization".
//!
//! # What is specialized
//!
//! For each `call` in the module (in function order, program order), the
//! facts of [`facts`]:
//!
//! * **shape**: `Vector[N]` called with a `Vector[3]`: `N = 3`;
//! * **value**: a scalar (or Bool) parameter called with a constant: a
//!   literal, or a nonzero number the exactness analysis proves is a single
//!   `f64` (`2 * 3`);
//! * **sign**: a scalar parameter called with a value proven strictly
//!   positive or negative. Kept only if it decides something (a comparison,
//!   or a check the analysis then proves cannot fail);
//! * **type**: an unannotated (`unknown`) parameter called with a value whose
//!   type is known exactly: `fib(30)` passes a dimensionless number. See
//!   "Types" below.
//!
//! The clone is a copy of the callee with the facts substituted (sizes in
//! every type; constant parameters removed from the signature and replaced
//! by a constant in the body), simplified by [`clean`] and its steps (`len`
//! of a static shape, decided comparisons, `[..][k]`, constant folding,
//! dead code, [`unroll`]ing loops of at most 8 constant iterations), and the
//! call is retargeted at it. The clone is appended to the module. The
//! generic function stays for the calls the pass did not specialize; when
//! none is left (nothing reachable from a function nobody calls reaches it)
//! it is removed, so a program does not carry both.
//!
//! Then the clone's own calls are looked at the same way (a `Vector[3]`
//! passed on is a fact for the next callee), until nothing is new.
//!
//! # Types
//!
//! An unannotated parameter is `unknown` in SIR, so every operation on it is
//! a run-time check, and the native engine runs it on boxed values. A call
//! that passes a value of known type (a number of a dimension and a concrete
//! shape, a Bool, a string) gets a clone whose parameter has that type, and
//! [`retype`] types the body again from it (`n - 1` is a number, `n < 2` a
//! Bool, and the checks that were only there to find that out go):
//!
//! ```text
//! def fib(n): if n < 2: return n ; return fib(n - 1) + fib(n - 2)
//! fib(30)          -->  call @"fib[n:Scalar]"    // n is a dimensionless number
//! ```
//!
//! The rules:
//!
//! * **Validity.** The clone is for calls whose argument types are known
//!   *exactly*; the generic function stays for every other call (an
//!   argument of unknown type, a tagged or symbolic-size array, a
//!   representation other than the default). The clone is made only if the
//!   verifier accepts it: an operation whose operands the language's rules
//!   reject (`n < 2` for a Bool `n`, `n + 1 m` for a dimensionless `n`) is a
//!   run-time error in the generic function, so the call that would reach it
//!   keeps the generic function and its error, unchanged. In a clone that is
//!   made, no check that could fail was removed: only checks that the
//!   argument's type already answers.
//! * **Dimensions.** A clone is per exact type, dimension included: `fib(3)`
//!   and `fib(3 m)` are `fib[n:Scalar]` and `fib[n:Scalar(length)]`. A
//!   dimension-polymorphic clone is not made: native code carries a scalar's
//!   dimension in its type, and each operation's dimension rule (`n + 1`
//!   needs a dimensionless `n`, `n * n` squares the dimension) is decided per
//!   dimension.
//! * **Deduplication.** The name is the identity (`f[n:Scalar,b:Bool]`), so a
//!   recursive call, or a second caller, with the same argument types finds
//!   the clone by name and recursion ends. The limits below apply.
//! * **Results.** The clone's declared result is the type of what it
//!   returns, computed to a fixed point over all the clones of the module
//!   together, so mutual recursion (`even`, `odd`) types too. Calls from
//!   outside the clone keep their `unknown` result type: the generic
//!   function's callers are not retyped, only boxed on return.
//! * **Not worth it.** A clone is dropped if it removes no run-time type
//!   check and passes no known type on to a call.
//!
//! # Names, duplicates, limits
//!
//! A clone is named `f[N=3]`, `f[x=2]`, `f[N=3,x>0]`: [`facts`] documents the
//! grammar, which is deterministic. The name is the identity: a second call
//! with the same facts reuses the clone (and says so, `SPEC-002`). Bounds,
//! all checked before a clone is made:
//!
//! * [`MAX_CLONES_PER_FUNCTION`] = 8 clones of one function (of `f`, counting
//!   `f[N=3]`, `f[x=2]` and `f[N=3,x=2]` together);
//! * [`MAX_CLONE_OPS`] = 300: no clone of a body with more operations;
//! * [`MAX_MODULE_CLONES`] = 64 clones, and [`MAX_CLONED_OPS`] = 4096
//!   operations cloned, in one module;
//! * unrolling adds at most [`unroll::MAX_UNROLL_OPS`] operations per clone,
//!   [`unroll::MAX_TRIPS`] iterations per loop.
//!
//! A call that would exceed one keeps calling the generic function, and the
//! first one of each function says so (`SPEC-004`).
//!
//! **Recursion terminates**: every clone counts against its function's
//! limit, and a recursive call with the facts of the clone it is in (`f[N=3]`
//! calling `f` on the same `Vector[3]`) finds itself by name. A recursion on
//! a constant (`fact(n - 1)` from `fact(5)`) makes one clone per value until
//! the base case or the limit (whichever is first); past the limit the
//! deepest clone calls the generic function, which is what it did before.
//!
//! # Semantics do not change
//!
//! A specialization is the same code with values known, so outputs, errors
//! and diagnostics are identical:
//!
//! * every operation the clone keeps has the kind, span and operand spans it
//!   had; the clone's constants are typed as the parameter was;
//! * a step only removes or replaces an operation when it computes what the
//!   interpreter would (folding never folds what fails), or proves it cannot
//!   fail ([`sky_ir::exact::Proof`]); a failure the generic function would
//!   report is still reported by the clone, at the same operation;
//! * the arguments of a call are still evaluated by the caller (they are
//!   operations before the call); only the call's operand list loses the
//!   constants the clone bakes in;
//! * call depth is unchanged (a clone calls what its function called), and a
//!   "recursion is too deep" diagnostic names `f`, not `f[N=3]`
//!   ([`sky_ir::source_name`]).
//!
//! The one thing trusted is what the pipeline trusts everywhere: the static
//! type of a value is right (a `Vector[3]` has three entries).
//!
//! # Explanations
//!
//! * `SPEC-001` "specialized `f` for N = 3 at this call: 4 checks removed",
//!   at the call ([`specialize_headline`] words it for `sky check
//!   --explain`); the notes list the facts and what they simplified;
//! * `SPEC-002` a call uses an existing clone;
//! * `SPEC-003` a loop of a clone was unrolled;
//! * `SPEC-004` a call was not specialized because of a limit;
//! * `SPEC-005` "specialized `fib` for n: Scalar (dimensionless): 3 run-time
//!   type checks removed", for a clone with a type fact.
//!
//! User code that becomes dead in a clone (`spare = x / 2` once `x` is a
//! constant and `spare` is unused) is reported in `dce`'s rules and words,
//! filed under this pass, so specializing never removes the user's code
//! silently. What folding does inside a clone is summarized in `SPEC-001`
//! ("3 operations folded to constants") and not listed.
//!
//! "Checks removed" counts operations that can fail at run time (division,
//! overflow, index, `len`, contracts, ...) that the clone no longer has or
//! that the exactness analysis now proves cannot fail, compared with the
//! generic function.

mod clean;
mod copy;
mod facts;
mod retype;
mod runtime;
pub mod unroll;

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet, VecDeque};

use sky_diag::Span;
use sky_ir::exact::{self, Range};
use sky_ir::{
    source_name, verify_function, Explanation, Function, Module, OpId, OpKind, ValueDef, ValueId,
};
use sky_types::{Shape, ShapeDim};

use self::clean::{checks, cleanup, Cleanup};
use self::facts::{clone_name, derive, type_words, Fact, Kind};
use self::retype::{apply, dynamic_checks, infer, typed_arguments, Env, Ty};
pub use self::runtime::{specialize_for, specialized_name, Specialization, SpecializationCache};
use self::unroll::MAX_UNROLL_OPS;
use crate::rewrite::Rewrite;
use crate::util::{reachable_ops, replace_uses};

/// The most clones of one function (counting every fact set together).
pub const MAX_CLONES_PER_FUNCTION: usize = 8;

/// The largest function body (operations, nested ones included) that is
/// cloned.
pub const MAX_CLONE_OPS: usize = 300;

/// The most clones the pass makes in one module.
pub const MAX_MODULE_CLONES: usize = 64;

/// The most operations, in total, that the clones of one module copy (the
/// sizes of the generic bodies).
pub const MAX_CLONED_OPS: usize = 4096;

/// A call was specialized.
pub const RULE_SPECIALIZED: &str = "SPEC-001";
/// A call uses an existing specialization.
pub const RULE_REUSED: &str = "SPEC-002";
/// A loop of a specialization was unrolled.
pub const RULE_UNROLLED: &str = "SPEC-003";
/// A call was not specialized because of a limit.
pub const RULE_LIMIT: &str = "SPEC-004";
/// A call was specialized on the types of its arguments.
pub const RULE_RETYPED: &str = "SPEC-005";

const PASS: &str = "specialize";

/// A specialized function, built and simplified.
pub(crate) struct Built {
    pub(crate) func: Function,
    pub(crate) cleanup: Cleanup,
    /// Run-time checks of the generic function, and of this clone.
    pub(crate) checks_before: usize,
    pub(crate) checks_after: usize,
    /// What the sign facts bought: comparisons decided plus checks proven.
    pub(crate) sign_gain: usize,
    /// Run-time type checks of the generic function ([`retype::dynamic_checks`]),
    /// and what the type facts bought: the checks the first typing removed,
    /// plus the calls that pass a known type to an unknown parameter. Zero
    /// without a type fact.
    pub(crate) type_checks_before: usize,
    pub(crate) type_gain: usize,
}

/// The number of operations of `f`, nested ones included.
fn size(f: &Function) -> usize {
    reachable_ops(f).len()
}

/// Copies `base` as `name`, with `facts` substituted, and simplifies it.
/// With `keep_params` a constant parameter stays in the signature (unused),
/// so the clone can be called exactly as `base` is.
pub(crate) fn build(
    base: &Function,
    name: String,
    facts: &[Fact],
    keep_params: bool,
    funcs: &[Function],
) -> Built {
    let mut f = base.clone();
    f.name = name;
    let checks_before = checks(base, &HashMap::new());

    // Sizes: every type mentioning a solved variable.
    let sizes: HashMap<u64, u64> = facts
        .iter()
        .filter_map(|x| match &x.kind {
            Kind::Size { var, n } => Some((var.id(), *n)),
            _ => None,
        })
        .collect();
    if !sizes.is_empty() {
        for v in &mut f.values {
            substitute(&mut v.ty, &sizes);
        }
        for t in f.params.iter_mut().chain(f.results.iter_mut()) {
            substitute(t, &sizes);
        }
    }

    // Constants and ranges of parameters.
    let params: Vec<ValueId> = f.param_values().to_vec();
    let entry = f.entry();
    let mut ranges: HashMap<ValueId, Range> = HashMap::new();
    let mut dropped = Vec::new();
    let mut inserted = 0;
    for fact in facts {
        match fact.kind {
            Kind::Const { param, value } => {
                let p = params[param];
                let ty = f.ty(p).clone();
                let kind = match value {
                    facts::Const::Rational(q) => OpKind::ConstRational(q),
                    facts::Const::Float(bits) => OpKind::ConstFloat { bits },
                    facts::Const::Bool(b) => OpKind::ConstBool(b),
                };
                let op = OpId(f.ops.len() as u32);
                let v = ValueId(f.values.len() as u32);
                // Unnamed: a named value that disappears is reported by
                // `dce` as user code removed, and this one is not the user's.
                f.values.push(sky_ir::ValueData {
                    ty,
                    def: ValueDef::OpResult { op, index: 0 },
                    name: None,
                });
                f.ops.push(sky_ir::OpData {
                    kind,
                    operands: Vec::new(),
                    results: vec![v],
                    regions: Vec::new(),
                    span: f.span,
                    operand_spans: Vec::new(),
                    parent: entry,
                });
                f.blocks[entry.index()].ops.insert(inserted, op);
                inserted += 1;
                replace_uses(&mut f, p, v);
                if !keep_params {
                    dropped.push(param);
                }
            }
            Kind::Sign { param, .. } => {
                if let Some(r) = fact.range() {
                    ranges.insert(params[param], r);
                }
            }
            Kind::Type { param, ref ty } => {
                f.params[param] = ty.clone();
                f.values[params[param].index()].ty = ty.clone();
            }
            Kind::Size { .. } => {}
        }
    }
    if !dropped.is_empty() {
        drop_params(&mut f, &dropped);
    }

    // Types: the parameters have theirs, so type the body again.
    let typed = facts.iter().any(Fact::is_type);
    let type_checks_before = if typed { dynamic_checks(base) } else { 0 };
    let mut typed_ok = true;
    if typed {
        let results = HashMap::new();
        let env = Env {
            funcs,
            results: &results,
        };
        let inferred = infer(&f, &env);
        apply(&mut f, &inferred, None);
        // The simplifications below assume well-typed operations: `dce`
        // drops a `max` of a length and a number because it "cannot fail",
        // where the generic function fails on it. So the language's rules
        // judge the typed body before anything simplifies it.
        typed_ok = verify_function(funcs, &f).is_ok();
    }

    let mut budget = MAX_UNROLL_OPS;
    let cleaned = cleanup(&mut f, &ranges, &mut budget);
    let checks_after = checks(&f, &ranges);
    let sign_gain = if ranges.is_empty() {
        0
    } else {
        cleaned.comparisons + checks(&f, &HashMap::new()).saturating_sub(checks_after)
    };
    let type_gain = if typed && typed_ok {
        type_checks_before.saturating_sub(dynamic_checks(&f)) + typed_arguments(&f, funcs)
    } else {
        0
    };
    Built {
        func: f,
        cleanup: cleaned,
        checks_before,
        checks_after,
        sign_gain,
        type_checks_before,
        type_gain,
    }
}

/// Replaces the size variables of `sizes` in a type.
fn substitute(t: &mut sky_ir::Type, sizes: &HashMap<u64, u64>) {
    let sky_ir::Type::Number(n) = t else { return };
    if !n
        .shape
        .dims()
        .iter()
        .any(|d| matches!(d, ShapeDim::Var(v) if sizes.contains_key(&v.id())))
    {
        return;
    }
    let dims = n
        .shape
        .dims()
        .iter()
        .map(|d| match d {
            ShapeDim::Var(v) => sizes
                .get(&v.id())
                .map_or_else(|| d.clone(), |k| ShapeDim::Known(*k)),
            known => known.clone(),
        })
        .collect();
    n.shape = Shape::new(dims);
}

/// Removes parameters `drop` (indices) from the signature and the entry
/// block, and renumbers the arguments that stay.
fn drop_params(f: &mut Function, drop: &[usize]) {
    let entry = f.entry();
    let keep = |i: &usize| !drop.contains(i);
    let args: Vec<ValueId> = f
        .block(entry)
        .args
        .iter()
        .enumerate()
        .filter(|(i, _)| keep(i))
        .map(|(_, v)| *v)
        .collect();
    f.params = f
        .params
        .iter()
        .enumerate()
        .filter(|(i, _)| keep(i))
        .map(|(_, t)| t.clone())
        .collect();
    for (i, v) in args.iter().enumerate() {
        f.values[v.index()].def = ValueDef::BlockArg {
            block: entry,
            index: i as u32,
        };
    }
    f.blocks[entry.index()].args = args;
}

/// What to do about one call.
enum Plan {
    /// Use the existing function of this name.
    Reuse { name: String, facts: Vec<Fact> },
    /// Make it.
    Create {
        name: String,
        facts: Vec<Fact>,
        built: Box<Built>,
    },
    /// Not specialized: the reason, when it is worth saying.
    Skip(Option<(&'static str, String)>),
}

struct Specializer<'m> {
    m: &'m mut Module,
    out: Vec<Rewrite>,
    /// Clones made so far, by the generic function's source name.
    families: HashMap<String, usize>,
    clones: usize,
    cloned_ops: usize,
    /// Names built and found not worth keeping.
    rejected: HashSet<String>,
    /// (function, reason) pairs already reported for `SPEC-004`.
    said: HashSet<(String, &'static str)>,
    queue: VecDeque<usize>,
    /// The clones with a type fact this run made, to settle their result
    /// types together once every call has been looked at.
    typed: Vec<TypedClone>,
}

/// A clone with a type fact, and where its explanation is.
struct TypedClone {
    name: String,
    /// The index of its `SPEC-005` record in the pass's output.
    note: usize,
    /// The run-time type checks of the generic function.
    checks_before: usize,
}

impl<'m> Specializer<'m> {
    fn new(m: &'m mut Module) -> Self {
        // Clones that already exist (a second run of the pipeline) count.
        let mut families: HashMap<String, usize> = HashMap::new();
        let mut clones = 0;
        for f in &m.functions {
            if f.name.contains('[') {
                *families
                    .entry(source_name(&f.name).to_string())
                    .or_default() += 1;
                clones += 1;
            }
        }
        Specializer {
            m,
            out: Vec::new(),
            families,
            clones,
            cloned_ops: 0,
            rejected: HashSet::new(),
            said: HashSet::new(),
            queue: VecDeque::new(),
            typed: Vec::new(),
        }
    }

    fn run(&mut self, start: Vec<usize>) {
        self.queue.extend(start);
        while let Some(i) = self.queue.pop_front() {
            self.scan(i);
        }
        self.settle();
    }

    /// Looks at every call of function `i`.
    fn scan(&mut self, i: usize) {
        let an = exact::analyze(&self.m.functions[i]);
        let calls: Vec<OpId> = {
            let f = &self.m.functions[i];
            reachable_ops(f)
                .into_iter()
                .filter(|o| matches!(f.op(*o).kind, OpKind::Call { .. }))
                .collect()
        };
        for op in calls {
            self.call(i, op, &an);
        }
    }

    fn call(&mut self, ci: usize, op: OpId, an: &exact::Analysis) {
        let (span, callee_name) = {
            let data = self.m.functions[ci].op(op);
            let OpKind::Call { callee } = &data.kind else {
                return;
            };
            (data.span, callee.clone())
        };
        let Some(callee_idx) = self.m.functions.iter().position(|g| g.name == callee_name) else {
            return;
        };
        let facts = derive(
            &self.m.functions[ci],
            an,
            self.m.functions[ci].op(op),
            &self.m.functions[callee_idx],
        );
        if facts.is_empty() {
            return;
        }
        match self.plan(callee_idx, facts) {
            Plan::Skip(None) => {}
            Plan::Skip(Some((what, why))) => {
                let family = source_name(&callee_name).to_string();
                if self.said.insert((family.clone(), what)) {
                    self.out.push(Rewrite {
                        pass: PASS,
                        rule: RULE_LIMIT,
                        span,
                        before: family,
                        after: what.to_string(),
                        because: vec![why],
                    });
                }
            }
            Plan::Reuse { name, facts } => {
                self.retarget(ci, op, &name, &facts);
                self.out.push(Rewrite {
                    pass: PASS,
                    rule: RULE_REUSED,
                    span,
                    before: source_name(&callee_name).to_string(),
                    after: name,
                    because: facts.iter().map(|f| f.why.clone()).collect(),
                });
            }
            Plan::Create { name, facts, built } => {
                let base_ops = size(&self.m.functions[callee_idx]);
                let family = source_name(&callee_name).to_string();
                *self.families.entry(family.clone()).or_default() += 1;
                self.clones += 1;
                self.cloned_ops += base_ops;
                self.retarget(ci, op, &name, &facts);
                let notes = self.notes(span, &family, &name, &facts, &built);
                if facts.iter().any(Fact::is_type) {
                    self.typed.push(TypedClone {
                        name: name.clone(),
                        note: self.out.len(),
                        checks_before: built.type_checks_before,
                    });
                }
                self.out.extend(notes);
                self.m.functions.push(built.func);
                self.queue.push_back(self.m.functions.len() - 1);
            }
        }
    }

    /// Decides what to do about a call of function `callee_idx` with `facts`.
    fn plan(&mut self, callee_idx: usize, mut facts: Vec<Fact>) -> Plan {
        let callee = &self.m.functions[callee_idx];
        let base_ops = size(callee);
        let family = source_name(&callee.name).to_string();
        loop {
            let name = clone_name(&self.m.functions[callee_idx].name, &facts);
            if self.m.functions.iter().any(|g| g.name == name) {
                return Plan::Reuse { name, facts };
            }
            if self.rejected.contains(&name) {
                // A type fact that made no valid or useful clone: the
                // other facts, if any, are what is left.
                if facts.iter().any(Fact::is_type) {
                    facts.retain(|f| !f.is_type());
                    if !facts.is_empty() {
                        continue;
                    }
                }
                return Plan::Skip(None);
            }
            let in_family = self.families.get(&family).copied().unwrap_or(0);
            if base_ops > MAX_CLONE_OPS {
                return Plan::Skip(Some((
                    "too big",
                    format!(
                        "`{family}` has {base_ops} operations; only bodies of at most \
                         {MAX_CLONE_OPS} are specialized"
                    ),
                )));
            }
            if in_family >= MAX_CLONES_PER_FUNCTION {
                return Plan::Skip(Some((
                    "too many",
                    format!(
                        "`{family}` already has {in_family} specializations, the most one \
                         function gets"
                    ),
                )));
            }
            if self.clones >= MAX_MODULE_CLONES || self.cloned_ops + base_ops > MAX_CLONED_OPS {
                return Plan::Skip(Some((
                    "module full",
                    format!(
                        "this program already has {} specializations of {} operations, the \
                         most one program gets",
                        self.clones, self.cloned_ops
                    ),
                )));
            }
            let built = build(
                &self.m.functions[callee_idx],
                name.clone(),
                &facts,
                false,
                &self.m.functions,
            );
            if facts.iter().any(Fact::is_type) {
                // Kept only if typing bought something and the result is
                // valid IR: the verifier is the judge of "the language's
                // rules accept these operand types".
                let valid =
                    built.type_gain > 0 && verify_function(&self.m.functions, &built.func).is_ok();
                if !valid {
                    self.rejected.insert(name);
                    facts.retain(|f| !f.is_type());
                    if facts.is_empty() {
                        return Plan::Skip(None);
                    }
                    continue;
                }
            }
            let has_sign = facts.iter().any(Fact::is_sign);
            if has_sign && built.sign_gain == 0 {
                // The sign decided nothing: without it, the same clone as
                // any other call with the remaining facts.
                facts.retain(|f| !f.is_sign());
                if facts.is_empty() {
                    self.rejected.insert(name);
                    return Plan::Skip(None);
                }
                continue;
            }
            return Plan::Create {
                name,
                facts,
                built: Box::new(built),
            };
        }
    }

    /// Points call `op` of function `ci` at clone `name`, and drops the
    /// operands the clone bakes in.
    fn retarget(&mut self, ci: usize, op: OpId, name: &str, facts: &[Fact]) {
        let dropped: Vec<usize> = facts
            .iter()
            .filter_map(|f| match f.kind {
                Kind::Const { param, .. } => Some(param),
                _ => None,
            })
            .collect();
        let data = &mut self.m.functions[ci].ops[op.index()];
        data.kind = OpKind::Call {
            callee: name.to_string(),
        };
        if dropped.is_empty() {
            return;
        }
        let spans_known = data.operand_spans.len() == data.operands.len();
        let keep = |i: &usize| !dropped.contains(i);
        data.operands = data
            .operands
            .iter()
            .enumerate()
            .filter(|(i, _)| keep(i))
            .map(|(_, v)| *v)
            .collect();
        if spans_known {
            data.operand_spans = data
                .operand_spans
                .iter()
                .enumerate()
                .filter(|(i, _)| keep(i))
                .map(|(_, s)| *s)
                .collect();
        }
    }

    /// The explanations of a new clone: the specialization at the call, and
    /// one per loop unrolled in it.
    fn notes(
        &self,
        span: Span,
        family: &str,
        name: &str,
        facts: &[Fact],
        built: &Built,
    ) -> Vec<Rewrite> {
        let removed = built.checks_before.saturating_sub(built.checks_after);
        let c = &built.cleanup;
        let typed = facts.iter().any(Fact::is_type);
        let mut because = vec![if typed {
            // The count is settled with the results (`Specializer::settle`).
            type_checks_line(family, name, 0)
        } else {
            format!(
                "{removed} run-time check{} removed: operations that can fail in `{family}` and \
                 cannot in `{name}`",
                if removed == 1 { "" } else { "s" }
            )
        }];
        because.extend(facts.iter().map(|f| f.why.clone()));
        let mut did = Vec::new();
        if c.lengths > 0 {
            did.push(count(c.lengths, "`len` is", "`len`s are") + " read from the now-known shape");
        }
        if c.comparisons > 0 {
            did.push(count(c.comparisons, "comparison", "comparisons") + " decided by the sign");
        }
        if c.branches > 0 {
            did.push(count(c.branches, "branch", "branches") + " decided");
        }
        if c.entries > 0 {
            did.push(count(c.entries, "index", "indices") + " of a literal read directly");
        }
        if c.folded > 0 {
            did.push(count(c.folded, "operation", "operations") + " folded to constants");
        }
        if !did.is_empty() {
            because.push(format!("in `{name}`: {}", did.join(", ")));
        }
        because.push(
            "the specialization is the same code with these values known: outputs, errors and \
             their spans are unchanged"
                .to_string(),
        );
        let mut out = vec![Rewrite {
            pass: PASS,
            rule: if typed {
                RULE_RETYPED
            } else {
                RULE_SPECIALIZED
            },
            span,
            before: family.to_string(),
            after: name.to_string(),
            because,
        }];
        for u in &c.loops {
            out.push(Rewrite {
                pass: PASS,
                rule: RULE_UNROLLED,
                span: u.span,
                before: name.to_string(),
                after: u.trips.to_string(),
                because: vec![
                    format!(
                        "in `{name}` the loop's condition is a constant on each check ({} \
                         iteration{})",
                        u.trips,
                        if u.trips == 1 { "" } else { "s" }
                    ),
                    "its operations run in the same order with the same spans, so errors and \
                     output are unchanged"
                        .to_string(),
                ],
            });
        }
        out.extend(c.removed.iter().map(|r| Rewrite {
            pass: PASS,
            ..r.clone()
        }));
        out
    }
}

/// The first reason of a `SPEC-005` record: how many run-time type checks
/// the clone has fewer than the generic function.
fn type_checks_line(family: &str, name: &str, removed: usize) -> String {
    format!(
        "{removed} run-time type check{} removed: operations of `{family}` that had to find \
         out at run time what a value is, and know it in `{name}`",
        if removed == 1 { "" } else { "s" }
    )
}

/// The most rounds the results of the typed clones are iterated.
const MAX_RESULT_ROUNDS: usize = 12;

impl Specializer<'_> {
    /// Types the clones with a type fact together, now that every call has
    /// been looked at (so every recursive call points at its clone).
    ///
    /// The result of a clone is what its `return` has, which depends on
    /// the results of the clones it calls, itself included. The iteration
    /// starts from "nothing returns yet" ([`Ty::Bot`]) and grows to a fixed
    /// point; `fib` returns its argument or a sum of `fib`s, and is a number
    /// if the sums are. Every clone is then typed with the fixed results,
    /// and the whole set is verified. A clone that does not verify (a call's
    /// now-known result meets an operation whose rules reject it: the
    /// generic function errors there at run time) is left as the first
    /// typing made it, with an `unknown` result, and the rest is iterated
    /// again without it.
    fn settle(&mut self) {
        if self.typed.is_empty() {
            return;
        }
        let bases: Vec<(usize, Function)> = self
            .typed
            .iter()
            .filter_map(|t| self.m.functions.iter().position(|g| g.name == t.name))
            .map(|i| (i, self.m.functions[i].clone()))
            .collect();
        // Whether the clone's result is left to inference: one result,
        // declared unknown.
        fn open(b: &Function) -> bool {
            b.results.len() == 1 && matches!(b.results[0], sky_ir::Type::Unknown)
        }
        let mut pinned: HashSet<usize> = HashSet::new();
        loop {
            let mut results: HashMap<String, Ty> = bases
                .iter()
                .enumerate()
                .map(|(i, (_, b))| {
                    let t = match b.results.as_slice() {
                        [_] if open(b) && !pinned.contains(&i) => Ty::Bot,
                        [t] => Ty::T(t.clone()),
                        _ => Ty::T(sky_ir::Type::Unknown),
                    };
                    (b.name.clone(), t)
                })
                .collect();
            let mut settled = false;
            for _ in 0..MAX_RESULT_ROUNDS {
                let mut changed = false;
                for (i, (_, b)) in bases.iter().enumerate() {
                    if pinned.contains(&i) || !open(b) {
                        continue;
                    }
                    let env = Env {
                        funcs: &self.m.functions,
                        results: &results,
                    };
                    let inferred = infer(b, &env);
                    let next = results[&b.name].join(&inferred.result);
                    if next != results[&b.name] {
                        results.insert(b.name.clone(), next);
                        changed = true;
                    }
                }
                if !changed {
                    settled = true;
                    break;
                }
            }
            if !settled {
                for (i, (_, b)) in bases.iter().enumerate() {
                    if !pinned.contains(&i) && open(b) {
                        results.insert(b.name.clone(), Ty::T(sky_ir::Type::Unknown));
                    }
                }
            }
            // Type every clone with the settled results, install them, and
            // check each.
            let trial: Vec<Function> = bases
                .iter()
                .enumerate()
                .map(|(i, (_, b))| {
                    if pinned.contains(&i) {
                        return b.clone();
                    }
                    let env = Env {
                        funcs: &self.m.functions,
                        results: &results,
                    };
                    let inferred = infer(b, &env);
                    let mut g = b.clone();
                    let result = open(b).then(|| results[&b.name].clone().into_type());
                    apply(&mut g, &inferred, result);
                    g
                })
                .collect();
            for ((idx, _), g) in bases.iter().zip(trial) {
                self.m.functions[*idx] = g;
            }
            let failed: Vec<usize> = bases
                .iter()
                .enumerate()
                .filter(|(i, (idx, _))| {
                    !pinned.contains(i)
                        && verify_function(&self.m.functions, &self.m.functions[*idx]).is_err()
                })
                .map(|(i, _)| i)
                .collect();
            if failed.is_empty() {
                break;
            }
            pinned.extend(failed);
        }
        // The counts and result types, now that the bodies are final.
        for t in &self.typed {
            let Some(g) = self.m.function(&t.name) else {
                continue;
            };
            let after = dynamic_checks(g);
            let removed = t.checks_before.saturating_sub(after);
            let family = source_name(&t.name).to_string();
            let returns = match g.results.as_slice() {
                [sky_ir::Type::Unknown] | [] => None,
                [r] => Some(type_words(r)),
                _ => None,
            };
            let note = &mut self.out[t.note];
            note.because[0] = type_checks_line(&family, &t.name, removed);
            if let Some(r) = returns {
                let at = 1.min(note.because.len());
                note.because
                    .insert(at, format!("`{}` returns a {r}", t.name));
            }
        }
    }
}

/// `1 branch`, `2 branches`.
fn count(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// Specializes calls in every function of `m` (see the module
/// documentation).
pub fn specialize(m: &mut Module) -> Vec<Rewrite> {
    let start: Vec<usize> = (0..m.functions.len()).collect();
    let existing: HashSet<String> = m.functions.iter().map(|f| f.name.clone()).collect();
    let called = callees(m);
    let mut s = Specializer::new(m);
    s.run(start);
    let out = s.out;
    remove_orphans(m, &existing, &called);
    out
}

/// The names of the functions some function of `m` calls.
fn callees(m: &Module) -> HashSet<String> {
    let mut out = HashSet::new();
    for f in &m.functions {
        for o in reachable_ops(f) {
            if let OpKind::Call { callee } = &f.op(o).kind {
                out.insert(callee.clone());
            }
        }
    }
    out
}

/// Removes what specialization left without a caller: a function that was
/// called before the pass (`called`) or is a clone the pass made, and that no
/// call reaches any more, starting from the functions that were never called
/// (`<main>`, and functions nobody calls, which stay as they are). A generic
/// function whose every call now goes to a clone is the usual case; without
/// this it would sit in the module unused, keeping its checks and its size.
/// (Recursion does not keep a function alive: reachability, not "has a
/// call".)
fn remove_orphans(m: &mut Module, existing: &HashSet<String>, called: &HashSet<String>) {
    let mut alive: HashSet<&str> = m
        .functions
        .iter()
        .filter(|f| existing.contains(&f.name) && !called.contains(&f.name))
        .map(|f| f.name.as_str())
        .collect();
    let mut work: Vec<&str> = alive.iter().copied().collect();
    while let Some(name) = work.pop() {
        let Some(f) = m.function(name) else { continue };
        for o in reachable_ops(f) {
            if let OpKind::Call { callee } = &f.op(o).kind {
                if let Some(g) = m.function(callee) {
                    if alive.insert(g.name.as_str()) {
                        work.push(g.name.as_str());
                    }
                }
            }
        }
    }
    let dead: HashSet<String> = m
        .functions
        .iter()
        .filter(|f| !alive.contains(f.name.as_str()))
        .map(|f| f.name.clone())
        .collect();
    m.functions.retain(|f| !dead.contains(&f.name));
}

/// The facts inside a specialization's name, as words: `N = 3, x > 0` for
/// `f[N=3,x>0]`, `n: Scalar (dimensionless)` for `f[n:Scalar]`; `None` for a
/// name that is not one.
pub(crate) fn facts_in_name(name: &str) -> Option<String> {
    let inner = name.split_once('[')?.1.strip_suffix(']')?;
    // Facts are separated by commas outside brackets (`Matrix[2,3]`).
    let mut parts: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut depth = 0usize;
    for c in inner.chars() {
        match c {
            '[' => depth += 1,
            ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(std::mem::take(&mut current));
                continue;
            }
            _ => {}
        }
        current.push(c);
    }
    parts.push(current);
    let words: Vec<String> = parts
        .iter()
        .map(|p| match p.split_once(':') {
            Some((param, ty)) => format!("{param}: {}", self::facts::words_of_type_key(ty)),
            None => p
                .replace('=', " = ")
                .replace('>', " > ")
                .replace('<', " < "),
        })
        .collect();
    Some(words.join(", "))
}

/// The headline `sky check --explain` gives an explanation of this pass, if
/// `x` is one: "[SPEC-001] specialized `f` for N = 3 at this call: 4 checks
/// removed". (The fields are those [`specialize`] records: `before` is the
/// function's source name, `after` the clone's name, and the first entry of
/// `because` starts with the number of checks.)
pub fn specialize_headline(x: &Explanation) -> Option<String> {
    let facts = || facts_in_name(&x.after);
    Some(match x.rule {
        RULE_SPECIALIZED => {
            let n = x
                .because
                .first()?
                .split(' ')
                .next()?
                .parse::<usize>()
                .ok()?;
            format!(
                "[{}] specialized `{}` for {} at this call: {n} check{} removed",
                x.rule,
                x.before,
                facts()?,
                if n == 1 { "" } else { "s" }
            )
        }
        RULE_RETYPED => {
            let n = x
                .because
                .first()?
                .split(' ')
                .next()?
                .parse::<usize>()
                .ok()?;
            format!(
                "[{}] specialized `{}` for {}: {n} run-time type check{} removed",
                x.rule,
                x.before,
                facts()?,
                if n == 1 { "" } else { "s" }
            )
        }
        RULE_REUSED => format!(
            "[{}] this call uses the existing specialization `{}`",
            x.rule, x.after
        ),
        RULE_UNROLLED => format!(
            "[{}] this loop runs exactly {} time{} in `{}`, so it is unrolled",
            x.rule,
            x.after,
            if x.after == "1" { "" } else { "s" },
            x.before
        ),
        RULE_LIMIT => format!(
            "[{}] this call of `{}` is not specialized: {}",
            x.rule,
            x.before,
            x.because.first()?
        ),
        _ => return None,
    })
}
