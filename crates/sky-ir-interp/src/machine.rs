//! The evaluator: a value table per activation, ops in order, regions by
//! recursion.
//!
//! The `match` over [`OpKind`] is exhaustive on purpose (no `_ =>` arm), so
//! an operation added to the IR fails to compile here instead of being
//! silently ignored.
//!
//! Every access to IR data goes through a checked accessor (`Act::op`,
//! `Act::get`, `fixed`, ...) that reports `malformed IR` instead of
//! indexing, so no input can make this module panic.

use std::io::Write;
use std::sync::{Mutex, PoisonError};

use rayon::prelude::*;
use sky_diag::Span;
use sky_interp::arith::{
    arith, arity_mismatch, equal, expect_bool, expect_number, order, pow, reciprocal,
    recursion_too_deep, sign, ArithOp, StackMeter, MAX_CALL_DEPTH, STACK_BUDGET,
};
use sky_interp::array::{self, Ix};
use sky_interp::contract;
use sky_interp::solve;
use sky_interp::Value;
use sky_ir::parallel::{CallGroup, Config, FnPlan, Role};
use sky_ir::{
    BlockData, BlockId, CmpOp, Function, Module, NumberType, OpData, OpId, OpKind, RegionId,
    SliceAxis, Type, ValueId,
};
use sky_units::{Dimension, Rational};

use crate::native::{Jit, JitOptions, Native};
use crate::par::ParRt;
use crate::IrRuntimeError;

type R<T> = Result<T, IrRuntimeError>;

/// How a block finished: by the terminator it ended in, with its operands'
/// values.
enum Exit {
    Yield(Vec<Value>),
    Condition(Vec<Value>),
    Return(Vec<Value>),
}

/// One function activation: the function and its value table, indexed by
/// [`ValueId`]. `None` is a value not defined (yet), which reading reports
/// as malformed IR. SSA scoping is not enforced here (a value defined in a
/// region stays readable afterwards); that is the verifier's job.
struct Act<'m> {
    func: &'m Function,
    vals: Vec<Option<Value>>,
    /// Where this activation was called from, for the "called here" label
    /// of a failed `requires` (as the tree-walker's frame has it); `None`
    /// for a run's entry call.
    call: Option<Span>,
    /// The call groups of `func` (see `sky_ir::parallel`), when the run is
    /// parallel and the function has any.
    plan: Option<&'m FnPlan>,
}

impl<'m> Act<'m> {
    fn new(func: &'m Function, call: Option<Span>) -> Self {
        Act {
            func,
            vals: vec![None; func.values.len()],
            call,
            plan: None,
        }
    }

    fn get(&self, v: ValueId, span: Span) -> R<&Value> {
        self.vals
            .get(v.index())
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                IrRuntimeError::malformed(format!("{v} is used before it is defined"), span)
            })
    }

    fn set(&mut self, v: ValueId, value: Value, span: Span) -> R<()> {
        match self.vals.get_mut(v.index()) {
            Some(slot) => {
                *slot = Some(value);
                Ok(())
            }
            None => Err(IrRuntimeError::malformed(
                format!("{v} is not a value of `{}`", self.func.name),
                span,
            )),
        }
    }

    fn values(&self, vs: &[ValueId], span: Span) -> R<Vec<Value>> {
        vs.iter().map(|v| self.get(*v, span).cloned()).collect()
    }

    /// Binds `op`'s results to `values`, which must match in number.
    fn define(&mut self, op: &OpData, values: Vec<Value>) -> R<()> {
        if op.results.len() != values.len() {
            return Err(IrRuntimeError::malformed(
                format!(
                    "`{}` has {} results, expected {}",
                    op.kind.mnemonic(),
                    op.results.len(),
                    values.len()
                ),
                op.span,
            ));
        }
        for (r, v) in op.results.iter().zip(values) {
            self.set(*r, v, op.span)?;
        }
        Ok(())
    }

    fn op(&self, id: OpId, span: Span) -> R<&'m OpData> {
        self.func
            .ops
            .get(id.index())
            .ok_or_else(|| IrRuntimeError::malformed(format!("no operation {id}"), span))
    }

    fn block(&self, id: BlockId, span: Span) -> R<&'m BlockData> {
        self.func
            .blocks
            .get(id.index())
            .ok_or_else(|| IrRuntimeError::malformed(format!("no block {id}"), span))
    }

    /// The single block of a region (v1 regions have exactly one).
    fn entry_of(&self, id: RegionId, span: Span) -> R<BlockId> {
        let region = self
            .func
            .regions
            .get(id.index())
            .ok_or_else(|| IrRuntimeError::malformed(format!("no region {id}"), span))?;
        match region.blocks[..] {
            [b] => Ok(b),
            [] => Err(IrRuntimeError::malformed(
                format!("{id} has no block"),
                span,
            )),
            _ => Err(IrRuntimeError::malformed(
                format!("{id} has several blocks, but v1 regions have one"),
                span,
            )),
        }
    }

    /// The type of `v`, if `v` exists.
    fn ty(&self, v: ValueId) -> Option<&'m Type> {
        self.func.values.get(v.index()).map(|d| &d.ty)
    }
}

/// `op`'s operands as exactly `N` values' ids.
fn fixed<const N: usize>(op: &OpData) -> R<[ValueId; N]> {
    op.operands.as_slice().try_into().map_err(|_| {
        IrRuntimeError::malformed(
            format!(
                "`{}` takes {N} operands, found {}",
                op.kind.mnemonic(),
                op.operands.len()
            ),
            op.span,
        )
    })
}

/// `op`'s regions as exactly `N` ids.
fn regions<const N: usize>(op: &OpData) -> R<[RegionId; N]> {
    op.regions.as_slice().try_into().map_err(|_| {
        IrRuntimeError::malformed(
            format!(
                "`{}` takes {N} regions, found {}",
                op.kind.mnemonic(),
                op.regions.len()
            ),
            op.span,
        )
    })
}

/// The dimension of a constant, from its result's type. An `Unknown`
/// result is dimensionless (a constant's value carries no other unit).
fn const_dim(act: &Act<'_>, op: &OpData) -> R<Dimension> {
    let [r] = op.results[..] else {
        return Err(IrRuntimeError::malformed(
            format!(
                "`{}` has {} results, expected 1",
                op.kind.mnemonic(),
                op.results.len()
            ),
            op.span,
        ));
    };
    match act.ty(r) {
        Some(Type::Number(NumberType { dim, shape, .. })) if shape.rank() == 0 => Ok(*dim),
        Some(Type::Number(_)) => Err(IrRuntimeError::malformed(
            format!("`{}` must have a scalar result", op.kind.mnemonic()),
            op.span,
        )),
        Some(Type::Unknown) => Ok(Dimension::DIMENSIONLESS),
        _ => Err(IrRuntimeError::malformed(
            format!("`{}` must have a number result", op.kind.mnemonic()),
            op.span,
        )),
    }
}

struct Machine<'m, 'w> {
    module: &'m Module,
    out: &'w mut dyn Write,
    /// User-function calls active now; see [`MAX_CALL_DEPTH`].
    depth: usize,
    stack: StackMeter,
    /// Native kernels for long `fused` operations (see `crate::native`).
    jit: Jit,
    /// The parallel runtime, when the run may use more than one thread.
    par: Option<&'m ParRt>,
    /// Ops left to run, in tests only: random malformed IR (a `while` whose
    /// condition is always true) must not hang the suite. Production has no
    /// step limit.
    #[cfg(test)]
    fuel: u64,
}

/// Calls `name` in `module` with `args`. Sequential: see [`run_par`].
pub(crate) fn call(
    module: &Module,
    name: &str,
    args: Vec<Value>,
    out: &mut dyn Write,
) -> R<Vec<Value>> {
    call_metered(module, name, args, out, StackMeter::new(), None)
}

/// [`call`], in parallel where `cfg` and the module allow it
/// (`sky_ir::parallel`).
pub(crate) fn call_par(
    module: &Module,
    name: &str,
    args: Vec<Value>,
    out: &mut dyn Write,
    cfg: &Config,
) -> R<Vec<Value>> {
    let rt = ParRt::new(module, cfg);
    let mut m = Machine {
        module,
        out,
        depth: 0,
        stack: StackMeter::new(),
        jit: Jit::new(JitOptions::default()),
        par: rt.as_ref(),
        #[cfg(test)]
        fuel: u64::MAX,
    };
    m.entry(name, args, Span::point(0), None)
}

/// [`call`] measuring the stack from `stack`, a meter another interpreter
/// started, so a run nested in it stays inside one budget.
pub(crate) fn call_metered(
    module: &Module,
    name: &str,
    args: Vec<Value>,
    out: &mut dyn Write,
    stack: StackMeter,
    call: Option<Span>,
) -> R<Vec<Value>> {
    call_with_jit(module, name, args, out, stack, call, JitOptions::default())
}

/// [`call_metered`] with the native-kernel options chosen.
pub(crate) fn call_with_jit(
    module: &Module,
    name: &str,
    args: Vec<Value>,
    out: &mut dyn Write,
    stack: StackMeter,
    call: Option<Span>,
    jit: JitOptions,
) -> R<Vec<Value>> {
    let mut m = Machine {
        module,
        out,
        depth: 0,
        stack,
        jit: Jit::new(jit),
        par: None,
        #[cfg(test)]
        fuel: u64::MAX,
    };
    m.entry(name, args, Span::point(0), call)
}

/// [`call`] with a bounded number of ops, for fuzzing.
#[cfg(test)]
pub(crate) fn call_with_fuel(
    module: &Module,
    name: &str,
    args: Vec<Value>,
    out: &mut dyn Write,
    fuel: u64,
) -> R<Vec<Value>> {
    let mut m = Machine {
        module,
        out,
        depth: 0,
        stack: StackMeter::new(),
        jit: Jit::new(JitOptions::default()),
        par: None,
        fuel,
    };
    m.entry(name, args, Span::point(0), None)
}

impl<'m> Machine<'m, '_> {
    /// Looks `name` up for a call with `argc` arguments made at `span`.
    fn resolve(&self, name: &str, argc: usize, span: Span) -> R<&'m Function> {
        let module = self.module;
        let Some(func) = module.function(name) else {
            return Err(IrRuntimeError::malformed(
                format!("call to unknown function `{name}`"),
                span,
            ));
        };
        if argc != func.params.len() {
            return Err(arity_mismatch(name, func.params.len(), argc, span).into());
        }
        Ok(func)
    }

    /// The entry call of a run. Like `sky run`'s top level, it is not a
    /// nested call and does not count against [`MAX_CALL_DEPTH`].
    fn entry(
        &mut self,
        name: &str,
        args: Vec<Value>,
        span: Span,
        call: Option<Span>,
    ) -> R<Vec<Value>> {
        let func = self.resolve(name, args.len(), span)?;
        self.activate(func, args, call)
    }

    /// A call made by a `call` op.
    fn call_named(&mut self, name: &str, args: Vec<Value>, span: Span) -> R<Vec<Value>> {
        let func = self.resolve(name, args.len(), span)?;
        if self.depth >= MAX_CALL_DEPTH {
            // A specialization `f[N=3]` is still `f` to the user.
            return Err(recursion_too_deep(sky_ir::source_name(name), span).into());
        }
        self.depth += 1;
        let result = self.activate(func, args, Some(span));
        self.depth -= 1;
        result
    }

    /// Runs `func`'s body in a new activation.
    fn activate(&mut self, func: &Function, args: Vec<Value>, call: Option<Span>) -> R<Vec<Value>> {
        let mut act = Act::new(func, call);
        act.plan = self.par.and_then(|p| p.plan.function(&func.name));
        let entry = act.entry_of(func.body, func.span)?;
        match self.block(&mut act, entry, args, func.span)? {
            Exit::Return(values) if values.len() == func.results.len() => Ok(values),
            Exit::Return(values) => Err(IrRuntimeError::malformed(
                format!(
                    "`{}` returns {} values, its signature has {}",
                    func.name,
                    values.len(),
                    func.results.len()
                ),
                func.span,
            )),
            Exit::Yield(_) | Exit::Condition(_) => Err(IrRuntimeError::malformed(
                format!("the body of `{}` must end in `return`", func.name),
                func.span,
            )),
        }
    }

    /// Runs a block: binds its arguments, runs its ops in order, and
    /// returns how it ended.
    fn block(&mut self, act: &mut Act<'_>, id: BlockId, args: Vec<Value>, span: Span) -> R<Exit> {
        self.stack.check(span)?;
        let block = act.block(id, span)?;
        if block.args.len() != args.len() {
            return Err(IrRuntimeError::malformed(
                format!(
                    "{id} takes {} arguments, given {}",
                    block.args.len(),
                    args.len()
                ),
                span,
            ));
        }
        for (a, v) in block.args.iter().zip(args) {
            act.set(*a, v, span)?;
        }
        let mut pos = 0;
        while pos < block.ops.len() {
            let op_id = block.ops[pos];
            let op = act.op(op_id, span)?;
            #[cfg(test)]
            {
                if self.fuel == 0 {
                    return Err(IrRuntimeError::malformed("out of fuel (test)", op.span));
                }
                self.fuel -= 1;
            }
            pos += 1;
            if let Some(group) = act.plan.and_then(|p| p.group_starting_at(op_id)) {
                let end = pos - 1 + group.ops.len();
                if block.ops.get(pos - 1..end) == Some(&group.ops[..])
                    && self.run_group(act, group)?
                {
                    pos = end;
                    continue;
                }
            }
            match op.kind {
                OpKind::Yield => return Ok(Exit::Yield(act.values(&op.operands, op.span)?)),
                OpKind::Condition => {
                    return Ok(Exit::Condition(act.values(&op.operands, op.span)?))
                }
                OpKind::Return => return Ok(Exit::Return(act.values(&op.operands, op.span)?)),
                _ => self.op(act, op)?,
            }
        }
        Err(IrRuntimeError::malformed(
            format!("{id} does not end in a terminator"),
            span,
        ))
    }

    /// Runs the (single) block of `region` with `args`.
    fn region(&mut self, act: &mut Act<'_>, id: RegionId, args: Vec<Value>, span: Span) -> R<Exit> {
        let entry = act.entry_of(id, span)?;
        self.block(act, entry, args, span)
    }

    /// Runs one operation that is not a terminator. Only operations that
    /// can recurse are handled here: `pure_op` has a large frame (one arm
    /// per operation), and keeping it off the recursion path is what lets
    /// [`MAX_CALL_DEPTH`] nested calls fit the stack budget in a debug
    /// build.
    fn op(&mut self, act: &mut Act<'_>, op: &OpData) -> R<()> {
        match &op.kind {
            OpKind::If => self.if_op(act, op),
            OpKind::While => self.while_op(act, op),
            OpKind::Call { callee } => {
                let args = act.values(&op.operands, op.span)?;
                let results = self.call_named(callee, args, op.span)?;
                act.define(op, results)
            }
            OpKind::Print => self.print(act, op),
            OpKind::Fused => fused_op(act, op, &mut self.jit, self.par),
            OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div | OpKind::Neg => match self.par {
                Some(rt) if elementwise(act, op, rt)? => Ok(()),
                _ => pure_op(act, op),
            },
            _ => pure_op(act, op),
        }
    }

    /// Runs a group of independent calls (`sky_ir::parallel::CallGroup`).
    ///
    /// Phase 1 starts the calls marked [`Role::Start`] on the pool, one after
    /// the other, and runs the operations marked [`Role::InPlace`] here,
    /// in order, meanwhile; the first of those to fail stops the phase (no
    /// call after it is started). Phase 2 replays the window in program
    /// order once the started calls are done: a started call gives its
    /// result or its error, a [`Role::Deferred`] operation runs, and the
    /// first failure is the error. The failure of an in-place operation
    /// comes after everything before it in the replay, so it is reported
    /// only if nothing before it failed: the sequential run's error, whichever
    /// thread finished first.
    ///
    /// Returns `false`, having done nothing, when the group must run
    /// sequentially: the caller is deep in the stack (see the module
    /// documentation of `sky_ir::parallel`).
    fn run_group(&mut self, act: &mut Act<'_>, group: &CallGroup) -> R<bool> {
        let Some((rt, pool)) = self.par.and_then(|rt| Some((rt, rt.pool()?))) else {
            return Ok(false);
        };
        if self.stack.used() > STACK_BUDGET / 2 {
            return Ok(false);
        }
        let s = act.func.span;
        let started = group.roles.iter().filter(|&&r| r == Role::Start).count();
        let slots: Vec<Mutex<Option<R<Vec<Value>>>>> =
            (0..started).map(|_| Mutex::new(None)).collect();
        let (module, depth) = (self.module, self.depth);
        // The position and error of the first in-place operation that failed.
        let mut here: Option<(usize, IrRuntimeError)> = None;
        pool.in_place_scope(|scope| {
            let mut next = 0;
            for (k, (&op_id, &role)) in group.ops.iter().zip(&group.roles).enumerate() {
                let mut run = || -> R<()> {
                    let op = act.op(op_id, s)?;
                    match role {
                        Role::Deferred => Ok(()),
                        Role::InPlace => self.op(act, op),
                        Role::Start => {
                            let OpKind::Call { callee } = &op.kind else {
                                return Err(IrRuntimeError::malformed(
                                    "expected a `call`",
                                    op.span,
                                ));
                            };
                            let args = act.values(&op.operands, op.span)?;
                            let slot = &slots[next];
                            next += 1;
                            let span = op.span;
                            scope.spawn(move |_| {
                                // The callee has no `io`: nothing is written.
                                let mut sink = std::io::sink();
                                let mut child = Machine {
                                    module,
                                    out: &mut sink,
                                    depth,
                                    stack: StackMeter::new(),
                                    jit: Jit::new(JitOptions::default()),
                                    par: Some(rt),
                                    #[cfg(test)]
                                    fuel: u64::MAX,
                                };
                                let r = child.call_named(callee, args, span);
                                *slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(r);
                            });
                            Ok(())
                        }
                    }
                };
                if let Err(e) = run() {
                    here = Some((k, e));
                    break;
                }
            }
        });
        // The replay. `end` is where phase 1 stopped: nothing at or after
        // it was started.
        let end = here.as_ref().map_or(group.ops.len(), |(k, _)| *k);
        let mut results = slots
            .into_iter()
            .map(|slot| slot.into_inner().unwrap_or_else(PoisonError::into_inner));
        for (&op_id, &role) in group.ops.iter().zip(&group.roles).take(end) {
            let op = act.op(op_id, s)?;
            match role {
                Role::InPlace => {}
                Role::Deferred => self.op(act, op)?,
                Role::Start => match results.next().flatten() {
                    Some(Ok(values)) => act.define(op, values)?,
                    Some(Err(e)) => return Err(e),
                    None => return Err(IrRuntimeError::malformed("a started call is missing", s)),
                },
            }
        }
        match here {
            Some((_, e)) => Err(e),
            None => Ok(true),
        }
    }

    /// `if`: run the arm the condition selects; its `yield` gives the
    /// results.
    fn if_op(&mut self, act: &mut Act<'_>, op: &OpData) -> R<()> {
        let s = op.span;
        let [c] = fixed(op)?;
        let [then, els] = regions(op)?;
        let arm = if expect_bool("if", act.get(c, s)?, s)? {
            then
        } else {
            els
        };
        match self.region(act, arm, Vec::new(), s)? {
            Exit::Yield(values) => act.define(op, values),
            Exit::Condition(_) | Exit::Return(_) => Err(IrRuntimeError::malformed(
                "an `if` region must end in `yield`",
                s,
            )),
        }
    }

    /// `while` (see `OpKind::While`): `before` computes the condition and
    /// the forwarded values; if true, `after` computes the next loop-carried
    /// values; if false, the forwarded values are the results.
    fn while_op(&mut self, act: &mut Act<'_>, op: &OpData) -> R<()> {
        let s = op.span;
        let [before, after] = regions(op)?;
        let mut carried = act.values(&op.operands, s)?;
        loop {
            let Exit::Condition(mut forwarded) = self.region(act, before, carried, s)? else {
                return Err(IrRuntimeError::malformed(
                    "a `while` `before` region must end in `condition`",
                    s,
                ));
            };
            if forwarded.is_empty() {
                return Err(IrRuntimeError::malformed(
                    "`condition` needs a Bool operand",
                    s,
                ));
            }
            let flag = forwarded.remove(0);
            if !expect_bool("while", &flag, s)? {
                return act.define(op, forwarded);
            }
            match self.region(act, after, forwarded, s)? {
                Exit::Yield(next) => carried = next,
                Exit::Condition(_) | Exit::Return(_) => {
                    return Err(IrRuntimeError::malformed(
                        "a `while` `after` region must end in `yield`",
                        s,
                    ));
                }
            }
        }
    }

    /// `print(a, b, ...)`: the values separated by single spaces, then a
    /// newline, written in one piece as `sky run` does.
    fn print(&mut self, act: &mut Act<'_>, op: &OpData) -> R<()> {
        let s = op.span;
        let mut line = String::new();
        for (i, v) in op.operands.iter().enumerate() {
            if i > 0 {
                line.push(' ');
            }
            line.push_str(&act.get(*v, s)?.print_text());
        }
        line.push('\n');
        self.out
            .write_all(line.as_bytes())
            .map_err(|e| sky_interp::RuntimeError::new(format!("cannot write output: {e}"), s))?;
        act.define(op, Vec::new())
    }
}

/// Runs an operation that never recurses into regions or calls.
fn pure_op(act: &mut Act<'_>, op: &OpData) -> R<()> {
    let s = op.span;
    match &op.kind {
        OpKind::ConstFloat { bits } => {
            let x = f64::from_bits(*bits);
            if !x.is_finite() {
                return Err(IrRuntimeError::malformed("a constant is not finite", s));
            }
            let dim = const_dim(act, op)?;
            act.define(op, vec![Value::number(x, dim)])
        }
        OpKind::ConstRational(q) => {
            let x = q.numer() as f64 / q.denom() as f64;
            if !x.is_finite() {
                return Err(IrRuntimeError::malformed(
                    "a rational constant is not finite",
                    s,
                ));
            }
            let dim = const_dim(act, op)?;
            act.define(op, vec![Value::number(x, dim)])
        }
        OpKind::ConstBool(b) => act.define(op, vec![Value::Bool(*b)]),
        OpKind::ConstStr(text) => act.define(op, vec![Value::Str(text.clone())]),
        OpKind::ConstNothing => act.define(op, vec![Value::None]),

        OpKind::Add => arith_op(act, op, ArithOp::Add),
        OpKind::Sub => arith_op(act, op, ArithOp::Sub),
        OpKind::Mul => arith_op(act, op, ArithOp::Mul),
        OpKind::Div => arith_op(act, op, ArithOp::Div),
        OpKind::Neg => {
            let [a] = fixed(op)?;
            let v = sign(true, act.get(a, s)?.clone(), op.operand_span(0))?;
            act.define(op, vec![v])
        }
        OpKind::Math(f) => {
            let vals = act.values(&op.operands, s)?;
            if vals.len() != f.arity() {
                return Err(IrRuntimeError::malformed(
                    format!(
                        "`{}` takes {} operands, found {}",
                        f.mnemonic(),
                        f.arity(),
                        vals.len()
                    ),
                    s,
                ));
            }
            let refs: Vec<&Value> = vals.iter().collect();
            let spans: Vec<Span> = (0..refs.len()).map(|i| op.operand_span(i)).collect();
            let v = crate::math::eval(*f, &refs, s, &spans)?;
            act.define(op, vec![v])
        }
        // The same function `sky run` calls for `v as unit`, so both print
        // the same digits and fail the same way.
        OpKind::Show { unit } => {
            let [a] = fixed(op)?;
            let v = sky_interp::show_unit_text(unit, act.get(a, s)?, s)?;
            act.define(op, vec![v])
        }
        OpKind::PowConst { exponent } => {
            let [a] = fixed(op)?;
            let e = Value::scalar(exponent.numer() as f64 / exponent.denom() as f64);
            // The exponent is an attribute: it has no operand, no span, and
            // no error of its own.
            let v = pow(
                act.get(a, s)?,
                &e,
                || Some(*exponent),
                s,
                op.operand_span(0),
                s,
            )?;
            act.define(op, vec![v])
        }
        OpKind::Pow => {
            let [a, e] = fixed(op)?;
            // A computed exponent gives a base with a unit no constant to
            // take the result's dimension from, as in `sky run`.
            let (l, r) = (op.operand_span(0), op.operand_span(1));
            let v = pow(act.get(a, s)?, act.get(e, s)?, || None::<Rational>, s, l, r)?;
            act.define(op, vec![v])
        }
        // The IR does not remember whether the source said `x⁻¹` or
        // `inv(x)`, so the (unreachable in checked programs) error for a
        // non-number always uses the `⁻¹` spelling.
        OpKind::Inverse => {
            let [a] = fixed(op)?;
            let v = reciprocal("⁻¹", act.get(a, s)?, s, op.operand_span(0))?;
            act.define(op, vec![v])
        }
        OpKind::MatMul => {
            let [a, b] = fixed(op)?;
            let (l, r) = (op.operand_span(0), op.operand_span(1));
            let v = array::matmul(act.get(a, s)?, act.get(b, s)?, s, l, r)?;
            act.define(op, vec![v])
        }
        // The functions `sky run` calls for `norm(x)` and `dot(a, b)`, with
        // the call's span and the arguments' spans, so the results and the
        // errors are the same bytes.
        OpKind::Norm => {
            let [a] = fixed(op)?;
            let v = sky_interp::vecfn::norm(act.get(a, s)?, s, op.operand_span(0))?;
            act.define(op, vec![v])
        }
        OpKind::Dot => {
            let [a, b] = fixed(op)?;
            let (l, r) = (op.operand_span(0), op.operand_span(1));
            let v = sky_interp::vecfn::dot(act.get(a, s)?, act.get(b, s)?, s, l, r)?;
            act.define(op, vec![v])
        }
        OpKind::Transpose => {
            let [a] = fixed(op)?;
            let v = array::transpose(act.get(a, s)?, s)?;
            act.define(op, vec![v])
        }
        // The Cholesky solve `sky run` performs where the checker rewrote
        // `A⁻¹ @ b`: the same function, so the same bits and errors.
        OpKind::Solve { method } => {
            let [a, b] = fixed(op)?;
            let (l, r) = (op.operand_span(0), op.operand_span(1));
            let v = solve::solve(*method, act.get(a, s)?, act.get(b, s)?, s, l, r)?;
            act.define(op, vec![v])
        }
        // `x @ A⁻¹`, the transposed solve: the same functions as the
        // tree-walker runs for a chain the checker planned.
        OpKind::SolveRight { method } => {
            let [x, a] = fixed(op)?;
            let (l, r) = (op.operand_span(0), op.operand_span(1));
            let v = solve::solve_right(*method, act.get(x, s)?, act.get(a, s)?, s, l, r)?;
            act.define(op, vec![v])
        }
        OpKind::MatMulEx { ta, tb, algorithm } => {
            let [a, b] = fixed(op)?;
            let (l, r) = (op.operand_span(0), op.operand_span(1));
            let v = solve::matmul_ex(
                *ta,
                *tb,
                *algorithm,
                act.get(a, s)?,
                act.get(b, s)?,
                s,
                l,
                r,
            )?;
            act.define(op, vec![v])
        }
        OpKind::DiagMul { side } => {
            let [a, b] = fixed(op)?;
            let (l, r) = (op.operand_span(0), op.operand_span(1));
            let (d, x) = if *side == sky_ir::Side::Left {
                (a, b)
            } else {
                (b, a)
            };
            let (dspan, xspan) = if *side == sky_ir::Side::Left {
                (l, r)
            } else {
                (r, l)
            };
            let v = solve::diag_mul(*side, act.get(d, s)?, act.get(x, s)?, s, dspan, xspan)?;
            act.define(op, vec![v])
        }
        OpKind::ArrayLiteral { cols } => array_literal(act, op, *cols),
        OpKind::Index => {
            let Some((&base, positions)) = op.operands.split_first() else {
                return Err(IrRuntimeError::malformed("`index` has no operands", s));
            };
            let mut items = Vec::with_capacity(positions.len());
            for (k, &p) in positions.iter().enumerate() {
                let at = op.operand_span(k + 1);
                items.push(Ix::At {
                    i: array::to_index(act.get(p, s)?, at)?,
                    span: at,
                });
            }
            let v = array::index(act.get(base, s)?, &items, s, op.operand_span(0))?;
            act.define(op, vec![v])
        }
        OpKind::Slice { axes } => slice(act, op, axes),
        OpKind::Zeros | OpKind::Identity => {
            let mut sizes = Vec::with_capacity(2);
            let name = op.kind.mnemonic();
            for (i, &v) in op.operands.iter().enumerate() {
                sizes.push(array::size_arg(name, act.get(v, s)?, op.operand_span(i))?);
            }
            let v = match (&op.kind, sizes.as_slice()) {
                (OpKind::Zeros, [_] | [_, _]) => array::zeros(&sizes, s)?,
                (OpKind::Identity, [n]) => array::identity(*n, s)?,
                _ => {
                    return Err(IrRuntimeError::malformed(
                        format!(
                            "`{name}` takes a size (`zeros`: two), found {}",
                            sizes.len()
                        ),
                        s,
                    ))
                }
            };
            act.define(op, vec![v])
        }
        OpKind::Len => {
            let [a] = fixed(op)?;
            let v = array::len(act.get(a, s)?, op.operand_span(0))?;
            act.define(op, vec![v])
        }

        OpKind::Cmp(c) => {
            let [a, b] = fixed(op)?;
            let (l, r) = (act.get(a, s)?, act.get(b, s)?);
            let (ls, rs) = (op.operand_span(0), op.operand_span(1));
            let holds = match c {
                CmpOp::Eq => equal(l, r, s, ls, rs)?,
                CmpOp::Ne => !equal(l, r, s, ls, rs)?,
                CmpOp::Lt => order("<", |x, y| x < y, l, r, s, ls, rs)?,
                CmpOp::Le => order("<=", |x, y| x <= y, l, r, s, ls, rs)?,
                CmpOp::Gt => order(">", |x, y| x > y, l, r, s, ls, rs)?,
                CmpOp::Ge => order(">=", |x, y| x >= y, l, r, s, ls, rs)?,
            };
            act.define(op, vec![Value::Bool(holds)])
        }
        OpKind::Not => {
            let [a] = fixed(op)?;
            let b = expect_bool("not", act.get(a, s)?, op.operand_span(0))?;
            act.define(op, vec![Value::Bool(!b)])
        }
        OpKind::LiteralError { error } => Err(match error {
            sky_ir::LiteralError::TooLarge => sky_interp::number_too_large(s),
            sky_ir::LiteralError::Overflow => sky_interp::arith::overflow(s),
        }
        .into()),
        OpKind::Input { name } => Err(sky_interp::arith::declared_without_value(name, s).into()),
        OpKind::ExpectBool { context } => {
            let [a] = fixed(op)?;
            let b = expect_bool(context.keyword(), act.get(a, s)?, s)?;
            act.define(op, vec![Value::Bool(b)])
        }
        OpKind::Assert { .. } => assert_op(act, op),

        // `+x` of a value the checker could not type: a number or an array
        // passes as it is, anything else fails as in `sky run`.
        OpKind::ExpectNumber { context } => {
            let [a] = fixed(op)?;
            let v = act.get(a, s)?.clone();
            expect_number(context.keyword(), &v, s)?;
            act.define(op, vec![v])
        }

        // Handled by `Machine::op` and `Machine::block`, which never send
        // them here; the arm keeps this `match` exhaustive.
        OpKind::If
        | OpKind::While
        | OpKind::Fused
        | OpKind::Call { .. }
        | OpKind::Print
        | OpKind::Yield
        | OpKind::Condition
        | OpKind::Return => Err(IrRuntimeError::malformed(
            format!("unexpected `{}`", op.kind.mnemonic()),
            s,
        )),
    }
}

/// `assert` (see `OpKind::Assert`): a false condition stops the run with
/// the failure `sky_interp::contract` builds, which is the tree-walker's,
/// word for word. Kept out of [`pure_op`]'s frame.
#[inline(never)]
fn assert_op(act: &mut Act<'_>, op: &OpData) -> R<()> {
    let s = op.span;
    let OpKind::Assert {
        kind,
        text,
        message,
        func,
        names,
    } = &op.kind
    else {
        return Err(IrRuntimeError::malformed("expected an `assert`", s));
    };
    let Some((cond, values)) = op.operands.split_first() else {
        return Err(IrRuntimeError::malformed("`assert` needs a condition", s));
    };
    if contract::expect_condition(*kind, act.get(*cond, s)?, s)? {
        return Ok(());
    }
    let mut shown: Vec<(&str, String)> = Vec::with_capacity(names.len());
    for (name, v) in names.iter().zip(values) {
        shown.push((name.as_str(), contract::show(act.get(*v, s)?)));
    }
    let mut err = contract::failure(*kind, func.as_deref(), text, message.as_deref(), &shown, s);
    if let (sky_ir::AssertKind::Requires, Some(call)) = (kind, act.call) {
        err = err.with_secondary(call, "called here");
    }
    Err(err.into())
}

/// `fused` (see `OpKind::Fused`): runs the scalar body once per element.
///
/// Each body operation is [`pure_op`], the code an unfused scalar operation
/// runs, which calls `sky_interp::arith`: for a finite `f64` every array
/// operation there is that same scalar computation per entry, so the
/// entries are bit-identical. The one exception is `neg`, which on arrays is
/// `0 - x` (see [`body_op`]).
///
/// The failure a chain of array operations reports is that of the first
/// operation, in program order, that fails at any element. Visiting elements
/// in order and stopping at the first failure could report a later
/// operation's failure at an earlier element instead. So a failure at body
/// operation `k` is remembered, and the remaining elements are run only up
/// to operation `k` (`limit`): an earlier operation failing at a later
/// element replaces it. The error left at the end is the earliest failing
/// operation's, which is the unfused chain's. (An error names its operation's
/// message and span, never an element.)
///
/// # In parallel
///
/// A big kernel (`Config::fused_work`: elements times body operations) is
/// cut into contiguous chunks that run that same loop on a value table of
/// their own. Each chunk ends with
/// what the loop over its elements leaves: its data, and its failure, if any,
/// as the index of the failing body operation and the error at the lowest
/// element of the chunk that fails it. The kernel's error is then the
/// smallest operation index, and among chunks that fail the same operation
/// the lowest chunk: the element the sequential loop meets first. So neither
/// the chunks nor the order the threads finish in can change the bits or the
/// error.
fn fused_op(act: &mut Act<'_>, op: &OpData, jit: &mut Jit, par: Option<&ParRt>) -> R<()> {
    let s = op.span;
    let [region] = regions(op)?;
    let entry = act.entry_of(region, s)?;
    let block = act.block(entry, s)?;
    let Some((&last, body)) = block.ops.split_last() else {
        return Err(IrRuntimeError::malformed("the body of `fused` is empty", s));
    };
    let last = act.op(last, s)?;
    let (OpKind::Yield, &[yielded]) = (&last.kind, &last.operands[..]) else {
        return Err(IrRuntimeError::malformed(
            "the body of `fused` must end in a `yield` of one value",
            s,
        ));
    };
    if block.args.len() != op.operands.len() {
        return Err(IrRuntimeError::malformed(
            format!(
                "the body of `fused` takes {} arguments, given {}",
                block.args.len(),
                op.operands.len()
            ),
            s,
        ));
    }
    let inputs = act.values(&op.operands, s)?;
    // The shape of the arrays; scalars broadcast.
    let mut shape: Option<&Vec<usize>> = None;
    for input in &inputs {
        match input {
            Value::Array { dims, .. } => match shape {
                Some(d) if d != dims => {
                    return Err(IrRuntimeError::malformed(
                        "the array operands of `fused` differ in shape",
                        s,
                    ))
                }
                _ => shape = Some(dims),
            },
            Value::Number { .. } => {}
            other => {
                return Err(IrRuntimeError::malformed(
                    format!("`fused` takes numbers, found {}", other.kind()),
                    s,
                ))
            }
        }
    }
    let Some(dims) = shape else {
        return Err(IrRuntimeError::malformed(
            "`fused` needs an array operand",
            s,
        ));
    };
    let count: usize = dims.iter().product();
    // A long array: a native kernel first, whose result and error are this
    // loop's (see `crate::native`); it declines what it does not compile,
    // and the interpreted loop below runs it, in parallel chunks when big.
    match jit.run(act.func, op, &inputs, count) {
        Native::Done(data, dim) => {
            let dims = dims.clone();
            return act.define(op, vec![Value::array(dims, data, dim)]);
        }
        Native::Failed(e) => return Err(e),
        Native::Declined => {}
    }
    let kernel = Kernel {
        s,
        block,
        body,
        yielded,
        inputs: &inputs,
    };
    let chunk = match par {
        Some(rt) if count >= 2 && count.saturating_mul(body.len()) >= rt.cfg.fused_work => {
            match rt.pool() {
                Some(pool) => kernel.run_parallel(act.func, count, rt, pool)?,
                None => kernel.run(act, 0, count)?,
            }
        }
        _ => kernel.run(act, 0, count)?,
    };
    if let Some((_, e)) = chunk.failure {
        return Err(e);
    }
    let Some(dim) = chunk.out_dim else {
        return Err(IrRuntimeError::malformed("`fused` computed no element", s));
    };
    let dims = dims.clone();
    act.define(op, vec![Value::array(dims, chunk.data, dim)])
}

/// A `fused` kernel ready to run: its body and its evaluated operands.
struct Kernel<'a> {
    s: Span,
    block: &'a BlockData,
    /// The body's operations, without the `yield`.
    body: &'a [OpId],
    yielded: ValueId,
    inputs: &'a [Value],
}

/// What running a range of elements of a kernel leaves.
struct Chunk {
    /// The elements computed (short if the range failed).
    data: Vec<f64>,
    out_dim: Option<Dimension>,
    /// The failing body operation with the lowest index, its index, and its
    /// error at the lowest element of the range that failed it.
    failure: Option<(usize, IrRuntimeError)>,
}

impl Kernel<'_> {
    /// Runs elements `lo..hi` in order, in `act`'s value table: the loop
    /// described at [`fused_op`].
    fn run(&self, act: &mut Act<'_>, lo: usize, hi: usize) -> R<Chunk> {
        let s = self.s;
        let mut data = Vec::with_capacity(hi - lo);
        let mut out_dim = None;
        let mut limit = self.body.len();
        let mut failure: Option<(usize, IrRuntimeError)> = None;
        for i in lo..hi {
            for (arg, input) in self.block.args.iter().zip(self.inputs) {
                let element = match input {
                    Value::Array { data, dim, .. } => Value::number(data[i], *dim),
                    scalar => scalar.clone(),
                };
                act.set(*arg, element, s)?;
            }
            let mut failed = false;
            for (k, id) in self.body[..limit].iter().enumerate() {
                if let Err(e) = body_op(act, act.op(*id, s)?) {
                    (limit, failure, failed) = (k, Some((k, e)), true);
                    break;
                }
            }
            if limit == 0 {
                break;
            }
            if failed || failure.is_some() {
                continue;
            }
            match act.get(self.yielded, s)? {
                Value::Number { value, dim } => {
                    out_dim.get_or_insert(*dim);
                    data.push(*value);
                }
                other => {
                    return Err(IrRuntimeError::malformed(
                        format!("`fused` must yield a number, found {}", other.kind()),
                        s,
                    ))
                }
            }
        }
        Ok(Chunk {
            data,
            out_dim,
            failure,
        })
    }

    /// Runs all `count` elements in chunks on the pool and merges them: see
    /// [`fused_op`], "In parallel". A malformed-IR error of the lowest chunk
    /// that has one is the result, as it is the first the loop meets.
    fn run_parallel(
        &self,
        func: &Function,
        count: usize,
        rt: &ParRt,
        pool: &rayon::ThreadPool,
    ) -> R<Chunk> {
        let chunks = rt.cfg.chunks(count);
        let per = count.div_ceil(chunks);
        let results: Vec<R<Chunk>> = pool.install(|| {
            (0..chunks)
                .into_par_iter()
                .map(|c| {
                    let (lo, hi) = (c * per, ((c + 1) * per).min(count));
                    if lo >= hi {
                        return Ok(Chunk {
                            data: Vec::new(),
                            out_dim: None,
                            failure: None,
                        });
                    }
                    // The body uses only its own arguments and results: a
                    // table of its own is all a chunk needs.
                    self.run(&mut Act::new(func, None), lo, hi)
                })
                .collect()
        });
        let mut parts = Vec::with_capacity(chunks);
        for r in results {
            parts.push(r?);
        }
        // Smallest failing operation, then lowest chunk: the tuple order.
        let first = parts
            .iter()
            .enumerate()
            .filter_map(|(c, p)| p.failure.as_ref().map(|(k, _)| (*k, c)))
            .min();
        if let Some((_, c)) = first {
            return Ok(Chunk {
                data: Vec::new(),
                out_dim: None,
                failure: parts.swap_remove(c).failure,
            });
        }
        let out_dim = parts.iter().find_map(|p| p.out_dim);
        let mut data = Vec::with_capacity(count);
        for p in parts {
            data.extend(p.data);
        }
        Ok(Chunk {
            data,
            out_dim,
            failure: None,
        })
    }
}

/// An elementwise operation on arrays, run on the pool when the array is big
/// (`Config::array_elements`). Returns whether it did; `false` leaves the
/// operation to [`pure_op`], which is `sky_interp::arith` and produces every
/// error, so this only ever takes the case that has no error before the
/// entries are computed:
///
/// * `+`, `-`: two arrays of one shape and one dimension;
/// * `*`: an array and a number whose dimensions multiply;
/// * `/`: an array and a nonzero number whose dimensions divide;
/// * `-x` of an array.
///
/// The entries are the same `f64` operations as the sequential loop's, so
/// the same bits. The one failure left is that an entry is not finite, one
/// error for the whole operation whichever entry (or thread) found it.
fn elementwise(act: &mut Act<'_>, op: &OpData, rt: &ParRt) -> R<bool> {
    let s = op.span;
    let min = rt.cfg.array_elements.max(2);
    if let OpKind::Neg = op.kind {
        let [a] = fixed(op)?;
        let Value::Array { dims, data, dim } = act.get(a, s)? else {
            return Ok(false);
        };
        if data.len() < min {
            return Ok(false);
        }
        // `0 - x`, as `sky_interp::array::negate`: a zero stays `+0`.
        let Some((out, _)) = rt.map1(data, |x| 0.0 - x) else {
            return Ok(false);
        };
        let v = Value::array(dims.clone(), out, *dim);
        act.define(op, vec![v])?;
        return Ok(true);
    }
    let [a, b] = fixed(op)?;
    let (l, r) = (act.get(a, s)?, act.get(b, s)?);
    let (dims, dim, mapped) = match (&op.kind, l, r) {
        (
            OpKind::Add | OpKind::Sub,
            Value::Array {
                dims: d1,
                data: x,
                dim: da,
            },
            Value::Array {
                dims: d2,
                data: y,
                dim: db,
            },
        ) if d1 == d2 && da == db && x.len() >= min => {
            let f = if matches!(op.kind, OpKind::Add) {
                |p: f64, q: f64| p + q
            } else {
                |p: f64, q: f64| p - q
            };
            (d1, *da, rt.map2(x, y, f))
        }
        (OpKind::Mul, Value::Number { value: k, dim: dk }, Value::Array { dims, data, dim })
        | (OpKind::Mul, Value::Array { dims, data, dim }, Value::Number { value: k, dim: dk })
            if data.len() >= min =>
        {
            let Ok(out) = dim.checked_mul(*dk) else {
                return Ok(false);
            };
            let k = *k;
            (dims, out, rt.map1(data, |x| x * k))
        }
        (OpKind::Div, Value::Array { dims, data, dim }, Value::Number { value: k, dim: dk })
            if data.len() >= min && *k != 0.0 =>
        {
            let Ok(out) = dim.checked_div(*dk) else {
                return Ok(false);
            };
            let k = *k;
            (dims, out, rt.map1(data, |x| x / k))
        }
        _ => return Ok(false),
    };
    let Some((out, finite)) = mapped else {
        return Ok(false);
    };
    if !finite {
        return Err(sky_interp::arith::overflow(s).into());
    }
    let v = Value::array(dims.clone(), out, dim);
    act.define(op, vec![v])?;
    Ok(true)
}

/// One operation of a `fused` body: [`pure_op`], except that `neg` is `0 - x`
/// as `sky_interp::array::negate` computes it, so that a zero stays `+0` and
/// the entry is bit-identical to the array `neg` the body replaced.
fn body_op(act: &mut Act<'_>, op: &OpData) -> R<()> {
    if let OpKind::Neg = op.kind {
        let [a] = fixed(op)?;
        let v = match act.get(a, op.span)?.clone() {
            Value::Number { value, dim } => Value::number(0.0 - value, dim),
            other => sign(true, other, op.operand_span(0))?,
        };
        return act.define(op, vec![v]);
    }
    pure_op(act, op)
}

fn arith_op(act: &mut Act<'_>, op: &OpData, kind: ArithOp) -> R<()> {
    let [a, b] = fixed(op)?;
    let s = op.span;
    let (l, r) = (op.operand_span(0), op.operand_span(1));
    let v = arith(kind, act.get(a, s)?, act.get(b, s)?, s, l, r)?;
    act.define(op, vec![v])
}

/// `array.literal`: the operands, in row-major order, arranged as `cols`
/// says (see [`OpKind::ArrayLiteral`]) and built by `sky_interp`'s literal
/// rule, which checks that every entry is a number of one dimension.
fn array_literal(act: &mut Act<'_>, op: &OpData, cols: Option<u64>) -> R<()> {
    let s = op.span;
    let entries: Vec<(Value, Span)> = act
        .values(&op.operands, s)?
        .into_iter()
        .enumerate()
        .map(|(i, v)| (v, op.operand_span(i)))
        .collect();
    let width = match cols {
        None => entries.len(),
        Some(c) => usize::try_from(c).unwrap_or(0),
    };
    if width == 0 || entries.len() % width != 0 {
        return Err(IrRuntimeError::malformed(
            format!(
                "`array.literal` cannot arrange {} operands in rows of {width}",
                entries.len()
            ),
            s,
        ));
    }
    let rows: Vec<Vec<(Value, Span)>> = entries.chunks(width).map(<[_]>::to_vec).collect();
    let v = array::literal(rows, cols.is_none(), s)?;
    act.define(op, vec![v])
}

/// `slice`: the base, then each item's operands in order.
fn slice(act: &mut Act<'_>, op: &OpData, axes: &[SliceAxis]) -> R<()> {
    let s = op.span;
    let Some((&base, rest)) = op.operands.split_first() else {
        return Err(IrRuntimeError::malformed("`slice` has no operands", s));
    };
    let mut rest = rest.iter();
    // The next bound's operand, and the span it was read at (operand 0 is
    // the base).
    let mut taken = 0;
    let mut bound = |act: &Act<'_>| -> R<(usize, Span)> {
        let Some(&v) = rest.next() else {
            return Err(IrRuntimeError::malformed(
                "`slice` has fewer operands than its items need",
                s,
            ));
        };
        taken += 1;
        let at = op.operand_span(taken);
        Ok((array::to_index(act.get(v, s)?, at)?, at))
    };
    let mut items = Vec::with_capacity(axes.len());
    for (k, axis) in axes.iter().enumerate() {
        items.push(match *axis {
            SliceAxis::At => {
                let (i, at) = bound(act)?;
                Ix::At { i, span: at }
            }
            SliceAxis::Range { start, stop } => {
                let start = if start { Some(bound(act)?.0) } else { None };
                let stop = if stop { Some(bound(act)?.0) } else { None };
                Ix::Slice {
                    start,
                    stop,
                    span: op.axis_span(k),
                }
            }
        });
    }
    if rest.next().is_some() {
        return Err(IrRuntimeError::malformed(
            "`slice` has more operands than its items need",
            s,
        ));
    }
    let v = array::index(act.get(base, s)?, &items, s, op.operand_span(0))?;
    act.define(op, vec![v])
}
