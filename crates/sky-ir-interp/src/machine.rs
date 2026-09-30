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
//!
//! # Buffers are moved at their last use
//!
//! Arrays are immutable values, and [`Value::Array`] owns its `Vec`. So an
//! operand is never *shared*: it is either cloned or moved out of its slot.
//! The last-use analysis of `sky_ir::lifetime` says which operands die at
//! their operation, and the interpreter moves exactly those (a slot that is
//! read again after a move would report `used before it is defined`, so an
//! analysis bug is loud, not a silently wrong number). An elementwise
//! operation, or a `fused` one, whose operand is an owned array of the
//! result's shape writes the result into that buffer: `v = v + dt * a` in a
//! loop allocates no array. Values that are not consumed by an operation are
//! dropped right after it, so a dead array does not keep its memory until
//! the function returns.
//!
//! The rest of the machine avoids allocating per operation as well, so a
//! loop over arrays runs without allocating in steady state: results are
//! defined without a temporary `Vec`, the values a region hands out travel in
//! a buffer that is reused, and a `fused` body runs in a value table that is
//! reused. (A call still allocates its argument list and its frame.)
//!
//! Nothing is moved while a call group runs (see `Machine::run_group`): its
//! operations run out of program order, so "the last use" is not a place in
//! the order they run in. The dying operands of the group are dropped when
//! it is done.

use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use rayon::prelude::*;
use sky_diag::Span;
use sky_interp::arith::{
    arith_owned, arity_mismatch, equal, expect_bool, expect_number, order, pow, reciprocal,
    recursion_too_deep, sign, ArithOp, Operand, StackMeter, MAX_CALL_DEPTH, STACK_BUDGET,
};
use sky_interp::array::{self, Ix};
use sky_interp::contract;
use sky_interp::solve;
use sky_interp::Value;
use sky_ir::parallel::{CallGroup, Config, FnPlan, Role};
use sky_ir::{
    BlockData, BlockId, CmpOp, Function, Lifetimes, Module, NumberType, OpData, OpId, OpKind,
    RegionId, SliceAxis, Type, ValueId,
};
use sky_units::{Dimension, Rational};

use crate::kernel::{ScalarKernel, MAX_ARGS};
use crate::native::{Jit, JitOptions, Native};
use crate::par::ParRt;
use crate::IrRuntimeError;

type R<T> = Result<T, IrRuntimeError>;

/// How a block finished: by the terminator it ended in. The terminator's
/// operands are in the buffer the block was given (see [`Machine::block`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Exit {
    Yield,
    Condition,
    Return,
}

/// The last-use analysis of every function of a module, by position, made the
/// first time a function runs. Shared by the machines of a parallel run (the
/// analysis is plain data).
type LifeTable = Arc<Vec<OnceLock<Arc<Lifetimes>>>>;

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
    /// The function's position in the module.
    fidx: usize,
    /// Which operands die where (see `sky_ir::lifetime`); `None` for a value
    /// table that only runs `fused` bodies, which are not analyzed.
    life: Option<Arc<Lifetimes>>,
    /// The operation that is running, if its operands' deaths apply.
    owner: Option<OpId>,
    /// Nothing is moved while set: operations are running out of program
    /// order (a call group).
    frozen: bool,
}

impl<'m> Act<'m> {
    fn new(
        func: &'m Function,
        fidx: usize,
        call: Option<Span>,
        life: Option<Arc<Lifetimes>>,
    ) -> Self {
        Act {
            func,
            vals: vec![None; func.values.len()],
            call,
            plan: None,
            fidx,
            life,
            owner: None,
            frozen: false,
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

    /// Moves `v` out of its slot: the caller is at its last use.
    fn take(&mut self, v: ValueId, span: Span) -> R<Value> {
        self.vals
            .get_mut(v.index())
            .and_then(Option::take)
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

    /// Whether operand `i` of the running operation is at its last use.
    fn dies(&self, i: usize) -> bool {
        !self.frozen
            && match (self.owner, &self.life) {
                (Some(o), Some(l)) => l.dies(o, i),
                _ => false,
            }
    }

    /// Operand `i` of `op` by value: moved out of its slot if it dies at
    /// `op`, else cloned.
    fn operand(&mut self, op: &OpData, i: usize) -> R<Value> {
        let Some(&v) = op.operands.get(i) else {
            return Err(IrRuntimeError::malformed(
                format!("`{}` has no operand {i}", op.kind.mnemonic()),
                op.span,
            ));
        };
        if self.dies(i) {
            self.take(v, op.span)
        } else {
            self.get(v, op.span).cloned()
        }
    }

    /// Operand `i` of `op`, moved out of its slot if it dies at `op` and is
    /// an array (a scalar is as cheap to read as to move), else `None`: the
    /// caller borrows it with [`Act::get`].
    fn take_array_if_dies(&mut self, op: &OpData, i: usize) -> R<Option<Value>> {
        let Some(&v) = op.operands.get(i) else {
            return Ok(None);
        };
        if self.dies(i) && matches!(self.get(v, op.span)?, Value::Array { .. }) {
            return self.take(v, op.span).map(Some);
        }
        Ok(None)
    }

    /// Binds `op`'s only result to `value`.
    fn define1(&mut self, op: &OpData, value: Value) -> R<()> {
        match op.results[..] {
            [r] => self.set(r, value, op.span),
            _ => Err(IrRuntimeError::malformed(
                format!(
                    "`{}` has {} results, expected 1",
                    op.kind.mnemonic(),
                    op.results.len()
                ),
                op.span,
            )),
        }
    }

    /// Binds `op`'s results to the values in `values` (which are taken out
    /// of it), which must match in number.
    fn define_from(&mut self, op: &OpData, values: &mut Vec<Value>) -> R<()> {
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
        for (r, v) in op.results.iter().zip(values.drain(..)) {
            self.set(*r, v, op.span)?;
        }
        Ok(())
    }

    /// Drops the operands of `op` that die at it and are still in their
    /// slots (an operation that only borrowed them), so a dead array does
    /// not keep its memory until the function returns.
    fn release(&mut self, op_id: OpId, op: &OpData) {
        if self.frozen {
            return;
        }
        let Some(life) = &self.life else {
            return;
        };
        for (i, v) in op.operands.iter().enumerate() {
            if life.dies(op_id, i) {
                if let Some(slot) = self.vals.get_mut(v.index()) {
                    *slot = None;
                }
            }
        }
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
    /// The last-use analysis of each function of the module.
    lifetimes: LifeTable,
    /// Buffers for the values a region hands out (`yield`, `condition`,
    /// `return`) and for call arguments, so a loop reuses them.
    bufs: Vec<Vec<Value>>,
    /// Value tables for `fused` bodies, so a loop reuses them.
    frames: Vec<Vec<Option<Value>>>,
    /// Register files for the scalar kernels of `fused` bodies, reused.
    regs: Vec<Vec<f64>>,
    /// The scalar kernel of each `fused` operation that has one (function
    /// position and operation index), compiled the first time it runs.
    kernels: HashMap<(usize, usize), Option<Arc<ScalarKernel>>>,
    /// Ops left to run, in tests only: random malformed IR (a `while` whose
    /// condition is always true) must not hang the suite. Production has no
    /// step limit.
    #[cfg(test)]
    fuel: u64,
}

impl<'m, 'w> Machine<'m, 'w> {
    /// A machine for a run of `module`. Each machine has its own buffers and
    /// kernels (none of it crosses threads: the machines of a parallel run
    /// are made inside the threads that use them); only the last-use
    /// analysis, which is plain data, is shared.
    fn new(
        module: &'m Module,
        out: &'w mut dyn Write,
        stack: StackMeter,
        jit: JitOptions,
        par: Option<&'m ParRt>,
        lifetimes: LifeTable,
    ) -> Self {
        Machine {
            module,
            out,
            depth: 0,
            stack,
            jit: Jit::new(jit),
            par,
            lifetimes,
            bufs: Vec::new(),
            frames: Vec::new(),
            regs: Vec::new(),
            kernels: HashMap::new(),
            #[cfg(test)]
            fuel: u64::MAX,
        }
    }
}

/// An empty analysis table for `module`.
fn life_table(module: &Module) -> LifeTable {
    Arc::new(
        (0..module.functions.len())
            .map(|_| OnceLock::new())
            .collect(),
    )
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
    let mut m = Machine::new(
        module,
        out,
        StackMeter::new(),
        JitOptions::default(),
        rt.as_ref(),
        life_table(module),
    );
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
    let mut m = Machine::new(module, out, stack, jit, None, life_table(module));
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
    let mut m = Machine::new(
        module,
        out,
        StackMeter::new(),
        JitOptions::default(),
        None,
        life_table(module),
    );
    m.fuel = fuel;
    m.entry(name, args, Span::point(0), None)
}

impl<'m> Machine<'m, '_> {
    /// Looks `name` up for a call with `argc` arguments made at `span`: its
    /// position in the module, and the function.
    fn resolve(&self, name: &str, argc: usize, span: Span) -> R<(usize, &'m Function)> {
        let module = self.module;
        let Some(index) = module.functions.iter().position(|f| f.name == name) else {
            return Err(IrRuntimeError::malformed(
                format!("call to unknown function `{name}`"),
                span,
            ));
        };
        let func = &module.functions[index];
        if argc != func.params.len() {
            return Err(arity_mismatch(name, func.params.len(), argc, span).into());
        }
        Ok((index, func))
    }

    /// A buffer for values in transit, empty.
    fn take_buf(&mut self) -> Vec<Value> {
        self.bufs.pop().unwrap_or_default()
    }

    /// Returns a buffer for [`Machine::take_buf`] to hand out again.
    fn give_buf(&mut self, mut buf: Vec<Value>) {
        buf.clear();
        self.bufs.push(buf);
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
        let (index, func) = self.resolve(name, args.len(), span)?;
        self.activate(index, func, args, call)
    }

    /// A call made by a `call` op.
    fn call_named(&mut self, name: &str, args: Vec<Value>, span: Span) -> R<Vec<Value>> {
        let (index, func) = self.resolve(name, args.len(), span)?;
        if self.depth >= MAX_CALL_DEPTH {
            // A specialization `f[N=3]` is still `f` to the user.
            return Err(recursion_too_deep(sky_ir::source_name(name), span).into());
        }
        self.depth += 1;
        let result = self.activate(index, func, args, Some(span));
        self.depth -= 1;
        result
    }

    /// Runs the function at position `index` in a new activation. `args`
    /// becomes the buffer of the results.
    fn activate(
        &mut self,
        index: usize,
        func: &Function,
        mut args: Vec<Value>,
        call: Option<Span>,
    ) -> R<Vec<Value>> {
        let life = match self.lifetimes.get(index) {
            Some(cell) => Arc::clone(cell.get_or_init(|| Arc::new(Lifetimes::analyze(func)))),
            None => Arc::new(Lifetimes::default()),
        };
        let mut act = Act::new(func, index, call, Some(life));
        act.plan = self.par.and_then(|p| p.plan.function(&func.name));
        let entry = act.entry_of(func.body, func.span)?;
        match self.block(&mut act, entry, &mut args, func.span)? {
            Exit::Return if args.len() == func.results.len() => Ok(args),
            Exit::Return => Err(IrRuntimeError::malformed(
                format!(
                    "`{}` returns {} values, its signature has {}",
                    func.name,
                    args.len(),
                    func.results.len()
                ),
                func.span,
            )),
            Exit::Yield | Exit::Condition => Err(IrRuntimeError::malformed(
                format!("the body of `{}` must end in `return`", func.name),
                func.span,
            )),
        }
    }

    /// Runs a block: binds its arguments (the values in `io`, which is left
    /// empty), runs its ops in order, and returns how it ended, with the
    /// terminator's operands in `io`.
    fn block(
        &mut self,
        act: &mut Act<'_>,
        id: BlockId,
        io: &mut Vec<Value>,
        span: Span,
    ) -> R<Exit> {
        self.stack.check(span)?;
        let block = act.block(id, span)?;
        if block.args.len() != io.len() {
            return Err(IrRuntimeError::malformed(
                format!(
                    "{id} takes {} arguments, given {}",
                    block.args.len(),
                    io.len()
                ),
                span,
            ));
        }
        for (a, v) in block.args.iter().zip(io.drain(..)) {
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
            act.owner = Some(op_id);
            let exit = match op.kind {
                OpKind::Yield => Exit::Yield,
                OpKind::Condition => Exit::Condition,
                OpKind::Return => Exit::Return,
                _ => {
                    self.op(act, op)?;
                    act.release(op_id, op);
                    continue;
                }
            };
            // A terminator hands its operands on, moving the ones that die.
            for i in 0..op.operands.len() {
                io.push(act.operand(op, i)?);
            }
            return Ok(exit);
        }
        Err(IrRuntimeError::malformed(
            format!("{id} does not end in a terminator"),
            span,
        ))
    }

    /// Runs the (single) block of `region` with the arguments in `io`.
    fn region(
        &mut self,
        act: &mut Act<'_>,
        id: RegionId,
        io: &mut Vec<Value>,
        span: Span,
    ) -> R<Exit> {
        let entry = act.entry_of(id, span)?;
        self.block(act, entry, io, span)
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
                // The arguments die at the call if the caller is done with
                // them, and then the callee owns their buffers.
                let mut args = self.take_buf();
                for i in 0..op.operands.len() {
                    args.push(act.operand(op, i)?);
                }
                let mut results = self.call_named(callee, args, op.span)?;
                let defined = act.define_from(op, &mut results);
                self.give_buf(results);
                defined
            }
            OpKind::Print => self.print(act, op),
            OpKind::Fused => self.fused_op(act, op),
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
    ///
    /// # Nothing is moved inside a group
    ///
    /// The operations of a group do not run in program order: the in-place
    /// ones run first, the deferred ones after the started calls are done. A
    /// value that dies at an in-place operation may be read by a deferred
    /// one that comes before it in the program, and it would be gone by then.
    /// So the activation is frozen while the group runs (no operand is moved,
    /// nested regions included, and the arguments of a started call are
    /// cloned), and when the group is done, and every one of its operations
    /// with it, the operands that die at them are dropped.
    fn run_group(&mut self, act: &mut Act<'_>, group: &CallGroup) -> R<bool> {
        let Some((rt, pool)) = self.par.and_then(|rt| Some((rt, rt.pool()?))) else {
            return Ok(false);
        };
        if self.stack.used() > STACK_BUDGET / 2 {
            return Ok(false);
        }
        let was_frozen = std::mem::replace(&mut act.frozen, true);
        let result = self.run_group_frozen(act, group, rt, pool);
        act.frozen = was_frozen;
        if let Ok(true) = result {
            for &op_id in &group.ops {
                if let Ok(op) = act.op(op_id, act.func.span) {
                    act.release(op_id, op);
                }
            }
        }
        result
    }

    /// [`Machine::run_group`] with the activation frozen.
    fn run_group_frozen(
        &mut self,
        act: &mut Act<'_>,
        group: &CallGroup,
        rt: &'m ParRt,
        pool: &rayon::ThreadPool,
    ) -> R<bool> {
        let s = act.func.span;
        let started = group.roles.iter().filter(|&&r| r == Role::Start).count();
        let slots: Vec<Mutex<Option<R<Vec<Value>>>>> =
            (0..started).map(|_| Mutex::new(None)).collect();
        let (module, depth) = (self.module, self.depth);
        let table = Arc::clone(&self.lifetimes);
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
                            let table = Arc::clone(&table);
                            scope.spawn(move |_| {
                                // The callee has no `io`: nothing is written.
                                let mut sink = std::io::sink();
                                // Made here, in the thread that uses it: its
                                // buffers never cross threads.
                                let mut child = Machine::new(
                                    module,
                                    &mut sink,
                                    StackMeter::new(),
                                    JitOptions::default(),
                                    Some(rt),
                                    table,
                                );
                                child.depth = depth;
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
        let mut io = self.take_buf();
        match self.region(act, arm, &mut io, s)? {
            Exit::Yield => {
                let defined = act.define_from(op, &mut io);
                self.give_buf(io);
                defined
            }
            Exit::Condition | Exit::Return => Err(IrRuntimeError::malformed(
                "an `if` region must end in `yield`",
                s,
            )),
        }
    }

    /// `while` (see `OpKind::While`): `before` computes the condition and
    /// the forwarded values; if true, `after` computes the next loop-carried
    /// values; if false, the forwarded values are the results. The
    /// loop-carried values travel in one buffer, and are moved from an
    /// iteration to the next.
    fn while_op(&mut self, act: &mut Act<'_>, op: &OpData) -> R<()> {
        let s = op.span;
        let [before, after] = regions(op)?;
        let mut io = self.take_buf();
        for i in 0..op.operands.len() {
            io.push(act.operand(op, i)?);
        }
        loop {
            if self.region(act, before, &mut io, s)? != Exit::Condition {
                return Err(IrRuntimeError::malformed(
                    "a `while` `before` region must end in `condition`",
                    s,
                ));
            }
            if io.is_empty() {
                return Err(IrRuntimeError::malformed(
                    "`condition` needs a Bool operand",
                    s,
                ));
            }
            let flag = io.remove(0);
            if !expect_bool("while", &flag, s)? {
                let defined = act.define_from(op, &mut io);
                self.give_buf(io);
                return defined;
            }
            if self.region(act, after, &mut io, s)? != Exit::Yield {
                return Err(IrRuntimeError::malformed(
                    "a `while` `after` region must end in `yield`",
                    s,
                ));
            }
        }
    }

    /// `print(a, b, ...)`: the values separated by single spaces, then a
    /// newline, written in one piece as `sky run` does. The values are only
    /// read: text is written straight from them, nothing is cloned.
    #[inline(never)]
    fn print(&mut self, act: &mut Act<'_>, op: &OpData) -> R<()> {
        use std::fmt::Write as _;
        let s = op.span;
        let mut line = String::new();
        for (i, v) in op.operands.iter().enumerate() {
            if i > 0 {
                line.push(' ');
            }
            // `Value::print_text`, without its intermediate `String`.
            match act.get(*v, s)? {
                Value::Str(text) => line.push_str(text),
                other => {
                    let _ = write!(line, "{other}");
                }
            }
        }
        line.push('\n');
        self.out
            .write_all(line.as_bytes())
            .map_err(|e| sky_interp::RuntimeError::new(format!("cannot write output: {e}"), s))?;
        act.define_from(op, &mut Vec::new())
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
    /// # Which loop runs it
    ///
    /// In this order:
    ///
    /// 1. **Native**, for an array long enough (`crate::native`): compiled
    ///    vector code, whose result and error are this loop's. It always
    ///    allocates its result (the compiled loop may not overlap its
    ///    input), which is nothing next to the elements it computes.
    /// 2. **Parallel**, when the run is parallel and the kernel big
    ///    (`Config::fused_work`): contiguous chunks run the interpreted loop
    ///    on value tables of their own (below).
    /// 3. **Here**: the scalar kernel for a body of `+ - * /`, negation and
    ///    constants (`crate::kernel`), else the interpreted loop. This is the
    ///    only path that writes into a dying operand's buffer, so it is the
    ///    one a loop over short arrays runs, without allocating.
    ///
    /// The scalar kernel comes after the parallel path because it is
    /// sequential: a kernel big enough to be worth chunks is worth more spread
    /// over the threads than run faster on one.
    ///
    /// # Buffers
    ///
    /// The body runs in a value table of its own, reused from one `fused` to
    /// the next, so the operands are read where they are: no array is copied
    /// to run a kernel. If an array operand dies here (the last use), its
    /// buffer holds the result: element `i` is read from every operand
    /// before element `i` of the result is written, so the write cannot be
    /// seen by a later read.
    ///
    /// # In parallel
    ///
    /// A big kernel is cut into contiguous chunks that run the interpreted
    /// loop on a value table of their own. Each chunk ends with what the loop
    /// over its elements leaves: its data, and its failure, if any, as the
    /// index of the failing body operation and the error at the lowest element
    /// of the chunk that fails it. The kernel's error is then the smallest
    /// operation index, and among chunks that fail the same operation the
    /// lowest chunk: the element the sequential loop meets first. So neither
    /// the chunks nor the order the threads finish in can change the bits or
    /// the error.
    #[inline(never)]
    fn fused_op(&mut self, act: &mut Act<'_>, op: &OpData) -> R<()> {
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
        // The shape of the arrays; scalars broadcast.
        let mut shape: Option<&Vec<usize>> = None;
        for &v in &op.operands {
            match act.get(v, s)? {
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
        // Native or parallel: these read the operands as a slice and build a
        // result of their own.
        let parallel = match self.par {
            Some(rt) if count >= 2 && count.saturating_mul(body.len()) >= rt.cfg.fused_work => {
                rt.pool().map(|pool| (rt, pool))
            }
            _ => None,
        };
        if self.jit.eligible(count) || parallel.is_some() {
            let dims = dims.clone();
            let done = {
                let inputs = op
                    .operands
                    .iter()
                    .map(|&v| act.get(v, s))
                    .collect::<R<Vec<&Value>>>()?;
                let native = if self.jit.eligible(count) {
                    self.jit.run(act.func, op, &inputs, count)
                } else {
                    Native::Declined
                };
                match native {
                    Native::Done(data, dim) => Some((data, dim)),
                    Native::Failed(e) => return Err(e),
                    Native::Declined => match parallel {
                        Some((rt, pool)) => {
                            let kernel = Kernel {
                                s,
                                block,
                                body,
                                yielded,
                                inputs: &inputs,
                            };
                            let chunk = kernel.run_parallel(act.func, count, rt, pool)?;
                            if let Some((_, e)) = chunk.failure {
                                return Err(e);
                            }
                            let Some(dim) = chunk.out_dim else {
                                return Err(IrRuntimeError::malformed(
                                    "`fused` computed no element",
                                    s,
                                ));
                            };
                            Some((chunk.data, dim))
                        }
                        None => None,
                    },
                }
            };
            if let Some((data, dim)) = done {
                return act.define1(op, Value::array(dims, data, dim));
            }
        }
        // The operand whose buffer holds the result: the first array that dies
        // here. It is moved out of its slot, so the loop reads it from here.
        let mut target: Option<(usize, Vec<usize>, Vec<f64>, Dimension)> = None;
        for i in 0..op.operands.len() {
            if let Some(Value::Array { dims, data, dim }) = act.take_array_if_dies(op, i)? {
                target = Some((i, dims, data, dim));
                break;
            }
        }
        let mut fresh = Vec::new();
        if target.is_none() {
            fresh.reserve_exact(count);
        }
        let mut frame = Act {
            func: act.func,
            vals: self.frames.pop().unwrap_or_default(),
            call: None,
            plan: None,
            fidx: act.fidx,
            life: None,
            owner: None,
            frozen: false,
        };
        let kernel = match act.owner {
            Some(id) => self
                .kernels
                .entry((act.fidx, id.index()))
                .or_insert_with(|| {
                    ScalarKernel::compile(act.func, block, body, yielded).map(Arc::new)
                })
                .clone(),
            None => None,
        };
        let mut regs = self.regs.pop().unwrap_or_default();
        // Stale entries are fine: a body operation defines its result before
        // any later operation reads it.
        frame.vals.resize(act.func.values.len(), None);
        let outcome = fused_loop(
            act,
            &mut frame,
            op,
            body,
            yielded,
            count,
            &mut target,
            &mut fresh,
            kernel.as_deref(),
            &mut regs,
        );
        self.frames.push(frame.vals);
        self.regs.push(regs);
        let out_dim = outcome?;
        let (dims, data) = match target {
            Some((_, dims, data, _)) => (dims, data),
            None => (act_dims(act, op, s)?, fresh),
        };
        act.define1(op, Value::array(dims, data, out_dim))
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
            act.define1(op, Value::number(x, dim))
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
            act.define1(op, Value::number(x, dim))
        }
        OpKind::ConstBool(b) => act.define1(op, Value::Bool(*b)),
        OpKind::ConstStr(text) => act.define1(op, Value::Str(text.clone())),
        OpKind::ConstNothing => act.define1(op, Value::None),

        OpKind::Add => arith_op(act, op, ArithOp::Add),
        OpKind::Sub => arith_op(act, op, ArithOp::Sub),
        OpKind::Mul => arith_op(act, op, ArithOp::Mul),
        OpKind::Div => arith_op(act, op, ArithOp::Div),
        OpKind::Neg => {
            let [_] = fixed(op)?;
            // An array that dies here is negated in its own buffer.
            let v = sign(true, act.operand(op, 0)?, op.operand_span(0))?;
            act.define1(op, v)
        }
        OpKind::Math(f) => {
            // At most two operands: on the stack, so a call in a loop
            // allocates nothing.
            let n = op.operands.len();
            if n != f.arity() || n > 2 {
                return Err(IrRuntimeError::malformed(
                    format!("`{}` takes {} operands, found {n}", f.mnemonic(), f.arity(),),
                    s,
                ));
            }
            let mut vals = [Value::None, Value::None];
            let mut spans = [s; 2];
            for (i, &o) in op.operands.iter().enumerate() {
                vals[i] = act.get(o, s)?.clone();
                spans[i] = op.operand_span(i);
            }
            let v = sky_interp::math::apply(*f, &vals[..n], s, &spans[..n])?;
            act.define1(op, v)
        }
        // The same function `sky run` calls for `v as unit`, so both print
        // the same digits and fail the same way.
        OpKind::Show { unit } => {
            let [a] = fixed(op)?;
            let v = sky_interp::show_unit_text(unit, act.get(a, s)?, s)?;
            act.define1(op, v)
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
            act.define1(op, v)
        }
        OpKind::Pow => {
            let [a, e] = fixed(op)?;
            // A computed exponent gives a base with a unit no constant to
            // take the result's dimension from, as in `sky run`.
            let (l, r) = (op.operand_span(0), op.operand_span(1));
            let v = pow(act.get(a, s)?, act.get(e, s)?, || None::<Rational>, s, l, r)?;
            act.define1(op, v)
        }
        // The IR does not remember whether the source said `x⁻¹` or
        // `inv(x)`, so the (unreachable in checked programs) error for a
        // non-number always uses the `⁻¹` spelling.
        OpKind::Inverse => {
            let [a] = fixed(op)?;
            let v = reciprocal("⁻¹", act.get(a, s)?, s, op.operand_span(0))?;
            act.define1(op, v)
        }
        OpKind::MatMul => {
            let [a, b] = fixed(op)?;
            let (l, r) = (op.operand_span(0), op.operand_span(1));
            let v = array::matmul(act.get(a, s)?, act.get(b, s)?, s, l, r)?;
            act.define1(op, v)
        }
        // The functions `sky run` calls for `norm(x)` and `dot(a, b)`, with
        // the call's span and the arguments' spans, so the results and the
        // errors are the same bytes.
        OpKind::Norm => {
            let [a] = fixed(op)?;
            let v = sky_interp::vecfn::norm(act.get(a, s)?, s, op.operand_span(0))?;
            act.define1(op, v)
        }
        OpKind::Dot => {
            let [a, b] = fixed(op)?;
            let (l, r) = (op.operand_span(0), op.operand_span(1));
            let v = sky_interp::vecfn::dot(act.get(a, s)?, act.get(b, s)?, s, l, r)?;
            act.define1(op, v)
        }
        OpKind::Transpose => {
            let [a] = fixed(op)?;
            let v = array::transpose(act.get(a, s)?, s)?;
            act.define1(op, v)
        }
        // The Cholesky solve `sky run` performs where the checker rewrote
        // `A⁻¹ @ b`: the same function, so the same bits and errors.
        OpKind::Solve { method } => {
            let [a, b] = fixed(op)?;
            let (l, r) = (op.operand_span(0), op.operand_span(1));
            let v = solve::solve(*method, act.get(a, s)?, act.get(b, s)?, s, l, r)?;
            act.define1(op, v)
        }
        // `x @ A⁻¹`, the transposed solve: the same functions as the
        // tree-walker runs for a chain the checker planned.
        OpKind::SolveRight { method } => {
            let [x, a] = fixed(op)?;
            let (l, r) = (op.operand_span(0), op.operand_span(1));
            let v = solve::solve_right(*method, act.get(x, s)?, act.get(a, s)?, s, l, r)?;
            act.define1(op, v)
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
            act.define1(op, v)
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
            act.define1(op, v)
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
            act.define1(op, v)
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
            act.define1(op, v)
        }
        OpKind::Len => {
            let [a] = fixed(op)?;
            let v = array::len(act.get(a, s)?, op.operand_span(0))?;
            act.define1(op, v)
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
            act.define1(op, Value::Bool(holds))
        }
        OpKind::Not => {
            let [a] = fixed(op)?;
            let b = expect_bool("not", act.get(a, s)?, op.operand_span(0))?;
            act.define1(op, Value::Bool(!b))
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
            act.define1(op, Value::Bool(b))
        }
        OpKind::Assert { .. } => assert_op(act, op),

        // `+x` of a value the checker could not type: a number or an array
        // passes as it is, anything else fails as in `sky run`.
        OpKind::ExpectNumber { context } => {
            let [_] = fixed(op)?;
            // Passes its operand on: the buffer moves if it dies here.
            let v = act.operand(op, 0)?;
            expect_number(context.keyword(), &v, s)?;
            act.define1(op, v)
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

/// The shape of the first array operand of `op`.
fn act_dims(act: &Act<'_>, op: &OpData, span: Span) -> R<Vec<usize>> {
    for &v in &op.operands {
        if let Value::Array { dims, .. } = act.get(v, span)? {
            return Ok(dims.clone());
        }
    }
    Err(IrRuntimeError::malformed(
        "`fused` needs an array operand",
        span,
    ))
}

/// The element loop of [`Machine::fused_op`], sequential: reads the operands
/// from `act` (and the one moved into `target`), runs `body` in `frame`, and
/// writes each element into `target`'s buffer or pushes it on `fresh`. Returns
/// the result's dimension.
///
/// The first element runs the body as SIR. If that succeeds and the body has
/// a scalar `kernel` (see `crate::kernel`), the other elements run on `regs`
/// instead, with the same rule for which failure is reported. (`Kernel::run`
/// below is the same loop over a range of elements, which collects its data,
/// for the parallel chunks.)
#[allow(clippy::too_many_arguments)]
fn fused_loop(
    act: &Act<'_>,
    frame: &mut Act<'_>,
    op: &OpData,
    body: &[OpId],
    yielded: ValueId,
    count: usize,
    target: &mut Option<(usize, Vec<usize>, Vec<f64>, Dimension)>,
    fresh: &mut Vec<f64>,
    kernel: Option<&ScalarKernel>,
    regs: &mut Vec<f64>,
) -> R<Dimension> {
    let s = op.span;
    let entry = act.entry_of(op.regions[0], s)?;
    let block = act.block(entry, s)?;
    let mut out_dim = None;
    let mut limit = body.len();
    let mut failure: Option<IrRuntimeError> = None;
    // Where the kernel reads each operand from: an array's entries (the
    // target's are read from `target`), or a scalar already in `regs`.
    let mut srcs: [Option<&[f64]>; MAX_ARGS] = [None; MAX_ARGS];
    let mut fast: Option<&ScalarKernel> = None;
    for i in 0..count {
        let mut failed = false;
        if let Some(kernel) = fast {
            for (k, src) in srcs.iter().enumerate().take(kernel.nargs) {
                if let Some(entries) = src {
                    regs[k] = entries[i];
                }
            }
            if let Some((t, _, data, _)) = target {
                regs[*t] = data[i];
            }
            if let Err((k, e)) = kernel.run(regs, limit) {
                (limit, failure, failed) = (k, Some(e), true);
            }
        } else {
            for (k, (arg, &operand)) in block.args.iter().zip(&op.operands).enumerate() {
                let element = match target {
                    Some((t, _, data, dim)) if *t == k => Value::number(data[i], *dim),
                    _ => match act.get(operand, s)? {
                        Value::Array { data, dim, .. } => Value::number(data[i], *dim),
                        scalar => scalar.clone(),
                    },
                };
                frame.set(*arg, element, s)?;
            }
            for (k, id) in body[..limit].iter().enumerate() {
                if let Err(e) = body_op(frame, act.op(*id, s)?) {
                    (limit, failure, failed) = (k, Some(e), true);
                    break;
                }
            }
        }
        if limit == 0 {
            break;
        }
        if failed || failure.is_some() {
            continue;
        }
        let value = match fast {
            Some(kernel) => regs[kernel.yield_reg],
            None => match frame.get(yielded, s)? {
                Value::Number { value, dim } => {
                    out_dim.get_or_insert(*dim);
                    *value
                }
                other => {
                    return Err(IrRuntimeError::malformed(
                        format!("`fused` must yield a number, found {}", other.kind()),
                        s,
                    ))
                }
            },
        };
        match target {
            Some((_, _, data, _)) => data[i] = value,
            None => fresh.push(value),
        }
        // The first element went through: the rest can use the kernel.
        if i == 0 {
            if let (Some(kernel), None) = (kernel, &failure) {
                regs.clear();
                regs.resize(kernel.nregs, 0.0);
                for (k, &operand) in op.operands.iter().enumerate() {
                    // The target was moved out of its slot: `target` has it.
                    if matches!(target, Some((t, ..)) if *t == k) {
                        continue;
                    }
                    match act.get(operand, s)? {
                        Value::Number { value, .. } => regs[k] = *value,
                        Value::Array { data, .. } => srcs[k] = Some(data),
                        _ => {}
                    }
                }
                fast = Some(kernel);
            }
        }
    }
    if let Some(e) = failure {
        return Err(e);
    }
    out_dim.ok_or_else(|| IrRuntimeError::malformed("`fused` computed no element", s))
}

/// A `fused` kernel ready to run in parallel chunks: its body and its
/// evaluated operands.
struct Kernel<'a> {
    s: Span,
    block: &'a BlockData,
    /// The body's operations, without the `yield`.
    body: &'a [OpId],
    yielded: ValueId,
    inputs: &'a [&'a Value],
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
    /// described at [`Machine::fused_op`].
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
                    scalar => (*scalar).clone(),
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
    /// [`Machine::fused_op`], "In parallel". A malformed-IR error of the lowest chunk
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
                    self.run(&mut Act::new(func, 0, None, None), lo, hi)
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
        act.define1(op, v)?;
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
    act.define1(op, v)?;
    Ok(true)
}

/// One operation of a `fused` body: [`pure_op`], except that `neg` is `0 - x`
/// as `sky_interp::array::negate` computes it, so that a zero stays `+0` and
/// the entry is bit-identical to the array `neg` the body replaced.
fn body_op(act: &mut Act<'_>, op: &OpData) -> R<()> {
    if let OpKind::Neg = op.kind {
        let [a] = fixed(op)?;
        let v = match act.get(a, op.span)? {
            Value::Number { value, dim } => Value::number(0.0 - value, *dim),
            other => sign(true, other.clone(), op.operand_span(0))?,
        };
        return act.define1(op, v);
    }
    pure_op(act, op)
}

fn arith_op(act: &mut Act<'_>, op: &OpData, kind: ArithOp) -> R<()> {
    let [a, b] = fixed(op)?;
    let s = op.span;
    let (l, r) = (op.operand_span(0), op.operand_span(1));
    // An array operand that dies here may hold the result; see
    // `sky_interp::array::arith_owned`.
    let (owned_l, owned_r) = (
        act.take_array_if_dies(op, 0)?,
        act.take_array_if_dies(op, 1)?,
    );
    let lhs = match owned_l {
        Some(v) => Operand::Owned(v),
        None => Operand::Borrowed(act.get(a, s)?),
    };
    let rhs = match owned_r {
        Some(v) => Operand::Owned(v),
        None => Operand::Borrowed(act.get(b, s)?),
    };
    let v = arith_owned(kind, lhs, rhs, s, l, r)?;
    act.define1(op, v)
}

/// `array.literal`: the operands, in row-major order, arranged as `cols`
/// says (see [`OpKind::ArrayLiteral`]) and built by `sky_interp`'s literal
/// rule, which checks that every entry is a number of one dimension.
fn array_literal(act: &mut Act<'_>, op: &OpData, cols: Option<u64>) -> R<()> {
    let s = op.span;
    let entries = op
        .operands
        .iter()
        .enumerate()
        .map(|(i, &v)| Ok((act.get(v, s)?.clone(), op.operand_span(i))))
        .collect::<R<Vec<(Value, Span)>>>()?;
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
    act.define1(op, v)
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
    act.define1(op, v)
}
