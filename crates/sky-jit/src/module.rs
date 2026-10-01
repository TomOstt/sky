//! Whole-module translation: every SIR function becomes a Cranelift function,
//! compiled in memory for the host CPU, and `sky run --engine=native` calls
//! `<main>`.
//!
//! # What native code computes
//!
//! A value's *representation* follows from its static type. A scalar number
//! is an `f64` (its dimension is the type's: it is not stored), a Bool is an
//! `i8`, a *small array* is `N` `f64`s (see below), and everything else
//! (larger arrays, strings, `none`, and any value of type `unknown`) is an
//! opaque handle to a value the runtime owns ([`crate::runtime`]). Scalar
//! arithmetic, comparisons, `if`, `while` and calls between functions are
//! inline machine code; anything else is a call into the runtime, which runs
//! the same functions the SIR interpreter does.
//!
//! # Small arrays
//!
//! A vector or matrix whose shape is known at compile time and has at most
//! [`small::MAX_SMALL`] entries is *scalar replaced*: it is `N` separate
//! `f64` values, in row-major order, and there is no handle, no allocation
//! and no runtime call. The dimension is the type's, as for scalars. The
//! operations on them (`array.literal`, `add`, `sub`, `neg`, scaling, `index`,
//! `dot`, `norm`, `matmul`, `transpose`, `fused`, `zeros`, `identity`, `len`)
//! are inline code, in [`small`]; the arithmetic of every entry is the shared
//! kernels' (same operations in the same order, no fused multiply-add), and
//! a failure re-runs the operation on the whole operands in the runtime, so
//! the diagnostic is the interpreter's.
//!
//! **ABI.** A small array is passed exactly as its `N` entries would be as
//! `N` consecutive `f64` scalars: as `N` parameters of a function, `N`
//! results of a function, `N` parameters of a block (`if` results, `while`
//! loop-carried values), in row-major order. A function with a `Vector[3]`
//! parameter therefore has three `f64` parameters in its native signature. A
//! small array is *materialized* (boxed into a runtime handle) when it goes
//! where only a handle can: a runtime-only operation (a solve, `show`, an
//! operation on an operand of unknown type), `print`, an `assert` message, a
//! slot of `unknown` type (a call argument, result or loop value). A handle
//! flowing back into a slot typed as a small array is *unpacked* with a
//! check that its shape and dimension are what the type says; if they are
//! not, the run deoptimizes, as for a scalar.
//!
//! # Vectorizing
//!
//! The entries of a small array, and pairs of scalar operations that follow
//! one another, are computed two at a time as `F64X2` vector operations
//! when that pays ([`slp`]: lane-wise IEEE, so the same bits; one check for
//! the pair, whose failure re-runs the scalar operations in program order).
//! `SKY_SLP=off|auto|always` picks the mode for [`compile`]; the default,
//! `auto`, packs only in loops and functions that are not latency bound.
//!
//! # Exactness
//!
//! * `fadd`, `fsub`, `fmul`, `fdiv`, `fneg`, `fabs` and `sqrt` are IEEE
//!   operations that give the interpreter's bits. Nothing is contracted into
//!   a fused multiply-add: Cranelift never does it unless asked. (Asked:
//!   see "Relaxed numerics" below; [`compile`] never asks.)
//! * The checks `sky_interp::arith` makes come in the same order: the
//!   dimension rules (static here: a dimension is part of the type), then a
//!   zero divisor, then the result's finiteness. A failing check calls a
//!   runtime function that re-runs the operation with the same operands the
//!   interpreter's way, so the diagnostic (message, spans, labels, notes) is
//!   the interpreter's.
//! * Small integer powers are the same multiplications `powi` makes;
//!   `sqrt`, `abs`, `floor`, `ceil`, `min` and `max` are inline (they are
//!   exact); every other math function calls `sky_interp::math::apply_f64`,
//!   the numeric core of `apply`.
//! * A run-time failure returns a status up the native stack: every function
//!   returns `(status, results...)`, and a caller that sees a nonzero status
//!   returns at once. Effects (printing) happen in program order.
//!
//! # Relaxed numerics
//!
//! [`compile_with`] with [`Numerics::Relaxed`] is the one place native code
//! may leave the interpreter's bits (spec "Relaxed numerics"). On a host
//! with a fused multiply-add instruction it contracts `a*b + c`, `a*b - c`
//! and `c - a*b` into one `fma` (one rounding), when the product is used by
//! nothing else and the sum is the next operation (only constants may sit
//! between them, so no operation that could fail or print is reordered),
//! and it compiles `fused` kernels with the same contraction
//! ([`crate::kernel::Options::contract`]). Everything else is as above.
//!
//! What still holds: dimensions are checked (statically, as before);
//! results must be finite; and a failure is the same error at the same
//! place. The fused value's finiteness is checked, and if it is not finite
//! the cold path replays the two operations unfused: a product that overflows
//! fails at the product, exactly as in reproducible numerics; a sum that
//! overflows fails at the sum; and if the unfused sum turns out finite (the
//! fused one rounded past the largest double, the unfused did not) the
//! unfused value is used and the run goes on. What relaxes: a product that
//! overflows while the fused sum is finite (`x*y - z` with `z` almost as big
//! as `x*y`, which is past the largest double) no longer fails, and a fused
//! result differs from the unfused one in the last bit.
//!
//! # Values whose type is unknown
//!
//! The checker types a few things `unknown` (an unannotated parameter, a call
//! through one), and the interpreters then check them dynamically: `x + 1 m`
//! fails at run time if `x` turns out to be a time. Native code does the same
//! for operations that *use* an unknown value (they run in the runtime, on
//! boxed values). Typed values flowing into unknown slots are boxed, which is
//! exact. An unknown value flowing into a *typed* slot (a `-> m` on a
//! function whose body computes an unknown, an argument for a typed
//! parameter) is checked when it arrives: a typed `f64` carries no dimension,
//! so if the value were not what the type says, native code would compute
//! something else. It nearly always is, and then nothing more happens. If it
//! is not, the run *deoptimizes*: native code stops, [`Compiled::run`] says so
//! ([`RunFailure::Deopt`]), and the caller runs the program on the SIR
//! interpreter, skipping the output native code already wrote
//! ([`SkipWriter`]); [`Compiled::run_or_rerun`] does that. Output is the only
//! effect and programs are deterministic, so the result is the SIR
//! interpreter's exactly.
//!
//! # Threads
//!
//! [`compile_par`] takes the thread configuration of `sky run` (`--threads`,
//! `SKY_THREADS`: `sky_ir::parallel::Config`); with one thread it is
//! [`compile_with_slp`], and with more, native code runs three kinds of work
//! on the rayon pool, the three the SIR interpreter does, with the same
//! results, the same bytes and the same errors on every thread count (the
//! rule and the reasons are the module documentation of `sky_ir::parallel`).
//!
//! * **Big `fused` kernels** run as the deterministic chunks of
//!   [`crate::kernel::Kernel::run_par`]: the compiled loop on each chunk, the
//!   failure of the lowest failing element of the first failing operation in
//!   body order.
//! * **Big elementwise array operations** run in chunks ([`crate::par`]); the
//!   one failure left is an overflow of the whole operation.
//! * **Independent pure calls** are the windows of `sky_ir::parallel`
//!   (`CallGroup`): a call the analysis marks `Start` is *spawned* instead of
//!   called, and the window's operations are otherwise translated in program
//!   order. The spawn packs the arguments in memory and hands the callee's
//!   *thunk* (a function generated per started callee, `define_thunk`: it
//!   unpacks the arguments, makes the ordinary native call, stores the
//!   results) to a worker. A started call's results are defined at the next
//!   *join*: before the first operation that reads one (`Role::Deferred`) and at
//!   the end of the window. Only callees whose parameters and results are
//!   numbers, Bools and small arrays are started (a handle is an index into one
//!   thread's runtime state); any other call of a window is called in place, in
//!   its place in program order.
//!
//! **Why native code can run on several threads.** The generated code has no
//! global or static data and no state that outlives a call: a function's frame
//! is its own, constants are immediates, and every other effect goes through
//! the runtime. The runtime's state is per thread (`runtime::STATE`: handles,
//! pending error, kernel cache), and a worker makes one of its own for each
//! started call, sharing with the caller only the program's plain data
//! (`Arc<Program>`: the SIR module, the tables, the configuration) and the
//! arguments it was given by value. So two calls cannot reach each other's
//! values. The code is finalized and never written again; the thunk and the
//! callee are ordinary reentrant functions. `Compiled` is not `Send` and
//! `Compiled::run` does not return before every started call has ended (every
//! path out of a window joins it, and the runtime waits for strays when the
//! run ends), so no worker runs native code whose memory is freed.
//!
//! **Why the output cannot depend on the thread count.** A window has no
//! `print` and its callees have no `io` (and no recursion), so the only thing a
//! call does that anyone can see is fail (or deoptimize), and the window is
//! resolved exactly as a replay in program order would: the operations run in
//! program order, a call that is started does not change what the operations
//! between it and its join compute (they do not read its results), and any
//! failure of the window (an in-place operation, a join) first waits for the
//! calls started so far and installs the outcome of the earliest one that did
//! not succeed, which precedes the failing operation in program order. A call
//! that fails is an `Err` outcome in the worker's own runtime state, carried
//! back whole (message, spans, labels, notes): it is the error the call gives
//! anywhere. Calls get the call depth the sequential run gives them (the
//! depth check of a call is made before the spawn, at the same operation), so
//! "recursion is too deep" arises at the same call; a started call runs on a
//! stack of half the run's budget and, if it runs out of it, is run again
//! on the calling thread at the join, where its outcome is the sequential
//! run's; and a call is started only while the caller has used at most half
//! the budget (otherwise it is called in place), as in the interpreter.
//!
//! What is worth a thread differs from the interpreter: native code runs an
//! operation so much faster that a call must cost `NATIVE_CALL_COST_SCALE`
//! times the estimate, and a kernel have `NATIVE_FUSED_WORK_SCALE` times the
//! work, to be worth one (`Config::for_native`; `sky-bench par --native`
//! measures the break-evens). Native code is compiled for one thread
//! configuration, which decides the windows (`sky build` executables have
//! none).
//!
//! # Limits
//!
//! Calls nest at most [`MAX_CALL_DEPTH`] deep, counted as the SIR interpreter
//! counts (the entry call does not count), with the same error at the same
//! call. The interpreters also stop at [`STACK_BUDGET`] bytes of Rust stack;
//! native code has the same budget, checked at the entry of every function
//! that makes calls. A native frame is far smaller than an interpreter's, so
//! the depth limit is always what a program meets first, except for
//! functions with thousands of live values.
//!
//! # Ownership
//!
//! Handles are reference counted by the generated code. Every value of
//! handle representation is *owned* by the block that defines it (an
//! operation's result or a block argument) and released at the end of that
//! block. A value handed to another block (a `yield`, `condition` or
//! `return` operand, a call argument) is retained first, and the receiver
//! owns that count. An operand of an operation is only borrowed. Scalar-only
//! code has no handles and pays nothing for this.
//!
//! # Unsafe
//!
//! Calling compiled code and freeing its memory are the two `unsafe`
//! operations; each is commented where it is made. (Flushing the instruction
//! cache, `crate::icache`, is the third, in a module of its own.)

#![allow(unsafe_code)]

use std::collections::HashMap;
use std::fmt;
use std::io::Write;
use std::sync::Arc;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    self, types, AbiParam, Block, BlockArg, FuncRef, InstBuilder, MemFlags, Signature,
    StackSlotData, StackSlotKind, UserFuncName, Value as Cv,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Linkage, Module as ClModule};
use sky_diag::Span;
use sky_interp::arith::{ArithOp, MAX_CALL_DEPTH, STACK_BUDGET};
use sky_interp::linalg::Numerics;
use sky_interp::RuntimeError;
use sky_ir::exact::{self, Analysis};
use sky_ir::parallel::{CallGroup, Config, Plan, Role};
use sky_ir::{
    BlockId, CmpOp, Function, MathFn, Module as SirModule, OpData, OpId, OpKind, Type, ValueId,
};
use sky_units::Dimension;

use crate::runtime::{self, Program, Rt, Site, Ty, FUNC_SITE};

mod slp;
mod small;
pub use slp::Slp;
use small::{small_shape, Sh};

/// A program native code cannot run (yet): what `sky run --engine=native`
/// reports, and `--engine=auto` answers by falling back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileError {
    /// Why, without the "cannot compile natively" lead.
    pub message: String,
    /// The operation the refusal is about, if there is one.
    pub span: Option<Span>,
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cannot compile natively: {}", self.message)
    }
}

impl std::error::Error for CompileError {}

type Res<T> = Result<T, CompileError>;

fn refuse<T>(message: impl Into<String>, span: Option<Span>) -> Res<T> {
    Err(CompileError {
        message: message.into(),
        span,
    })
}

/// A module compiled to machine code. Run it with [`Compiled::run`].
///
/// Not `Send`: the code lives in the memory of the thread's compiler, which
/// is freed with it. Compile and run on the same thread (under
/// `sky_ast::with_deep_stack`, like every executor).
pub struct Compiled {
    jit: Option<JITModule>,
    program: Arc<Program>,
    /// The address of `<main>`, of type `extern "C" fn(i32, i64, i32) -> i32`.
    main: usize,
}

/// Why a native run did not finish.
#[derive(Clone, Debug, PartialEq)]
pub enum RunFailure {
    /// The program stopped with a run-time error: the diagnostic
    /// `sky_ir_interp::run` gives, whole.
    Error(RuntimeError),
    /// A value of unknown type was not what a declaration says (see the
    /// module docs), so native code cannot go on exactly. `written` bytes of
    /// output are already written; the caller runs the program again on the
    /// SIR interpreter without them.
    Deopt { written: usize },
}

impl fmt::Display for RunFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunFailure::Error(e) => e.fmt(f),
            RunFailure::Deopt { written } => write!(
                f,
                "native code stopped after {written} bytes of output: a value was not what its type says"
            ),
        }
    }
}

impl std::error::Error for RunFailure {}

/// A writer that drops the first `skip` bytes written to it, and passes the
/// rest on: what a rerun of a program uses after native code wrote that much.
pub struct SkipWriter<'a> {
    inner: &'a mut dyn Write,
    skip: usize,
}

impl<'a> SkipWriter<'a> {
    pub fn new(inner: &'a mut dyn Write, skip: usize) -> Self {
        SkipWriter { inner, skip }
    }
}

impl Write for SkipWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.skip == 0 {
            return self.inner.write(buf);
        }
        let n = self.skip.min(buf.len());
        self.skip -= n;
        if n == buf.len() {
            return Ok(n);
        }
        Ok(n + self.inner.write(&buf[n..])?)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

impl Compiled {
    /// Runs the module's `<main>`, writing `print` output to `out`. Exactly
    /// what `sky_ir_interp::run` does: the same output bytes, and the same
    /// failure, as a whole diagnostic; except that it may
    /// [deoptimize](RunFailure::Deopt).
    ///
    /// Run this under `sky_ast::with_deep_stack`: the stack budget assumes its
    /// stack.
    pub fn run(&self, out: &mut dyn Write) -> Result<(), RunFailure> {
        // SAFETY: `self.main` is `<main>`'s address (see `Compiled::main`),
        // and `self.jit`, which owns the code, is alive for the whole call.
        unsafe { run_entry(self.program.clone(), self.main, out) }
    }

    /// [`Compiled::run`], and on a deoptimization `rerun` (the SIR
    /// interpreter on the same module) with a writer that skips the output
    /// already written. The result is the run's, and whether it deoptimized.
    pub fn run_or_rerun<E: From<RuntimeError>>(
        &self,
        out: &mut dyn Write,
        rerun: impl FnOnce(&mut dyn Write) -> Result<(), E>,
    ) -> (Result<(), E>, bool) {
        match self.run(out) {
            Ok(()) => (Ok(()), false),
            Err(RunFailure::Error(e)) => (Err(E::from(e)), false),
            Err(RunFailure::Deopt { written }) => (rerun(&mut SkipWriter::new(out, written)), true),
        }
    }
}

/// Runs the compiled `<main>` at `main` with the runtime state of `program`
/// installed, writing `print` output to `out`: [`Compiled::run`] for code in
/// memory, and the entry point of an executable built ahead of time.
///
/// # Safety
///
/// `main` must be the address of `<main>` compiled by [`translate_module`]
/// from `program.module`, with the signature `(i32 depth, i64 stack limit,
/// i32 call site) -> i32 status` in the platform's default calling
/// convention, and its code must stay valid for the whole call.
pub(crate) unsafe fn run_entry(
    program: Arc<Program>,
    main: usize,
    out: &mut dyn Write,
) -> Result<(), RunFailure> {
    let guard = runtime::begin(program, out);
    let limit = sky_interp::arith::stack_address().saturating_sub(STACK_BUDGET) as i64;
    // SAFETY: the caller promises that `main` has this signature and is live.
    let entry: extern "C" fn(i32, i64, i32) -> i32 =
        unsafe { std::mem::transmute(main as *const u8) };
    let status = entry(0, limit, -1);
    let done = runtime::finish();
    drop(guard);
    // A finished run has released everything it made: a check of the
    // ownership rules, on in the tests.
    debug_assert!(
        status != 0 || done.live == 0,
        "native code leaked {} handles",
        done.live
    );
    match (status, done.error) {
        (0, _) => Ok(()),
        (_, _) if done.deopt => Err(RunFailure::Deopt {
            written: done.written,
        }),
        (_, Some(e)) => Err(RunFailure::Error(e)),
        (_, None) => Err(RunFailure::Error(RuntimeError::new(
            "internal error in the native engine: a failure without a diagnostic",
            Span::point(0),
        ))),
    }
}

impl Drop for Compiled {
    fn drop(&mut self) {
        if let Some(jit) = self.jit.take() {
            // SAFETY: nothing calls into this code after `self` is gone (the
            // only address kept, `main`, is dropped with it), and `run`
            // returned before any `&self` borrow ended.
            unsafe { jit.free_memory() };
        }
    }
}

/// Compiles every function of `module` (which must verify) to machine code
/// with reproducible numerics: the interpreter's bits.
pub fn compile(module: &SirModule) -> Result<Compiled, CompileError> {
    compile_with(module, Numerics::Reproducible)
}

/// [`compile`] with the numerics chosen: [`Numerics::Relaxed`] lets native
/// code contract `a*b + c` into a fused multiply-add (see the module docs).
/// Vectorization follows `SKY_SLP` (see [`Slp`]).
pub fn compile_with(module: &SirModule, numerics: Numerics) -> Result<Compiled, CompileError> {
    compile_with_slp(module, numerics, Slp::from_env())
}

/// [`compile_with`] with an explicit vectorization mode (see [`Slp`]).
pub fn compile_with_slp(
    module: &SirModule,
    numerics: Numerics,
    slp: Slp,
) -> Result<Compiled, CompileError> {
    compile_par(module, numerics, slp, &Config::sequential())
}

/// [`compile_with_slp`] for a run that may use the threads `par` allows
/// (`--threads=N`; see "Threads" below): big `fused` kernels and big
/// elementwise array operations run in chunks on the pool, exactly as the
/// SIR interpreter's do. With one thread (`par.threads == 1`) this is
/// [`compile_with_slp`].
pub fn compile_par(
    module: &SirModule,
    numerics: Numerics,
    slp: Slp,
    par: &Config,
) -> Result<Compiled, CompileError> {
    let built = build(module, false, numerics == Numerics::Relaxed, slp, par)?;
    let jit = built.jit;
    match (jit, built.main) {
        (Some(jit), Some(main)) => Ok(Compiled {
            jit: Some(jit),
            program: Arc::new(built.program),
            main,
        }),
        _ => refuse("the module has no `<main>`", None),
    }
}

/// The Cranelift IR of every function of `module`, as text
/// (`sky check --emit=clif`), before Cranelift optimizes it.
pub fn emit_clif(module: &SirModule) -> Result<String, CompileError> {
    emit_clif_with(module, Numerics::Reproducible)
}

/// [`emit_clif`] with the numerics chosen (relaxed shows the `fma`s).
pub fn emit_clif_with(module: &SirModule, numerics: Numerics) -> Result<String, CompileError> {
    emit_clif_with_slp(module, numerics, Slp::from_env())
}

/// [`emit_clif_with`] with an explicit vectorization mode (see [`Slp`]).
pub fn emit_clif_with_slp(
    module: &SirModule,
    numerics: Numerics,
    slp: Slp,
) -> Result<String, CompileError> {
    Ok(build(
        module,
        true,
        numerics == Numerics::Relaxed,
        slp,
        &Config::sequential(),
    )?
    .clif)
}

/// What [`build`] made.
struct Built {
    jit: Option<JITModule>,
    program: Program,
    main: Option<usize>,
    clif: String,
}

/// How a value is held in native code.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Rep {
    /// A scalar number: an `f64`.
    F64,
    /// A Bool: an `i8`, 0 or 1.
    Bool,
    /// A handle to a runtime-owned value: an `i64`, never 0.
    Ref,
    /// A small array of known shape: one `f64` per entry, row-major (see the
    /// module docs).
    Small(Sh),
}

fn rep_of(t: &Type) -> Rep {
    match t {
        Type::Number(n) if n.shape.rank() == 0 => Rep::F64,
        Type::Number(_) => small_shape(t).map_or(Rep::Ref, Rep::Small),
        Type::Bool => Rep::Bool,
        _ => Rep::Ref,
    }
}

/// The native types a value of representation `r` is held (and passed) as.
fn rep_tys(r: Rep) -> Vec<ir::Type> {
    match r {
        Rep::F64 => vec![types::F64],
        Rep::Bool => vec![types::I8],
        Rep::Ref => vec![types::I64],
        Rep::Small(sh) => vec![types::F64; sh.len()],
    }
}

fn cl_ty(t: Ty) -> ir::Type {
    match t {
        Ty::I32 => types::I32,
        Ty::I64 => types::I64,
        Ty::F64 => types::F64,
    }
}

/// The dimension of a scalar-number type.
fn dim_of_type(t: &Type) -> Option<Dimension> {
    match t {
        Type::Number(n) if n.shape.rank() == 0 => Some(n.dim),
        _ => None,
    }
}

/// The dimension of a number type of any shape.
fn number_dim(t: &Type) -> Option<Dimension> {
    match t {
        Type::Number(n) => Some(n.dim),
        _ => None,
    }
}

/// A value while it is being defined: what an operation computed, before it
/// is held as the representation its result's type asks for.
#[derive(Clone)]
enum Comp {
    F64(Cv, Dimension),
    Bool(Cv),
    /// A fresh handle (the definer owns it). If the result is typed as a
    /// scalar it is unpacked with a check (see the module docs).
    Ref(Cv),
    /// The entries of a small array, row-major, and its dimension.
    Small(Vec<Cv>, Sh, Dimension),
}

/// A defined value.
#[derive(Clone, Copy)]
struct Slot {
    v: Cv,
    rep: Rep,
}

/// State shared by the functions of a module while they are translated.
pub(crate) struct Shared<'m> {
    m: &'m SirModule,
    pub fn_ids: Vec<FuncId>,
    rt_ids: HashMap<Rt, FuncId>,
    pub dims: Vec<Dimension>,
    dim_ids: HashMap<Dimension, i32>,
    pub sites: Vec<Site>,
    /// Contract `a*b + c` into a fused multiply-add (relaxed numerics on a
    /// host that has the instruction).
    contract: bool,
    /// Pack pairs of `f64` operations into vector lanes (see [`Slp`]).
    slp: Slp,
    /// The calls that may run concurrently (see "Threads"); `None`: none do.
    plan: Option<&'m Plan>,
    /// The thunk of each function a window starts (see "Threads").
    thunks: Vec<Option<FuncId>>,
}

fn build(module: &SirModule, emit: bool, relaxed: bool, slp: Slp, par: &Config) -> Res<Built> {
    precheck(module)?;
    let isa = crate::host_isa().map_err(|e| CompileError {
        message: format!("no code generator for this machine ({e})"),
        span: None,
    })?;
    let call_conv = isa.default_call_conv();
    let contract = relaxed && crate::kernel::has_fma(&isa);
    let mut builder =
        crate::jit_builder(isa, crate::MODULE_ARENA_BYTES).map_err(|e| CompileError {
            message: format!("no memory for native code ({e})"),
            span: None,
        })?;
    for rt in Rt::ALL {
        builder.symbol(rt.name(), rt.address());
    }
    let mut jit = JITModule::new(builder);

    // Windows of independent calls, for a run with threads. Native code runs
    // a call in a fraction of the time the interpreter does, so a call must
    // be that much bigger to be worth starting elsewhere.
    let plan =
        (par.is_parallel() && !emit).then(|| sky_ir::parallel::analyze(module, &par.for_native()));
    let done = translate_module(
        module,
        &mut jit,
        call_conv,
        emit,
        contract,
        slp,
        plan.as_ref(),
    )?;
    let (sh, code_sizes, clif) = (done.sh, done.code_sizes, done.clif);
    let thunk_code = done.thunk_code;
    if emit {
        return Ok(Built {
            jit: None,
            program: Program {
                module: SirModule::default(),
                dims: Vec::new(),
                sites: Vec::new(),
                relaxed: false,
                par: Config::sequential(),
            },
            main: None,
            clif,
        });
    }
    jit.finalize_definitions()
        .map_err(|e| internal("the module", &e))?;
    for (id, size) in sh.fn_ids.iter().zip(&code_sizes) {
        crate::icache::flush(jit.get_finalized_function(*id), *size);
    }
    for (id, size) in &thunk_code {
        crate::icache::flush(jit.get_finalized_function(*id), *size);
    }
    let main_index = module
        .functions
        .iter()
        .position(|f| f.name == SirModule::MAIN)
        .unwrap_or(0);
    let main = jit.get_finalized_function(sh.fn_ids[main_index]) as usize;
    Ok(Built {
        jit: Some(jit),
        program: Program {
            module: module.clone(),
            dims: sh.dims,
            sites: sh.sites,
            relaxed,
            // The thresholds native code splits at (see `Config::for_native`).
            par: par.for_native(),
        },
        main: Some(main),
        clif,
    })
}

/// The module must verify and have a `<main>` without parameters or results.
pub(crate) fn precheck(module: &SirModule) -> Res<()> {
    if let Err(errors) = sky_ir::verify(module) {
        let first = errors.first();
        return refuse(
            format!(
                "the module does not verify ({})",
                first.map_or("no detail", |e| e.message.as_str())
            ),
            first.map(|e| e.span),
        );
    }
    let main = module.function(SirModule::MAIN);
    match main {
        Some(f) if f.params.is_empty() && f.results.is_empty() => {}
        Some(f) => return refuse("`<main>` takes parameters or returns values", Some(f.span)),
        None => return refuse("the module has no `<main>`", None),
    }
    Ok(())
}

pub(crate) fn internal(what: &str, e: &dyn fmt::Display) -> CompileError {
    CompileError {
        message: format!("the code generator failed on {what}: {e}"),
        span: None,
    }
}

/// What [`translate_module`] left: the shared tables and, per function, the
/// size of its machine code and (when only the IR was asked for) its text.
pub(crate) struct Translated<'m> {
    /// The ids of the functions, and the tables native code names by index.
    pub sh: Shared<'m>,
    /// The size of each function's machine code, to flush the instruction
    /// cache.
    pub code_sizes: Vec<usize>,
    /// The thunks (see "Threads"), with the size of each one's machine code.
    pub thunk_code: Vec<(FuncId, usize)>,
    pub clif: String,
}

/// Declares the runtime functions and every function of `module` in `jit`
/// (a JIT or an object file: the translation is the same), and translates
/// and defines each. `emit` only produces the Cranelift IR text; `contract`
/// fuses `a*b + c` (relaxed numerics); `slp` packs vector lanes.
pub(crate) fn translate_module<'m, M: ClModule>(
    module: &'m SirModule,
    jit: &mut M,
    call_conv: cranelift_codegen::isa::CallConv,
    emit: bool,
    contract: bool,
    slp: Slp,
    plan: Option<&'m Plan>,
) -> Res<Translated<'m>> {
    let mut sh = Shared {
        m: module,
        fn_ids: Vec::new(),
        rt_ids: HashMap::new(),
        dims: Vec::new(),
        dim_ids: HashMap::new(),
        sites: Vec::new(),
        contract,
        slp,
        plan,
        thunks: vec![None; module.functions.len()],
    };
    for rt in Rt::ALL {
        let (params, ret) = rt.signature();
        let mut sig = Signature::new(call_conv);
        sig.params
            .extend(params.iter().map(|t| AbiParam::new(cl_ty(*t))));
        if let Some(r) = ret {
            sig.returns.push(AbiParam::new(cl_ty(r)));
        }
        let id = jit
            .declare_function(rt.name(), Linkage::Import, &sig)
            .map_err(|e| internal(rt.name(), &e))?;
        sh.rt_ids.insert(rt, id);
    }
    for (i, f) in module.functions.iter().enumerate() {
        let sig = function_signature(f, call_conv);
        let id = jit
            .declare_function(&symbol_name(i, &f.name), Linkage::Local, &sig)
            .map_err(|e| internal(&f.name, &e))?;
        sh.fn_ids.push(id);
    }
    // A thunk for each function a window of the plan starts.
    let started = started_callees(module, plan);
    for (i, f) in module.functions.iter().enumerate() {
        if started[i] {
            let name = symbol_name(i, &f.name).replacen("sky_f", "sky_t", 1);
            let id = jit
                .declare_function(&name, Linkage::Local, &thunk_signature(call_conv))
                .map_err(|e| internal(&f.name, &e))?;
            sh.thunks[i] = Some(id);
        }
    }

    let mut clif = String::new();
    let mut code_sizes = Vec::new();
    let mut fctx = FunctionBuilderContext::new();
    let mut ctx = jit.make_context();
    for (i, f) in module.functions.iter().enumerate() {
        ctx.func.signature = function_signature(f, call_conv);
        ctx.func.name = UserFuncName::testcase(symbol_name(i, &f.name));
        FnTx::translate(&mut ctx.func, &mut fctx, jit, &mut sh, f, i as u32)?;
        if emit {
            clif.push_str(&format!("; {}\n{}\n", describe(f), ctx.func));
        } else {
            jit.define_function(sh.fn_ids[i], &mut ctx)
                .map_err(|e| internal(&f.name, &e))?;
            code_sizes.push(ctx.compiled_code().map_or(0, |c| c.code_buffer().len()));
        }
        jit.clear_context(&mut ctx);
    }
    let mut thunk_code = Vec::new();
    for i in 0..module.functions.len() {
        if let Some(id) = sh.thunks[i] {
            define_thunk(jit, &sh, i, call_conv, &mut fctx, &mut ctx)?;
            thunk_code.push((id, ctx.compiled_code().map_or(0, |c| c.code_buffer().len())));
            jit.clear_context(&mut ctx);
        }
    }
    Ok(Translated {
        sh,
        code_sizes,
        thunk_code,
        clif,
    })
}

// ----- threads -----

/// Whether a call of `g` may be started on another thread: its parameters
/// and results are numbers, Bools and small arrays, which are plain words. A
/// handle is an index into the runtime state of one thread (see
/// `runtime::tasks`) and cannot cross.
fn startable(g: &Function) -> bool {
    g.params
        .iter()
        .chain(&g.results)
        .all(|t| rep_of(t) != Rep::Ref)
}

/// Which functions some window of `plan` starts a call of, and can: the ones
/// that get a thunk.
fn started_callees(module: &SirModule, plan: Option<&Plan>) -> Vec<bool> {
    let mut started = vec![false; module.functions.len()];
    let Some(plan) = plan else {
        return started;
    };
    for f in &module.functions {
        let Some(fp) = plan.function(&f.name) else {
            continue;
        };
        for group in &fp.groups {
            for (&id, &role) in group.ops.iter().zip(&group.roles) {
                if role != Role::Start {
                    continue;
                }
                let Some(op) = f.ops.get(id.index()) else {
                    continue;
                };
                if let OpKind::Call { callee } = &op.kind {
                    if let Some(ci) = module.functions.iter().position(|g| &g.name == callee) {
                        if startable(&module.functions[ci]) {
                            started[ci] = true;
                        }
                    }
                }
            }
        }
    }
    started
}

/// `(depth: i32, stack limit: i64, call site: i32, args: i64, results: i64)
/// -> status: i32`: the signature of a thunk.
fn thunk_signature(call_conv: cranelift_codegen::isa::CallConv) -> Signature {
    let mut sig = Signature::new(call_conv);
    sig.params.push(AbiParam::new(types::I32));
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(types::I32));
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(types::I64));
    sig.returns.push(AbiParam::new(types::I32));
    sig
}

/// Defines the thunk of function `ci`: what a started call runs on a worker
/// (see "Threads"). It reads the callee's parameters, each one 8-byte word
/// (an `f64`, a Bool in the low byte, or the entries of a small array one word
/// each, in the order of the callee's native parameters) from `args`, makes
/// the ordinary call with the depth, stack limit and call site it was given,
/// and writes the results in the same way to `results` (a function that
/// returns through memory gets `results` as its result area, which has the
/// same layout). It returns the callee's status.
fn define_thunk<M: ClModule>(
    jit: &mut M,
    sh: &Shared<'_>,
    ci: usize,
    call_conv: cranelift_codegen::isa::CallConv,
    fctx: &mut FunctionBuilderContext,
    ctx: &mut cranelift_codegen::Context,
) -> Res<()> {
    let g = &sh.m.functions[ci];
    let Some(id) = sh.thunks[ci] else {
        return Ok(());
    };
    ctx.func.signature = thunk_signature(call_conv);
    ctx.func.name = UserFuncName::testcase(symbol_name(ci, &g.name).replacen("sky_f", "sky_t", 1));
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, fctx);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        let p = b.block_params(entry).to_vec();
        let mut args = vec![p[0], p[1], p[2]];
        let mut at: i32 = 0;
        for t in &g.params {
            for ty in rep_tys(rep_of(t)) {
                let v = b.ins().load(ty, MemFlags::trusted(), p[3], 8 * at);
                args.push(v);
                at += 1;
            }
        }
        let in_memory = results_in_memory(g);
        if in_memory {
            args.push(p[4]);
        }
        let callee = jit.declare_func_in_func(sh.fn_ids[ci], b.func);
        let call = b.ins().call(callee, &args);
        let results = b.inst_results(call).to_vec();
        if !in_memory {
            for (i, v) in results[1..].iter().enumerate() {
                b.ins().store(MemFlags::trusted(), *v, p[4], (8 * i) as i32);
            }
        }
        b.ins().return_(&[results[0]]);
        b.seal_all_blocks();
        b.finalize();
    }
    jit.define_function(id, ctx)
        .map_err(|e| internal(&g.name, &e))
}

fn describe(f: &Function) -> String {
    format!("sky function `{}`", f.name)
}

/// A symbol name for function number `i`: readable, and unique.
fn symbol_name(i: usize, name: &str) -> String {
    let clean: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    format!("sky_f{i}_{clean}")
}

/// Whether `f` returns its results through memory: a function with a small
/// array result has more result values than the machine returns in registers
/// (a 3 by 3 matrix is nine `f64`s; Cranelift refuses more than the ABI's
/// return registers), so its caller passes the address of an area of 8 bytes
/// per value as one more parameter, after the others, and the function stores
/// its results there, in order, before it returns `status` 0. Everything
/// else, including every function without a small array result, returns its
/// results in registers, as always.
fn results_in_memory(f: &Function) -> bool {
    f.results.iter().any(|r| matches!(rep_of(r), Rep::Small(_)))
        && f.results
            .iter()
            .map(|r| rep_tys(rep_of(r)).len())
            .sum::<usize>()
            > 1
}

/// `(depth: i32, stack limit: i64, call site: i32, params..., [result area:
/// i64]) -> (status: i32, results...)`; the result area and the results in
/// registers are as [`results_in_memory`] says.
fn function_signature(f: &Function, call_conv: cranelift_codegen::isa::CallConv) -> Signature {
    let mut sig = Signature::new(call_conv);
    sig.params.push(AbiParam::new(types::I32));
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(types::I32));
    for p in &f.params {
        sig.params
            .extend(rep_tys(rep_of(p)).into_iter().map(AbiParam::new));
    }
    let in_memory = results_in_memory(f);
    if in_memory {
        sig.params.push(AbiParam::new(types::I64));
    }
    sig.returns.push(AbiParam::new(types::I32));
    if !in_memory {
        for r in &f.results {
            sig.returns
                .extend(rep_tys(rep_of(r)).into_iter().map(AbiParam::new));
        }
    }
    sig
}

/// The sum that can fuse the product `body[i]`, if it is one: a scalar `mul`
/// whose result has no other use, feeding the next operation that is not a
/// constant, which must be an `add` or `sub` whose other operand and result
/// have the product's dimension. Only constants may sit between them, so no
/// operation that could fail, print or call is reordered. Everything is
/// decided from types, so a product that passes is always fused by
/// [`FnTx::fused_sum`].
fn fusable_product(f: &Function, uses: &[u32], body: &[OpId], i: usize) -> Option<OpId> {
    let mul = f.op(body[i]);
    let ([x, y], [m]) = (&mul.operands[..], &mul.results[..]) else {
        return None;
    };
    if !matches!(mul.kind, OpKind::Mul) || uses.get(m.index()) != Some(&1) {
        return None;
    }
    let (Some(dx), Some(dy), Some(dm)) = (
        dim_of_type(f.ty(*x)),
        dim_of_type(f.ty(*y)),
        dim_of_type(f.ty(*m)),
    ) else {
        return None;
    };
    if dx.checked_mul(dy).ok() != Some(dm) {
        return None;
    }
    let next = *body[i + 1..].iter().find(|id| {
        !matches!(
            f.op(**id).kind,
            OpKind::ConstFloat { .. } | OpKind::ConstRational(_) | OpKind::ConstBool(_)
        )
    })?;
    let sum = f.op(next);
    let ([l, r], [s]) = (&sum.operands[..], &sum.results[..]) else {
        return None;
    };
    if !matches!(sum.kind, OpKind::Add | OpKind::Sub) || l == r || (l != m && r != m) {
        return None;
    }
    let other = if l == m { r } else { l };
    (dim_of_type(f.ty(*other)) == Some(dm) && dim_of_type(f.ty(*s)) == Some(dm)).then_some(next)
}

/// The `a*b + c` patterns [`compile_with`] fuses under [`Numerics::Relaxed`]
/// (on a host with a fused multiply-add): for each, the source span of the
/// product and of the sum. For `sky check --explain`.
pub fn fused_sums(module: &SirModule) -> Vec<(Span, Span)> {
    let mut found = Vec::new();
    for f in &module.functions {
        let uses = use_counts(f);
        for block in &f.blocks {
            let Some((_, body)) = block.ops.split_last() else {
                continue;
            };
            for i in 0..body.len() {
                if let Some(sum) = fusable_product(f, &uses, body, i) {
                    found.push((f.op(body[i]).span, f.op(sum).span));
                }
            }
        }
    }
    found
}

/// How many operands refer to each value of `f`, in every operation of its
/// arena (a value used in a nested region counts too).
fn use_counts(f: &Function) -> Vec<u32> {
    let mut uses = vec![0u32; f.values.len()];
    for op in &f.ops {
        for v in &op.operands {
            if let Some(n) = uses.get_mut(v.index()) {
                *n += 1;
            }
        }
    }
    uses
}

fn args_of(vs: &[Cv]) -> Vec<BlockArg> {
    vs.iter().map(|v| BlockArg::Value(*v)).collect()
}

/// The translation of one function.
struct FnTx<'a, 'm, M: ClModule> {
    b: FunctionBuilder<'a>,
    jit: &'a mut M,
    sh: &'a mut Shared<'m>,
    f: &'m Function,
    fidx: u32,
    vals: Vec<Option<Slot>>,
    /// The entries of each value held as a small array (`Rep::Small`), by
    /// value.
    smalls: Vec<Option<Vec<Cv>>>,
    /// The number each value is known to be (a constant's), by value.
    consts: Vec<Option<f64>>,
    /// The handle values owned by each block being translated, innermost
    /// last.
    owned: Vec<Vec<ValueId>>,
    /// The one exit for every failure: returns status 1.
    err: Block,
    depth: Cv,
    stack_limit: Cv,
    call_site: Cv,
    rt_refs: HashMap<Rt, FuncRef>,
    fn_refs: HashMap<usize, FuncRef>,
    site_ids: HashMap<u32, i32>,
    /// Where each scalar that is a lane of a packed vector came from
    /// (see [`slp`]).
    lanes: HashMap<Cv, (Cv, u8)>,
    /// Whether array operations are vectorized in the loop or function body
    /// being translated, innermost last (see [`slp`]).
    slp_regions: Vec<bool>,
    /// What the exactness analysis proves about the function: an operation
    /// that can fail by kind and provably cannot needs no check.
    facts: Analysis,
    /// How many operands refer to each value (only when contracting).
    uses: Vec<u32>,
    /// Products whose emission waits for the sum that fuses them.
    pending: HashMap<ValueId, Pending>,
    /// The windows of independent calls being translated, innermost last
    /// (see "Threads").
    wins: Vec<Win>,
}

/// A window of independent calls being translated (see "Threads").
struct Win {
    /// The runtime group: an `i64` handle.
    gid: Cv,
    /// The failure exit the function had before the window opened.
    outer_err: Block,
    /// The failure exit inside the window: waits for the calls started so
    /// far (which installs the error of the earliest failing one as the
    /// pending error, see `runtime::tasks`), ends the group, and goes on to
    /// `outer_err`.
    err: Block,
    /// What each operation of the window does (see `sky_ir::parallel`).
    roles: HashMap<OpId, Role>,
    /// Calls started since the last join, in order: the call's position in
    /// the group, its operation and its callee.
    pending: Vec<(i32, OpId, usize)>,
    /// How many calls the window has started.
    spawned: i32,
}

/// A product `x * y` (the operation `site`) not yet emitted: the sum that
/// uses it will fuse it (relaxed numerics, see the module docs).
#[derive(Clone, Copy)]
struct Pending {
    x: Cv,
    y: Cv,
    site: OpId,
}

impl<'a, 'm, M: ClModule> FnTx<'a, 'm, M> {
    fn translate(
        func: &'a mut ir::Function,
        fctx: &'a mut FunctionBuilderContext,
        jit: &'a mut M,
        sh: &'a mut Shared<'m>,
        f: &'m Function,
        fidx: u32,
    ) -> Res<()> {
        let tx_contract = sh.contract;
        let mut b = FunctionBuilder::new(func, fctx);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        let err = b.create_block();
        let params = b.block_params(entry).to_vec();
        let mut tx = FnTx {
            b,
            jit,
            sh,
            f,
            fidx,
            vals: vec![None; f.values.len()],
            smalls: vec![None; f.values.len()],
            consts: vec![None; f.values.len()],
            owned: Vec::new(),
            err,
            depth: params[0],
            stack_limit: params[1],
            call_site: params[2],
            rt_refs: HashMap::new(),
            fn_refs: HashMap::new(),
            site_ids: HashMap::new(),
            lanes: HashMap::new(),
            slp_regions: Vec::new(),
            facts: exact::analyze(f),
            uses: if tx_contract {
                use_counts(f)
            } else {
                Vec::new()
            },
            pending: HashMap::new(),
            wins: Vec::new(),
        };
        let limit = params[1];
        let pack = tx.sh.slp == Slp::Auto && !slp::body_latency_bound(f);
        tx.slp_regions.push(pack);
        tx.owned.push(Vec::new());
        let entry_block = f.entry();
        tx.bind_all(f.param_values(), &params[3..]);
        if tx.has_calls() {
            tx.stack_check(limit);
        }
        let term = tx.block_body(entry_block)?;
        if !matches!(term.kind, OpKind::Return) || term.operands.len() != f.results.len() {
            return refuse(
                format!("the body of `{}` must end in a matching `return`", f.name),
                Some(f.span),
            );
        }
        let mut outs = vec![tx.iconst(types::I32, 0)];
        let mut values = Vec::new();
        for (i, &v) in term.operands.iter().enumerate() {
            let out = tx.transfer(term, v, &f.results[i], "a returned value")?;
            values.extend(out);
        }
        if results_in_memory(f) {
            let area = params[params.len() - 1];
            for (i, v) in values.into_iter().enumerate() {
                tx.b.ins()
                    .store(MemFlags::trusted(), v, area, (8 * i) as i32);
            }
        } else {
            outs.extend(values);
        }
        tx.release_owned();
        tx.b.ins().return_(&outs);

        // The one place a failure ends: status 1, no results.
        tx.b.switch_to_block(tx.err);
        tx.b.set_cold_block(tx.err);
        let mut outs = vec![tx.iconst(types::I32, 1)];
        if !results_in_memory(f) {
            for r in &f.results {
                let z = tx.zeros(rep_of(r));
                outs.extend(z);
            }
        }
        tx.b.ins().return_(&outs);
        tx.b.seal_all_blocks();
        tx.b.finalize();
        Ok(())
    }

    fn has_calls(&self) -> bool {
        self.f
            .ops
            .iter()
            .any(|o| matches!(o.kind, OpKind::Call { .. }))
    }

    // ----- small emitters -----

    fn iconst(&mut self, ty: ir::Type, n: i64) -> Cv {
        self.b.ins().iconst(ty, n)
    }

    fn zero(&mut self, rep: Rep) -> Cv {
        match rep {
            Rep::F64 | Rep::Small(_) => self.b.ins().f64const(0.0),
            Rep::Bool => self.iconst(types::I8, 0),
            Rep::Ref => self.iconst(types::I64, 0),
        }
    }

    /// A zero of every native value `rep` is held as.
    fn zeros(&mut self, rep: Rep) -> Vec<Cv> {
        (0..rep_tys(rep).len()).map(|_| self.zero(rep)).collect()
    }

    fn rt(&mut self, rt: Rt) -> FuncRef {
        if let Some(r) = self.rt_refs.get(&rt) {
            return *r;
        }
        let id = self.sh.rt_ids[&rt];
        let r = self.jit.declare_func_in_func(id, self.b.func);
        self.rt_refs.insert(rt, r);
        r
    }

    /// Calls a runtime function; its result, if it has one.
    fn call(&mut self, rt: Rt, args: &[Cv]) -> Option<Cv> {
        let r = self.rt(rt);
        let inst = self.b.ins().call(r, args);
        self.b.inst_results(inst).first().copied()
    }

    fn site(&mut self, op: OpId) -> i32 {
        if let Some(s) = self.site_ids.get(&op.0) {
            return *s;
        }
        let id = self.sh.sites.len() as i32;
        self.sh.sites.push(Site {
            func: self.fidx,
            op: op.0,
        });
        self.site_ids.insert(op.0, id);
        id
    }

    fn func_site(&mut self) -> i32 {
        let id = self.sh.sites.len() as i32;
        self.sh.sites.push(Site {
            func: self.fidx,
            op: FUNC_SITE,
        });
        id
    }

    fn dim_id(&mut self, d: Dimension) -> i32 {
        if let Some(i) = self.sh.dim_ids.get(&d) {
            return *i;
        }
        let i = self.sh.dims.len() as i32;
        self.sh.dims.push(d);
        self.sh.dim_ids.insert(d, i);
        i
    }

    fn c32(&mut self, n: i32) -> Cv {
        self.iconst(types::I32, i64::from(n))
    }

    /// Jumps to the failure exit if `bad` (an integer) is nonzero.
    fn fail_if(&mut self, bad: Cv) {
        let cont = self.b.create_block();
        let err = self.err;
        self.b.ins().brif(bad, err, &[], cont, &[]);
        self.b.switch_to_block(cont);
    }

    /// If `bad`, calls the failure function `rt` (which stores the error) and
    /// jumps to the failure exit.
    fn fail_with(&mut self, bad: Cv, rt: Rt, args: &[Cv]) {
        let cold = self.b.create_block();
        let cont = self.b.create_block();
        self.b.ins().brif(bad, cold, &[], cont, &[]);
        self.b.set_cold_block(cold);
        self.b.switch_to_block(cold);
        self.call(rt, args);
        let err = self.err;
        self.b.ins().jump(err, &[]);
        self.b.switch_to_block(cont);
    }

    /// Fails unconditionally: calls `rt` and leaves; what follows is
    /// unreachable and goes in a block of its own.
    fn fail_now(&mut self, rt: Rt, args: &[Cv]) {
        self.call(rt, args);
        let err = self.err;
        self.b.ins().jump(err, &[]);
        let dead = self.b.create_block();
        self.b.switch_to_block(dead);
    }

    /// Jumps to the failure exit unless `ok` (an integer) is nonzero.
    fn fail_unless(&mut self, ok: Cv) {
        let cont = self.b.create_block();
        let err = self.err;
        self.b.ins().brif(ok, cont, &[], err, &[]);
        self.b.switch_to_block(cont);
    }

    /// Fails if the number `r` a runtime function returned is NaN (which
    /// means it stored an error).
    fn nan_fail(&mut self, r: Cv) {
        let ok = self.b.ins().fcmp(FloatCC::Equal, r, r);
        self.fail_unless(ok);
    }

    /// Fails if the status `s` (an i32) is nonzero.
    fn status_fail(&mut self, s: Cv) {
        let bad = self.b.ins().icmp_imm(IntCC::NotEqual, s, 0);
        self.fail_if(bad);
    }

    /// Whether the exactness analysis proves that the operation `id`, which
    /// can fail by kind, cannot with these operands: no check is needed.
    fn proven(&self, id: OpId) -> bool {
        self.facts.proof(id).is_some()
    }

    /// Whether the analysis proves the scalar `v` is never zero.
    fn nonzero(&self, v: ValueId) -> bool {
        self.facts.known(v).is_some_and(|k| k.is_nonzero())
    }

    /// Whether `x` is finite (an `i8`): false for infinity and for NaN.
    ///
    /// `x - x` is 0 for a finite `x` and NaN for the rest, and "ordered"
    /// (not NaN) is one flag test on both machines this runs on. Testing
    /// `|x| < inf` instead needs the infinity in a register, which the
    /// register allocator rebuilds in every block (two more instructions
    /// per check, and a `fabs`).
    fn is_finite(&mut self, x: Cv) -> Cv {
        let d = self.b.ins().fsub(x, x);
        self.b.ins().fcmp(FloatCC::Ordered, d, d)
    }

    /// Fails if `x` (a result that must be finite) is not: the overflow
    /// check of `sky_interp::arith::finite`.
    fn finite_or(&mut self, x: Cv, rt: Rt, args: &[Cv]) {
        let ok = self.is_finite(x);
        let (cold, cont) = (self.b.create_block(), self.b.create_block());
        self.b.ins().brif(ok, cont, &[], cold, &[]);
        self.b.set_cold_block(cold);
        self.b.switch_to_block(cold);
        self.call(rt, args);
        let err = self.err;
        self.b.ins().jump(err, &[]);
        self.b.switch_to_block(cont);
    }

    /// The stack budget check at the entry of a function that makes calls.
    fn stack_check(&mut self, limit: Cv) {
        let sp = self.b.ins().get_stack_pointer(types::I64);
        let bad = self.b.ins().icmp(IntCC::UnsignedLessThan, sp, limit);
        let site = self.func_site();
        let site = self.c32(site);
        self.fail_with(bad, Rt::FailStack, &[site]);
    }

    // ----- values -----

    fn bind(&mut self, v: ValueId, s: Slot) {
        if s.rep == Rep::Ref {
            if let Some(top) = self.owned.last_mut() {
                top.push(v);
            }
        }
        self.vals[v.index()] = Some(s);
    }

    /// Binds `v` to the native values `cvs` a value of its type is held as
    /// (one for a scalar, a Bool or a handle, `N` for a small array).
    fn bind_flat(&mut self, v: ValueId, cvs: &[Cv]) {
        match rep_of(self.f.ty(v)) {
            Rep::Small(sh) => self.bind_small(v, sh, cvs.to_vec()),
            rep => self.bind(v, Slot { v: cvs[0], rep }),
        }
    }

    /// Binds each of `values` to its share of the flat list `cvs`.
    fn bind_all(&mut self, values: &[ValueId], cvs: &[Cv]) {
        let mut at = 0;
        for &v in values {
            let n = rep_tys(rep_of(self.f.ty(v))).len();
            self.bind_flat(v, &cvs[at..at + n]);
            at += n;
        }
    }

    /// Holds `v` as the small array with entries `elems`.
    fn bind_small(&mut self, v: ValueId, sh: Sh, elems: Vec<Cv>) {
        self.vals[v.index()] = Some(Slot {
            v: elems[0],
            rep: Rep::Small(sh),
        });
        self.smalls[v.index()] = Some(elems);
    }

    /// The entries of the small array `v`.
    fn small_elems(&self, v: ValueId, op: &OpData) -> Res<Vec<Cv>> {
        match self.smalls.get(v.index()).cloned().flatten() {
            Some(e) => Ok(e),
            None => refuse(format!("{v} is not a small array"), Some(op.span)),
        }
    }

    fn slot(&self, v: ValueId, op: &OpData) -> Res<Slot> {
        self.vals.get(v.index()).copied().flatten().map_or_else(
            || refuse(format!("{v} is used before it is defined"), Some(op.span)),
            Ok,
        )
    }

    fn operand(&self, op: &OpData, i: usize) -> Res<Slot> {
        match op.operands.get(i) {
            Some(v) => self.slot(*v, op),
            None => refuse(
                format!("`{}` is missing operand {i}", op.kind.mnemonic()),
                Some(op.span),
            ),
        }
    }

    /// The dimension of the scalar operand `i` (from its type).
    fn operand_dim(&self, op: &OpData, i: usize) -> Option<Dimension> {
        dim_of_type(self.f.ty(*op.operands.get(i)?))
    }

    /// Holds `c`, what `op` computed, as its result `i`, in the
    /// representation the result's type asks for.
    fn define(&mut self, op: &OpData, i: usize, c: Comp) -> Res<()> {
        let Some(&r) = op.results.get(i) else {
            return refuse(
                format!("`{}` has no result {i}", op.kind.mnemonic()),
                Some(op.span),
            );
        };
        let ty = self.f.ty(r);
        let slot = match (c, rep_of(ty)) {
            (Comp::F64(v, d), Rep::F64) => {
                if dim_of_type(ty) != Some(d) {
                    return refuse(
                        format!(
                            "the result of `{}` is typed with a different dimension than it computes",
                            op.kind.mnemonic()
                        ),
                        Some(op.span),
                    );
                }
                Slot { v, rep: Rep::F64 }
            }
            (Comp::Bool(v), Rep::Bool) => Slot { v, rep: Rep::Bool },
            (Comp::Ref(v), Rep::Ref) => Slot { v, rep: Rep::Ref },
            (Comp::F64(v, d), Rep::Ref) => {
                let d = self.dim_id(d);
                let d = self.c32(d);
                let h = self.call1(Rt::BoxNum, &[d, v]);
                self.check_handle(h);
                Slot {
                    v: h,
                    rep: Rep::Ref,
                }
            }
            (Comp::Bool(v), Rep::Ref) => {
                let w = self.b.ins().uextend(types::I32, v);
                let h = self.call1(Rt::BoxBool, &[w]);
                self.check_handle(h);
                Slot {
                    v: h,
                    rep: Rep::Ref,
                }
            }
            (Comp::Ref(h), Rep::F64) => {
                let d = dim_of_type(ty).unwrap_or(Dimension::DIMENSIONLESS);
                let d = self.dim_id(d);
                let d = self.c32(d);
                let x = self.call1(Rt::TakeNum, &[h, d]);
                self.nan_fail(x);
                Slot {
                    v: x,
                    rep: Rep::F64,
                }
            }
            (Comp::Ref(h), Rep::Bool) => {
                let x = self.call1(Rt::TakeBool, &[h]);
                let bad = self.b.ins().icmp_imm(IntCC::SignedLessThan, x, 0);
                self.fail_if(bad);
                let b = self.b.ins().icmp_imm(IntCC::NotEqual, x, 0);
                Slot {
                    v: b,
                    rep: Rep::Bool,
                }
            }
            (Comp::Small(elems, sh, d), Rep::Small(want)) => {
                if sh != want || number_dim(ty) != Some(d) {
                    return refuse(
                        format!(
                            "the result of `{}` is typed with a different shape or dimension than it computes",
                            op.kind.mnemonic()
                        ),
                        Some(op.span),
                    );
                }
                self.bind_small(r, sh, elems);
                return Ok(());
            }
            (Comp::Small(elems, sh, d), Rep::Ref) => {
                let h = self.box_small(&elems, sh, d);
                Slot {
                    v: h,
                    rep: Rep::Ref,
                }
            }
            (Comp::Ref(h), Rep::Small(sh)) => {
                let d = number_dim(ty).unwrap_or(Dimension::DIMENSIONLESS);
                let elems = self.unpack_small(h, sh, d, true);
                self.bind_small(r, sh, elems);
                return Ok(());
            }
            (Comp::F64(..), Rep::Bool | Rep::Small(_))
            | (Comp::Bool(_), Rep::F64 | Rep::Small(_))
            | (Comp::Small(..), Rep::F64 | Rep::Bool) => {
                return refuse(
                    format!("the result of `{}` has the wrong type", op.kind.mnemonic()),
                    Some(op.span),
                )
            }
        };
        self.bind(r, slot);
        Ok(())
    }

    /// Fails if a runtime call that should have returned a handle returned 0.
    fn check_handle(&mut self, h: Cv) {
        let bad = self.b.ins().icmp_imm(IntCC::Equal, h, 0);
        self.fail_if(bad);
    }

    /// The value `v` (an operand of `op`) as a value of type `to` that its
    /// receiver owns: a handle is retained, a scalar boxed if `to` is a
    /// handle type. `what` names the place, for a refusal.
    ///
    /// The result is the flat list of native values the receiver holds it
    /// as: one, or `N` for a small array.
    fn transfer(&mut self, op: &OpData, v: ValueId, to: &Type, what: &str) -> Res<Vec<Cv>> {
        let s = self.slot(v, op)?;
        let want = rep_of(to);
        match (s.rep, want) {
            (Rep::F64, Rep::F64) => {
                if dim_of_type(self.f.ty(v)) != dim_of_type(to) {
                    return refuse(
                        format!("{what} has a different dimension than its destination"),
                        Some(op.span),
                    );
                }
                Ok(vec![s.v])
            }
            (Rep::Bool, Rep::Bool) => Ok(vec![s.v]),
            (Rep::Ref, Rep::Ref) => {
                self.call(Rt::Retain, &[s.v]);
                Ok(vec![s.v])
            }
            (Rep::Small(a), Rep::Small(b)) => {
                if a != b || number_dim(self.f.ty(v)) != number_dim(to) {
                    return refuse(
                        format!("{what} has a different shape or dimension than its destination"),
                        Some(op.span),
                    );
                }
                self.small_elems(v, op)
            }
            (Rep::Small(sh), Rep::Ref) => {
                let d = number_dim(self.f.ty(v)).unwrap_or(Dimension::DIMENSIONLESS);
                let elems = self.small_elems(v, op)?;
                Ok(vec![self.box_small(&elems, sh, d)])
            }
            // A value of unknown type where a typed one is expected: check it.
            (Rep::Ref, Rep::Small(sh)) => {
                let d = number_dim(to).unwrap_or(Dimension::DIMENSIONLESS);
                Ok(self.unpack_small(s.v, sh, d, false))
            }
            (Rep::F64, Rep::Ref) => {
                let d = dim_of_type(self.f.ty(v)).unwrap_or(Dimension::DIMENSIONLESS);
                let d = self.dim_id(d);
                let d = self.c32(d);
                let h = self.call1(Rt::BoxNum, &[d, s.v]);
                self.check_handle(h);
                Ok(vec![h])
            }
            (Rep::Bool, Rep::Ref) => {
                let w = self.b.ins().uextend(types::I32, s.v);
                let h = self.call1(Rt::BoxBool, &[w]);
                self.check_handle(h);
                Ok(vec![h])
            }
            // A value of unknown type where a typed one is expected: check it.
            (Rep::Ref, Rep::F64) => {
                let d = dim_of_type(to).unwrap_or(Dimension::DIMENSIONLESS);
                let d = self.dim_id(d);
                let d = self.c32(d);
                let x = self.call1(Rt::PeekNum, &[s.v, d]);
                self.nan_fail(x);
                Ok(vec![x])
            }
            (Rep::Ref, Rep::Bool) => {
                let x = self.call1(Rt::PeekBool, &[s.v]);
                let bad = self.b.ins().icmp_imm(IntCC::SignedLessThan, x, 0);
                self.fail_if(bad);
                Ok(vec![self.b.ins().icmp_imm(IntCC::NotEqual, x, 0)])
            }
            _ => refuse(
                format!("{what} has a different type than its destination"),
                Some(op.span),
            ),
        }
    }

    /// Releases the handle values the innermost block owns, and forgets the
    /// block.
    fn release_owned(&mut self) {
        self.emit_release_owned();
        self.owned.pop();
    }

    fn emit_release_owned(&mut self) {
        let vs = self.owned.last().cloned().unwrap_or_default();
        for v in vs {
            if let Some(s) = self.vals[v.index()] {
                self.call(Rt::Release, &[s.v]);
            }
        }
    }

    // ----- blocks and operations -----

    /// Translates the operations of `bid` before its terminator, and returns
    /// the terminator.
    fn block_body(&mut self, bid: BlockId) -> Res<&'m OpData> {
        let f = self.f;
        let block = f.block(bid);
        let Some((term, body)) = block.ops.split_last() else {
            return refuse("a block has no terminator", Some(f.span));
        };
        // Windows of independent calls (see "Threads"): open before their
        // first operation, closed after their last.
        let windows = self.windows_in(bid, body);
        let mut next_window = 0;
        let mut open_end: Option<usize> = None;
        let mut i = 0;
        loop {
            if open_end == Some(i) {
                self.close_window()?;
                open_end = None;
            }
            if i >= body.len() {
                break;
            }
            if open_end.is_none() {
                if let Some(&(start, end, group)) = windows.get(next_window) {
                    if start == i {
                        self.open_window(group);
                        open_end = Some(end);
                        next_window += 1;
                    }
                }
            }
            if open_end.is_some() {
                match self.role_of(body[i]) {
                    // Reads a result that is not there yet: wait for the calls.
                    Some(Role::Deferred) => self.join_window()?,
                    Some(Role::Start) if self.can_start(body[i]) => {
                        self.spawn_op(body[i])?;
                        i += 1;
                        continue;
                    }
                    _ => {}
                }
            }
            // Relaxed numerics: a product that fuses into its sum is left for
            // the sum. Otherwise pairs of independent operations may share
            // vector lanes (SLP), and the rest are translated one by one.
            if self.sh.contract && fusable_product(f, &self.uses, body, i).is_some() {
                self.defer_product(body[i])?;
                i += 1;
                continue;
            }
            // A sum that reads a deferred product must be the fused one, so
            // it is never offered to SLP (which would read the product as a
            // value that was never emitted).
            let reads_pending = |id: OpId| {
                f.op(id)
                    .operands
                    .iter()
                    .any(|v| self.pending.contains_key(v))
            };
            // In a window, only two operations that both run in place are
            // paired: one that waits for a started call must not move before
            // the join, and none may straddle the end of the window.
            let in_zone = match open_end {
                None => true,
                Some(end) => {
                    i + 1 < end
                        && self.role_of(body[i]) == Some(Role::InPlace)
                        && self.role_of(body[i + 1]) == Some(Role::InPlace)
                }
            };
            let pairable = i + 1 < body.len()
                && in_zone
                && !reads_pending(body[i])
                && !reads_pending(body[i + 1]);
            if pairable && self.scalar_pair(body[i], body[i + 1])? {
                i += 2;
            } else {
                self.op(body[i])?;
                i += 1;
            }
        }
        Ok(f.op(*term))
    }

    // ----- threads -----
    //
    // A *window* is a run of operations of a block that `sky_ir::parallel`
    // found to hold independent pure calls (a `CallGroup`). Its operations are
    // translated in program order, but the calls marked `Role::Start` are
    // *spawned* (`runtime::tasks`: their arguments are packed in memory and
    // the callee's thunk runs on the pool) instead of called, and their
    // results are defined at the next *join*: before the first operation that
    // reads one (`Role::Deferred`), and at the end of the window. A failure
    // anywhere in the window goes to the window's own exit, which joins first:
    // the earliest failing call, which is earlier in program order than
    // the operation that failed (the calls started before it, and none after
    // it, are the window's calls so far), replaces the failure. So the
    // failure reported is the sequential run's.

    /// The role `id` has in the innermost open window.
    fn role_of(&self, id: OpId) -> Option<Role> {
        self.wins.last().and_then(|w| w.roles.get(&id).copied())
    }

    /// Whether the call `id` has a thunk, so that it can be started.
    fn can_start(&self, id: OpId) -> bool {
        let OpKind::Call { callee } = &self.f.op(id).kind else {
            return false;
        };
        self.sh
            .m
            .functions
            .iter()
            .position(|g| &g.name == callee)
            .is_some_and(|ci| self.sh.thunks[ci].is_some())
    }

    /// The windows of the plan in block `bid`: where each starts and ends in
    /// `body` (the block without its terminator) and its group, in order. A
    /// window is only opened if it starts some call natively.
    fn windows_in(&self, bid: BlockId, body: &[OpId]) -> Vec<(usize, usize, &'m CallGroup)> {
        let Some(plan) = self.sh.plan else {
            return Vec::new();
        };
        let Some(fp) = plan.function(&self.f.name) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for g in &fp.groups {
            if g.block != bid {
                continue;
            }
            let Some(start) = g.ops.first().and_then(|o| body.iter().position(|b| b == o)) else {
                continue;
            };
            let end = start + g.ops.len();
            if body.get(start..end) != Some(&g.ops[..]) {
                continue;
            }
            let starts_one = g
                .ops
                .iter()
                .zip(&g.roles)
                .any(|(&o, &r)| r == Role::Start && self.can_start(o));
            if starts_one {
                out.push((start, end, g));
            }
        }
        out.sort_by_key(|w| w.0);
        out
    }

    /// Opens a window: a new group, and a failure exit of its own.
    fn open_window(&mut self, group: &CallGroup) {
        let gid = self.call1(Rt::GroupBegin, &[]);
        let err = self.b.create_block();
        let outer_err = std::mem::replace(&mut self.err, err);
        self.wins.push(Win {
            gid,
            outer_err,
            err,
            roles: group
                .ops
                .iter()
                .copied()
                .zip(group.roles.iter().copied())
                .collect(),
            pending: Vec::new(),
            spawned: 0,
        });
    }

    /// Closes the innermost window: waits for its calls, ends the group, and
    /// writes the window's failure exit.
    fn close_window(&mut self) -> Res<()> {
        self.join_window()?;
        let Some(win) = self.wins.pop() else {
            return Ok(());
        };
        self.call(Rt::GroupEnd, &[win.gid]);
        let after = self.b.create_block();
        self.b.ins().jump(after, &[]);
        // The failure exit: join (which replaces the pending error by an
        // earlier call's), end the group, and fail as the function does.
        self.b.switch_to_block(win.err);
        self.b.set_cold_block(win.err);
        self.call(Rt::Join, &[win.gid]);
        self.call(Rt::GroupEnd, &[win.gid]);
        self.b.ins().jump(win.outer_err, &[]);
        self.b.switch_to_block(after);
        self.err = win.outer_err;
        Ok(())
    }

    /// Waits for the calls the innermost window has started since the last
    /// join, fails if one failed, and defines their results.
    fn join_window(&mut self) -> Res<()> {
        let Some(win) = self.wins.last_mut() else {
            return Ok(());
        };
        if win.pending.is_empty() {
            return Ok(());
        }
        let gid = win.gid;
        let pending = std::mem::take(&mut win.pending);
        let status = self.call1(Rt::Join, &[gid]);
        self.status_fail(status);
        let m = self.sh.m;
        let f = self.f;
        for (k, id, ci) in pending {
            let op = f.op(id);
            let g = &m.functions[ci];
            let kv = self.c32(k);
            let ptr = self.call1(Rt::TaskResult, &[gid, kv]);
            let tys: Vec<ir::Type> = g.results.iter().flat_map(|r| rep_tys(rep_of(r))).collect();
            // Position 0 stands for the status, which `bind_call_results`
            // skips.
            let mut results = vec![status];
            for (i, t) in tys.into_iter().enumerate() {
                let v = self
                    .b
                    .ins()
                    .load(t, MemFlags::trusted(), ptr, (8 * i) as i32);
                results.push(v);
            }
            self.bind_call_results(op, g, &results)?;
        }
        Ok(())
    }

    /// Starts the call `id` of a window: the depth check of a call, the
    /// arguments packed in memory, and a spawn of the callee's thunk.
    fn spawn_op(&mut self, id: OpId) -> Res<()> {
        let f = self.f;
        let m = self.sh.m;
        let op = f.op(id);
        let OpKind::Call { callee } = &op.kind else {
            return refuse(
                "a window starts something that is not a call",
                Some(op.span),
            );
        };
        let Some(ci) = m.functions.iter().position(|g| &g.name == callee) else {
            return refuse(
                format!("a call to unknown function `{callee}`"),
                Some(op.span),
            );
        };
        let g = &m.functions[ci];
        let Some(thunk) = self.sh.thunks[ci] else {
            return self.call_op(id, op, callee);
        };
        if op.operands.len() != g.params.len() || op.results.len() != g.results.len() {
            return refuse(
                format!("a call to `{callee}` does not match its signature"),
                Some(op.span),
            );
        }
        let site = self.site(id);
        let site_v = self.c32(site);
        let limit = self.c32(MAX_CALL_DEPTH as i32);
        let bad = self
            .b
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, self.depth, limit);
        self.fail_with(bad, Rt::FailDepth, &[site_v]);
        let depth = self.b.ins().iadd_imm(self.depth, 1);
        let mut words = Vec::new();
        for (i, &v) in op.operands.iter().enumerate() {
            words.extend(self.transfer(
                op,
                v,
                &g.params[i],
                &format!("argument {i} of a call to `{callee}`"),
            )?);
        }
        let slot = self.b.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            8 * words.len().max(1) as u32,
            3,
        ));
        let addr = self.b.ins().stack_addr(types::I64, slot, 0);
        for (k, w) in words.iter().enumerate() {
            self.b
                .ins()
                .store(MemFlags::trusted(), *w, addr, (8 * k) as i32);
        }
        let nres: usize = g.results.iter().map(|r| rep_tys(rep_of(r)).len()).sum();
        let tref = self.jit.declare_func_in_func(thunk, self.b.func);
        let taddr = self.b.ins().func_addr(types::I64, tref);
        let gid = match self.wins.last_mut() {
            Some(w) => {
                w.pending.push((w.spawned, id, ci));
                w.spawned += 1;
                w.gid
            }
            None => return refuse("a call is started outside a window", Some(op.span)),
        };
        let nargs = self.c32(words.len() as i32);
        let nres = self.c32(nres as i32);
        let stack_limit = self.stack_limit;
        self.call(
            Rt::Spawn,
            &[gid, taddr, depth, stack_limit, site_v, addr, nargs, nres],
        );
        Ok(())
    }

    // ----- relaxed numerics: contraction -----

    /// Holds the product `id` back for the sum that will fuse it.
    fn defer_product(&mut self, id: OpId) -> Res<()> {
        let op = self.f.op(id);
        let (x, y) = (self.operand(op, 0)?, self.operand(op, 1)?);
        self.pending.insert(
            op.results[0],
            Pending {
                x: x.v,
                y: y.v,
                site: id,
            },
        );
        Ok(())
    }

    /// `x*y + c`, `x*y - c` or `c - x*y` as one fused multiply-add, where
    /// `product` is which operand of the sum was the deferred product.
    fn fused_sum(&mut self, id: OpId, op: &OpData, kind: ArithOp) -> Res<()> {
        let product_is_lhs = self.pending.contains_key(&op.operands[0]);
        let (pv, ci) = if product_is_lhs {
            (op.operands[0], 1)
        } else {
            (op.operands[1], 0)
        };
        let Some(p) = self.pending.remove(&pv) else {
            return refuse("a fused product is missing", Some(op.span));
        };
        let c = self.operand(op, ci)?.v;
        let r = match (kind, product_is_lhs) {
            (ArithOp::Sub, true) => {
                let nc = self.b.ins().fneg(c);
                self.b.ins().fma(p.x, p.y, nc)
            }
            (ArithOp::Sub, false) => {
                let nx = self.b.ins().fneg(p.x);
                self.b.ins().fma(nx, p.y, c)
            }
            _ => self.b.ins().fma(p.x, p.y, c),
        };
        let r = if self.proven(p.site) && self.proven(id) {
            r
        } else {
            self.finite_or_unfused(r, p, c, kind, product_is_lhs, id)
        };
        let dim = dim_of_type(self.f.ty(op.results[0])).unwrap_or(Dimension::DIMENSIONLESS);
        self.define(op, 0, Comp::F64(r, dim))
    }

    /// The fused value `r` if it is finite. Otherwise the two operations are
    /// replayed unfused, as reproducible numerics would run them: a product
    /// that is not finite fails as the product, a sum that is not finite
    /// fails as the sum (both with the interpreter's error), and a finite
    /// unfused sum (the fused one rounded past the largest double, the
    /// unfused one did not) is the value.
    fn finite_or_unfused(
        &mut self,
        r: Cv,
        p: Pending,
        c: Cv,
        kind: ArithOp,
        product_is_lhs: bool,
        id: OpId,
    ) -> Cv {
        let ok = self.is_finite(r);
        let (cold, cont) = (self.b.create_block(), self.b.create_block());
        self.b.append_block_param(cont, types::F64);
        self.b
            .ins()
            .brif(ok, cont, &[BlockArg::Value(r)], cold, &[]);
        self.b.set_cold_block(cold);
        self.b.switch_to_block(cold);
        let prod = self.b.ins().fmul(p.x, p.y);
        let prod_ok = self.is_finite(prod);
        let (mul_bad, sum_check) = (self.b.create_block(), self.b.create_block());
        self.b.ins().brif(prod_ok, sum_check, &[], mul_bad, &[]);
        self.b.set_cold_block(mul_bad);
        self.b.switch_to_block(mul_bad);
        let site = self.site(p.site);
        let site = self.c32(site);
        self.call(Rt::Fail2, &[site, p.x, p.y]);
        let err = self.err;
        self.b.ins().jump(err, &[]);
        self.b.set_cold_block(sum_check);
        self.b.switch_to_block(sum_check);
        let (l, rr) = if product_is_lhs { (prod, c) } else { (c, prod) };
        let sum = match kind {
            ArithOp::Sub => self.b.ins().fsub(l, rr),
            _ => self.b.ins().fadd(l, rr),
        };
        let sum_ok = self.is_finite(sum);
        let sum_bad = self.b.create_block();
        self.b
            .ins()
            .brif(sum_ok, cont, &[BlockArg::Value(sum)], sum_bad, &[]);
        self.b.set_cold_block(sum_bad);
        self.b.switch_to_block(sum_bad);
        let site = self.site(id);
        let site = self.c32(site);
        self.call(Rt::Fail2, &[site, l, rr]);
        self.b.ins().jump(err, &[]);
        self.b.switch_to_block(cont);
        self.b.block_params(cont)[0]
    }

    fn call1(&mut self, rt: Rt, args: &[Cv]) -> Cv {
        match self.call(rt, args) {
            Some(v) => v,
            None => self.zero(Rep::Ref),
        }
    }

    /// The dimension of a constant, from its result's type (`unknown` is
    /// dimensionless, as in the interpreter).
    fn const_dim(&self, op: &OpData) -> Res<Dimension> {
        let Some(&r) = op.results.first() else {
            return refuse("a constant has no result", Some(op.span));
        };
        match self.f.ty(r) {
            Type::Number(n) if n.shape.rank() == 0 => Ok(n.dim),
            Type::Unknown => Ok(Dimension::DIMENSIONLESS),
            _ => refuse("a constant must have a scalar number result", Some(op.span)),
        }
    }

    /// A stand-in for a value that is never used (what follows an operation
    /// that always fails).
    fn dummy(&mut self, ty: &Type) -> Comp {
        match rep_of(ty) {
            Rep::F64 => {
                let v = self.b.ins().f64const(0.0);
                Comp::F64(v, dim_of_type(ty).unwrap_or(Dimension::DIMENSIONLESS))
            }
            Rep::Bool => Comp::Bool(self.iconst(types::I8, 0)),
            Rep::Ref => Comp::Ref(self.iconst(types::I64, 0)),
            Rep::Small(sh) => {
                let elems = self.zeros(Rep::Small(sh));
                Comp::Small(
                    elems,
                    sh,
                    number_dim(ty).unwrap_or(Dimension::DIMENSIONLESS),
                )
            }
        }
    }

    fn op(&mut self, id: OpId) -> Res<()> {
        let f = self.f;
        let op = f.op(id);
        if self.small_op(id, op)? {
            return Ok(());
        }
        match &op.kind {
            OpKind::ConstFloat { bits } => {
                let x = f64::from_bits(*bits);
                if !x.is_finite() {
                    return refuse("a constant is not finite", Some(op.span));
                }
                let d = self.const_dim(op)?;
                let v = self.b.ins().f64const(x);
                self.note_const(op, x);
                self.define(op, 0, Comp::F64(v, d))
            }
            OpKind::ConstRational(q) => {
                let x = q.numer() as f64 / q.denom() as f64;
                if !x.is_finite() {
                    return refuse("a rational constant is not finite", Some(op.span));
                }
                let d = self.const_dim(op)?;
                let v = self.b.ins().f64const(x);
                self.note_const(op, x);
                self.define(op, 0, Comp::F64(v, d))
            }
            OpKind::ConstBool(x) => {
                let v = self.iconst(types::I8, i64::from(*x));
                self.define(op, 0, Comp::Bool(v))
            }
            OpKind::ConstStr(_) => {
                let site = self.site(id);
                let site = self.c32(site);
                let h = self.call1(Rt::ConstStr, &[site]);
                self.check_handle(h);
                self.define(op, 0, Comp::Ref(h))
            }
            OpKind::ConstNothing => {
                let h = self.call1(Rt::ConstNone, &[]);
                self.check_handle(h);
                self.define(op, 0, Comp::Ref(h))
            }
            OpKind::LiteralError { .. } | OpKind::Input { .. } => {
                let site = self.site(id);
                let site = self.c32(site);
                self.fail_now(Rt::FailAlways, &[site]);
                for (i, &r) in op.results.iter().enumerate() {
                    let z = self.dummy(f.ty(r));
                    self.define(op, i, z)?;
                }
                Ok(())
            }
            OpKind::Add => self.arith(id, op, ArithOp::Add),
            OpKind::Sub => self.arith(id, op, ArithOp::Sub),
            OpKind::Mul => self.arith(id, op, ArithOp::Mul),
            OpKind::Div => self.arith(id, op, ArithOp::Div),
            OpKind::Neg => {
                let a = self.operand(op, 0)?;
                if let (Rep::F64, Some(d)) = (a.rep, self.operand_dim(op, 0)) {
                    let r = self.b.ins().fneg(a.v);
                    return self.define(op, 0, Comp::F64(r, d));
                }
                self.generic(id, op)
            }
            OpKind::PowConst { exponent } => {
                let x = exponent.numer() as f64 / exponent.denom() as f64;
                self.pow_const(id, op, x)
            }
            OpKind::Pow => self.pow(id, op),
            OpKind::Inverse => self.inverse(id, op),
            OpKind::Math(m) => self.math(id, op, *m),
            OpKind::Cmp(c) => self.cmp(id, op, *c),
            OpKind::Not => {
                let a = self.operand(op, 0)?;
                if a.rep == Rep::Bool {
                    let r = self.b.ins().bxor_imm(a.v, 1);
                    return self.define(op, 0, Comp::Bool(r));
                }
                self.generic(id, op)
            }
            OpKind::ExpectBool { .. } => {
                let a = self.operand(op, 0)?;
                if a.rep == Rep::Bool {
                    return self.define(op, 0, Comp::Bool(a.v));
                }
                self.generic(id, op)
            }
            OpKind::ExpectRange { .. } => self.generic(id, op),
            OpKind::ExpectNumber { .. } => {
                let a = self.operand(op, 0)?;
                if let (Rep::F64, Some(d)) = (a.rep, self.operand_dim(op, 0)) {
                    return self.define(op, 0, Comp::F64(a.v, d));
                }
                self.generic(id, op)
            }
            OpKind::Fused => self.fused(id, op),
            OpKind::Show { .. }
            | OpKind::MatMul
            | OpKind::Norm
            | OpKind::Dot
            | OpKind::Transpose
            | OpKind::Assume
            | OpKind::Solve { .. }
            | OpKind::SolveRight { .. }
            | OpKind::MatMulEx { .. }
            | OpKind::DiagMul { .. }
            | OpKind::ArrayLiteral { .. }
            | OpKind::ConstArray { .. }
            | OpKind::Index
            | OpKind::Slice { .. }
            | OpKind::Zeros
            | OpKind::Identity
            | OpKind::Len => self.generic(id, op),
            OpKind::If => self.if_op(id, op),
            OpKind::While => self.while_op(id, op),
            OpKind::Call { callee } => self.call_op(id, op, callee),
            OpKind::Print => self.print_op(id, op),
            OpKind::Assert { .. } => self.assert_op(id, op),
            OpKind::Yield | OpKind::Condition | OpKind::Return => refuse(
                format!("`{}` in the middle of a block", op.kind.mnemonic()),
                Some(op.span),
            ),
        }
    }

    /// `fused`: the runtime runs it (`runtime::fused`, reached through
    /// [`FnTx::generic`]: the operands are pushed, the result is a handle).
    /// Once an operation has run about 512 elements, the runtime compiles its
    /// body with `crate::kernel::compile_kernel` and runs the SIMD kernel,
    /// which has the interpreted loop's exact result and error.
    fn fused(&mut self, id: OpId, op: &'m OpData) -> Res<()> {
        self.generic(id, op)
    }

    /// An operation the runtime runs on boxed operands (see the module docs):
    /// arrays, strings, values of unknown type, and the cases whose
    /// dimensions make them fail. Result 0 is the handle the runtime returns.
    fn generic(&mut self, id: OpId, op: &OpData) -> Res<()> {
        if op.results.len() != 1 {
            return refuse(
                format!("`{}` has no single result", op.kind.mnemonic()),
                Some(op.span),
            );
        }
        self.push_operands(op, 0..op.operands.len())?;
        let site = self.site(id);
        let site = self.c32(site);
        let h = self.call1(Rt::Op, &[site]);
        self.check_handle(h);
        self.define(op, 0, Comp::Ref(h))
    }

    /// Pushes operands `idx` for the next runtime operation.
    fn push_operands(&mut self, op: &OpData, idx: std::ops::Range<usize>) -> Res<()> {
        for i in idx {
            let s = self.operand(op, i)?;
            match s.rep {
                Rep::F64 => {
                    let d = self.operand_dim(op, i).unwrap_or(Dimension::DIMENSIONLESS);
                    let d = self.dim_id(d);
                    let d = self.c32(d);
                    self.call(Rt::PushNum, &[d, s.v]);
                }
                Rep::Bool => {
                    let w = self.b.ins().uextend(types::I32, s.v);
                    self.call(Rt::PushBool, &[w]);
                }
                Rep::Ref => {
                    self.call(Rt::PushRef, &[s.v]);
                }
                Rep::Small(sh) => {
                    let v = op.operands[i];
                    let d = number_dim(self.f.ty(v)).unwrap_or(Dimension::DIMENSIONLESS);
                    let elems = self.small_elems(v, op)?;
                    self.push_small(&elems, sh, d);
                }
            }
        }
        Ok(())
    }

    // ----- scalar arithmetic -----

    /// `a + b`, `a - b`, `a * b`, `a / b` on typed scalars, in the order
    /// `sky_interp::arith::arith` checks: the dimension rule (static: a case
    /// that fails goes to the runtime, which fails it), a zero divisor, then
    /// the finiteness of the result.
    fn arith(&mut self, id: OpId, op: &OpData, kind: ArithOp) -> Res<()> {
        if !self.pending.is_empty()
            && matches!(kind, ArithOp::Add | ArithOp::Sub)
            && op.operands.iter().any(|v| self.pending.contains_key(v))
        {
            return self.fused_sum(id, op, kind);
        }
        let (a, b) = (self.operand(op, 0)?, self.operand(op, 1)?);
        if let (Rep::F64, Rep::F64, Some(da), Some(db)) = (
            a.rep,
            b.rep,
            self.operand_dim(op, 0),
            self.operand_dim(op, 1),
        ) {
            let dim = match kind {
                ArithOp::Add | ArithOp::Sub => (da == db).then_some(da),
                ArithOp::Mul => da.checked_mul(db).ok(),
                ArithOp::Div => da.checked_div(db).ok(),
            };
            if let Some(dim) = dim {
                let site = self.site(id);
                let site = self.c32(site);
                let fail = [site, a.v, b.v];
                let proven = self.proven(id);
                let r = match kind {
                    ArithOp::Add => self.b.ins().fadd(a.v, b.v),
                    ArithOp::Sub => self.b.ins().fsub(a.v, b.v),
                    ArithOp::Mul => self.b.ins().fmul(a.v, b.v),
                    ArithOp::Div => {
                        if !proven && !self.nonzero(op.operands[1]) {
                            let zero = self.b.ins().f64const(0.0);
                            let bad = self.b.ins().fcmp(FloatCC::Equal, b.v, zero);
                            self.fail_with(bad, Rt::Fail2, &fail);
                        }
                        self.b.ins().fdiv(a.v, b.v)
                    }
                };
                if !proven {
                    self.finite_or(r, Rt::Fail2, &fail);
                }
                return self.define(op, 0, Comp::F64(r, dim));
            }
        }
        self.generic(id, op)
    }

    /// The multiplications `powi` makes for the integer `n` (compiler-rt's
    /// `__powidf2`, which is what Rust's `f64::powi` calls): squaring, a
    /// product for each set bit of `|n|`, and a reciprocal at the end for a
    /// negative `n`.
    fn powi(&mut self, base: Cv, n: i32) -> Cv {
        if n == 0 {
            return self.b.ins().f64const(1.0);
        }
        let (mut k, mut a) = (n.unsigned_abs(), base);
        let mut r: Option<Cv> = None;
        loop {
            if k & 1 == 1 {
                r = Some(match r {
                    None => a,
                    Some(r) => self.b.ins().fmul(r, a),
                });
            }
            k /= 2;
            if k == 0 {
                break;
            }
            a = self.b.ins().fmul(a, a);
        }
        let r = r.unwrap_or(base);
        if n < 0 {
            let one = self.b.ins().f64const(1.0);
            self.b.ins().fdiv(one, r)
        } else {
            r
        }
    }

    /// `a ^ e` for a constant exponent `x` (see `sky_interp::arith::pow`).
    fn pow_const(&mut self, id: OpId, op: &OpData, x: f64) -> Res<()> {
        let a = self.operand(op, 0)?;
        let OpKind::PowConst { exponent } = op.kind else {
            return self.generic(id, op);
        };
        if let (Rep::F64, Some(db)) = (a.rep, self.operand_dim(op, 0)) {
            let dim = if db.is_dimensionless() {
                Some(Dimension::DIMENSIONLESS)
            } else {
                db.pow_rat(exponent).ok()
            };
            if let Some(dim) = dim {
                let site = self.site(id);
                let site = self.c32(site);
                let r = if x.fract() == 0.0 && x.abs() <= sky_interp::arith::POWI_MAX {
                    let proven = self.proven(id);
                    if x < 0.0 && !proven && !self.nonzero(op.operands[0]) {
                        let zero = self.b.ins().f64const(0.0);
                        let bad = self.b.ins().fcmp(FloatCC::Equal, a.v, zero);
                        self.fail_with(bad, Rt::Fail1, &[site, a.v]);
                    }
                    let r = self.powi(a.v, x as i32);
                    if !proven {
                        self.finite_or(r, Rt::Fail1, &[site, a.v]);
                    }
                    r
                } else {
                    let xv = self.b.ins().f64const(x);
                    let r = self.call1(Rt::PowScalar, &[site, a.v, xv]);
                    self.nan_fail(r);
                    r
                };
                return self.define(op, 0, Comp::F64(r, dim));
            }
        }
        self.generic(id, op)
    }

    /// `a ^ e` with a computed exponent: dimensionless scalars.
    fn pow(&mut self, id: OpId, op: &OpData) -> Res<()> {
        let (a, e) = (self.operand(op, 0)?, self.operand(op, 1)?);
        if let (Rep::F64, Rep::F64, Some(da), Some(de)) = (
            a.rep,
            e.rep,
            self.operand_dim(op, 0),
            self.operand_dim(op, 1),
        ) {
            if da.is_dimensionless() && de.is_dimensionless() {
                let site = self.site(id);
                let site = self.c32(site);
                let r = self.call1(Rt::PowScalar, &[site, a.v, e.v]);
                self.nan_fail(r);
                return self.define(op, 0, Comp::F64(r, Dimension::DIMENSIONLESS));
            }
        }
        self.generic(id, op)
    }

    /// `a⁻¹` of a typed scalar: a zero fails, then the finiteness check.
    fn inverse(&mut self, id: OpId, op: &OpData) -> Res<()> {
        let a = self.operand(op, 0)?;
        if let (Rep::F64, Some(da)) = (a.rep, self.operand_dim(op, 0)) {
            if let Ok(dim) = da.inverse() {
                let site = self.site(id);
                let site = self.c32(site);
                let proven = self.proven(id);
                if !proven && !self.nonzero(op.operands[0]) {
                    let zero = self.b.ins().f64const(0.0);
                    let bad = self.b.ins().fcmp(FloatCC::Equal, a.v, zero);
                    self.fail_with(bad, Rt::Fail1, &[site, a.v]);
                }
                let one = self.b.ins().f64const(1.0);
                let r = self.b.ins().fdiv(one, a.v);
                if !proven {
                    self.finite_or(r, Rt::Fail1, &[site, a.v]);
                }
                return self.define(op, 0, Comp::F64(r, dim));
            }
        }
        self.generic(id, op)
    }

    /// A math function of typed scalars: the exact ones inline, the rest in
    /// `sky_interp::math::apply_f64`, the core of `apply`.
    fn math(&mut self, id: OpId, op: &OpData, m: MathFn) -> Res<()> {
        let n = m.arity();
        if op.operands.len() != n {
            return self.generic(id, op);
        }
        let (mut xs, mut dims) = (Vec::new(), Vec::new());
        for i in 0..n {
            let s = self.operand(op, i)?;
            match (s.rep, self.operand_dim(op, i)) {
                (Rep::F64, Some(d)) => {
                    xs.push(s.v);
                    dims.push(d);
                }
                _ => return self.generic(id, op),
            }
        }
        let Ok(dim) = m.result_dim(&dims) else {
            return self.generic(id, op);
        };
        let site = self.site(id);
        let site = self.c32(site);
        let proven = self.proven(id);
        // `apply` ends with `+ 0.0`, which turns `-0` into `0`.
        let plus_zero = |tx: &mut Self, r: Cv| {
            let z = tx.b.ins().f64const(0.0);
            tx.b.ins().fadd(r, z)
        };
        let r = match m {
            MathFn::Sqrt => {
                if !proven {
                    let zero = self.b.ins().f64const(0.0);
                    let bad = self.b.ins().fcmp(FloatCC::LessThan, xs[0], zero);
                    self.fail_with(bad, Rt::Fail1, &[site, xs[0]]);
                }
                let r = self.b.ins().sqrt(xs[0]);
                plus_zero(self, r)
            }
            MathFn::Abs => {
                let r = self.b.ins().fabs(xs[0]);
                plus_zero(self, r)
            }
            MathFn::Floor => {
                let r = self.b.ins().floor(xs[0]);
                plus_zero(self, r)
            }
            MathFn::Ceil => {
                let r = self.b.ins().ceil(xs[0]);
                plus_zero(self, r)
            }
            MathFn::Min => {
                let r = self.b.ins().fmin(xs[0], xs[1]);
                plus_zero(self, r)
            }
            MathFn::Max => {
                let r = self.b.ins().fmax(xs[0], xs[1]);
                plus_zero(self, r)
            }
            _ => {
                // The domain first, as `apply_f64` checks it, then the value
                // (`core_f64`, the same function), then the finiteness check.
                // A failure re-runs the operation the interpreter's way for
                // its diagnostic.
                let fail = [site, xs[0], xs.get(1).copied().unwrap_or(xs[0])];
                let fail = &fail[..=n.min(2)];
                let outside = if proven {
                    None
                } else {
                    self.math_domain(m, &xs)
                };
                if let Some(bad) = outside {
                    let rt = if n == 1 { Rt::Fail1 } else { Rt::Fail2 };
                    self.fail_with(bad, rt, fail);
                }
                let fid = MathFn::ALL
                    .iter()
                    .position(|x| *x == m)
                    .unwrap_or(usize::MAX);
                let fid = self.c32(fid as i32);
                let r = if n == 1 {
                    self.call1(Rt::Libm1, &[fid, xs[0]])
                } else {
                    self.call1(Rt::Libm2, &[fid, xs[0], xs[1]])
                };
                if !proven {
                    let rt = if n == 1 { Rt::Fail1 } else { Rt::Fail2 };
                    self.finite_or(r, rt, fail);
                }
                plus_zero(self, r)
            }
        };
        self.define(op, 0, Comp::F64(r, dim))
    }

    /// Whether `m(xs)` is outside its domain (an `i8`, nonzero if so): the
    /// conditions of `sky_interp::math::apply_f64`, which says which error.
    /// `None` for a function defined on every finite number.
    fn math_domain(&mut self, m: MathFn, xs: &[Cv]) -> Option<Cv> {
        let konst = |tx: &mut Self, x: f64| tx.b.ins().f64const(x);
        Some(match m {
            MathFn::Log | MathFn::Log10 | MathFn::Log2 => {
                let zero = konst(self, 0.0);
                self.b.ins().fcmp(FloatCC::LessThanOrEqual, xs[0], zero)
            }
            MathFn::Log1p => {
                let lo = konst(self, -1.0);
                self.b.ins().fcmp(FloatCC::LessThanOrEqual, xs[0], lo)
            }
            MathFn::Acosh => {
                let one = konst(self, 1.0);
                self.b.ins().fcmp(FloatCC::LessThan, xs[0], one)
            }
            MathFn::Asin | MathFn::Acos => {
                let a = self.b.ins().fabs(xs[0]);
                let one = konst(self, 1.0);
                self.b.ins().fcmp(FloatCC::GreaterThan, a, one)
            }
            MathFn::Atanh => {
                let a = self.b.ins().fabs(xs[0]);
                let one = konst(self, 1.0);
                self.b.ins().fcmp(FloatCC::GreaterThanOrEqual, a, one)
            }
            MathFn::Atan2 => {
                let zero = konst(self, 0.0);
                let x0 = self.b.ins().fcmp(FloatCC::Equal, xs[0], zero);
                let y0 = self.b.ins().fcmp(FloatCC::Equal, xs[1], zero);
                self.b.ins().band(x0, y0)
            }
            _ => return None,
        })
    }

    fn cmp(&mut self, id: OpId, op: &OpData, c: CmpOp) -> Res<()> {
        let (a, b) = (self.operand(op, 0)?, self.operand(op, 1)?);
        match (a.rep, b.rep) {
            (Rep::F64, Rep::F64) => {
                let (da, db) = (self.operand_dim(op, 0), self.operand_dim(op, 1));
                if da.is_some() && da == db {
                    let cc = match c {
                        CmpOp::Eq => FloatCC::Equal,
                        CmpOp::Ne => FloatCC::NotEqual,
                        CmpOp::Lt => FloatCC::LessThan,
                        CmpOp::Le => FloatCC::LessThanOrEqual,
                        CmpOp::Gt => FloatCC::GreaterThan,
                        CmpOp::Ge => FloatCC::GreaterThanOrEqual,
                    };
                    let r = self.b.ins().fcmp(cc, a.v, b.v);
                    return self.define(op, 0, Comp::Bool(r));
                }
            }
            (Rep::Bool, Rep::Bool) if matches!(c, CmpOp::Eq | CmpOp::Ne) => {
                let cc = if c == CmpOp::Eq {
                    IntCC::Equal
                } else {
                    IntCC::NotEqual
                };
                let r = self.b.ins().icmp(cc, a.v, b.v);
                return self.define(op, 0, Comp::Bool(r));
            }
            _ => {}
        }
        self.generic(id, op)
    }

    // ----- control flow -----

    fn single_block(&self, region: sky_ir::RegionId, op: &OpData) -> Res<BlockId> {
        match self.f.region(region).blocks[..] {
            [b] => Ok(b),
            _ => refuse("a region has not exactly one block", Some(op.span)),
        }
    }

    /// The Bool `c` as an `i8` for a branch: a native Bool as it is, a handle
    /// through the runtime, which fails with `sky run`'s error unless it
    /// holds a Bool.
    fn condition(&mut self, id: OpId, op: &OpData, c: Slot) -> Res<Cv> {
        match c.rep {
            Rep::Bool => Ok(c.v),
            Rep::Ref => {
                let site = self.site(id);
                let site = self.c32(site);
                let r = self.call1(Rt::Cond, &[site, c.v]);
                let bad = self.b.ins().icmp_imm(IntCC::SignedLessThan, r, 0);
                self.fail_if(bad);
                Ok(self.b.ins().icmp_imm(IntCC::NotEqual, r, 0))
            }
            Rep::F64 | Rep::Small(_) => refuse("a condition is a number", Some(op.span)),
        }
    }

    /// Whether `id` can run whether or not its branch is taken: it cannot
    /// fail, print or call, and it reads and makes only numbers and Bools
    /// that native code holds unboxed (a comparison of same-dimension
    /// scalars or of Bools, a `not`, a negation, a constant).
    fn speculable(&self, id: OpId) -> bool {
        let f = self.f;
        let op = f.op(id);
        let scalar = |v: &ValueId| matches!(rep_of(f.ty(*v)), Rep::F64 | Rep::Bool);
        if !op.operands.iter().chain(&op.results).all(scalar) {
            return false;
        }
        let same_dim = |op: &OpData| {
            let d = |i: usize| dim_of_type(f.ty(op.operands[i]));
            d(0).is_some() && d(0) == d(1)
        };
        match &op.kind {
            OpKind::ConstFloat { bits } => f64::from_bits(*bits).is_finite(),
            OpKind::ConstBool(_) => true,
            OpKind::Cmp(c) => match op.operands[..] {
                [a, b] if matches!(f.ty(a), Type::Bool) => {
                    matches!(f.ty(b), Type::Bool) && matches!(c, CmpOp::Eq | CmpOp::Ne)
                }
                [_, _] => same_dim(op),
                _ => false,
            },
            OpKind::Not => op.operands.len() == 1 && matches!(f.ty(op.operands[0]), Type::Bool),
            OpKind::Neg => op.operands.len() == 1 && dim_of_type(f.ty(op.operands[0])).is_some(),
            _ => false,
        }
    }

    /// An `if` whose arms only compute comparisons and constants (what a
    /// short-circuit `and`, `or`, or `break` flag lowers to) as straight-line
    /// code: both arms run, and each result is a `select`. There is no branch
    /// to mispredict and no `i8` handed through a merge block for the next
    /// branch to test again. Nothing observable changes, since an arm that
    /// runs when it should not can neither fail nor print. `Ok(false)` if
    /// the `if` is not of that shape.
    fn if_select(&mut self, op: &'m OpData, cond: Cv) -> Res<bool> {
        /// More work than this is not worth doing on both paths.
        const MAX_OPS: usize = 6;
        let f = self.f;
        let [then_r, else_r] = op.regions[..] else {
            return Ok(false);
        };
        let (Ok(then_b), Ok(else_b)) =
            (self.single_block(then_r, op), self.single_block(else_r, op))
        else {
            return Ok(false);
        };
        let mut arms = Vec::new();
        for bid in [then_b, else_b] {
            let Some((term, body)) = f.block(bid).ops.split_last() else {
                return Ok(false);
            };
            let term = f.op(*term);
            if !matches!(term.kind, OpKind::Yield)
                || term.operands.len() != op.results.len()
                || body.len() > MAX_OPS
                || !body.iter().all(|o| self.speculable(*o))
            {
                return Ok(false);
            }
            // The results are unboxed scalars of the type each arm yields.
            for (&y, &r) in term.operands.iter().zip(&op.results) {
                let (ty, want) = (f.ty(y), f.ty(r));
                let ok = match (rep_of(ty), rep_of(want)) {
                    (Rep::F64, Rep::F64) => dim_of_type(ty) == dim_of_type(want),
                    (Rep::Bool, Rep::Bool) => true,
                    _ => false,
                };
                if !ok {
                    return Ok(false);
                }
            }
            arms.push((body, term));
        }
        let mut chosen: Vec<Vec<Cv>> = Vec::new();
        for (body, term) in arms {
            for id in body {
                self.op(*id)?;
            }
            let mut vs = Vec::new();
            for &y in &term.operands {
                vs.push(self.slot(y, term)?.v);
            }
            chosen.push(vs);
        }
        for (i, &r) in op.results.iter().enumerate() {
            let v = self.b.ins().select(cond, chosen[0][i], chosen[1][i]);
            self.bind(
                r,
                Slot {
                    v,
                    rep: rep_of(f.ty(r)),
                },
            );
        }
        Ok(true)
    }

    fn if_op(&mut self, id: OpId, op: &'m OpData) -> Res<()> {
        let f = self.f;
        let c = self.operand(op, 0)?;
        let cond = self.condition(id, op, c)?;
        if c.rep == Rep::Bool && self.if_select(op, cond)? {
            return Ok(());
        }
        let [then_r, else_r] = op.regions[..] else {
            return refuse("an `if` needs two regions", Some(op.span));
        };
        let merge = self.b.create_block();
        for r in &op.results {
            for t in rep_tys(rep_of(f.ty(*r))) {
                self.b.append_block_param(merge, t);
            }
        }
        let (then_b, else_b) = (self.b.create_block(), self.b.create_block());
        self.b.ins().brif(cond, then_b, &[], else_b, &[]);
        for (blk, region) in [(then_b, then_r), (else_b, else_r)] {
            self.b.switch_to_block(blk);
            let bid = self.single_block(region, op)?;
            self.owned.push(Vec::new());
            let term = self.block_body(bid)?;
            if !matches!(term.kind, OpKind::Yield) || term.operands.len() != op.results.len() {
                return refuse("an `if` arm must `yield` its results", Some(op.span));
            }
            let mut outs = Vec::new();
            for (i, &v) in term.operands.iter().enumerate() {
                outs.extend(self.transfer(term, v, f.ty(op.results[i]), "a yielded value")?);
            }
            self.release_owned();
            self.b.ins().jump(merge, &args_of(&outs));
        }
        self.b.switch_to_block(merge);
        let params = self.b.block_params(merge).to_vec();
        self.bind_all(&op.results, &params);
        Ok(())
    }

    /// `while`: a header block (the `before` region) with the loop-carried
    /// values as parameters, a body block (`after`), and an exit block whose
    /// parameters are the results. The values a branch hands on are
    /// transferred in a block of its own, so the `before` block's handles are
    /// released on both paths.
    fn while_op(&mut self, id: OpId, op: &'m OpData) -> Res<()> {
        let f = self.f;
        self.slp_region(
            || !slp::loop_latency_bound(f, op),
            |tx| tx.while_loop(id, op),
        )
    }

    fn while_loop(&mut self, id: OpId, op: &'m OpData) -> Res<()> {
        let f = self.f;
        let [before_r, after_r] = op.regions[..] else {
            return refuse("a `while` needs two regions", Some(op.span));
        };
        let before = self.single_block(before_r, op)?;
        let after = self.single_block(after_r, op)?;
        let (bargs, aargs) = (&f.block(before).args, &f.block(after).args);
        if bargs.len() != op.operands.len() || op.results.len() != op.operands.len() {
            return refuse(
                "a `while` carries a different number of values",
                Some(op.span),
            );
        }
        let header = self.b.create_block();
        for &a in bargs {
            for t in rep_tys(rep_of(f.ty(a))) {
                self.b.append_block_param(header, t);
            }
        }
        let mut inits = Vec::new();
        for (i, &v) in op.operands.iter().enumerate() {
            inits.extend(self.transfer(op, v, f.ty(bargs[i]), "a loop-carried value")?);
        }
        self.b.ins().jump(header, &args_of(&inits));

        // The `before` region: computes the condition and the forwarded values.
        self.b.switch_to_block(header);
        self.owned.push(Vec::new());
        let params = self.b.block_params(header).to_vec();
        self.bind_all(bargs, &params);
        let term = self.block_body(before)?;
        if !matches!(term.kind, OpKind::Condition) || term.operands.is_empty() {
            return refuse(
                "a `while` `before` region must end in `condition`",
                Some(op.span),
            );
        }
        let fwd = &term.operands[1..];
        if fwd.len() != aargs.len() {
            return refuse(
                "a `while` forwards a different number of values",
                Some(op.span),
            );
        }
        let c = self.slot(term.operands[0], term)?;
        let cond = self.condition(id, op, c)?;
        let (after_pre, exit_pre) = (self.b.create_block(), self.b.create_block());
        let (after_b, exit_b) = (self.b.create_block(), self.b.create_block());
        for &a in aargs {
            for t in rep_tys(rep_of(f.ty(a))) {
                self.b.append_block_param(after_b, t);
            }
        }
        for &r in &op.results {
            for t in rep_tys(rep_of(f.ty(r))) {
                self.b.append_block_param(exit_b, t);
            }
        }
        self.b.ins().brif(cond, after_pre, &[], exit_pre, &[]);
        self.b.switch_to_block(after_pre);
        let mut outs = Vec::new();
        for (i, &v) in fwd.iter().enumerate() {
            outs.extend(self.transfer(term, v, f.ty(aargs[i]), "a forwarded value")?);
        }
        self.emit_release_owned();
        self.b.ins().jump(after_b, &args_of(&outs));
        self.b.switch_to_block(exit_pre);
        let mut outs = Vec::new();
        for (i, &v) in fwd.iter().enumerate() {
            outs.extend(self.transfer(term, v, f.ty(op.results[i]), "a loop result")?);
        }
        self.emit_release_owned();
        self.b.ins().jump(exit_b, &args_of(&outs));
        self.owned.pop();

        // The `after` region: the next loop-carried values.
        self.b.switch_to_block(after_b);
        self.owned.push(Vec::new());
        let params = self.b.block_params(after_b).to_vec();
        self.bind_all(aargs, &params);
        let term = self.block_body(after)?;
        if !matches!(term.kind, OpKind::Yield) || term.operands.len() != bargs.len() {
            return refuse(
                "a `while` `after` region must `yield` the next values",
                Some(op.span),
            );
        }
        let mut outs = Vec::new();
        for (i, &v) in term.operands.iter().enumerate() {
            outs.extend(self.transfer(term, v, f.ty(bargs[i]), "a loop-carried value")?);
        }
        self.release_owned();
        self.b.ins().jump(header, &args_of(&outs));

        // After the loop.
        self.b.switch_to_block(exit_b);
        let params = self.b.block_params(exit_b).to_vec();
        self.bind_all(&op.results, &params);
        Ok(())
    }

    /// A call to another function of the module, natively. The depth check
    /// is the interpreter's: a call made at depth 1000 or more fails, at this
    /// call, before the callee runs.
    fn call_op(&mut self, id: OpId, op: &OpData, callee: &str) -> Res<()> {
        let m = self.sh.m;
        let Some(ci) = m.functions.iter().position(|g| g.name == callee) else {
            return refuse(
                format!("a call to unknown function `{callee}`"),
                Some(op.span),
            );
        };
        let g = &m.functions[ci];
        if op.operands.len() != g.params.len() || op.results.len() != g.results.len() {
            return refuse(
                format!("a call to `{callee}` does not match its signature"),
                Some(op.span),
            );
        }
        let site = self.site(id);
        let site_v = self.c32(site);
        let limit = self.c32(MAX_CALL_DEPTH as i32);
        let bad = self
            .b
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, self.depth, limit);
        self.fail_with(bad, Rt::FailDepth, &[site_v]);
        let depth = self.b.ins().iadd_imm(self.depth, 1);
        let mut args = vec![depth, self.stack_limit, site_v];
        for (i, &v) in op.operands.iter().enumerate() {
            args.extend(self.transfer(
                op,
                v,
                &g.params[i],
                &format!("argument {i} of a call to `{callee}`"),
            )?);
        }
        let fref = match self.fn_refs.get(&ci) {
            Some(r) => *r,
            None => {
                let r = self
                    .jit
                    .declare_func_in_func(self.sh.fn_ids[ci], self.b.func);
                self.fn_refs.insert(ci, r);
                r
            }
        };
        // A callee that returns through memory gets the address of a stack
        // area for its results (see `results_in_memory`).
        let area = if results_in_memory(g) {
            let n: usize = g.results.iter().map(|r| rep_tys(rep_of(r)).len()).sum();
            let slot = self.b.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                8 * n as u32,
                3,
            ));
            let addr = self.b.ins().stack_addr(types::I64, slot, 0);
            args.push(addr);
            Some(addr)
        } else {
            None
        };
        let inst = self.b.ins().call(fref, &args);
        let mut results = self.b.inst_results(inst).to_vec();
        self.status_fail(results[0]);
        if let Some(addr) = area {
            let tys: Vec<ir::Type> = g.results.iter().flat_map(|r| rep_tys(rep_of(r))).collect();
            for (i, t) in tys.into_iter().enumerate() {
                let v = self
                    .b
                    .ins()
                    .load(t, MemFlags::trusted(), addr, (8 * i) as i32);
                results.push(v);
            }
        }
        self.bind_call_results(op, g, &results)
    }

    /// Defines the results of the call `op` of `g` from `results`: the
    /// callee's status (skipped) and then its results, flat, as native values.
    fn bind_call_results(&mut self, op: &OpData, g: &Function, results: &[Cv]) -> Res<()> {
        let mut at = 1;
        for (i, ty) in g.results.iter().enumerate() {
            let v = results[at];
            let rep = rep_of(ty);
            let c = match rep {
                Rep::F64 => Comp::F64(v, dim_of_type(ty).unwrap_or(Dimension::DIMENSIONLESS)),
                Rep::Bool => Comp::Bool(v),
                Rep::Ref => Comp::Ref(v),
                Rep::Small(sh) => Comp::Small(
                    results[at..at + sh.len()].to_vec(),
                    sh,
                    number_dim(ty).unwrap_or(Dimension::DIMENSIONLESS),
                ),
            };
            at += rep_tys(rep).len();
            self.define(op, i, c)?;
        }
        Ok(())
    }

    fn print_op(&mut self, id: OpId, op: &OpData) -> Res<()> {
        for i in 0..op.operands.len() {
            let s = self.operand(op, i)?;
            match s.rep {
                Rep::F64 => {
                    let d = self.operand_dim(op, i).unwrap_or(Dimension::DIMENSIONLESS);
                    let d = self.dim_id(d);
                    let d = self.c32(d);
                    self.call(Rt::PrintNum, &[d, s.v]);
                }
                Rep::Bool => {
                    let w = self.b.ins().uextend(types::I32, s.v);
                    self.call(Rt::PrintBool, &[w]);
                }
                Rep::Ref => {
                    self.call(Rt::PrintRef, &[s.v]);
                }
                Rep::Small(sh) => {
                    let v = op.operands[i];
                    let d = number_dim(self.f.ty(v)).unwrap_or(Dimension::DIMENSIONLESS);
                    let elems = self.small_elems(v, op)?;
                    self.print_small(&elems, sh, d);
                }
            }
        }
        let site = self.site(id);
        let site = self.c32(site);
        let status = self.call1(Rt::PrintEnd, &[site]);
        self.status_fail(status);
        Ok(())
    }

    /// `assert`: a false condition pushes the values of its names and calls
    /// the runtime, which builds the contract's failure.
    fn assert_op(&mut self, id: OpId, op: &OpData) -> Res<()> {
        let c = self.operand(op, 0)?;
        let site = self.site(id);
        let site_v = self.c32(site);
        let (fail_b, cont) = (self.b.create_block(), self.b.create_block());
        match c.rep {
            Rep::Bool => {
                self.b.ins().brif(c.v, cont, &[], fail_b, &[]);
            }
            Rep::Ref => {
                let r = self.call1(Rt::AssertDyn, &[site_v, c.v]);
                let bad = self.b.ins().icmp_imm(IntCC::SignedLessThan, r, 0);
                self.fail_if(bad);
                let holds = self.b.ins().icmp_imm(IntCC::Equal, r, 1);
                self.b.ins().brif(holds, cont, &[], fail_b, &[]);
            }
            Rep::F64 | Rep::Small(_) => {
                return refuse("an `assert` condition is a number", Some(op.span))
            }
        }
        self.b.set_cold_block(fail_b);
        self.b.switch_to_block(fail_b);
        self.push_operands(op, 1..op.operands.len())?;
        let call_site = self.call_site;
        self.call(Rt::AssertFail, &[site_v, call_site]);
        let err = self.err;
        self.b.ins().jump(err, &[]);
        self.b.switch_to_block(cont);
        Ok(())
    }
}
