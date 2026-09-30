//! Whole-module translation: every SIR function becomes a Cranelift function,
//! compiled in memory for the host CPU, and `sky run --engine=native` calls
//! `<main>`.
//!
//! # What native code computes
//!
//! A value's *representation* follows from its static type. A scalar number
//! is an `f64` (its dimension is the type's: it is not stored), a Bool is an
//! `i8`, and everything else (arrays, strings, `none`, and any value of type
//! `unknown`) is an opaque handle to a value the runtime owns
//! ([`crate::runtime`]). Scalar arithmetic, comparisons, `if`, `while` and
//! calls between functions are inline machine code; anything else is a call
//! into the runtime, which runs the same functions the SIR interpreter does.
//!
//! # Exactness
//!
//! * `fadd`, `fsub`, `fmul`, `fdiv`, `fneg`, `fabs` and `sqrt` are IEEE
//!   operations that give the interpreter's bits. Nothing is contracted into
//!   a fused multiply-add: Cranelift never does it unless asked.
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
    self, types, AbiParam, Block, BlockArg, FuncRef, InstBuilder, Signature, UserFuncName,
    Value as Cv,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Linkage, Module as ClModule};
use sky_diag::Span;
use sky_interp::arith::{ArithOp, MAX_CALL_DEPTH, STACK_BUDGET};
use sky_interp::RuntimeError;
use sky_ir::{
    BlockId, CmpOp, Function, MathFn, Module as SirModule, OpData, OpId, OpKind, Type, ValueId,
};
use sky_units::Dimension;

use crate::runtime::{self, Program, Rt, Site, Ty, FUNC_SITE};

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
        let guard = runtime::begin(self.program.clone(), out);
        let limit = sky_interp::arith::stack_address().saturating_sub(STACK_BUDGET) as i64;
        // SAFETY: `main` is the address of `<main>` compiled with the
        // signature `(i32 depth, i64 stack limit, i32 call site) -> i32
        // status` (checked when it was built: `<main>` has no parameters and
        // no results), in the platform's default calling convention, and
        // `self.jit`, which owns the code, is alive for the whole call.
        let entry: extern "C" fn(i32, i64, i32) -> i32 =
            unsafe { std::mem::transmute(self.main as *const u8) };
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

/// Compiles every function of `module` (which must verify) to machine code.
pub fn compile(module: &SirModule) -> Result<Compiled, CompileError> {
    let built = build(module, false)?;
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
    Ok(build(module, true)?.clif)
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
}

fn rep_of(t: &Type) -> Rep {
    match t {
        Type::Number(n) if n.shape.rank() == 0 => Rep::F64,
        Type::Bool => Rep::Bool,
        _ => Rep::Ref,
    }
}

fn rep_ty(r: Rep) -> ir::Type {
    match r {
        Rep::F64 => types::F64,
        Rep::Bool => types::I8,
        Rep::Ref => types::I64,
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

/// A value while it is being defined: what an operation computed, before it
/// is held as the representation its result's type asks for.
#[derive(Clone, Copy)]
enum Comp {
    F64(Cv, Dimension),
    Bool(Cv),
    /// A fresh handle (the definer owns it). If the result is typed as a
    /// scalar it is unpacked with a check (see the module docs).
    Ref(Cv),
}

/// A defined value.
#[derive(Clone, Copy)]
struct Slot {
    v: Cv,
    rep: Rep,
}

/// State shared by the functions of a module while they are translated.
struct Shared<'m> {
    m: &'m SirModule,
    fn_ids: Vec<FuncId>,
    rt_ids: HashMap<Rt, FuncId>,
    dims: Vec<Dimension>,
    dim_ids: HashMap<Dimension, i32>,
    sites: Vec<Site>,
}

fn build(module: &SirModule, emit: bool) -> Res<Built> {
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

    let isa = crate::host_isa().map_err(|e| CompileError {
        message: format!("no code generator for this machine ({e})"),
        span: None,
    })?;
    let call_conv = isa.default_call_conv();
    let mut builder =
        crate::jit_builder(isa, crate::MODULE_ARENA_BYTES).map_err(|e| CompileError {
            message: format!("no memory for native code ({e})"),
            span: None,
        })?;
    for rt in Rt::ALL {
        builder.symbol(rt.name(), rt.address());
    }
    let mut jit = JITModule::new(builder);

    let internal = |what: &str, e: &dyn fmt::Display| CompileError {
        message: format!("the code generator failed on {what}: {e}"),
        span: None,
    };
    let mut sh = Shared {
        m: module,
        fn_ids: Vec::new(),
        rt_ids: HashMap::new(),
        dims: Vec::new(),
        dim_ids: HashMap::new(),
        sites: Vec::new(),
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

    let mut clif = String::new();
    // The size of each function's machine code, to flush the instruction cache.
    let mut code_sizes = Vec::new();
    let mut fctx = FunctionBuilderContext::new();
    let mut ctx = jit.make_context();
    for (i, f) in module.functions.iter().enumerate() {
        ctx.func.signature = function_signature(f, call_conv);
        ctx.func.name = UserFuncName::testcase(symbol_name(i, &f.name));
        FnTx::translate(&mut ctx.func, &mut fctx, &mut jit, &mut sh, f, i as u32)?;
        if emit {
            clif.push_str(&format!("; {}\n{}\n", describe(f), ctx.func));
        } else {
            jit.define_function(sh.fn_ids[i], &mut ctx)
                .map_err(|e| internal(&f.name, &e))?;
            code_sizes.push(ctx.compiled_code().map_or(0, |c| c.code_buffer().len()));
        }
        jit.clear_context(&mut ctx);
    }
    if emit {
        return Ok(Built {
            jit: None,
            program: Program {
                module: SirModule::default(),
                dims: Vec::new(),
                sites: Vec::new(),
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
        },
        main: Some(main),
        clif,
    })
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

/// `(depth: i32, stack limit: i64, call site: i32, params...) -> (status:
/// i32, results...)`.
fn function_signature(f: &Function, call_conv: cranelift_codegen::isa::CallConv) -> Signature {
    let mut sig = Signature::new(call_conv);
    sig.params.push(AbiParam::new(types::I32));
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(types::I32));
    for p in &f.params {
        sig.params.push(AbiParam::new(rep_ty(rep_of(p))));
    }
    sig.returns.push(AbiParam::new(types::I32));
    for r in &f.results {
        sig.returns.push(AbiParam::new(rep_ty(rep_of(r))));
    }
    sig
}

fn args_of(vs: &[Cv]) -> Vec<BlockArg> {
    vs.iter().map(|v| BlockArg::Value(*v)).collect()
}

/// The translation of one function.
struct FnTx<'a, 'm> {
    b: FunctionBuilder<'a>,
    jit: &'a mut JITModule,
    sh: &'a mut Shared<'m>,
    f: &'m Function,
    fidx: u32,
    vals: Vec<Option<Slot>>,
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
}

impl<'a, 'm> FnTx<'a, 'm> {
    fn translate(
        func: &'a mut ir::Function,
        fctx: &'a mut FunctionBuilderContext,
        jit: &'a mut JITModule,
        sh: &'a mut Shared<'m>,
        f: &'m Function,
        fidx: u32,
    ) -> Res<()> {
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
            owned: Vec::new(),
            err,
            depth: params[0],
            stack_limit: params[1],
            call_site: params[2],
            rt_refs: HashMap::new(),
            fn_refs: HashMap::new(),
            site_ids: HashMap::new(),
        };
        let limit = params[1];
        tx.owned.push(Vec::new());
        let entry_block = f.entry();
        for (i, &p) in f.param_values().iter().enumerate() {
            let rep = rep_of(f.ty(p));
            tx.bind(
                p,
                Slot {
                    v: params[3 + i],
                    rep,
                },
            );
        }
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
        for (i, &v) in term.operands.iter().enumerate() {
            let out = tx.transfer(term, v, &f.results[i], "a returned value")?;
            outs.push(out);
        }
        tx.release_owned();
        tx.b.ins().return_(&outs);

        // The one place a failure ends: status 1, no results.
        tx.b.switch_to_block(tx.err);
        tx.b.set_cold_block(tx.err);
        let mut outs = vec![tx.iconst(types::I32, 1)];
        for r in &f.results {
            let z = tx.zero(rep_of(r));
            outs.push(z);
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
            Rep::F64 => self.b.ins().f64const(0.0),
            Rep::Bool => self.iconst(types::I8, 0),
            Rep::Ref => self.iconst(types::I64, 0),
        }
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

    /// Fails if `x` (a result that must be finite) is not: the overflow
    /// check of `sky_interp::arith::finite`.
    fn finite_or(&mut self, x: Cv, rt: Rt, args: &[Cv]) {
        let abs = self.b.ins().fabs(x);
        let inf = self.b.ins().f64const(f64::INFINITY);
        // False for infinity and for NaN (an ordered comparison).
        let ok = self.b.ins().fcmp(FloatCC::LessThan, abs, inf);
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
            (Comp::F64(..), Rep::Bool) | (Comp::Bool(_), Rep::F64) => {
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
    fn transfer(&mut self, op: &OpData, v: ValueId, to: &Type, what: &str) -> Res<Cv> {
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
                Ok(s.v)
            }
            (Rep::Bool, Rep::Bool) => Ok(s.v),
            (Rep::Ref, Rep::Ref) => {
                self.call(Rt::Retain, &[s.v]);
                Ok(s.v)
            }
            (Rep::F64, Rep::Ref) => {
                let d = dim_of_type(self.f.ty(v)).unwrap_or(Dimension::DIMENSIONLESS);
                let d = self.dim_id(d);
                let d = self.c32(d);
                let h = self.call1(Rt::BoxNum, &[d, s.v]);
                self.check_handle(h);
                Ok(h)
            }
            (Rep::Bool, Rep::Ref) => {
                let w = self.b.ins().uextend(types::I32, s.v);
                let h = self.call1(Rt::BoxBool, &[w]);
                self.check_handle(h);
                Ok(h)
            }
            // A value of unknown type where a typed one is expected: check it.
            (Rep::Ref, Rep::F64) => {
                let d = dim_of_type(to).unwrap_or(Dimension::DIMENSIONLESS);
                let d = self.dim_id(d);
                let d = self.c32(d);
                let x = self.call1(Rt::PeekNum, &[s.v, d]);
                self.nan_fail(x);
                Ok(x)
            }
            (Rep::Ref, Rep::Bool) => {
                let x = self.call1(Rt::PeekBool, &[s.v]);
                let bad = self.b.ins().icmp_imm(IntCC::SignedLessThan, x, 0);
                self.fail_if(bad);
                Ok(self.b.ins().icmp_imm(IntCC::NotEqual, x, 0))
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
        for id in body {
            self.op(*id)?;
        }
        Ok(f.op(*term))
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
        }
    }

    fn op(&mut self, id: OpId) -> Res<()> {
        let f = self.f;
        let op = f.op(id);
        match &op.kind {
            OpKind::ConstFloat { bits } => {
                let x = f64::from_bits(*bits);
                if !x.is_finite() {
                    return refuse("a constant is not finite", Some(op.span));
                }
                let d = self.const_dim(op)?;
                let v = self.b.ins().f64const(x);
                self.define(op, 0, Comp::F64(v, d))
            }
            OpKind::ConstRational(q) => {
                let x = q.numer() as f64 / q.denom() as f64;
                if !x.is_finite() {
                    return refuse("a rational constant is not finite", Some(op.span));
                }
                let d = self.const_dim(op)?;
                let v = self.b.ins().f64const(x);
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
            | OpKind::Solve { .. }
            | OpKind::SolveRight { .. }
            | OpKind::MatMulEx { .. }
            | OpKind::DiagMul { .. }
            | OpKind::ArrayLiteral { .. }
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
                let r = match kind {
                    ArithOp::Add => self.b.ins().fadd(a.v, b.v),
                    ArithOp::Sub => self.b.ins().fsub(a.v, b.v),
                    ArithOp::Mul => self.b.ins().fmul(a.v, b.v),
                    ArithOp::Div => {
                        let zero = self.b.ins().f64const(0.0);
                        let bad = self.b.ins().fcmp(FloatCC::Equal, b.v, zero);
                        self.fail_with(bad, Rt::Fail2, &fail);
                        self.b.ins().fdiv(a.v, b.v)
                    }
                };
                self.finite_or(r, Rt::Fail2, &fail);
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
                    if x < 0.0 {
                        let zero = self.b.ins().f64const(0.0);
                        let bad = self.b.ins().fcmp(FloatCC::Equal, a.v, zero);
                        self.fail_with(bad, Rt::Fail1, &[site, a.v]);
                    }
                    let r = self.powi(a.v, x as i32);
                    self.finite_or(r, Rt::Fail1, &[site, a.v]);
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
                let zero = self.b.ins().f64const(0.0);
                let bad = self.b.ins().fcmp(FloatCC::Equal, a.v, zero);
                self.fail_with(bad, Rt::Fail1, &[site, a.v]);
                let one = self.b.ins().f64const(1.0);
                let r = self.b.ins().fdiv(one, a.v);
                self.finite_or(r, Rt::Fail1, &[site, a.v]);
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
        // `apply` ends with `+ 0.0`, which turns `-0` into `0`.
        let plus_zero = |tx: &mut Self, r: Cv| {
            let z = tx.b.ins().f64const(0.0);
            tx.b.ins().fadd(r, z)
        };
        let r = match m {
            MathFn::Sqrt => {
                let zero = self.b.ins().f64const(0.0);
                let bad = self.b.ins().fcmp(FloatCC::LessThan, xs[0], zero);
                self.fail_with(bad, Rt::Fail1, &[site, xs[0]]);
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
                let r = if n == 1 {
                    self.call1(Rt::Math1, &[site, xs[0]])
                } else {
                    self.call1(Rt::Math2, &[site, xs[0], xs[1]])
                };
                self.nan_fail(r);
                r
            }
        };
        self.define(op, 0, Comp::F64(r, dim))
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
            Rep::F64 => refuse("a condition is a number", Some(op.span)),
        }
    }

    fn if_op(&mut self, id: OpId, op: &'m OpData) -> Res<()> {
        let f = self.f;
        let c = self.operand(op, 0)?;
        let cond = self.condition(id, op, c)?;
        let [then_r, else_r] = op.regions[..] else {
            return refuse("an `if` needs two regions", Some(op.span));
        };
        let reps: Vec<Rep> = op.results.iter().map(|r| rep_of(f.ty(*r))).collect();
        let merge = self.b.create_block();
        for r in &reps {
            self.b.append_block_param(merge, rep_ty(*r));
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
                outs.push(self.transfer(term, v, f.ty(op.results[i]), "a yielded value")?);
            }
            self.release_owned();
            self.b.ins().jump(merge, &args_of(&outs));
        }
        self.b.switch_to_block(merge);
        let params = self.b.block_params(merge).to_vec();
        for (i, (&r, p)) in op.results.iter().zip(params).enumerate() {
            self.bind(r, Slot { v: p, rep: reps[i] });
        }
        Ok(())
    }

    /// `while`: a header block (the `before` region) with the loop-carried
    /// values as parameters, a body block (`after`), and an exit block whose
    /// parameters are the results. The values a branch hands on are
    /// transferred in a block of its own, so the `before` block's handles are
    /// released on both paths.
    fn while_op(&mut self, id: OpId, op: &'m OpData) -> Res<()> {
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
            self.b.append_block_param(header, rep_ty(rep_of(f.ty(a))));
        }
        let mut inits = Vec::new();
        for (i, &v) in op.operands.iter().enumerate() {
            inits.push(self.transfer(op, v, f.ty(bargs[i]), "a loop-carried value")?);
        }
        self.b.ins().jump(header, &args_of(&inits));

        // The `before` region: computes the condition and the forwarded values.
        self.b.switch_to_block(header);
        self.owned.push(Vec::new());
        let params = self.b.block_params(header).to_vec();
        for (&a, p) in bargs.iter().zip(params) {
            self.bind(
                a,
                Slot {
                    v: p,
                    rep: rep_of(f.ty(a)),
                },
            );
        }
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
            self.b.append_block_param(after_b, rep_ty(rep_of(f.ty(a))));
        }
        let res_reps: Vec<Rep> = op.results.iter().map(|r| rep_of(f.ty(*r))).collect();
        for r in &res_reps {
            self.b.append_block_param(exit_b, rep_ty(*r));
        }
        self.b.ins().brif(cond, after_pre, &[], exit_pre, &[]);
        self.b.switch_to_block(after_pre);
        let mut outs = Vec::new();
        for (i, &v) in fwd.iter().enumerate() {
            outs.push(self.transfer(term, v, f.ty(aargs[i]), "a forwarded value")?);
        }
        self.emit_release_owned();
        self.b.ins().jump(after_b, &args_of(&outs));
        self.b.switch_to_block(exit_pre);
        let mut outs = Vec::new();
        for (i, &v) in fwd.iter().enumerate() {
            outs.push(self.transfer(term, v, f.ty(op.results[i]), "a loop result")?);
        }
        self.emit_release_owned();
        self.b.ins().jump(exit_b, &args_of(&outs));
        self.owned.pop();

        // The `after` region: the next loop-carried values.
        self.b.switch_to_block(after_b);
        self.owned.push(Vec::new());
        let params = self.b.block_params(after_b).to_vec();
        for (&a, p) in aargs.iter().zip(params) {
            self.bind(
                a,
                Slot {
                    v: p,
                    rep: rep_of(f.ty(a)),
                },
            );
        }
        let term = self.block_body(after)?;
        if !matches!(term.kind, OpKind::Yield) || term.operands.len() != bargs.len() {
            return refuse(
                "a `while` `after` region must `yield` the next values",
                Some(op.span),
            );
        }
        let mut outs = Vec::new();
        for (i, &v) in term.operands.iter().enumerate() {
            outs.push(self.transfer(term, v, f.ty(bargs[i]), "a loop-carried value")?);
        }
        self.release_owned();
        self.b.ins().jump(header, &args_of(&outs));

        // After the loop.
        self.b.switch_to_block(exit_b);
        let params = self.b.block_params(exit_b).to_vec();
        for (i, (&r, p)) in op.results.iter().zip(params).enumerate() {
            self.bind(
                r,
                Slot {
                    v: p,
                    rep: res_reps[i],
                },
            );
        }
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
            args.push(self.transfer(
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
        let inst = self.b.ins().call(fref, &args);
        let results = self.b.inst_results(inst).to_vec();
        self.status_fail(results[0]);
        for (i, ty) in g.results.iter().enumerate() {
            let v = results[1 + i];
            let c = match rep_of(ty) {
                Rep::F64 => Comp::F64(v, dim_of_type(ty).unwrap_or(Dimension::DIMENSIONLESS)),
                Rep::Bool => Comp::Bool(v),
                Rep::Ref => Comp::Ref(v),
            };
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
            Rep::F64 => return refuse("an `assert` condition is a number", Some(op.span)),
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
