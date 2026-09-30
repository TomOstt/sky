//! The runtime native code calls: `extern "C"` functions over runtime-owned
//! values, and the state they share.
//!
//! # Values
//!
//! Native code keeps a number as an `f64` and a Bool as an `i8`. Everything
//! else (strings, arrays, values the checker could not type) is an *opaque
//! handle*: an index into a table of reference-counted
//! [`sky_interp::Value`]s that lives here, so native code never sees a
//! pointer to one. Handles are reference counted by the generated code (see
//! `module.rs`, "Ownership"); every count is dropped when the run ends.
//!
//! Every array operation is the function of [`sky_interp::array`] and
//! [`sky_interp::arith`] that the SIR interpreter calls (through the glue in
//! [`eval_op`], which follows `sky-ir-interp`'s `machine.rs` operation by
//! operation), so results and diagnostics are the interpreters'.
//!
//! # Calls and errors
//!
//! Operations that take a variable number of operands push them first
//! (`push_*`) and then call [`sky_rt_op`] with a *site*: an index into the
//! program's table of the SIR operations that can fail, which gives the
//! runtime the operation's kind, spans and static types. A failure is stored
//! as the pending error (a [`RuntimeError`], with all its spans and labels)
//! and the call reports it in its result (`0` for a handle, NaN for a
//! number, `-1` for a status); generated code then returns a status up the
//! native stack and [`crate::Compiled::run`] takes the pending error.
//!
//! Inline arithmetic that finds a failing case (a division by zero, a
//! non-finite result) calls one of the `sky_rt_fail_*` functions with the
//! same operands: they re-run the operation the interpreter's way, which
//! builds its exact diagnostic.
//!
//! # Values that are not what their type says
//!
//! A value the checker could not type (`unknown`) can flow into a slot a
//! declaration types: `-> m` on a function whose body computes an unknown.
//! The interpreters check dynamically and go on with whatever the value is;
//! native code holds the typed slot as a bare `f64` and cannot. So it *checks*
//! at that moment (`take_*`, `peek_*`): if the value is what the type says
//! (almost always) nothing else happens; if it is not, the run *deoptimizes*:
//! native code stops at once (no diagnostic), [`crate::Compiled::run`]
//! reports [`crate::RunFailure::Deopt`] with the number of output bytes
//! already written, and the caller runs the program again on the SIR
//! interpreter, skipping that many bytes ([`crate::SkipWriter`]). Output is
//! the only effect and the program is deterministic, so the result is the
//! interpreter's exactly.
//!
//! # Unsafe
//!
//! The only `unsafe` here is the pointer to the run's output writer, which
//! must outlive the borrow `Compiled::run` was given; see [`begin`].

#![allow(unsafe_code)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Write;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;

use sky_diag::Span;
use sky_interp::arith::{
    arith, equal, expect_bool, expect_number, order, pow, pow_scalar, range_bound, range_step,
    reciprocal, recursion_too_deep, sign, stack_exhausted, ArithOp,
};
use sky_interp::array::{self, Ix};
use sky_interp::contract;
use sky_interp::solve;
use sky_interp::{RuntimeError, Value};
use sky_ir::{BlockData, CmpOp, Function, Module, OpData, OpKind, RangePart, SliceAxis, Type};
use sky_units::{Dimension, MathFn, Rational};

type R<T> = Result<T, RuntimeError>;

/// A native function's view of the module: everything a runtime call needs
/// to know about the operation that called it.
pub(crate) struct Program {
    /// The SIR module the native code was made from.
    pub module: Module,
    /// Dimensions native code names by index (a number's dimension is not
    /// stored in the `f64`).
    pub dims: Vec<Dimension>,
    /// The operations native code refers to by index.
    pub sites: Vec<Site>,
}

/// One entry of [`Program::sites`]: an operation of a function, or (`op ==
/// [`FUNC_SITE`]`) the function itself.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Site {
    pub func: u32,
    pub op: u32,
}

/// The `op` of a site that stands for a whole function.
pub(crate) const FUNC_SITE: u32 = u32::MAX;

impl Program {
    fn function(&self, site: Site) -> Option<&Function> {
        self.module.functions.get(site.func as usize)
    }

    /// The operation of `site`, if it is one.
    fn op(&self, site: i32) -> Option<(&Function, &OpData)> {
        let s = *self.sites.get(usize::try_from(site).ok()?)?;
        let f = self.function(s)?;
        Some((f, f.ops.get(s.op as usize)?))
    }

    /// The span a site stands for: an operation's, or its function's.
    fn span(&self, site: i32) -> Span {
        let Some(s) = usize::try_from(site)
            .ok()
            .and_then(|i| self.sites.get(i))
            .copied()
        else {
            return Span::point(0);
        };
        match self.function(s) {
            Some(f) if s.op == FUNC_SITE => f.span,
            Some(f) => f.ops.get(s.op as usize).map_or(f.span, |o| o.span),
            None => Span::point(0),
        }
    }
}

/// An operand pushed for the next [`sky_rt_op`].
enum Arg {
    Val(Value),
    Shared(Rc<Value>),
}

impl Arg {
    fn get(&self) -> &Value {
        match self {
            Arg::Val(v) => v,
            Arg::Shared(v) => v,
        }
    }
}

/// One entry of the handle table.
struct Cell {
    rc: u32,
    value: Option<Rc<Value>>,
}

struct State {
    program: Arc<Program>,
    cells: Vec<Cell>,
    free: Vec<u32>,
    args: Vec<Arg>,
    /// The line `print` is building, and whether its next item is the first.
    line: String,
    first: bool,
    error: Option<RuntimeError>,
    /// A value was not what its type said (see the module docs).
    deopt: bool,
    /// Bytes `print` has written so far.
    written: usize,
    /// The value table of a `fused` body, kept between calls.
    scratch: Vec<Option<Value>>,
    /// Compiled `fused` kernels of this run, by the operation's address
    /// (stable while the run borrows the module; the state, and so the
    /// cache, is per run).
    kernels: HashMap<usize, KernelEntry>,
    out: NonNull<dyn Write>,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// Ends the run's runtime state when dropped.
pub(crate) struct RunGuard(());

impl Drop for RunGuard {
    fn drop(&mut self) {
        // The state (and with it the pointer to the writer) is gone before
        // `begin`'s borrow of the writer ends.
        let old = STATE.try_with(|s| s.try_borrow_mut().ok().and_then(|mut s| s.take()));
        drop(old);
    }
}

/// Starts a run: installs the runtime state for this thread, with `out` as
/// the writer `print` writes to. The state is removed when the guard drops.
///
/// The guard must be dropped before `out`'s borrow ends, which the caller
/// (`Compiled::run`) does by holding it for the duration of the call.
pub(crate) fn begin(program: Arc<Program>, out: &mut dyn Write) -> RunGuard {
    // SAFETY: only the lifetime bound of the trait object is erased. The
    // pointer is used only by runtime calls made while the returned guard is
    // alive, and `Compiled::run` keeps the guard (which removes the state
    // holding the pointer) inside the borrow of `out`.
    let out: NonNull<dyn Write> = unsafe {
        std::mem::transmute::<NonNull<dyn Write + '_>, NonNull<dyn Write + 'static>>(NonNull::from(
            out,
        ))
    };
    STATE.with(|s| {
        *s.borrow_mut() = Some(State {
            program,
            cells: Vec::new(),
            free: Vec::new(),
            args: Vec::new(),
            line: String::new(),
            first: true,
            error: None,
            deopt: false,
            written: 0,
            scratch: Vec::new(),
            kernels: HashMap::new(),
            out,
        });
    });
    RunGuard(())
}

/// How a run stopped.
pub(crate) struct Finish {
    /// The error it left, if any.
    pub error: Option<RuntimeError>,
    /// A value was not what its type said.
    pub deopt: bool,
    /// Bytes written.
    pub written: usize,
    /// Handles still alive: 0 after a run that finished, if every count was
    /// balanced (the generated code's ownership rules, see `module.rs`).
    pub live: usize,
}

pub(crate) fn finish() -> Finish {
    STATE
        .try_with(|s| {
            s.try_borrow_mut().ok().and_then(|mut s| {
                s.as_mut().map(|s| Finish {
                    error: s.error.take(),
                    deopt: s.deopt,
                    written: s.written,
                    live: s.cells.iter().filter(|c| c.rc > 0).count(),
                })
            })
        })
        .ok()
        .flatten()
        .unwrap_or(Finish {
            error: None,
            deopt: false,
            written: 0,
            live: 0,
        })
}

/// Runs `f` on the state, or gives `dflt` if there is none (a runtime call
/// outside a run, which generated code never makes).
///
/// A panic in `f` (a bug in code this calls) must not unwind into generated
/// code, which has no unwind information: it is caught, recorded as the run's
/// error, and `dflt` (which every caller's generated code treats as failure)
/// is returned.
fn with<T>(dflt: T, f: impl FnOnce(&mut State) -> T) -> T {
    let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        STATE
            .try_with(|s| match s.try_borrow_mut() {
                Ok(mut s) => s.as_mut().map(f),
                Err(_) => None,
            })
            .ok()
            .flatten()
    }));
    match ran {
        Ok(Some(v)) => v,
        Ok(None) => dflt,
        Err(_) => {
            // The state's borrow ended with the unwind.
            let _ = STATE.try_with(|s| {
                if let Ok(mut s) = s.try_borrow_mut() {
                    if let Some(st) = s.as_mut() {
                        st.fail(internal("a runtime function panicked"));
                    }
                }
            });
            dflt
        }
    }
}

fn internal(message: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::new(
        format!("internal error in the native engine: {message}"),
        Span::point(0),
    )
}

impl State {
    fn fail(&mut self, e: RuntimeError) {
        // The first failure is the one that stops the run.
        if self.error.is_none() {
            self.error = Some(e);
        }
    }

    fn alloc(&mut self, v: Rc<Value>) -> i64 {
        let cell = Cell {
            rc: 1,
            value: Some(v),
        };
        let index = match self.free.pop() {
            Some(i) => {
                self.cells[i as usize] = cell;
                i as usize
            }
            None => {
                self.cells.push(cell);
                self.cells.len() - 1
            }
        };
        index as i64 + 1
    }

    fn cell(&self, h: i64) -> Option<&Cell> {
        let i = usize::try_from(h.checked_sub(1)?).ok()?;
        self.cells.get(i).filter(|c| c.rc > 0)
    }

    fn value(&self, h: i64) -> Option<Rc<Value>> {
        self.cell(h)?.value.clone()
    }

    fn release(&mut self, h: i64) {
        let Some(i) = usize::try_from(h.wrapping_sub(1)).ok() else {
            return;
        };
        let Some(c) = self.cells.get_mut(i) else {
            return;
        };
        if c.rc == 0 {
            return;
        }
        c.rc -= 1;
        if c.rc == 0 {
            c.value = None;
            self.free.push(i as u32);
        }
    }

    /// The pushed operand `i` as a handle's value, or a failure.
    fn take_value(&mut self, h: i64) -> Option<Rc<Value>> {
        match self.value(h) {
            Some(v) => Some(v),
            None => {
                self.fail(internal(format!("handle {h} is not a live value")));
                None
            }
        }
    }
}

// ----- boxing and handles -----

pub(crate) extern "C" fn sky_rt_box_num(dim: i32, x: f64) -> i64 {
    with(0, |st| {
        let d = st.program.dims.get(dim as usize).copied();
        match d {
            Some(d) => st.alloc(Rc::new(Value::number(x, d))),
            None => {
                st.fail(internal("unknown dimension"));
                0
            }
        }
    })
}

pub(crate) extern "C" fn sky_rt_box_bool(b: i32) -> i64 {
    with(0, |st| st.alloc(Rc::new(Value::Bool(b != 0))))
}

pub(crate) extern "C" fn sky_rt_const_str(site: i32) -> i64 {
    with(0, |st| {
        let text = match st.program.op(site) {
            Some((
                _,
                OpData {
                    kind: OpKind::ConstStr(t),
                    ..
                },
            )) => t.clone(),
            _ => {
                st.fail(internal("`const.str` site"));
                return 0;
            }
        };
        st.alloc(Rc::new(Value::Str(text)))
    })
}

pub(crate) extern "C" fn sky_rt_const_none() -> i64 {
    with(0, |st| st.alloc(Rc::new(Value::None)))
}

pub(crate) extern "C" fn sky_rt_retain(h: i64) {
    with((), |st| {
        let i = usize::try_from(h.wrapping_sub(1)).unwrap_or(usize::MAX);
        if let Some(c) = st.cells.get_mut(i).filter(|c| c.rc > 0) {
            c.rc += 1;
        }
    });
}

pub(crate) extern "C" fn sky_rt_release(h: i64) {
    with((), |st| st.release(h));
}

/// The number in handle `h` if it is a number of dimension `dim`, else NaN
/// and the run deoptimizes. With `release`, `h` is released either way.
fn number_of(h: i64, dim: i32, release: bool) -> f64 {
    with(f64::NAN, |st| {
        let want = st.program.dims.get(dim as usize).copied();
        let v = st.value(h);
        if release {
            st.release(h);
        }
        match (v.as_deref(), want) {
            (Some(Value::Number { value, dim }), Some(want)) if *dim == want => *value,
            _ => {
                st.deopt = true;
                f64::NAN
            }
        }
    })
}

/// The Bool in handle `h` as 0 or 1 if it is a Bool, else -1 and the run
/// deoptimizes. With `release`, `h` is released either way.
fn bool_of(h: i64, release: bool) -> i32 {
    with(-1, |st| {
        let v = st.value(h);
        if release {
            st.release(h);
        }
        match v.as_deref() {
            Some(Value::Bool(b)) => i32::from(*b),
            _ => {
                st.deopt = true;
                -1
            }
        }
    })
}

/// The number in handle `h` (which is released): NaN and a deoptimization if
/// it is not a number of dimension `dim`.
pub(crate) extern "C" fn sky_rt_take_num(h: i64, dim: i32) -> f64 {
    number_of(h, dim, true)
}

/// [`sky_rt_take_num`] without releasing `h`.
pub(crate) extern "C" fn sky_rt_peek_num(h: i64, dim: i32) -> f64 {
    number_of(h, dim, false)
}

/// The Bool in handle `h` (which is released) as 0 or 1, or -1 and a
/// deoptimization if it is not a Bool.
pub(crate) extern "C" fn sky_rt_take_bool(h: i64) -> i32 {
    bool_of(h, true)
}

/// [`sky_rt_take_bool`] without releasing `h`.
pub(crate) extern "C" fn sky_rt_peek_bool(h: i64) -> i32 {
    bool_of(h, false)
}

/// The condition of the `if` or `while` at `site` held in handle `h`: 0 or 1,
/// or -1 with `sky run`'s error for a condition that is not a Bool.
pub(crate) extern "C" fn sky_rt_cond(site: i32, h: i64) -> i32 {
    with(-1, |st| {
        let Some(v) = st.take_value(h) else {
            return -1;
        };
        let keyword = match st.program.op(site) {
            Some((_, o)) if matches!(o.kind, OpKind::While) => "while",
            _ => "if",
        };
        let span = st.program.span(site);
        match expect_bool(keyword, &v, span) {
            Ok(b) => i32::from(b),
            Err(e) => {
                st.fail(e);
                -1
            }
        }
    })
}

// ----- operands and generic operations -----

pub(crate) extern "C" fn sky_rt_push_num(dim: i32, x: f64) {
    with((), |st| match st.program.dims.get(dim as usize).copied() {
        Some(d) => st.args.push(Arg::Val(Value::number(x, d))),
        None => st.fail(internal("unknown dimension")),
    });
}

pub(crate) extern "C" fn sky_rt_push_bool(b: i32) {
    with((), |st| st.args.push(Arg::Val(Value::Bool(b != 0))));
}

pub(crate) extern "C" fn sky_rt_push_ref(h: i64) {
    with((), |st| {
        if let Some(v) = st.take_value(h) {
            st.args.push(Arg::Shared(v));
        }
    });
}

/// Runs the operation at `site` on the operands pushed since the last
/// operation, and returns a handle to its result (0 on failure).
pub(crate) extern "C" fn sky_rt_op(site: i32) -> i64 {
    with(0, |st| {
        let args = std::mem::take(&mut st.args);
        let mut scratch = std::mem::take(&mut st.scratch);
        let mut kernels = std::mem::take(&mut st.kernels);
        let result = match st.program.op(site) {
            Some((f, op)) => eval_op(f, op, &args, &mut scratch, &mut kernels),
            None => Err(internal("unknown site")),
        };
        let mut args = args;
        args.clear();
        st.args = args;
        st.scratch = scratch;
        st.kernels = kernels;
        match result {
            Ok(v) => st.alloc(Rc::new(v)),
            Err(e) => {
                st.fail(e);
                0
            }
        }
    })
}

/// Re-runs the operation at `site`, which failed inline, on one scalar
/// operand of dimension `dim`... to produce its error.
fn fail_with(st: &mut State, site: i32, values: &[f64]) {
    let Some((f, op)) = st.program.op(site) else {
        st.fail(internal("unknown site"));
        return;
    };
    let mut args = Vec::with_capacity(values.len());
    for (i, x) in values.iter().enumerate() {
        let dim = op
            .operands
            .get(i)
            .and_then(|v| match f.ty(*v) {
                Type::Number(n) => Some(n.dim),
                _ => None,
            })
            .unwrap_or(Dimension::DIMENSIONLESS);
        args.push(Arg::Val(Value::number(*x, dim)));
    }
    match eval_op(f, op, &args, &mut Vec::new(), &mut HashMap::new()) {
        Err(e) => st.fail(e),
        Ok(_) => st.fail(internal(format!(
            "`{}` failed inline but succeeds in the runtime",
            op.kind.mnemonic()
        ))),
    }
}

/// The operation at `site` failed inline on the operand `a`.
pub(crate) extern "C" fn sky_rt_fail1(site: i32, a: f64) {
    with((), |st| fail_with(st, site, &[a]));
}

/// The operation at `site` failed inline on the operands `a` and `b`.
pub(crate) extern "C" fn sky_rt_fail2(site: i32, a: f64, b: f64) {
    with((), |st| fail_with(st, site, &[a, b]));
}

/// The call at `site` is nested more than `MAX_CALL_DEPTH` deep.
pub(crate) extern "C" fn sky_rt_fail_depth(site: i32) {
    with((), |st| {
        let span = st.program.span(site);
        let callee = match st.program.op(site) {
            Some((
                _,
                OpData {
                    kind: OpKind::Call { callee },
                    ..
                },
            )) => callee.clone(),
            _ => String::new(),
        };
        st.fail(recursion_too_deep(&callee, span));
    });
}

/// The function at `site` was entered with the stack budget used up.
pub(crate) extern "C" fn sky_rt_fail_stack(site: i32) {
    with((), |st| {
        let span = st.program.span(site);
        st.fail(stack_exhausted(span));
    });
}

/// `literal.error` and `input`: operations that fail whenever they run.
pub(crate) extern "C" fn sky_rt_fail_always(site: i32) {
    with((), |st| {
        let s = st.program.span(site);
        let e = match st.program.op(site) {
            Some((_, o)) => match &o.kind {
                OpKind::LiteralError {
                    error: sky_ir::LiteralError::TooLarge,
                } => sky_interp::number_too_large(s),
                OpKind::LiteralError {
                    error: sky_ir::LiteralError::Overflow,
                } => sky_interp::arith::overflow(s),
                OpKind::Input { name } => sky_interp::arith::declared_without_value(name, s),
                _ => internal("`fail_always` site"),
            },
            None => internal("unknown site"),
        };
        st.fail(e);
    });
}

// ----- scalar math -----

/// `f(x)` for the math function at `site`, in the interpreters' own code;
/// NaN with the pending error set on failure.
pub(crate) extern "C" fn sky_rt_math1(site: i32, x: f64) -> f64 {
    math(site, x, 0.0)
}

/// `f(x, y)` for the two-argument math function at `site`.
pub(crate) extern "C" fn sky_rt_math2(site: i32, x: f64, y: f64) -> f64 {
    math(site, x, y)
}

/// `f(x)` for the math function number `f` (its place in `MathFn::ALL`):
/// the value only, with no lookup of the site and no state. Native code has
/// already checked the domain, and checks the result, inline.
pub(crate) extern "C" fn sky_rt_libm1(f: i32, x: f64) -> f64 {
    libm(f, x, 0.0)
}

/// [`sky_rt_libm1`] for a two-argument function.
pub(crate) extern "C" fn sky_rt_libm2(f: i32, x: f64, y: f64) -> f64 {
    libm(f, x, y)
}

fn libm(f: i32, x: f64, y: f64) -> f64 {
    match MathFn::ALL.get(f as usize) {
        Some(f) => sky_interp::math::core_f64(*f, x, y),
        None => f64::NAN,
    }
}

fn math(site: i32, x: f64, y: f64) -> f64 {
    with(f64::NAN, |st| {
        let span = st.program.span(site);
        let Some((_, op)) = st.program.op(site) else {
            st.fail(internal("unknown site"));
            return f64::NAN;
        };
        let OpKind::Math(f) = op.kind else {
            st.fail(internal("`math` site"));
            return f64::NAN;
        };
        match sky_interp::math::apply_f64(f, x, y, span, op.operand_span(0)) {
            Ok(r) => r,
            Err(e) => {
                st.fail(e);
                f64::NAN
            }
        }
    })
}

/// `b ^ x` for the `pow.const` or `pow` at `site`: `sky_interp`'s
/// `pow_scalar`; NaN with the pending error set on failure.
pub(crate) extern "C" fn sky_rt_pow_scalar(site: i32, b: f64, x: f64) -> f64 {
    with(f64::NAN, |st| {
        let span = st.program.span(site);
        let base_span = st
            .program
            .op(site)
            .map_or(span, |(_, op)| op.operand_span(0));
        match pow_scalar(b, x, span, base_span) {
            Ok(r) => r,
            Err(e) => {
                st.fail(e);
                f64::NAN
            }
        }
    })
}

// ----- printing -----

fn print_item(text: String) {
    with((), |st| {
        if !std::mem::take(&mut st.first) {
            st.line.push(' ');
        }
        st.line.push_str(&text);
    });
}

pub(crate) extern "C" fn sky_rt_print_num(dim: i32, x: f64) {
    let d = with(None, |st| st.program.dims.get(dim as usize).copied());
    if let Some(d) = d {
        print_item(Value::number(x, d).print_text());
    }
}

pub(crate) extern "C" fn sky_rt_print_bool(b: i32) {
    print_item(Value::Bool(b != 0).print_text());
}

pub(crate) extern "C" fn sky_rt_print_ref(h: i64) {
    let v = with(None, |st| st.value(h));
    if let Some(v) = v {
        print_item(v.print_text());
    }
}

/// Ends the `print` at `site`: writes the line in one piece, like the SIR
/// interpreter. 0 on success, 1 with the pending error set otherwise.
pub(crate) extern "C" fn sky_rt_print_end(site: i32) -> i32 {
    with(1, |st| {
        let mut line = std::mem::take(&mut st.line);
        st.first = true;
        line.push('\n');
        // SAFETY: `begin` stored a pointer to the writer `Compiled::run` was
        // given, and that borrow lasts as long as this state does (the guard
        // removes the state first). No other reference to the writer exists
        // while native code runs, and this is the only place that uses it.
        let out = unsafe { st.out.as_mut() };
        match out.write_all(line.as_bytes()) {
            Ok(()) => {
                st.written += line.len();
                0
            }
            Err(e) => {
                let span = st.program.span(site);
                st.fail(RuntimeError::new(format!("cannot write output: {e}"), span));
                1
            }
        }
    })
}

// ----- contracts -----

/// The condition of the `assert` at `site` held in handle `h`: 1 if it holds,
/// 0 if it is false (the caller then pushes the values of the names and
/// calls [`sky_rt_assert_fail`]), -1 with the pending error set if it is not
/// a Bool.
pub(crate) extern "C" fn sky_rt_assert_dyn(site: i32, h: i64) -> i32 {
    with(-1, |st| {
        let Some(v) = st.take_value(h) else {
            return -1;
        };
        let Some((_, op)) = st.program.op(site) else {
            st.fail(internal("unknown site"));
            return -1;
        };
        let OpKind::Assert { kind, .. } = &op.kind else {
            st.fail(internal("`assert` site"));
            return -1;
        };
        match contract::expect_condition(*kind, &v, op.span) {
            Ok(b) => i32::from(b),
            Err(e) => {
                st.fail(e);
                -1
            }
        }
    })
}

/// The contract at `site` is false. The values of its names were pushed;
/// `call_site` is the site of the call that entered the function (or -1),
/// which a failed `requires` labels "called here".
pub(crate) extern "C" fn sky_rt_assert_fail(site: i32, call_site: i32) {
    with((), |st| {
        let args = std::mem::take(&mut st.args);
        let call = (call_site >= 0).then(|| st.program.span(call_site));
        let e = match st.program.op(site) {
            Some((
                _,
                OpData {
                    kind:
                        OpKind::Assert {
                            kind,
                            text,
                            message,
                            func,
                            names,
                        },
                    span,
                    ..
                },
            )) => {
                let shown: Vec<(&str, String)> = names
                    .iter()
                    .zip(&args)
                    .map(|(n, v)| (n.as_str(), contract::show(v.get())))
                    .collect();
                let mut err = contract::failure(
                    *kind,
                    func.as_deref(),
                    text,
                    message.as_deref(),
                    &shown,
                    *span,
                );
                if let (sky_ir::AssertKind::Requires, Some(call)) = (kind, call) {
                    err = err.with_secondary(call, "called here");
                }
                err
            }
            _ => internal("`assert` site"),
        };
        let mut args = args;
        args.clear();
        st.args = args;
        st.fail(e);
    });
}

// ----- the operations on values -----

fn malformed(what: impl std::fmt::Display, span: Span) -> RuntimeError {
    RuntimeError::new(format!("malformed IR: {what}"), span)
}

/// Evaluates a non-control-flow operation on evaluated operands: the glue of
/// `pure_op` in `sky-ir-interp`'s `machine.rs`, over the same functions.
fn eval_op(
    func: &Function,
    op: &OpData,
    a: &[Arg],
    scratch: &mut Vec<Option<Value>>,
    kernels: &mut HashMap<usize, KernelEntry>,
) -> R<Value> {
    let s = op.span;
    let arg = |i: usize| -> R<&Value> {
        a.get(i).map(Arg::get).ok_or_else(|| {
            malformed(
                format!("`{}` is missing operand {i}", op.kind.mnemonic()),
                s,
            )
        })
    };
    let (l, r) = (op.operand_span(0), op.operand_span(1));
    match &op.kind {
        OpKind::Add => arith(ArithOp::Add, arg(0)?, arg(1)?, s, l, r),
        OpKind::Sub => arith(ArithOp::Sub, arg(0)?, arg(1)?, s, l, r),
        OpKind::Mul => arith(ArithOp::Mul, arg(0)?, arg(1)?, s, l, r),
        OpKind::Div => arith(ArithOp::Div, arg(0)?, arg(1)?, s, l, r),
        OpKind::Neg => sign(true, arg(0)?.clone(), l),
        OpKind::Math(f) => {
            let vals: Vec<Value> = a.iter().map(|x| x.get().clone()).collect();
            let spans: Vec<Span> = (0..vals.len()).map(|i| op.operand_span(i)).collect();
            sky_interp::math::apply(*f, &vals, s, &spans)
        }
        OpKind::Show { unit } => sky_interp::show_unit_text(unit, arg(0)?, s),
        OpKind::PowConst { exponent } => {
            let e = Value::scalar(exponent.numer() as f64 / exponent.denom() as f64);
            pow(arg(0)?, &e, || Some(*exponent), s, l, s)
        }
        OpKind::Pow => pow(arg(0)?, arg(1)?, || None::<Rational>, s, l, r),
        OpKind::Inverse => reciprocal("⁻¹", arg(0)?, s, l),
        OpKind::MatMul => array::matmul(arg(0)?, arg(1)?, s, l, r),
        OpKind::Norm => sky_interp::vecfn::norm(arg(0)?, s, l),
        OpKind::Dot => sky_interp::vecfn::dot(arg(0)?, arg(1)?, s, l, r),
        OpKind::Transpose => array::transpose(arg(0)?, s),
        OpKind::Solve { method } => solve::solve(*method, arg(0)?, arg(1)?, s, l, r),
        OpKind::SolveRight { method } => solve::solve_right(*method, arg(0)?, arg(1)?, s, l, r),
        OpKind::MatMulEx { ta, tb, algorithm } => {
            solve::matmul_ex(*ta, *tb, *algorithm, arg(0)?, arg(1)?, s, l, r)
        }
        OpKind::DiagMul { side } => {
            // The diagonal is the left operand for `Side::Left`.
            let (d, x, dspan, xspan) = if *side == sky_ir::Side::Left {
                (arg(0)?, arg(1)?, l, r)
            } else {
                (arg(1)?, arg(0)?, r, l)
            };
            solve::diag_mul(*side, d, x, s, dspan, xspan)
        }
        OpKind::ArrayLiteral { cols } => array_literal(op, a, *cols),
        OpKind::Index => {
            let mut items = Vec::with_capacity(a.len().saturating_sub(1));
            for k in 1..a.len() {
                let at = op.operand_span(k);
                items.push(Ix::At {
                    i: array::to_index(arg(k)?, at)?,
                    span: at,
                });
            }
            array::index(arg(0)?, &items, s, l)
        }
        OpKind::Slice { axes } => slice(op, a, axes),
        OpKind::Zeros | OpKind::Identity => {
            let name = op.kind.mnemonic();
            let mut sizes = Vec::with_capacity(2);
            for (i, v) in a.iter().enumerate() {
                sizes.push(array::size_arg(name, v.get(), op.operand_span(i))?);
            }
            match (&op.kind, sizes.as_slice()) {
                (OpKind::Zeros, [_] | [_, _]) => array::zeros(&sizes, s),
                (OpKind::Identity, [n]) => array::identity(*n, s),
                _ => Err(malformed(
                    format!(
                        "`{name}` takes a size (`zeros`: two), found {}",
                        sizes.len()
                    ),
                    s,
                )),
            }
        }
        OpKind::Len => array::len(arg(0)?, l),
        OpKind::Cmp(c) => {
            let (x, y) = (arg(0)?, arg(1)?);
            let holds = match c {
                CmpOp::Eq => equal(x, y, s, l, r)?,
                CmpOp::Ne => !equal(x, y, s, l, r)?,
                CmpOp::Lt => order("<", |p, q| p < q, x, y, s, l, r)?,
                CmpOp::Le => order("<=", |p, q| p <= q, x, y, s, l, r)?,
                CmpOp::Gt => order(">", |p, q| p > q, x, y, s, l, r)?,
                CmpOp::Ge => order(">=", |p, q| p >= q, x, y, s, l, r)?,
            };
            Ok(Value::Bool(holds))
        }
        OpKind::Not => Ok(Value::Bool(!expect_bool("not", arg(0)?, l)?)),
        OpKind::ExpectBool { context } => {
            Ok(Value::Bool(expect_bool(context.keyword(), arg(0)?, s)?))
        }
        OpKind::Assume => Ok(arg(0)?.clone()),
        OpKind::ExpectRange { part } => match part {
            RangePart::Bound => range_bound(arg(0)?, s),
            RangePart::Step => range_step(arg(0)?, s),
        },
        OpKind::ExpectNumber { context } => {
            let v = arg(0)?;
            expect_number(context.keyword(), v, s)?;
            Ok(v.clone())
        }
        OpKind::Fused => fused(func, op, a, scratch, kernels),
        other => Err(malformed(
            format!("unexpected `{}` in the native runtime", other.mnemonic()),
            s,
        )),
    }
}

/// `array.literal`: the operands in row-major order, arranged as `cols` says.
fn array_literal(op: &OpData, a: &[Arg], cols: Option<u64>) -> R<Value> {
    let s = op.span;
    let entries: Vec<(Value, Span)> = a
        .iter()
        .enumerate()
        .map(|(i, v)| (v.get().clone(), op.operand_span(i)))
        .collect();
    let width = match cols {
        None => entries.len(),
        Some(c) => usize::try_from(c).unwrap_or(0),
    };
    if width == 0 || entries.len() % width != 0 {
        return Err(malformed(
            format!(
                "`array.literal` cannot arrange {} operands in rows of {width}",
                entries.len()
            ),
            s,
        ));
    }
    let rows: Vec<Vec<(Value, Span)>> = entries.chunks(width).map(<[_]>::to_vec).collect();
    array::literal(rows, cols.is_none(), s)
}

/// `slice`: the base, then each item's operands in order.
fn slice(op: &OpData, a: &[Arg], axes: &[SliceAxis]) -> R<Value> {
    let s = op.span;
    let Some((base, rest)) = a.split_first() else {
        return Err(malformed("`slice` has no operands", s));
    };
    let mut rest = rest.iter();
    let mut taken = 0;
    let mut bound = || -> R<(usize, Span)> {
        let Some(v) = rest.next() else {
            return Err(malformed(
                "`slice` has fewer operands than its items need",
                s,
            ));
        };
        taken += 1;
        let at = op.operand_span(taken);
        Ok((array::to_index(v.get(), at)?, at))
    };
    let mut items = Vec::with_capacity(axes.len());
    for (k, axis) in axes.iter().enumerate() {
        items.push(match *axis {
            SliceAxis::At => {
                let (i, at) = bound()?;
                Ix::At { i, span: at }
            }
            SliceAxis::Range { start, stop } => {
                let start = if start { Some(bound()?.0) } else { None };
                let stop = if stop { Some(bound()?.0) } else { None };
                Ix::Slice {
                    start,
                    stop,
                    span: op.axis_span(k),
                }
            }
        });
    }
    if rest.next().is_some() {
        return Err(malformed(
            "`slice` has more operands than its items need",
            s,
        ));
    }
    array::index(base.get(), &items, s, op.operand_span(0))
}

// ----- fused -----

/// `fused`: runs the scalar body once per element, the interpreter's way
/// (`fused_op` in `sky-ir-interp`): each body operation is the operation of
/// the same kind on numbers, except `neg`, which is `0 - x`. This is the
/// entry point a compiled kernel replaces.
/// What the run knows of one `fused` operation: the operand kinds its kernel
/// was compiled (or refused) for, the kernel, and how many elements the
/// interpreted loop has run for it.
#[derive(Default)]
pub(crate) struct KernelEntry {
    sigs: Vec<crate::kernel::InputSig>,
    kernel: Option<crate::kernel::Kernel>,
    interpreted: usize,
}

/// Elements a `fused` operation runs interpreted before its body is
/// compiled: compiling costs about what interpreting this many does (the
/// SIR interpreter's threshold, `JitOptions::DEFAULT_THRESHOLD`).
const KERNEL_THRESHOLD: usize = 512;

/// A compiled kernel for `op` if one pays, run on the operands: its result
/// and error are the loop's below (see `crate::kernel`). `None`: run the loop.
fn fused_native(
    op: &OpData,
    func: &Function,
    a: &[Arg],
    count: usize,
    kernels: &mut HashMap<usize, KernelEntry>,
) -> Option<R<(Vec<f64>, Dimension)>> {
    use crate::kernel::{compile_kernel, Input, InputSig};
    let mut sigs = Vec::with_capacity(a.len());
    let mut args = Vec::with_capacity(a.len());
    for input in a {
        match input.get() {
            Value::Array { data, dim, .. } => {
                sigs.push(InputSig {
                    dim: *dim,
                    array: true,
                });
                args.push(Input::Array(data));
            }
            Value::Number { value, dim } => {
                sigs.push(InputSig {
                    dim: *dim,
                    array: false,
                });
                args.push(Input::Scalar(*value));
            }
            _ => return None,
        }
    }
    let entry = kernels.entry(op as *const OpData as usize).or_default();
    if entry.sigs != sigs {
        entry.interpreted = entry.interpreted.saturating_add(count);
        if entry.interpreted < KERNEL_THRESHOLD {
            return None;
        }
        entry.kernel = compile_kernel(func, op, &sigs).ok();
        entry.sigs = sigs;
    }
    let kernel = entry.kernel.as_ref()?;
    Some(
        kernel
            .run_alloc(&args, count)
            .map(|data| (data, kernel.output_dim())),
    )
}

fn fused(
    func: &Function,
    op: &OpData,
    a: &[Arg],
    vals: &mut Vec<Option<Value>>,
    kernels: &mut HashMap<usize, KernelEntry>,
) -> R<Value> {
    let s = op.span;
    let [region] = op.regions[..] else {
        return Err(malformed("`fused` takes one region", s));
    };
    let block: &BlockData = region_block(func, region, s)?;
    let Some((&last, body)) = block.ops.split_last() else {
        return Err(malformed("the body of `fused` is empty", s));
    };
    let last = func
        .ops
        .get(last.index())
        .ok_or_else(|| malformed("no such operation", s))?;
    let (OpKind::Yield, &[yielded]) = (&last.kind, &last.operands[..]) else {
        return Err(malformed(
            "the body of `fused` must end in a `yield` of one value",
            s,
        ));
    };
    if block.args.len() != a.len() {
        return Err(malformed(
            format!(
                "the body of `fused` takes {} arguments, given {}",
                block.args.len(),
                a.len()
            ),
            s,
        ));
    }
    let mut shape: Option<&Vec<usize>> = None;
    for input in a {
        match input.get() {
            Value::Array { dims, .. } => match shape {
                Some(d) if d != dims => {
                    return Err(malformed(
                        "the array operands of `fused` differ in shape",
                        s,
                    ))
                }
                _ => shape = Some(dims),
            },
            Value::Number { .. } => {}
            other => {
                return Err(malformed(
                    format!("`fused` takes numbers, found {}", other.kind()),
                    s,
                ))
            }
        }
    }
    let Some(dims) = shape else {
        return Err(malformed("`fused` needs an array operand", s));
    };
    let count: usize = dims.iter().product();
    if let Some(done) = fused_native(op, func, a, count, kernels) {
        let (data, dim) = done?;
        return Ok(Value::array(dims.clone(), data, dim));
    }
    // The body's values, in a buffer the runtime keeps between calls (a value
    // is always set before it is read, so nothing is cleared).
    if vals.len() < func.values.len() {
        vals.resize(func.values.len(), None);
    }
    let mut data = Vec::with_capacity(count);
    let mut out_dim = None;
    let mut limit = body.len();
    let mut failure: Option<RuntimeError> = None;
    for i in 0..count {
        for (arg, input) in block.args.iter().zip(a) {
            let element = match input.get() {
                Value::Array { data, dim, .. } => Value::number(data[i], *dim),
                scalar => scalar.clone(),
            };
            set(vals, *arg, element, s)?;
        }
        let mut failed = false;
        for (k, id) in body[..limit].iter().enumerate() {
            let body_op = func
                .ops
                .get(id.index())
                .ok_or_else(|| malformed("no such operation", s))?;
            if let Err(e) = body_eval(func, body_op, vals) {
                (limit, failure, failed) = (k, Some(e), true);
                break;
            }
        }
        if limit == 0 {
            break;
        }
        if failed || failure.is_some() {
            continue;
        }
        match vals.get(yielded.index()).and_then(Option::as_ref) {
            Some(Value::Number { value, dim }) => {
                out_dim.get_or_insert(*dim);
                data.push(*value);
            }
            Some(other) => {
                return Err(malformed(
                    format!("`fused` must yield a number, found {}", other.kind()),
                    s,
                ))
            }
            None => return Err(malformed("`fused` yields an undefined value", s)),
        }
    }
    if let Some(e) = failure {
        return Err(e);
    }
    let Some(dim) = out_dim else {
        return Err(malformed("`fused` computed no element", s));
    };
    Ok(Value::array(dims.clone(), data, dim))
}

fn region_block(func: &Function, region: sky_ir::RegionId, span: Span) -> R<&BlockData> {
    let r = func
        .regions
        .get(region.index())
        .ok_or_else(|| malformed("no such region", span))?;
    let [b] = r.blocks[..] else {
        return Err(malformed("a region has one block in v1", span));
    };
    func.blocks
        .get(b.index())
        .ok_or_else(|| malformed("no such block", span))
}

fn set(vals: &mut [Option<Value>], v: sky_ir::ValueId, x: Value, span: Span) -> R<()> {
    match vals.get_mut(v.index()) {
        Some(slot) => {
            *slot = Some(x);
            Ok(())
        }
        None => Err(malformed(format!("{v} is not a value"), span)),
    }
}

/// One operation of a `fused` body, its operands read from `vals`.
fn body_eval(func: &Function, op: &OpData, vals: &mut [Option<Value>]) -> R<()> {
    let s = op.span;
    let result = match &op.kind {
        OpKind::ConstFloat { bits } => {
            let x = f64::from_bits(*bits);
            if !x.is_finite() {
                return Err(malformed("a constant is not finite", s));
            }
            Value::number(x, const_dim(func, op)?)
        }
        OpKind::ConstRational(q) => {
            let x = q.numer() as f64 / q.denom() as f64;
            if !x.is_finite() {
                return Err(malformed("a rational constant is not finite", s));
            }
            Value::number(x, const_dim(func, op)?)
        }
        OpKind::ConstBool(b) => Value::Bool(*b),
        OpKind::Neg => {
            let a = *op
                .operands
                .first()
                .ok_or_else(|| malformed("`neg` has no operand", s))?;
            match value_of(vals, a, s)? {
                Value::Number { value, dim } => Value::number(0.0 - value, *dim),
                other => sign(true, other.clone(), op.operand_span(0))?,
            }
        }
        // The four operations a fusion makes, on the values in place: the
        // body runs once per element, so it must not allocate.
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div => {
            let kind = match op.kind {
                OpKind::Add => ArithOp::Add,
                OpKind::Sub => ArithOp::Sub,
                OpKind::Mul => ArithOp::Mul,
                _ => ArithOp::Div,
            };
            let (Some(&a), Some(&b)) = (op.operands.first(), op.operands.get(1)) else {
                return Err(malformed("an arithmetic operation has two operands", s));
            };
            arith(
                kind,
                value_of(vals, a, s)?,
                value_of(vals, b, s)?,
                s,
                op.operand_span(0),
                op.operand_span(1),
            )?
        }
        _ => {
            let mut args = Vec::with_capacity(op.operands.len());
            for v in &op.operands {
                args.push(Arg::Val(value_of(vals, *v, s)?.clone()));
            }
            eval_op(func, op, &args, &mut Vec::new(), &mut HashMap::new())?
        }
    };
    let [r] = op.results[..] else {
        return Err(malformed("a body operation has one result", s));
    };
    set(vals, r, result, s)
}

fn value_of(vals: &[Option<Value>], v: sky_ir::ValueId, span: Span) -> R<&Value> {
    vals.get(v.index())
        .and_then(Option::as_ref)
        .ok_or_else(|| malformed(format!("{v} is used before it is defined"), span))
}

/// The dimension of a constant, from its result's type (`Unknown` is
/// dimensionless), as `machine.rs` reads it.
fn const_dim(func: &Function, op: &OpData) -> R<Dimension> {
    let [r] = op.results[..] else {
        return Err(malformed("a constant has one result", op.span));
    };
    match func.values.get(r.index()).map(|d| &d.ty) {
        Some(Type::Number(n)) if n.shape.rank() == 0 => Ok(n.dim),
        Some(Type::Unknown) => Ok(Dimension::DIMENSIONLESS),
        _ => Err(malformed(
            "a constant must have a scalar number result",
            op.span,
        )),
    }
}

// ----- the table of runtime functions -----

/// A type at the native/runtime boundary.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Ty {
    I32,
    I64,
    F64,
}

/// The runtime functions native code can call.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Rt {
    BoxNum,
    BoxBool,
    ConstStr,
    ConstNone,
    Retain,
    Release,
    TakeNum,
    TakeBool,
    PeekNum,
    PeekBool,
    Cond,
    PushNum,
    PushBool,
    PushRef,
    Op,
    Fail1,
    Fail2,
    FailDepth,
    FailStack,
    FailAlways,
    Math1,
    Math2,
    Libm1,
    Libm2,
    PowScalar,
    PrintNum,
    PrintBool,
    PrintRef,
    PrintEnd,
    AssertDyn,
    AssertFail,
}

impl Rt {
    pub const ALL: [Rt; 31] = [
        Rt::BoxNum,
        Rt::BoxBool,
        Rt::ConstStr,
        Rt::ConstNone,
        Rt::Retain,
        Rt::Release,
        Rt::TakeNum,
        Rt::TakeBool,
        Rt::PeekNum,
        Rt::PeekBool,
        Rt::Cond,
        Rt::PushNum,
        Rt::PushBool,
        Rt::PushRef,
        Rt::Op,
        Rt::Fail1,
        Rt::Fail2,
        Rt::FailDepth,
        Rt::FailStack,
        Rt::FailAlways,
        Rt::Math1,
        Rt::Math2,
        Rt::Libm1,
        Rt::Libm2,
        Rt::PowScalar,
        Rt::PrintNum,
        Rt::PrintBool,
        Rt::PrintRef,
        Rt::PrintEnd,
        Rt::AssertDyn,
        Rt::AssertFail,
    ];

    /// The symbol name, also what `--emit=clif` shows.
    pub fn name(self) -> &'static str {
        match self {
            Rt::BoxNum => "sky_rt_box_num",
            Rt::BoxBool => "sky_rt_box_bool",
            Rt::ConstStr => "sky_rt_const_str",
            Rt::ConstNone => "sky_rt_const_none",
            Rt::Retain => "sky_rt_retain",
            Rt::Release => "sky_rt_release",
            Rt::TakeNum => "sky_rt_take_num",
            Rt::TakeBool => "sky_rt_take_bool",
            Rt::PeekNum => "sky_rt_peek_num",
            Rt::PeekBool => "sky_rt_peek_bool",
            Rt::Cond => "sky_rt_cond",
            Rt::PushNum => "sky_rt_push_num",
            Rt::PushBool => "sky_rt_push_bool",
            Rt::PushRef => "sky_rt_push_ref",
            Rt::Op => "sky_rt_op",
            Rt::Fail1 => "sky_rt_fail1",
            Rt::Fail2 => "sky_rt_fail2",
            Rt::FailDepth => "sky_rt_fail_depth",
            Rt::FailStack => "sky_rt_fail_stack",
            Rt::FailAlways => "sky_rt_fail_always",
            Rt::Math1 => "sky_rt_math1",
            Rt::Math2 => "sky_rt_math2",
            Rt::Libm1 => "sky_rt_libm1",
            Rt::Libm2 => "sky_rt_libm2",
            Rt::PowScalar => "sky_rt_pow_scalar",
            Rt::PrintNum => "sky_rt_print_num",
            Rt::PrintBool => "sky_rt_print_bool",
            Rt::PrintRef => "sky_rt_print_ref",
            Rt::PrintEnd => "sky_rt_print_end",
            Rt::AssertDyn => "sky_rt_assert_dyn",
            Rt::AssertFail => "sky_rt_assert_fail",
        }
    }

    /// The function's address.
    pub fn address(self) -> *const u8 {
        match self {
            Rt::BoxNum => sky_rt_box_num as *const u8,
            Rt::BoxBool => sky_rt_box_bool as *const u8,
            Rt::ConstStr => sky_rt_const_str as *const u8,
            Rt::ConstNone => sky_rt_const_none as *const u8,
            Rt::Retain => sky_rt_retain as *const u8,
            Rt::Release => sky_rt_release as *const u8,
            Rt::TakeNum => sky_rt_take_num as *const u8,
            Rt::TakeBool => sky_rt_take_bool as *const u8,
            Rt::PeekNum => sky_rt_peek_num as *const u8,
            Rt::PeekBool => sky_rt_peek_bool as *const u8,
            Rt::Cond => sky_rt_cond as *const u8,
            Rt::PushNum => sky_rt_push_num as *const u8,
            Rt::PushBool => sky_rt_push_bool as *const u8,
            Rt::PushRef => sky_rt_push_ref as *const u8,
            Rt::Op => sky_rt_op as *const u8,
            Rt::Fail1 => sky_rt_fail1 as *const u8,
            Rt::Fail2 => sky_rt_fail2 as *const u8,
            Rt::FailDepth => sky_rt_fail_depth as *const u8,
            Rt::FailStack => sky_rt_fail_stack as *const u8,
            Rt::FailAlways => sky_rt_fail_always as *const u8,
            Rt::Math1 => sky_rt_math1 as *const u8,
            Rt::Math2 => sky_rt_math2 as *const u8,
            Rt::Libm1 => sky_rt_libm1 as *const u8,
            Rt::Libm2 => sky_rt_libm2 as *const u8,
            Rt::PowScalar => sky_rt_pow_scalar as *const u8,
            Rt::PrintNum => sky_rt_print_num as *const u8,
            Rt::PrintBool => sky_rt_print_bool as *const u8,
            Rt::PrintRef => sky_rt_print_ref as *const u8,
            Rt::PrintEnd => sky_rt_print_end as *const u8,
            Rt::AssertDyn => sky_rt_assert_dyn as *const u8,
            Rt::AssertFail => sky_rt_assert_fail as *const u8,
        }
    }

    /// The parameter types and the result type (`None`: no result).
    pub fn signature(self) -> (&'static [Ty], Option<Ty>) {
        use Ty::*;
        match self {
            Rt::BoxNum => (&[I32, F64], Some(I64)),
            Rt::BoxBool => (&[I32], Some(I64)),
            Rt::ConstStr => (&[I32], Some(I64)),
            Rt::ConstNone => (&[], Some(I64)),
            Rt::Retain | Rt::Release => (&[I64], None),
            Rt::TakeNum | Rt::PeekNum => (&[I64, I32], Some(F64)),
            Rt::TakeBool | Rt::PeekBool => (&[I64], Some(I32)),
            Rt::Cond => (&[I32, I64], Some(I32)),
            Rt::PushNum => (&[I32, F64], None),
            Rt::PushBool => (&[I32], None),
            Rt::PushRef => (&[I64], None),
            Rt::Op => (&[I32], Some(I64)),
            Rt::Fail1 => (&[I32, F64], None),
            Rt::Fail2 => (&[I32, F64, F64], None),
            Rt::FailDepth | Rt::FailStack | Rt::FailAlways => (&[I32], None),
            Rt::Math1 => (&[I32, F64], Some(F64)),
            Rt::Math2 => (&[I32, F64, F64], Some(F64)),
            Rt::Libm1 => (&[I32, F64], Some(F64)),
            Rt::Libm2 => (&[I32, F64, F64], Some(F64)),
            Rt::PowScalar => (&[I32, F64, F64], Some(F64)),
            Rt::PrintNum => (&[I32, F64], None),
            Rt::PrintBool => (&[I32], None),
            Rt::PrintRef => (&[I64], None),
            Rt::PrintEnd => (&[I32], Some(I32)),
            Rt::AssertDyn => (&[I32, I64], Some(I32)),
            Rt::AssertFail => (&[I32, I32], None),
        }
    }
}
