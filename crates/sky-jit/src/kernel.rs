//! Compiled `fused` kernels: the scalar body of a SIR `fused` operation
//! turned into a native loop over the elements.
//!
//! [`compile_kernel`] takes a `fused` operation and what is known of its
//! operands (their dimensions, and which are arrays and which broadcast
//! scalars) and returns a [`Kernel`]. [`Kernel::run`] computes the result
//! for `n` elements.
//!
//! # Exactness
//!
//! A kernel gives the bytes and the errors of `sky-ir-interp`'s `fused_op`,
//! which runs the body once per element through the code of the unfused
//! operations (`sky_interp::arith`):
//!
//! * `add`, `sub`, `mul`, `div` are the plain IEEE instructions. Nothing is
//!   ever contracted into a fused multiply-add: the code generator has no
//!   such rewrite, and the tests below compare bits, including `-0`.
//! * `neg` is `0 - x`, as in the interpreter's kernel body (a zero stays
//!   `+0`).
//! * A constant is the `f64` the interpreter builds (`bits`, or
//!   `numer as f64 / denom as f64`).
//! * `sqrt` and `abs` are the hardware instructions (correctly rounded and
//!   exact, so equal to Rust's), followed by the interpreter's `+ 0.0` that
//!   turns a `-0` into `0`. Every other math function is a call to
//!   `sky_interp::math::apply` (see `kernel_rt`).
//! * Dimensions are settled when the kernel is compiled: the result
//!   dimension is computed with the interpreter's own rules, and a body
//!   whose dimensions do not agree (an error the interpreter would report at
//!   run time), or that has an operation a kernel does not implement, is
//!   **refused** ([`Refusal`]) so the caller keeps using the interpreter.
//!
//! # Errors: fast path, then the interpreter's answer
//!
//! Sky numbers are always finite, and every failure of these operations
//! (overflow, division by zero, a domain error) shows as a non-finite
//! result. So the fast path computes every element without stopping, and
//! collects one bit: "some operation produced a non-finite value". It is
//! collected in the vector registers as `acc += r - r` (NaN if `r` is
//! infinite or NaN, else `+0`), one accumulator per unrolled vector, and
//! looked at once after the loop. Only if it is set does the kernel run the
//! *reference path*: the same algorithm as the interpreter's `fused_op`
//! (elements in index order, remembering the earliest failing operation and
//! running later elements only up to it), on `Value`s, through
//! `sky_interp::arith::arith` and `sky_interp::math::apply`. So the error is
//! the interpreter's, with its message, labels and the spans of the body's
//! operations, and "the first failing element in index order" is defined
//! by that one algorithm and not re-derived in machine code. A failing
//! program pays the interpreter's price once; a program that does not fail
//! (the case that matters) pays for one vector add per checked value.
//!
//! Not every operation needs its own check. A non-finite operand of `add`,
//! `sub`, `mul`, `neg`, `sqrt`, `abs` or the numerator of `div` gives a
//! non-finite result, so if such a value flows to the result, the check of
//! the result covers it. Only a value that reaches the result through a
//! divisor or a call of a math function (which can hide it: `x / inf` is
//! `0`) is checked itself. The yield is always checked. A check, taken by
//! itself, never fires on a value the interpreter would accept (no false
//! alarms, and if one fired the reference path would simply compute the
//! result), and never misses one it would reject (operands are finite by
//! the invariant of `Value`).
//!
//! # Speed
//!
//! * Values that do not depend on an array element (constants, scalar
//!   operands, and operations on only those) are computed once before the
//!   loop, and splat into a vector.
//! * The loop works on `F64X2` (Cranelift's widest vector; both its `aarch64`
//!   and `x86_64` backends lower vector types of at most 128 bits), four
//!   vectors (eight elements) per iteration so independent chains overlap,
//!   and a scalar loop for the rest. `-0`, `NaN` and rounding behave the
//!   same in a lane and in a scalar, so the tail agrees with the body.
//! * Loads and stores are unaligned-tolerant vector accesses: on the CPUs
//!   Cranelift targets they cost the same as aligned ones unless they cross
//!   a cache line, and a `Vec<f64>` is only guaranteed 8-byte alignment.
//! * Calls of math functions are scalar calls, lane by lane.
//!
//! # Unsafe code
//!
//! This is the module that calls the compiled code (the only `unsafe` in the
//! crate besides [`crate::smoke`]); each block says why it is sound.

#![allow(unsafe_code)]

use std::collections::HashMap;
use std::fmt;
use std::sync::OnceLock;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    types, AbiParam, InstBuilder, MemFlags, Type as ClifType, Value as Cv,
};
use cranelift_codegen::isa::OwnedTargetIsa;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use sky_diag::Span;
use sky_interp::arith::{arith, ArithOp};
use sky_interp::{RuntimeError, Value};
use sky_ir::{Function, OpData, OpKind, Type, ValueId};
use sky_units::{Dimension, MathFn};

use crate::kernel_rt;

/// What a kernel is compiled for about one operand of the `fused` operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InputSig {
    /// The operand's dimension (of an array, that of its entries).
    pub dim: Dimension,
    /// An array operand (`true`), or a scalar that broadcasts (`false`).
    pub array: bool,
}

/// Why a `fused` operation was not compiled. Nothing is wrong with it: the
/// caller keeps using the interpreter, which also reports the error a body
/// that is not well typed gives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal(pub String);

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "not compiled: {}", self.0)
    }
}

impl std::error::Error for Refusal {}

fn refuse<T>(why: impl Into<String>) -> Result<T, Refusal> {
    Err(Refusal(why.into()))
}

/// How a kernel is built. The default is the shipping configuration; the
/// rest exist to measure what each part costs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    /// Use 128-bit vectors (`F64X2`). Off: a scalar loop.
    pub simd: bool,
    /// Vectors (or scalars) per iteration of the main loop, at least 1.
    pub unroll: usize,
    /// Collect the non-finite check. **Off is for measuring only**: the
    /// kernel then never reports a failure, so on an overflow it returns
    /// `inf` or `NaN` in the output, which Sky never allows. The interpreter
    /// integration never turns this off.
    pub check_finite: bool,
    /// Collect the check with a fused multiply-add where the host has one
    /// (one instruction instead of two). Off exercises the portable form.
    pub fma_check: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            simd: true,
            unroll: 4,
            check_finite: true,
            fma_check: true,
        }
    }
}

/// One operand's data for [`Kernel::run`].
#[derive(Clone, Copy, Debug)]
pub enum Input<'a> {
    /// The entries of an array operand: at least as many as the output.
    Array(&'a [f64]),
    /// A scalar operand, the same for every element.
    Scalar(f64),
}

// ----- the plan: the body as a checked dataflow graph -----

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bin {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    /// Operand `k` of the `fused` operation.
    Input(usize),
    Const(f64),
    Bin(Bin, usize, usize),
    /// `0 - x`.
    Neg(usize),
    Sqrt(usize),
    Abs(usize),
    /// A call through `kernel_rt::math`; the second operand is `None` for a
    /// function of one argument.
    Math(MathFn, usize, Option<usize>),
}

impl Kind {
    fn operands(&self) -> [Option<usize>; 2] {
        match *self {
            Kind::Input(_) | Kind::Const(_) => [None, None],
            Kind::Bin(_, a, b) => [Some(a), Some(b)],
            Kind::Neg(a) | Kind::Sqrt(a) | Kind::Abs(a) => [Some(a), None],
            Kind::Math(_, a, b) => [Some(a), b],
        }
    }

    /// Which operands, if non-finite, make the result non-finite.
    fn monotone(&self) -> [bool; 2] {
        match self {
            Kind::Bin(Bin::Div, ..) => [true, false],
            Kind::Bin(..) => [true, true],
            Kind::Neg(_) | Kind::Sqrt(_) | Kind::Abs(_) => [true, false],
            Kind::Input(_) | Kind::Const(_) | Kind::Math(..) => [false, false],
        }
    }

    /// Whether the operation can fail (its result can be non-finite).
    fn fallible(&self) -> bool {
        matches!(self, Kind::Bin(..) | Kind::Sqrt(_) | Kind::Math(..))
    }
}

#[derive(Clone, Copy, Debug)]
struct Node {
    kind: Kind,
    /// Does not depend on an array element: computed once, before the loop.
    uniform: bool,
    /// This node's result is checked for being finite.
    check: bool,
}

struct Plan {
    nodes: Vec<Node>,
    yielded: usize,
    out_dim: Dimension,
    /// Uniform nodes a varying node or the store reads, which the loop
    /// needs splat.
    splat: Vec<bool>,
    uses_math: bool,
}

/// The body of the `fused` operation as the reference path runs it.
struct Reference {
    span: Span,
    ops: Vec<(OpData, Dimension)>,
    args: Vec<ValueId>,
    yielded: ValueId,
    values: usize,
    sigs: Vec<InputSig>,
}

fn const_dim(func: &Function, op: &OpData) -> Result<Dimension, Refusal> {
    let [r] = op.results[..] else {
        return refuse("a constant with several results");
    };
    match func.values.get(r.index()).map(|v| &v.ty) {
        Some(Type::Number(n)) if n.shape.rank() == 0 => Ok(n.dim),
        Some(Type::Unknown) => Ok(Dimension::DIMENSIONLESS),
        _ => refuse("a constant that is not a scalar number"),
    }
}

/// Analyses the body: validates it, computes dimensions, and decides what
/// is uniform and what is checked.
fn plan(func: &Function, fused: &OpData, sigs: &[InputSig]) -> Result<(Plan, Reference), Refusal> {
    if fused.kind != OpKind::Fused {
        return refuse("not a `fused` operation");
    }
    let [region] = fused.regions[..] else {
        return refuse("the body of `fused` is not one region");
    };
    let Some(region) = func.regions.get(region.index()) else {
        return refuse("a region that does not exist");
    };
    let [block] = region.blocks[..] else {
        return refuse("the body of `fused` is not one block");
    };
    let Some(block) = func.blocks.get(block.index()) else {
        return refuse("a block that does not exist");
    };
    if block.args.len() != sigs.len() || fused.operands.len() != sigs.len() {
        return refuse("the body's arguments do not match the operands");
    }
    let op_of = |id: sky_ir::OpId| {
        func.ops
            .get(id.index())
            .ok_or_else(|| Refusal("an operation that does not exist".into()))
    };
    let Some((&last, body)) = block.ops.split_last() else {
        return refuse("the body of `fused` is empty");
    };
    let last = op_of(last)?;
    let (OpKind::Yield, &[yielded]) = (&last.kind, &last.operands[..]) else {
        return refuse("the body of `fused` does not end in a `yield` of one value");
    };

    let mut nodes: Vec<Node> = Vec::new();
    let mut dims: Vec<Dimension> = Vec::new();
    let mut node_of: HashMap<ValueId, usize> = HashMap::new();
    for (k, (arg, sig)) in block.args.iter().zip(sigs).enumerate() {
        node_of.insert(*arg, nodes.len());
        nodes.push(Node {
            kind: Kind::Input(k),
            uniform: !sig.array,
            check: false,
        });
        dims.push(sig.dim);
    }
    if !sigs.iter().any(|s| s.array) {
        return refuse("no array operand");
    }
    let mut ops = Vec::with_capacity(body.len());
    let mut uses_math = false;
    for id in body {
        let op = op_of(*id)?;
        let [result] = op.results[..] else {
            return refuse("an operation without exactly one result");
        };
        let operand = |i: usize| -> Result<usize, Refusal> {
            op.operands
                .get(i)
                .and_then(|v| node_of.get(v))
                .copied()
                .ok_or_else(|| Refusal("an operand that is not defined".into()))
        };
        let want = |n: usize| -> Result<(), Refusal> {
            if op.operands.len() == n {
                Ok(())
            } else {
                refuse(format!(
                    "`{}` with the wrong operand count",
                    op.kind.mnemonic()
                ))
            }
        };
        let (kind, dim) = match &op.kind {
            OpKind::ConstFloat { bits } => {
                want(0)?;
                let x = f64::from_bits(*bits);
                if !x.is_finite() {
                    return refuse("a constant that is not finite");
                }
                (Kind::Const(x), const_dim(func, op)?)
            }
            OpKind::ConstRational(q) => {
                want(0)?;
                let x = q.numer() as f64 / q.denom() as f64;
                if !x.is_finite() {
                    return refuse("a constant that is not finite");
                }
                (Kind::Const(x), const_dim(func, op)?)
            }
            k @ (OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div) => {
                want(2)?;
                let (a, b) = (operand(0)?, operand(1)?);
                let dim = match k {
                    OpKind::Add | OpKind::Sub => {
                        if dims[a] != dims[b] {
                            return refuse("operands of different dimensions");
                        }
                        dims[a]
                    }
                    OpKind::Mul => match dims[a].checked_mul(dims[b]) {
                        Ok(d) => d,
                        Err(_) => return refuse("dimension exponent too large"),
                    },
                    _ => match dims[a].checked_div(dims[b]) {
                        Ok(d) => d,
                        Err(_) => return refuse("dimension exponent too large"),
                    },
                };
                let bin = match k {
                    OpKind::Add => Bin::Add,
                    OpKind::Sub => Bin::Sub,
                    OpKind::Mul => Bin::Mul,
                    _ => Bin::Div,
                };
                (Kind::Bin(bin, a, b), dim)
            }
            OpKind::Neg => {
                want(1)?;
                let a = operand(0)?;
                (Kind::Neg(a), dims[a])
            }
            OpKind::Math(f) => {
                want(f.arity())?;
                let a = operand(0)?;
                let b = if f.arity() == 2 {
                    Some(operand(1)?)
                } else {
                    None
                };
                let args: Vec<Dimension> =
                    [Some(a), b].iter().flatten().map(|i| dims[*i]).collect();
                let Ok(dim) = f.result_dim(&args) else {
                    return refuse("a math function whose arguments do not fit");
                };
                let kind = match f {
                    MathFn::Sqrt => Kind::Sqrt(a),
                    MathFn::Abs => Kind::Abs(a),
                    _ => {
                        uses_math = true;
                        Kind::Math(*f, a, b)
                    }
                };
                (kind, dim)
            }
            other => return refuse(format!("`{}` is not elementwise", other.mnemonic())),
        };
        let uniform = kind.operands().iter().flatten().all(|i| nodes[*i].uniform);
        node_of.insert(result, nodes.len());
        nodes.push(Node {
            kind,
            uniform,
            check: false,
        });
        dims.push(dim);
        ops.push((op.clone(), const_dim_or_default(func, op)));
    }
    let Some(&y) = node_of.get(&yielded) else {
        return refuse("the body yields a value it does not define");
    };

    // Which values need their own finiteness check: going backwards, a
    // value that can fail and is not covered by a later check on a value
    // it flows into monotonically is checked itself.
    let mut detected = vec![false; nodes.len()];
    detected[y] = true;
    nodes[y].check = true;
    for i in (0..nodes.len()).rev() {
        if nodes[i].kind.fallible() && !detected[i] {
            nodes[i].check = true;
            detected[i] = true;
        }
        if detected[i] {
            let mono = nodes[i].kind.monotone();
            for (o, m) in nodes[i].kind.operands().iter().zip(mono) {
                if let (Some(o), true) = (o, m) {
                    detected[*o] = true;
                }
            }
        }
    }
    let mut splat = vec![false; nodes.len()];
    splat[y] = nodes[y].uniform;
    for n in nodes.iter().filter(|n| !n.uniform) {
        for o in n.kind.operands().iter().flatten() {
            splat[*o] = nodes[*o].uniform;
        }
    }
    let out_dim = dims[y];
    Ok((
        Plan {
            nodes,
            yielded: y,
            out_dim,
            splat,
            uses_math,
        },
        Reference {
            span: fused.span,
            ops,
            args: block.args.clone(),
            yielded,
            values: func.values.len(),
            sigs: sigs.to_vec(),
        },
    ))
}

/// A constant's dimension for the reference path (others do not use it).
fn const_dim_or_default(func: &Function, op: &OpData) -> Dimension {
    const_dim(func, op).unwrap_or(Dimension::DIMENSIONLESS)
}

// ----- code generation -----

/// The type of the values one loop works on: `F64X2` or `F64`.
#[derive(Clone, Copy)]
struct Mode {
    ty: ClifType,
}

impl Mode {
    fn lanes(self) -> usize {
        self.ty.lane_count() as usize
    }
}

/// Everything the loops read that is computed before them.
struct Pre {
    /// The pointer to each operand's data.
    ptrs: Vec<Cv>,
    /// The scalar value of each uniform node.
    scalar: Vec<Option<Cv>>,
    /// The vector splat of a uniform node the vector loop reads.
    splat: Vec<Option<Cv>>,
    zero_s: Cv,
    zero_v: Cv,
    math: Option<cranelift_codegen::ir::FuncRef>,
}

/// Where an iteration reads and writes: the output pointer and the pointer
/// of each array operand (by operand index).
struct Cursor {
    out: Cv,
    ins: Vec<Option<Cv>>,
}

struct Gen<'p> {
    plan: &'p Plan,
    /// Accumulators per unrolled vector (0: no check in the loop, else 1
    /// or 2, so that two checks do not wait on one another).
    groups: usize,
    /// The host has a fused multiply-add instruction, so a check is one
    /// operation (`acc + r * 0`, which is NaN exactly if `r` is infinite or
    /// NaN, and `acc` otherwise). Only the check uses it, never a value.
    fma: bool,
}

impl Gen<'_> {
    /// The result of node `i`'s operation on `args` (already of type
    /// `mode.ty`), and its finiteness term if it is checked.
    fn op(
        &self,
        b: &mut FunctionBuilder,
        pre: &Pre,
        mode: Mode,
        kind: Kind,
        args: [Option<Cv>; 2],
    ) -> Cv {
        let zero = if mode.lanes() == 1 {
            pre.zero_s
        } else {
            pre.zero_v
        };
        let (x, y) = (args[0].unwrap_or(zero), args[1].unwrap_or(zero));
        match kind {
            Kind::Bin(Bin::Add, ..) => b.ins().fadd(x, y),
            Kind::Bin(Bin::Sub, ..) => b.ins().fsub(x, y),
            Kind::Bin(Bin::Mul, ..) => b.ins().fmul(x, y),
            Kind::Bin(Bin::Div, ..) => b.ins().fdiv(x, y),
            Kind::Neg(_) => b.ins().fsub(zero, x),
            Kind::Sqrt(_) => {
                let r = b.ins().sqrt(x);
                // The interpreter's `+ 0.0` after a math function.
                b.ins().fadd(r, zero)
            }
            Kind::Abs(_) => b.ins().fabs(x),
            Kind::Math(f, ..) => {
                let Some(func) = pre.math else {
                    return zero;
                };
                let id = b
                    .ins()
                    .iconst(types::I32, i64::from(kernel_rt::math_index(f)));
                let mut out: Option<Cv> = None;
                for lane in 0..mode.lanes() {
                    let (lx, ly) = if mode.lanes() == 1 {
                        (x, y)
                    } else {
                        (
                            b.ins().extractlane(x, lane as u8),
                            b.ins().extractlane(y, lane as u8),
                        )
                    };
                    let call = b.ins().call(func, &[id, lx, ly]);
                    let r = b.inst_results(call)[0];
                    out = Some(match out {
                        None if mode.lanes() == 1 => r,
                        None => b.ins().scalar_to_vector(mode.ty, r),
                        Some(v) => b.ins().insertlane(v, r, lane as u8),
                    });
                }
                out.unwrap_or(zero)
            }
            Kind::Input(_) | Kind::Const(_) => zero,
        }
    }

    /// Adds to `acc` something that is NaN if `r` is not finite and `+0`
    /// if it is (see [`Gen::fma`]).
    fn check(&self, b: &mut FunctionBuilder, pre: &Pre, mode: Mode, acc: Cv, r: Cv) -> Cv {
        let zero = if mode.lanes() == 1 {
            pre.zero_s
        } else {
            pre.zero_v
        };
        if self.fma {
            b.ins().fma(r, zero, acc)
        } else {
            let d = b.ins().fsub(r, r);
            b.ins().fadd(acc, d)
        }
    }

    /// One iteration's body for `slots` groups of `mode` values starting at
    /// the cursor (`out`, and for each array operand `k` the pointer
    /// `ins[k]`). `accs` are the finiteness accumulators, `self.groups` per
    /// slot; the updated ones are returned.
    fn iteration(
        &self,
        b: &mut FunctionBuilder,
        pre: &Pre,
        mode: Mode,
        at: &Cursor,
        slots: usize,
        accs: &[Cv],
    ) -> Vec<Cv> {
        let (out, ins) = (at.out, &at.ins);
        let flags = MemFlags::new().with_notrap();
        let mut next_accs = Vec::with_capacity(accs.len());
        for slot in 0..slots {
            let offset = (slot * mode.lanes() * 8) as i32;
            let mut vals: Vec<Option<Cv>> = vec![None; self.plan.nodes.len()];
            let mut acc: Vec<Cv> = accs[slot * self.groups..(slot + 1) * self.groups].to_vec();
            let mut checked = 0;
            for (idx, node) in self.plan.nodes.iter().enumerate() {
                if node.uniform {
                    continue;
                }
                let get = |vals: &[Option<Cv>], o: usize| -> Cv {
                    if self.plan.nodes[o].uniform {
                        if mode.lanes() == 1 {
                            pre.scalar[o].unwrap_or(pre.zero_s)
                        } else {
                            pre.splat[o].unwrap_or(pre.zero_v)
                        }
                    } else {
                        vals[o].unwrap_or(pre.zero_v)
                    }
                };
                let r = match node.kind {
                    Kind::Input(k) => match ins.get(k).copied().flatten() {
                        Some(base) => b.ins().load(mode.ty, flags, base, offset),
                        None => pre.zero_v,
                    },
                    kind => {
                        let ops = kind.operands();
                        let args = [ops[0].map(|o| get(&vals, o)), ops[1].map(|o| get(&vals, o))];
                        self.op(b, pre, mode, kind, args)
                    }
                };
                vals[idx] = Some(r);
                if node.check && self.groups > 0 {
                    let g = checked % self.groups;
                    acc[g] = self.check(b, pre, mode, acc[g], r);
                    checked += 1;
                }
            }
            let y = self.plan.yielded;
            let result = if self.plan.nodes[y].uniform {
                if mode.lanes() == 1 {
                    pre.scalar[y].unwrap_or(pre.zero_s)
                } else {
                    pre.splat[y].unwrap_or(pre.zero_v)
                }
            } else {
                vals[y].unwrap_or(pre.zero_v)
            };
            b.ins().store(flags, result, out, offset);
            next_accs.extend(acc);
        }
        next_accs
    }
}

fn compile_function(
    plan: &Plan,
    opts: Options,
    fma: bool,
    module: &mut JITModule,
) -> Result<cranelift_module::FuncId, String> {
    let mut ctx = module.make_context();
    ctx.func.signature.params = vec![AbiParam::new(types::I64); 3];
    ctx.func.signature.returns = vec![AbiParam::new(types::I64)];
    let math_id = if plan.uses_math {
        let mut sig = module.make_signature();
        sig.params = vec![
            AbiParam::new(types::I32),
            AbiParam::new(types::F64),
            AbiParam::new(types::F64),
        ];
        sig.returns = vec![AbiParam::new(types::F64)];
        Some(
            module
                .declare_function("sky_kernel_math", Linkage::Import, &sig)
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    let math = math_id.map(|id| module.declare_func_in_func(id, &mut ctx.func));

    let varying_checks = plan.nodes.iter().filter(|n| n.check && !n.uniform).count();
    let groups = if opts.check_finite {
        varying_checks.min(2)
    } else {
        0
    };
    let gen = Gen { plan, groups, fma };
    let mut fctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);
        let (n, ins, out) = {
            let p = b.block_params(entry);
            (p[0], p[1], p[2])
        };

        let vmode = Mode {
            ty: if opts.simd { types::F64X2 } else { types::F64 },
        };
        let smode = Mode { ty: types::F64 };
        let flags = MemFlags::trusted();

        // Before the loops: operand pointers, and everything uniform.
        let mut ptrs = Vec::new();
        for (idx, node) in plan.nodes.iter().enumerate() {
            if let Kind::Input(k) = node.kind {
                debug_assert_eq!(k, idx);
                ptrs.push(b.ins().load(types::I64, flags, ins, (8 * k) as i32));
            }
        }
        let zero_s = b.ins().f64const(0.0);
        let zero_v = if vmode.lanes() == 1 {
            zero_s
        } else {
            b.ins().splat(vmode.ty, zero_s)
        };
        let mut pre = Pre {
            ptrs,
            scalar: vec![None; plan.nodes.len()],
            splat: vec![None; plan.nodes.len()],
            zero_s,
            zero_v,
            math,
        };
        let mut pre_terms: Option<Cv> = None;
        for (idx, node) in plan.nodes.iter().enumerate() {
            if !node.uniform {
                continue;
            }
            let r = match node.kind {
                Kind::Input(k) => b.ins().load(types::F64, flags, pre.ptrs[k], 0),
                Kind::Const(x) => b.ins().f64const(x),
                kind => {
                    let ops = kind.operands();
                    let args = [
                        ops[0].and_then(|o| pre.scalar[o]),
                        ops[1].and_then(|o| pre.scalar[o]),
                    ];
                    gen.op(&mut b, &pre, smode, kind, args)
                }
            };
            pre.scalar[idx] = Some(r);
            if plan.splat[idx] {
                pre.splat[idx] = Some(if vmode.lanes() == 1 {
                    r
                } else {
                    b.ins().splat(vmode.ty, r)
                });
            }
            if node.check && opts.check_finite {
                let d = b.ins().fsub(r, r);
                pre_terms = Some(match pre_terms {
                    Some(t) => b.ins().fadd(t, d),
                    None => d,
                });
            }
        }
        let pre_acc = pre_terms.unwrap_or(zero_s);

        // The array operands are read through pointers that advance with the
        // output's (rather than by an index), so that every load and store
        // is a base register plus a constant offset.
        let arrays: Vec<usize> = (0..plan.nodes.len())
            .filter(|&k| matches!(plan.nodes[k].kind, Kind::Input(_)) && !plan.nodes[k].uniform)
            .collect();
        let nbytes = b.ins().ishl_imm(n, 3);
        let out_end = b.ins().iadd(out, nbytes);
        let cursor_of = |params: &[Cv]| -> Cursor {
            let mut ins = vec![None; plan.nodes.len()];
            for (k, p) in arrays.iter().zip(&params[1..]) {
                ins[*k] = Some(*p);
            }
            Cursor {
                out: params[0],
                ins,
            }
        };

        // Main loop: `unroll` vectors per iteration, with `groups`
        // accumulators each.
        let unroll = opts.unroll.max(1);
        let step = (unroll * vmode.lanes() * 8) as i64;
        let head = b.create_block();
        let main = b.create_block();
        let mid = b.create_block();
        for _ in 0..=arrays.len() {
            b.append_block_param(head, types::I64);
        }
        for _ in 0..unroll * groups {
            b.append_block_param(head, vmode.ty);
        }
        let mut first: Vec<Cv> = vec![out];
        first.extend(arrays.iter().map(|k| pre.ptrs[*k]));
        first.extend(std::iter::repeat(zero_v).take(unroll * groups));
        b.ins().jump(head, &block_args(&first));

        b.switch_to_block(head);
        let params: Vec<Cv> = b.block_params(head).to_vec();
        let (cursor, accs) = params.split_at(1 + arrays.len());
        let next_out = b.ins().iadd_imm(cursor[0], step);
        let fits = b
            .ins()
            .icmp(IntCC::UnsignedLessThanOrEqual, next_out, out_end);
        b.ins().brif(fits, main, &[], mid, &[]);

        b.switch_to_block(main);
        b.seal_block(main);
        let new_accs = gen.iteration(&mut b, &pre, vmode, &cursor_of(cursor), unroll, accs);
        let mut again: Vec<Cv> = vec![next_out];
        for p in &cursor[1..] {
            again.push(b.ins().iadd_imm(*p, step));
        }
        again.extend(new_accs);
        b.ins().jump(head, &block_args(&again));
        b.seal_block(head);

        // Between: fold the vector accumulators into one scalar.
        b.switch_to_block(mid);
        b.seal_block(mid);
        let mut total = pre_acc;
        for acc in accs {
            for lane in 0..vmode.lanes() {
                let l = if vmode.lanes() == 1 {
                    *acc
                } else {
                    b.ins().extractlane(*acc, lane as u8)
                };
                total = b.ins().fadd(total, l);
            }
        }

        // Tail: the remaining elements, one at a time.
        let thead = b.create_block();
        let tmain = b.create_block();
        let done = b.create_block();
        let tail_accs = groups.max(1);
        for _ in 0..=arrays.len() {
            b.append_block_param(thead, types::I64);
        }
        for _ in 0..tail_accs {
            b.append_block_param(thead, types::F64);
        }
        let mut tfirst: Vec<Cv> = cursor.to_vec();
        tfirst.push(total);
        tfirst.extend(std::iter::repeat(zero_s).take(tail_accs - 1));
        b.ins().jump(thead, &block_args(&tfirst));
        b.switch_to_block(thead);
        let tparams: Vec<Cv> = b.block_params(thead).to_vec();
        let (tcursor, taccs) = tparams.split_at(1 + arrays.len());
        let more = b.ins().icmp(IntCC::UnsignedLessThan, tcursor[0], out_end);
        b.ins().brif(more, tmain, &[], done, &[]);
        b.switch_to_block(tmain);
        b.seal_block(tmain);
        let new_taccs = gen.iteration(
            &mut b,
            &pre,
            smode,
            &cursor_of(tcursor),
            1,
            &taccs[..groups],
        );
        let mut tagain: Vec<Cv> = Vec::new();
        for p in tcursor {
            tagain.push(b.ins().iadd_imm(*p, 8));
        }
        tagain.extend(new_taccs);
        tagain.extend(&taccs[groups..]);
        b.ins().jump(thead, &block_args(&tagain));
        b.seal_block(thead);

        b.switch_to_block(done);
        b.seal_block(done);
        let mut tacc = taccs[0];
        for t in &taccs[1..] {
            tacc = b.ins().fadd(tacc, *t);
        }
        let bad = b.ins().fcmp(FloatCC::Unordered, tacc, tacc);
        let bad = b.ins().uextend(types::I64, bad);
        b.ins().return_(&[bad]);
        b.finalize();
    }
    let id = module
        .declare_function("sky_kernel", Linkage::Local, &ctx.func.signature)
        .map_err(|e| e.to_string())?;
    // `SKY_JIT_DUMP=clif` prints the kernel's Cranelift IR, `=asm` also its
    // machine code (to stderr), for looking at what a body became.
    let dump = std::env::var("SKY_JIT_DUMP").unwrap_or_default();
    if dump == "clif" || dump == "asm" {
        eprintln!("{}", ctx.func.display());
    }
    ctx.set_disasm(dump == "asm");
    module
        .define_function(id, &mut ctx)
        .map_err(|e| e.to_string())?;
    if dump == "asm" {
        if let Some(code) = ctx.compiled_code() {
            eprintln!("{}", code.vcode.as_deref().unwrap_or(""));
        }
    }
    module.clear_context(&mut ctx);
    Ok(id)
}

fn block_args(values: &[Cv]) -> Vec<cranelift_codegen::ir::BlockArg> {
    values.iter().map(|v| (*v).into()).collect()
}

/// Whether the host has a fused multiply-add instruction (so that
/// `fma` is one instruction and not a library call).
fn has_fma(isa: &OwnedTargetIsa) -> bool {
    isa.triple().architecture.to_string().starts_with("aarch64")
        || isa
            .isa_flags()
            .iter()
            .any(|v| v.name == "has_fma" && v.as_bool() == Some(true))
}

fn isa() -> Result<OwnedTargetIsa, String> {
    static ISA: OnceLock<Result<OwnedTargetIsa, String>> = OnceLock::new();
    ISA.get_or_init(crate::host_isa).clone()
}

// ----- the kernel -----

type KernelFn = unsafe extern "C" fn(usize, *const *const f64, *mut f64) -> u64;

/// A compiled `fused` body. See the module docs.
pub struct Kernel {
    /// Owns the machine code `entry` points into.
    module: Option<JITModule>,
    entry: KernelFn,
    reference: Reference,
    out_dim: Dimension,
    options: Options,
}

impl fmt::Debug for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Kernel")
            .field("inputs", &self.reference.sigs)
            .field("out_dim", &self.out_dim)
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        if let Some(module) = self.module.take() {
            // SAFETY: `entry` points into this module's memory and is not
            // called after this: the kernel is being dropped, and `run`
            // borrows it.
            unsafe { module.free_memory() };
        }
    }
}

/// Compiles the body of `fused` (an `OpKind::Fused` operation of `func`) for
/// operands described by `inputs`, one per operand of the operation.
pub fn compile_kernel(
    func: &Function,
    fused: &OpData,
    inputs: &[InputSig],
) -> Result<Kernel, Refusal> {
    compile_kernel_with(func, fused, inputs, Options::default())
}

/// [`compile_kernel`] with the build [`Options`] chosen.
pub fn compile_kernel_with(
    func: &Function,
    fused: &OpData,
    inputs: &[InputSig],
    options: Options,
) -> Result<Kernel, Refusal> {
    let (plan, reference) = plan(func, fused, inputs)?;
    let out_dim = plan.out_dim;
    let isa = isa().map_err(|e| Refusal(format!("no code generator for this host: {e}")))?;
    let fma = options.fma_check && has_fma(&isa);
    let mut builder = JITBuilder::with_isa(isa, default_libcall_names());
    builder.symbol("sky_kernel_math", kernel_rt::math as *const u8);
    let mut module = JITModule::new(builder);
    let id = match compile_function(&plan, options, fma, &mut module) {
        Ok(id) => id,
        Err(e) => {
            // SAFETY: nothing was called from this module.
            unsafe { module.free_memory() };
            return refuse(format!("code generation failed: {e}"));
        }
    };
    if let Err(e) = module.finalize_definitions() {
        // SAFETY: nothing was called from this module.
        unsafe { module.free_memory() };
        return refuse(format!("code generation failed: {e}"));
    }
    let code = module.get_finalized_function(id);
    // SAFETY: `code` was compiled above as `extern "C"`-compatible
    // `(usize, *const *const f64, *mut f64) -> u64` (three pointer-sized
    // integer parameters and one integer result, in the host's default
    // calling convention, which is the C one), and it stays valid until
    // `Kernel::drop` frees the module.
    let entry: KernelFn = unsafe { std::mem::transmute(code) };
    Ok(Kernel {
        module: Some(module),
        entry,
        reference,
        out_dim,
        options,
    })
}

impl Kernel {
    /// The dimension of the result.
    pub fn output_dim(&self) -> Dimension {
        self.out_dim
    }

    /// What the kernel was compiled for.
    pub fn inputs(&self) -> &[InputSig] {
        &self.reference.sigs
    }

    /// Computes `out.len()` elements: element `i` of `out` is the body on
    /// element `i` of each array input (and the scalar inputs, whole).
    ///
    /// `inputs` match [`Kernel::inputs`] in number and in array/scalar. On
    /// an error the contents of `out` are unspecified, and the error is the
    /// interpreter's for the same elements.
    pub fn run(&self, inputs: &[Input<'_>], out: &mut [f64]) -> Result<(), RuntimeError> {
        if !self.run_fast(inputs, out)? {
            return Ok(());
        }
        let data = self.reference.run(inputs, out.len())?;
        out.copy_from_slice(&data);
        Ok(())
    }

    /// Like [`Kernel::run`], into a new vector of `n` elements, which is not
    /// zeroed first (the loop writes every element).
    pub fn run_alloc(&self, inputs: &[Input<'_>], n: usize) -> Result<Vec<f64>, RuntimeError> {
        let mut out: Vec<f64> = Vec::with_capacity(n);
        // SAFETY: `out` has capacity for `n` elements, so `as_mut_ptr` is
        // valid for `n` writes of `f64`, and nothing else uses it meanwhile.
        let failed = unsafe { self.fast(inputs, n, out.as_mut_ptr())? };
        if failed {
            return self.reference.run(inputs, n);
        }
        // SAFETY: on success the loop wrote all `n` elements, so they are
        // initialized (every bit pattern is a valid `f64`).
        unsafe { out.set_len(n) };
        Ok(out)
    }

    /// The compiled loop alone: `true` if it saw a non-finite value, which
    /// means an element fails (then `out` is not the result), `false` if
    /// `out` is the result.
    fn run_fast(&self, inputs: &[Input<'_>], out: &mut [f64]) -> Result<bool, RuntimeError> {
        // SAFETY: `out` is a live unique slice, valid for `out.len()` writes.
        unsafe { self.fast(inputs, out.len(), out.as_mut_ptr()) }
    }

    /// # Safety
    ///
    /// `out` must be valid for writes of `n` `f64`s, and not overlap an input.
    unsafe fn fast(
        &self,
        inputs: &[Input<'_>],
        n: usize,
        out: *mut f64,
    ) -> Result<bool, RuntimeError> {
        let span = self.reference.span;
        let sigs = &self.reference.sigs;
        if inputs.len() != sigs.len() {
            return Err(RuntimeError::new(
                format!(
                    "malformed IR: a kernel for {} operands was run on {}",
                    sigs.len(),
                    inputs.len()
                ),
                span,
            ));
        }
        // The pointers the compiled code reads; a scalar is read through a
        // pointer to its copy in `scalars`. Both live on the stack for the
        // usual few operands: a call must not allocate (it is a few hundred
        // nanoseconds for a thousand elements).
        const INLINE: usize = 8;
        let k = inputs.len();
        let mut inline_scalars = [0.0f64; INLINE];
        let mut heap_scalars;
        let scalars: &mut [f64] = if k <= INLINE {
            &mut inline_scalars[..k]
        } else {
            heap_scalars = vec![0.0f64; k];
            &mut heap_scalars[..]
        };
        for (slot, input) in scalars.iter_mut().zip(inputs) {
            if let Input::Scalar(x) = input {
                *slot = *x;
            }
        }
        let mut inline_ptrs = [std::ptr::null::<f64>(); INLINE];
        let mut heap_ptrs;
        let ptrs: &mut [*const f64] = if k <= INLINE {
            &mut inline_ptrs[..k]
        } else {
            heap_ptrs = vec![std::ptr::null::<f64>(); k];
            &mut heap_ptrs[..]
        };
        for (((slot, input), sig), scalar) in ptrs.iter_mut().zip(inputs).zip(sigs).zip(&*scalars) {
            match input {
                Input::Array(a) if sig.array && a.len() >= n => *slot = a.as_ptr(),
                Input::Scalar(_) if !sig.array => *slot = scalar,
                _ => {
                    return Err(RuntimeError::new(
                        "malformed IR: a kernel was run on operands of another shape",
                        span,
                    ))
                }
            }
        }
        // SAFETY: the compiled function reads `n` elements through each
        // array pointer (checked above to have at least `n`), one element
        // through each scalar pointer, and writes `n` elements through
        // `out` (the caller's contract). `ptrs` has one pointer per
        // operand, as compiled. The function does not retain any pointer,
        // and the module holding the code is alive (`&self`).
        let failed = unsafe { (self.entry)(n, ptrs.as_ptr(), out) };
        Ok(failed != 0)
    }
}

// ----- the reference path -----

impl Reference {
    /// The interpreter's `fused_op` over the elements (see the module docs),
    /// on `Value`s: the result, or the error of the earliest failing
    /// operation.
    fn run(&self, inputs: &[Input<'_>], n: usize) -> Result<Vec<f64>, RuntimeError> {
        let malformed = |what: &str| RuntimeError::new(format!("malformed IR: {what}"), self.span);
        let mut vals: Vec<Option<Value>> = vec![None; self.values];
        let mut data = Vec::with_capacity(n);
        let mut limit = self.ops.len();
        let mut failure: Option<RuntimeError> = None;
        for i in 0..n {
            for ((arg, input), sig) in self.args.iter().zip(inputs).zip(&self.sigs) {
                let x = match input {
                    Input::Array(a) => a[i],
                    Input::Scalar(x) => *x,
                };
                vals[arg.index()] = Some(Value::number(x, sig.dim));
            }
            let mut failed = false;
            for (k, (op, dim)) in self.ops[..limit].iter().enumerate() {
                if let Err(e) = step(&mut vals, op, *dim, self.span) {
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
            match vals.get(self.yielded.index()) {
                Some(Some(Value::Number { value, .. })) => data.push(*value),
                _ => return Err(malformed("the body of `fused` must yield a number")),
            }
        }
        match failure {
            Some(e) => Err(e),
            None => Ok(data),
        }
    }
}

/// One body operation, as `sky-ir-interp`'s `body_op` runs it.
fn step(
    vals: &mut [Option<Value>],
    op: &OpData,
    dim: Dimension,
    span: Span,
) -> Result<(), RuntimeError> {
    let s = op.span;
    let get = |vals: &[Option<Value>], i: usize| -> Result<Value, RuntimeError> {
        op.operands
            .get(i)
            .and_then(|v| vals.get(v.index()))
            .and_then(Clone::clone)
            .ok_or_else(|| RuntimeError::new("malformed IR: an operand is not defined", span))
    };
    let value = match &op.kind {
        OpKind::ConstFloat { bits } => Value::number(f64::from_bits(*bits), dim),
        OpKind::ConstRational(q) => Value::number(q.numer() as f64 / q.denom() as f64, dim),
        kind @ (OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div) => {
            let arith_op = match kind {
                OpKind::Add => ArithOp::Add,
                OpKind::Sub => ArithOp::Sub,
                OpKind::Mul => ArithOp::Mul,
                _ => ArithOp::Div,
            };
            arith(
                arith_op,
                &get(vals, 0)?,
                &get(vals, 1)?,
                s,
                op.operand_span(0),
                op.operand_span(1),
            )?
        }
        OpKind::Neg => match get(vals, 0)? {
            Value::Number { value, dim } => Value::number(0.0 - value, dim),
            other => sky_interp::arith::sign(true, other, op.operand_span(0))?,
        },
        OpKind::Math(f) => {
            let args: Vec<Value> = (0..op.operands.len())
                .map(|i| get(vals, i))
                .collect::<Result<_, _>>()?;
            let spans: Vec<Span> = (0..args.len()).map(|i| op.operand_span(i)).collect();
            sky_interp::math::apply(*f, &args, s, &spans)?
        }
        other => {
            return Err(RuntimeError::new(
                format!("malformed IR: `{}` in a kernel", other.mnemonic()),
                span,
            ))
        }
    };
    let slot = op
        .results
        .first()
        .and_then(|r| vals.get_mut(r.index()))
        .ok_or_else(|| RuntimeError::new("malformed IR: an operation has no result", span))?;
    *slot = Some(value);
    Ok(())
}

#[cfg(test)]
#[path = "kernel_tests.rs"]
mod tests;
