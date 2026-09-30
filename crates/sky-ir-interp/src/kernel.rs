//! Scalar kernels for `fused` bodies.
//!
//! A `fused` body is scalar code run once per element. Interpreting it as
//! ordinary SIR (a value table of `Value`s, an `arith` call with its
//! dimension checks per operation) costs far more than the arithmetic
//! itself. When the body is nothing but `add`, `sub`, `mul`, `div`, `neg`
//! and constants, the per-element work is a few `f64` operations on
//! registers, and this module compiles the body to exactly that.
//!
//! The kernel changes nothing about what the program means. It is used only
//! for the elements *after* the first, and only if the first, which runs the
//! ordinary way, succeeded: that element checks the dimensions and finds the
//! result's, which do not depend on the element, so what is left per element
//! is the value, the overflow check and the zero check. Each failure is built
//! by the code the ordinary path calls (`sky_interp::arith::overflow`,
//! `div_zero`) with the same spans, and the interpreter's rule for which
//! failure is reported (the earliest operation in body order) is applied to
//! the operation index [`ScalarKernel::run`] returns.

use sky_diag::Span;
use sky_interp::arith::{div_zero, overflow};
use sky_ir::{BlockData, Function, OpId, OpKind, ValueId};

use crate::IrRuntimeError;

/// Most operands a kernel takes (the interpreter keeps their sources on the
/// stack).
pub(crate) const MAX_ARGS: usize = 8;

enum Kind {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Const(f64),
}

struct Step {
    kind: Kind,
    /// Registers of the operands (`b` is unused by `neg` and constants).
    a: usize,
    b: usize,
    /// The register that holds the result.
    dst: usize,
    /// The operation's span, and its second operand's (the divisor).
    span: Span,
    rspan: Span,
}

/// A compiled `fused` body: registers `0..nargs` hold the element's operands,
/// each step defines the next register.
pub(crate) struct ScalarKernel {
    pub(crate) nargs: usize,
    pub(crate) nregs: usize,
    pub(crate) yield_reg: usize,
    steps: Vec<Step>,
}

impl ScalarKernel {
    /// The kernel for a body, or `None` if it has anything but the operations
    /// above (or more than [`MAX_ARGS`] operands, or anything malformed): the
    /// ordinary path then runs it.
    pub(crate) fn compile(
        func: &Function,
        block: &BlockData,
        body: &[OpId],
        yielded: ValueId,
    ) -> Option<ScalarKernel> {
        let nargs = block.args.len();
        if nargs > MAX_ARGS {
            return None;
        }
        // The value in each register.
        let mut regs: Vec<ValueId> = block.args.clone();
        let reg = |regs: &[ValueId], v: ValueId| regs.iter().position(|&w| w == v);
        let mut steps = Vec::with_capacity(body.len());
        for id in body {
            let op = func.ops.get(id.index())?;
            let [result] = op.results[..] else {
                return None;
            };
            let dst = regs.len();
            let (kind, a, b) = match (&op.kind, &op.operands[..]) {
                (OpKind::Add, &[x, y]) => (Kind::Add, reg(&regs, x)?, reg(&regs, y)?),
                (OpKind::Sub, &[x, y]) => (Kind::Sub, reg(&regs, x)?, reg(&regs, y)?),
                (OpKind::Mul, &[x, y]) => (Kind::Mul, reg(&regs, x)?, reg(&regs, y)?),
                (OpKind::Div, &[x, y]) => (Kind::Div, reg(&regs, x)?, reg(&regs, y)?),
                (OpKind::Neg, &[x]) => {
                    let a = reg(&regs, x)?;
                    (Kind::Neg, a, a)
                }
                (OpKind::ConstFloat { bits }, []) => {
                    let c = f64::from_bits(*bits);
                    if !c.is_finite() {
                        return None;
                    }
                    (Kind::Const(c), 0, 0)
                }
                (OpKind::ConstRational(q), []) => {
                    let c = q.numer() as f64 / q.denom() as f64;
                    if !c.is_finite() {
                        return None;
                    }
                    (Kind::Const(c), 0, 0)
                }
                _ => return None,
            };
            steps.push(Step {
                kind,
                a,
                b,
                dst,
                span: op.span,
                rspan: op.operand_span(1),
            });
            regs.push(result);
        }
        let yield_reg = reg(&regs, yielded)?;
        Some(ScalarKernel {
            nargs,
            nregs: regs.len(),
            yield_reg,
            steps,
        })
    }

    /// Runs the first `limit` steps on `regs` (at least [`ScalarKernel::nregs`]
    /// long, with the operands in the first [`ScalarKernel::nargs`]). On a failure:
    /// the index of the step and the error.
    pub(crate) fn run(
        &self,
        regs: &mut [f64],
        limit: usize,
    ) -> Result<(), (usize, IrRuntimeError)> {
        for (k, s) in self.steps[..limit.min(self.steps.len())].iter().enumerate() {
            let (a, b) = (regs[s.a], regs[s.b]);
            let x = match s.kind {
                Kind::Add => a + b,
                Kind::Sub => a - b,
                Kind::Mul => a * b,
                Kind::Div => {
                    if b == 0.0 {
                        return Err((k, div_zero(s.span, s.rspan).into()));
                    }
                    a / b
                }
                // As on arrays: `0 - x`, so a zero stays `+0`.
                Kind::Neg => 0.0 - a,
                Kind::Const(c) => c,
            };
            if !x.is_finite() {
                return Err((k, overflow(s.span).into()));
            }
            regs[s.dst] = x;
        }
        Ok(())
    }
}
