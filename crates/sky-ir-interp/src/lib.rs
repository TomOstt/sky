//! The reference interpreter for SIR: the executable specification every
//! optimization and backend is tested against (`docs/design/ir.md`,
//! "Reference semantics and testing").
//!
//! It is deliberately simple: one value table per function activation, ops
//! evaluated in order, regions run by recursion. It takes no shortcuts, so
//! when a pass or a backend disagrees with it, the pass or backend is wrong.
//!
//! # Same behaviour as `sky run`
//!
//! A SIR program must behave exactly like the tree-walking interpreter:
//! same output bytes, same errors. Values are [`sky_interp::Value`] and
//! print through its `Display`/`print_text`, and every arithmetic rule
//! (finite checks, `powi`/`powf`, reciprocal, comparisons, limits) is the
//! function in [`sky_interp::arith`] that `sky run` itself calls, so the two
//! cannot drift.
//!
//! # Malformed IR
//!
//! The verifier is a separate component and this interpreter must not rely
//! on it: IR it would reject (a missing value, a wrong operand count, a
//! missing region, a block without a terminator) is a
//! `malformed IR: ...` [`IrRuntimeError`], never a panic.
//!
//! # Limits
//!
//! Nested calls are capped at [`sky_interp::MAX_CALL_DEPTH`] and Rust stack
//! use is measured against [`sky_interp::arith::STACK_BUDGET`], both with
//! the same errors as `sky run`. Loops have no step limit (an infinite loop
//! runs forever, as in Python). Callers must run [`run`] and [`call`] under
//! `sky_ast::with_deep_stack`, because the stack budget assumes its stack.

#![forbid(unsafe_code)]

mod machine;
mod math;

#[cfg(test)]
mod tests;

use std::fmt;
use std::io::Write;
use std::ops::Deref;

use sky_diag::{Diagnostic, Span};
use sky_interp::RuntimeError;
pub use sky_interp::Value;
use sky_ir::Module;

/// A failure while running SIR, located at the failing operation.
///
/// It wraps [`sky_interp::RuntimeError`], so a failure that is the same as
/// one of `sky run`'s has the same message; `err.message` and `err.span`
/// work through `Deref`.
#[derive(Clone, Debug, PartialEq)]
pub struct IrRuntimeError(RuntimeError);

impl IrRuntimeError {
    /// IR that breaks a rule the verifier would have caught.
    pub(crate) fn malformed(what: impl fmt::Display, span: Span) -> Self {
        IrRuntimeError(RuntimeError::new(format!("malformed IR: {what}"), span))
    }

    /// The error as a diagnostic, worded like `sky run`'s.
    pub fn to_diagnostic(&self) -> Diagnostic {
        self.0.to_diagnostic()
    }
}

impl From<RuntimeError> for IrRuntimeError {
    fn from(e: RuntimeError) -> Self {
        IrRuntimeError(e)
    }
}

impl From<IrRuntimeError> for Diagnostic {
    fn from(e: IrRuntimeError) -> Self {
        e.to_diagnostic()
    }
}

impl Deref for IrRuntimeError {
    type Target = RuntimeError;

    fn deref(&self) -> &RuntimeError {
        &self.0
    }
}

impl fmt::Display for IrRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for IrRuntimeError {}

/// Runs the module's [`Module::MAIN`] function, writing `print` output to
/// `out`. Run this under `sky_ast::with_deep_stack`.
///
/// Lowering turns a file's value-less declarations (`A: Matrix[3, 3]`, an
/// input with no value) into parameters of `<main>`. Nothing can supply
/// them in v0.1, so a `<main>` with parameters fails with the same error
/// `sky run` gives on the same file.
pub fn run(module: &Module, out: &mut impl Write) -> Result<(), IrRuntimeError> {
    if let Some(main) = module.function(Module::MAIN) {
        if let Some(&first) = main.param_values().first() {
            let name = main.value(first).name.as_deref().unwrap_or("an input");
            return Err(sky_interp::arith::declared_without_value(name, main.span).into());
        }
    }
    machine::call(module, Module::MAIN, Vec::new(), out).map(|_| ())
}

/// Calls the module function `name` with `args` and returns its results.
/// For tests; run this under `sky_ast::with_deep_stack`.
pub fn call(
    module: &Module,
    name: &str,
    args: Vec<Value>,
    out: &mut impl Write,
) -> Result<Vec<Value>, IrRuntimeError> {
    machine::call(module, name, args, out)
}
