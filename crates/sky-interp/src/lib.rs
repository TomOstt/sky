//! Tree-walking interpreter for Sky v0.1.
//!
//! [`run_module`] executes a module that the checker accepted. A checked
//! program should never hit a type error here, but the interpreter still
//! re-checks every operand dynamically and reports a [`RuntimeError`]
//! (never a panic), because the checker types some things as unknown
//! (strings, calls to names it has no rule for) and lets them through.
//!
//! # Values
//!
//! A [`Value`] is a number with a dimension (`Value::Number`), an array
//! (`Value::Array`), a Bool, or a string. A unit literal is converted to SI
//! once, at the literal (spec "Unit values"), so a number is a plain `f64`
//! plus the [`Dimension`] it is measured in; no value remembers its unit. An
//! array is a vector or matrix of such numbers sharing one dimension, stored
//! row-major; a [`Value::Number`] is the rank-0 case. The rules for arrays
//! (elementwise operations, `@`, the inverse, the Cholesky solve, indexing)
//! are in [`array`], on the dense numerics in [`linalg`].
//!
//! # Rewrites
//!
//! The checker decides which `A⁻¹ @ b` become Cholesky solves and records
//! their spans (`sky_types::Checked::cholesky_solves`); [`run_module_with`]
//! takes that set and honours it. [`run_module`] runs without one.
//!
//! # Arithmetic
//!
//! Every value the interpreter holds is finite. Division by zero, overflow
//! (a finite operation whose result does not fit an `f64`), and operations
//! with no real result (`(-8)^(1/3)`) are runtime errors on the failing
//! expression instead of `inf` or `NaN`. See the spec's "Running programs".
//!
//! # Depth
//!
//! Evaluation recurses once per AST level and the parser caps nesting, so
//! callers run [`run_module`] under `sky_ast::with_deep_stack`, like the
//! checker.

#![forbid(unsafe_code)]

pub mod arith;
pub mod array;
mod env;
mod error;
mod eval;
mod literal;
pub mod math;
mod value;

#[cfg(test)]
mod array_tests;
#[cfg(test)]
mod tests;

pub use arith::MAX_CALL_DEPTH;
pub use env::RuntimeEnv;
pub use error::{ErrorDetails, RuntimeError};
pub use eval::{run_module, run_module_with};
pub use literal::{number_literal, unit_literal_si};
/// The dense numerics, shared with the checker (see `sky_types::linalg`).
pub use sky_types::linalg;
pub use sky_units::Dimension;
pub use value::{format_number, si_unit, Value};
