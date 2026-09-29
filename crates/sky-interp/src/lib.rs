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
//! A [`Value`] is a number with a dimension (`Value::Number`), a Bool, or a
//! string. A unit literal is converted to SI once, at the literal (spec
//! "Unit values"), so a number is a plain `f64` plus the [`Dimension`] it
//! is measured in; no value remembers its unit. Arrays will arrive as an
//! element dimension plus a shape (the checker's `ValueTy` shape); a
//! [`Value::Number`] is then the rank-0 case. Until then an expression that
//! would need one (indexing, `@`) is a runtime error, not a panic.
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
mod env;
mod error;
mod eval;
pub mod math;
mod value;

#[cfg(test)]
mod tests;

pub use arith::MAX_CALL_DEPTH;
pub use env::RuntimeEnv;
pub use error::{ErrorDetails, RuntimeError};
pub use eval::run_module;
pub use sky_units::Dimension;
pub use value::{format_number, si_unit, Value};
