//! Dimensions as rational exponent vectors over the SI base dimensions, plus unit tables.
//!
//! This crate is pure: it performs no I/O and never evaluates user code. It provides
//!
//! * [`Rational`]: a small exact rational number (checked `i64` arithmetic),
//! * [`Dimension`]: seven rational exponents over L, M, T, I, Θ, N, J,
//! * [`Unit`], [`lookup`], and [`is_unit_symbol`]: the unit table (a dimension
//!   plus a scale factor) and SI prefixes, the single source of truth for unit names,
//! * [`Constant`] and [`CONSTANTS`]: CODATA 2018 physical constants, which the checker
//!   binds as names in scope (they are values, not units),
//! * [`MathFn`]: the built-in math functions and their dimension rules,
//! * [`DimUnifier`]: unification of dimension terms containing dimension variables.
//!
//! Unification only ever compares dimensions; unit scale factors never take part.
//!
//! Every checker-facing operation returns `Result` and reports exponent
//! overflow as [`DimError::ExponentOverflow`] (or [`UnifyError::Overflow`])
//! instead of panicking. The arithmetic operators on [`Rational`] and
//! [`Dimension`] panic on overflow and are meant only for constants and tests.

#![forbid(unsafe_code)]

pub mod constants;
pub mod dimension;
pub mod math;
pub mod rational;
pub mod unify;
pub mod unit;

pub use constants::{lookup_constant, Constant, CONSTANTS};
pub use dimension::{BaseDim, DimError, Dimension};
pub use math::{DimRule, MathDimError, MathFn};
pub use rational::{Rational, RationalError};
pub use unify::{DimMismatch, DimTerm, DimUnifier, DimVar, UnifyError};
pub use unit::{is_unit_symbol, lookup, suggest_unit, Prefix, Resolved, Unit, PREFIXES, UNITS};
