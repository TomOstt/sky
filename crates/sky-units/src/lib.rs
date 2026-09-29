//! Dimensions as rational exponent vectors over the SI base dimensions, plus unit tables.
//!
//! This crate is pure: it performs no I/O and never evaluates user code. It provides
//!
//! * [`Rational`]: a small exact rational number (checked `i64` arithmetic),
//! * [`Dimension`]: seven rational exponents over L, M, T, I, Θ, N, J,
//! * [`Unit`], [`lookup`], and [`is_unit_symbol`]: the v0.1 unit table (a dimension
//!   plus a scale factor), the single source of truth shared by the parser and checker,
//! * [`DimUnifier`]: unification of dimension terms containing dimension variables.
//!
//! Unification only ever compares dimensions; unit scale factors never take part.
//!
//! Every checker-facing operation returns `Result` and reports exponent
//! overflow as [`DimError::ExponentOverflow`] (or [`UnifyError::Overflow`])
//! instead of panicking. The arithmetic operators on [`Rational`] and
//! [`Dimension`] panic on overflow and are meant only for constants and tests.

#![forbid(unsafe_code)]

pub mod dimension;
pub mod rational;
pub mod unify;
pub mod unit;

pub use dimension::{BaseDim, DimError, Dimension};
pub use rational::{Rational, RationalError};
pub use unify::{DimMismatch, DimTerm, DimUnifier, DimVar, UnifyError};
pub use unit::{is_unit_symbol, lookup, Unit, UNITS};
