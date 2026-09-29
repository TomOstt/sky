//! Bidirectional type checker: units, symbolic shapes, and matrix-property tags. Never executes user code.
//!
//! This crate is pure: no I/O, no evaluation of user code, and no panics on
//! checker-facing paths (every fallible operation returns `Result`).
//!
//! * [`check`]: the walk over the parsed AST ([`check_module`],
//!   [`check_module_with`]), producing `sky-diag` diagnostics and the
//!   rewrite decisions ([`RewriteSite`]). Everything below is AST-free.
//! * [`shape`]: symbolic shapes ([`Shape`], [`ShapeDim`]) and equality-only
//!   unification ([`ShapeUnifier`]). There is no dimension arithmetic such as
//!   `N+1` in v0.1.
//! * [`tags`]: matrix-property tags ([`Tag`], [`TagSet`]) and their subtyping lattice.
//! * [`rules`]: typing rules for `@`, elementwise operators, transpose, and inverse
//!   over [`TensorType`] (a shape plus tags).
//! * [`rewrite`]: the one v0.1 rewrite, `A⁻¹ @ b` to a Cholesky solve for a
//!   positive-definite `A`.
//! * [`scale`]: tag rules for scaling by a scalar and for negation.
//! * [`tag_ops`]: the single source of truth for tag propagation
//!   ([`TagOp`], [`tag_result`]) and for when a Symmetric result must be
//!   mirrored ([`needs_mirroring`]).

#![forbid(unsafe_code)]

pub mod check;
mod quote;
pub mod rewrite;
pub mod rules;
pub mod scale;
pub mod shape;
pub mod tag_ops;
pub mod tags;

pub use check::{
    check_module, check_module_with, Checked, Env, RewriteSite, Ty, ValueTy, MAX_STORED_REWRITES,
};
pub use quote::QUOTE_MAX_CHARS;
pub use rewrite::{cholesky_rewrite, MatmulLhs, Rewrite};
pub use rules::{elementwise, inverse, matmul, transpose, TensorType};
pub use scale::{negated_tags, scaled_tags};
pub use shape::{
    DimConflict, DimValue, Shape, ShapeDim, ShapeError, ShapeErrorKind, ShapeUnifier, ShapeVar,
};
pub use tag_ops::{needs_mirroring, tag_result, TagOp};
pub use tags::{matmul_tags, preserved_by_inverse, preserved_by_transpose, Tag, TagSet};
