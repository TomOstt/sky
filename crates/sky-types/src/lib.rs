//! Bidirectional type checker: units, symbolic shapes, and matrix-property tags. Never executes user code.
//!
//! This crate is pure: no I/O, no evaluation of user code, and no panics on
//! checker-facing paths (every fallible operation returns `Result`).
//!
//! * [`check`]: the walk over the parsed AST ([`check_module`],
//!   [`check_module_with`]), producing `sky-diag` diagnostics and the
//!   rewrite decisions ([`RewriteSite`]). Everything below is AST-free.
//!   Blocks, control flow, and functions (scoping, merges, calls to `def`s)
//!   are in the private `flow` module, documented at its top.
//! * `provenance`: why a name has its dimension, for the derivation notes on
//!   dimension-mismatch errors (private; see its module docs for the bounds).
//! * [`linalg`]: dense linear algebra on `f64` slices (inverse, Cholesky),
//!   shared with the interpreter.
//! * [`numerics`] and `cancellation`: warnings that teach numerical
//!   analysis (condition numbers of constant matrices, catastrophic
//!   cancellation), and the estimator behind the first.
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

mod array;
mod builtins;
mod cancellation;
pub mod check;
pub mod constant;
mod flow;
pub mod linalg;
pub mod numerics;
mod provenance;
pub mod rewrite;
pub mod rules;
pub mod scale;
pub mod shape;
mod shape_diag;
pub mod tag_ops;
pub mod tags;

pub use check::{
    check_module, check_module_with, const_exponent, Checked, Env, RewriteSite, Ty, ValueTy,
    MAX_STORED_REWRITES,
};
pub use constant::{
    apply_unit, entry_value, parse_number, NumberError, UnitError, MAX_ARRAY_ELEMENTS,
};
pub use flow::FuncSig;
pub use rewrite::{cholesky_rewrite, MatmulLhs, Rewrite};
pub use rules::{elementwise, inverse, matmul, transpose, TensorType};
pub use scale::{negated_tags, scaled_tags};
pub use shape::{
    Binding, DimConflict, DimValue, Origin, Shape, ShapeContext, ShapeDim, ShapeError,
    ShapeErrorKind, ShapeUnifier, ShapeVar, Transpose, MAX_BINDINGS_PER_VAR,
};
/// Most characters of user source text quoted in one place (see
/// [`sky_diag::display_capped`]).
pub use sky_diag::DISPLAY_CAP_CHARS as QUOTE_MAX_CHARS;
pub use tag_ops::{needs_mirroring, tag_result, TagOp};
pub use tags::{matmul_tags, preserved_by_inverse, preserved_by_transpose, Tag, TagSet};
