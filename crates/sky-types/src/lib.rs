//! Bidirectional type checker: units, symbolic shapes, and matrix-property tags. Never executes user code.
//!
//! This crate is pure: no I/O, no evaluation of user code, and no panics on
//! checker-facing paths (every fallible operation returns `Result`). The parts
//! here do not depend on the AST yet:
//!
//! * [`shape`]: symbolic shapes ([`Shape`], [`ShapeDim`]) and equality-only
//!   unification ([`ShapeUnifier`]). There is no dimension arithmetic such as
//!   `N+1` in v0.1.
//! * [`tags`]: matrix-property tags ([`Tag`], [`TagSet`]) and their subtyping lattice.
//! * [`rules`]: typing rules for `@`, elementwise operators, transpose, and inverse
//!   over [`TensorType`] (a shape plus tags).
//! * [`rewrite`]: the one v0.1 rewrite, `A⁻¹ @ b` to a Cholesky solve for a
//!   positive-definite `A`.

#![forbid(unsafe_code)]

pub mod rewrite;
pub mod rules;
pub mod shape;
pub mod tags;

pub use rewrite::{cholesky_rewrite, MatmulLhs, Rewrite};
pub use rules::{elementwise, inverse, matmul, transpose, TensorType};
pub use shape::{
    DimConflict, DimValue, Shape, ShapeDim, ShapeError, ShapeErrorKind, ShapeUnifier, ShapeVar,
};
pub use tags::{matmul_tags, preserved_by_inverse, preserved_by_transpose, Tag, TagSet};
