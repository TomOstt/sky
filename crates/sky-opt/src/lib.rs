//! Optimization passes over SIR (`docs/design/ir.md`, "Passes") and the
//! default pipeline.
//!
//! Each pass has a function form, `fn(&mut Module) -> Vec<Rewrite>`, and a
//! [`sky_ir::Pass`] form in [`pipeline`]. A [`Rewrite`] has the fields of
//! `sky_ir::Explanation` (minus `effect_on_semantics`, which is `Unchanged`
//! for every pass here); [`Rewrite::explain`] is the one place that turns it
//! into an explanation, so rule ids and texts survive unchanged.
//!
//! * [`fold`]: constant folding, exact on rationals, bit-identical to the
//!   reference interpreter on floats.
//! * [`dce`]: removes pure operations whose results are unused.
//! * [`cholesky`]: `A⁻¹ @ b` with a PositiveDefinite `A` becomes
//!   `solve[cholesky](A, b)`.
//! * [`default_pipeline`]: `fold`, then the e-graph (`sky-egraph`, under the
//!   `Reproducible` contract), then `dce`, then `cholesky`.
//!
//! The rule for all of them: the optimized program behaves exactly like the
//! original under `sky_ir_interp`, including which error is reported where.

#![forbid(unsafe_code)]

mod cholesky;
mod dce;
mod fold;
pub mod pipeline;
mod rewrite;
mod util;

#[cfg(test)]
mod differential;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod testutil;

pub use cholesky::{cholesky, cholesky_text, CHOLESKY_RULE};
pub use dce::dce;
pub use fold::{fold, fold_with, FoldMode};
pub use pipeline::{
    default_passes, default_pipeline, CholeskyPass, DcePass, FoldPass, DEFAULT_PASSES,
};
pub use rewrite::Rewrite;
