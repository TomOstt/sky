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
//! * [`contracts`]: removes an `assert`, `requires` or `ensures` whose
//!   condition is provably true, from the constants and the calls in the
//!   module.
//! * [`dce`]: removes pure operations whose results are unused.
//! * [`fuse`]: a tree of elementwise array operations becomes one `fused`
//!   operation (one kernel, no temporary arrays).
//! * [`cholesky`]: `A⁻¹ @ b` with a PositiveDefinite `A` becomes
//!   `solve[cholesky](A, b)`.
//! * [`default_pipeline`]: `fold`, then the e-graph (`sky-egraph`, under the
//!   `Reproducible` contract), then `contracts`, then `dce`, then `fuse`,
//!   then `cholesky`.
//!
//! The rule for all of them: the optimized program behaves exactly like the
//! original under `sky_ir_interp`, including which error is reported where.

#![forbid(unsafe_code)]

mod cholesky;
mod contracts;
mod dce;
mod fold;
mod fuse;
pub mod pipeline;
mod rewrite;
mod util;

#[cfg(test)]
mod contracts_tests;
#[cfg(test)]
mod differential;
#[cfg(test)]
mod exact_tests;
#[cfg(test)]
mod fuse_tests;
#[cfg(test)]
mod roundoff_ref;
#[cfg(test)]
mod roundoff_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod testutil;

pub use cholesky::{cholesky, cholesky_text, CHOLESKY_RULE};
pub use contracts::{contract_headline, contract_rule, contracts};
pub use dce::dce;
pub use fold::{fold, fold_with, FoldMode};
pub use fuse::{fuse, FUSE_RULE};
pub use pipeline::{
    default_passes, default_pipeline, CholeskyPass, ContractsPass, DcePass, FoldPass, FusePass,
    DEFAULT_PASSES,
};
pub use rewrite::Rewrite;
