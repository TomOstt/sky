//! Optimization passes over SIR (`docs/design/ir.md`, "Passes").
//!
//! Each pass is a function `fn(&mut Module) -> Vec<Rewrite>`. The pass
//! manager and `Explanation` type do not exist on this branch yet; a
//! [`Rewrite`] has the fields of the design's `Explanation` (minus
//! `effect_on_semantics`, which is `Unchanged` for every pass here), so the
//! adapter at merge time is a field-for-field copy.
//!
//! * [`fold`]: constant folding, exact on rationals, bit-identical to the
//!   reference interpreter on floats.
//! * [`dce`]: removes pure operations whose results are unused.
//! * [`cholesky`]: `A⁻¹ @ b` with a PositiveDefinite `A` becomes
//!   `solve[cholesky](A, b)`.
//!
//! The rule for all of them: the optimized program behaves exactly like the
//! original under `sky_ir_interp`, including which error is reported where.

#![forbid(unsafe_code)]

mod cholesky;
mod dce;
mod fold;
mod rewrite;
mod util;

#[cfg(test)]
mod differential;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod testutil;

pub use cholesky::cholesky;
pub use dce::dce;
pub use fold::{fold, fold_with, FoldMode};
pub use rewrite::Rewrite;
