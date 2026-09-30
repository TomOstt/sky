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
//! * [`powers`]: `x^n` for a constant integer `2 <= n <= 14` becomes the
//!   multiplications the interpreter's `powi` performs, bit for bit.
//! * [`transposes`], [`structure`] (module `linalg`): transposes that are
//!   not materialized (`matmul[ta]`), Gram products, Diagonal scaling and
//!   Symmetric transposes. All exact.
//! * [`algorithms`]: records the algorithm class a policy chooses for each
//!   `matmul`; the hook the kernel and tuning agents plug into.
//! * [`specialize`]: a call whose arguments have facts the callee's signature
//!   lacks (a concrete shape for `Vector[N]`, a constant, a proven sign) gets
//!   a clone of the callee that knows them, simplified (`len` of a known
//!   shape, decided branches, small constant loops unrolled). [`specialize_for`]
//!   and [`SpecializationCache`] do the same for values an engine sees at
//!   run time.
//! * [`default_pipeline`]: `specialize`, then `fold`, then the e-graph (`sky-egraph`, under the
//!   `Reproducible` contract), then `contracts`, then `dce`, then `powers`,
//!   then `fuse`, then `cholesky`, then `transposes`, `structure` and
//!   `algorithms`.
//!
//! The algorithm-choosing rewrites that change rounding (the order of an
//! `@` chain, a solve instead of an inverse) are made by the checker
//! (`sky_types::chain`), not here, so that the tree-walker follows the same
//! choice. Everything in this crate is exact.
//!
//! The rule for all of them: the optimized program behaves exactly like the
//! original under `sky_ir_interp`, including which error is reported where.

#![forbid(unsafe_code)]

pub mod algorithm;
mod cholesky;
mod contracts;
mod dce;
mod fold;
mod fuse;
mod linalg;
pub mod pipeline;
pub mod powers;
mod rewrite;
pub mod specialize;
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

pub use algorithm::{
    algorithms, default_policy, MatMulPolicy, MatMulShape, ShapeClass, ALGORITHM_RULE,
};
pub use cholesky::{cholesky, cholesky_text, CHOLESKY_RULE};
pub use contracts::{contract_headline, contract_rule, contracts};
pub use dce::dce;
pub use fold::{fold, fold_with, FoldMode};
pub use fuse::{fuse, FUSE_RULE};
pub use linalg::{structure, transposes, DIAGONAL_RULE, GRAM_RULE, SYMMETRIC_RULE, TRANSPOSE_RULE};
pub use pipeline::{
    default_passes, default_passes_without_specialize, default_pipeline, AlgorithmPass,
    CholeskyPass, ContractsPass, DcePass, FoldPass, FusePass, PowersPass, SpecializePass,
    StructurePass, TransposesPass, DEFAULT_PASSES,
};
pub use powers::{powers, POWER_RULE};
pub use rewrite::Rewrite;
pub use specialize::{
    specialize, specialize_for, specialize_headline, specialized_name, Specialization,
    SpecializationCache,
};
