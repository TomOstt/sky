//! The passes as [`sky_ir::Pass`]es, and the default pipeline.
//!
//! [`fold_with`], [`dce`] and [`cholesky`] keep their function form (the
//! tests and other tools call them directly and inspect the [`Rewrite`]
//! records); the pass types here run them and report every record through
//! [`Rewrite::explain`], the one conversion into `sky_ir::Explanation`, so
//! rule ids and texts reach `sky check --explain` unchanged.
//!
//! # The default pipeline
//!
//! [`default_pipeline`] runs, in this order (`sky check` and the tests use
//! it):
//!
//! 1. `fold`: constant folding in [`FoldMode::BitIdentical`], so no printed
//!    digit changes;
//! 2. `egraph`: e-graph rewriting under the `Reproducible` contract, which
//!    enables only exact and conditional rules (`@fast` rules are never on
//!    by default);
//! 3. `contracts`: removes an `assert`, `requires` or `ensures` that the
//!    exactness analysis proves true (from constants, and from the
//!    arguments of every call of the function), and says so. It runs after
//!    the e-graph, so it sees the simplified conditions, and before `dce`,
//!    which then removes the comparison the check leaves unused;
//! 4. `dce`: removes dead pure operations, and says so when named
//!    (user-written) values disappear. It runs after the e-graph, which
//!    leaves the operations it replaced behind;
//! 5. `fuse`: trees of elementwise array operations become `fused` kernels.
//!    It runs after `dce`, because a dead operation left by the e-graph
//!    would count as a second use of its operands and keep a temporary
//!    alive, and after the e-graph, because a `fused` body is opaque to it
//!    (fusing first would hide the arithmetic from the rewrite rules). It
//!    runs before `cholesky` only by convention: they touch different
//!    operations, and `cholesky`'s result must stay last so it is never
//!    rewritten.
//! 6. `cholesky`: `A⁻¹ @ b` with a PositiveDefinite `A` becomes
//!    `solve[cholesky](A, b)`. It runs last so the earlier passes see the
//!    program as the user wrote it, and its result is never rewritten.
//!
//! The manager verifies the module before the first pass and after every
//! pass ([`default_pipeline`] leaves that at the manager's default, on in
//! debug builds; callers turn it on with [`PassManager::verify_each`]).

use sky_egraph::EgraphPass;
use sky_ir::{Changed, Module, Pass, PassCx, PassManager};

use crate::rewrite::Rewrite;
use crate::{cholesky, contracts, dce, fold_with, fuse, FoldMode};

/// The pass names of [`default_pipeline`], in order: what
/// `sky check --emit-ir=after:<name>` accepts.
pub const DEFAULT_PASSES: [&str; 6] = ["fold", "egraph", "contracts", "dce", "fuse", "cholesky"];

/// Runs `f`, explains its records, and says whether the module changed.
/// (A pass can change a module without a record: dropping an unnamed
/// temporary is not news, so the records cannot tell.)
fn run_reporting(
    m: &mut Module,
    cx: &mut PassCx,
    f: impl FnOnce(&mut Module) -> Vec<Rewrite>,
) -> Changed {
    let before = m.clone();
    for rw in f(m) {
        rw.explain(cx);
    }
    Changed::from(*m != before)
}

/// Constant folding ([`fold_with`]) as a pass named `"fold"`.
#[derive(Clone, Copy, Default, Debug)]
pub struct FoldPass {
    pub mode: FoldMode,
}

impl Pass for FoldPass {
    fn name(&self) -> &'static str {
        "fold"
    }

    fn run(&self, m: &mut Module, cx: &mut PassCx) -> Changed {
        run_reporting(m, cx, |m| fold_with(m, self.mode))
    }
}

/// Static discharge of contracts ([`contracts`]) as a pass named
/// `"contracts"`.
#[derive(Clone, Copy, Default, Debug)]
pub struct ContractsPass;

impl Pass for ContractsPass {
    fn name(&self) -> &'static str {
        "contracts"
    }

    fn run(&self, m: &mut Module, cx: &mut PassCx) -> Changed {
        run_reporting(m, cx, contracts)
    }
}

/// Dead-code elimination ([`dce`]) as a pass named `"dce"`.
#[derive(Clone, Copy, Default, Debug)]
pub struct DcePass;

impl Pass for DcePass {
    fn name(&self) -> &'static str {
        "dce"
    }

    fn run(&self, m: &mut Module, cx: &mut PassCx) -> Changed {
        run_reporting(m, cx, dce)
    }
}

/// Kernel fusion ([`fuse`]) as a pass named `"fuse"`.
#[derive(Clone, Copy, Default, Debug)]
pub struct FusePass;

impl Pass for FusePass {
    fn name(&self) -> &'static str {
        "fuse"
    }

    fn run(&self, m: &mut Module, cx: &mut PassCx) -> Changed {
        run_reporting(m, cx, fuse)
    }
}

/// The Cholesky solve selection ([`cholesky`]) as a pass named
/// `"cholesky"`.
#[derive(Clone, Copy, Default, Debug)]
pub struct CholeskyPass;

impl Pass for CholeskyPass {
    fn name(&self) -> &'static str {
        "cholesky"
    }

    fn run(&self, m: &mut Module, cx: &mut PassCx) -> Changed {
        run_reporting(m, cx, cholesky)
    }
}

/// Appends the default passes (see the module documentation) to `manager`.
pub fn default_passes(manager: PassManager) -> PassManager {
    manager
        .with(FoldPass::default())
        .with(EgraphPass::default())
        .with(ContractsPass)
        .with(DcePass)
        .with(FusePass)
        .with(CholeskyPass)
}

/// The default pipeline in a [`PassManager::new`] manager.
pub fn default_pipeline() -> PassManager {
    default_passes(PassManager::new())
}
