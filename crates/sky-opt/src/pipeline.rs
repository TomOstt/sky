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
//! 1. `specialize`: a call whose arguments have facts the callee's signature
//!    lacks (a `Vector[3]` for a `Vector[N]`, a constant, a proven sign)
//!    gets a clone of the callee that knows them, simplified. It runs first
//!    for three reasons. It reads the program as lowered, where the constant
//!    arguments are `const` operations or provable by the exactness analysis
//!    (`2 * 3`), and it retargets the calls before anything looks at them.
//!    Its clones are ordinary functions to every later pass, so `fold`,
//!    `egraph`, `contracts`, `dce` and `fuse` all see the constants and the
//!    concrete shapes (`contracts`, which reasons from the arguments of every
//!    call of a function, now reasons per clone, and discharges what only
//!    one kind of call satisfies). And it needs no other pass's output: it
//!    folds and cleans each clone itself, which it must, to know whether the
//!    clone is worth keeping and what the clone's own calls pass on.
//! 2. `fold`: constant folding in [`FoldMode::BitIdentical`], so no printed
//!    digit changes;
//! 3. `egraph`: e-graph rewriting under the `Reproducible` contract, which
//!    enables only exact and conditional rules (`@fast` rules are never on
//!    by default);
//! 4. `contracts`: removes an `assert`, `requires` or `ensures` that the
//!    exactness analysis proves true (from constants, and from the
//!    arguments of every call of the function), and says so. It runs after
//!    the e-graph, so it sees the simplified conditions, and before `dce`,
//!    which then removes the comparison the check leaves unused;
//! 5. `dce`: removes dead pure operations, and says so when named
//!    (user-written) values disappear. It runs after the e-graph, which
//!    leaves the operations it replaced behind;
//! 6. `powers`: `x^n` for a constant integer `2 <= n <= 14` becomes the
//!    multiplications `powi` performs, bit for bit (see [`crate::powers`]).
//!    It runs before `fuse`, so the multiplications can fuse into an
//!    elementwise kernel (`pow.const` does not), and after the e-graph,
//!    which knows `pow.const`;
//! 7. `fuse`: trees of elementwise array operations become `fused` kernels.
//!    It runs after `dce`, because a dead operation left by the e-graph
//!    would count as a second use of its operands and keep a temporary
//!    alive, and after the e-graph, because a `fused` body is opaque to it
//!    (fusing first would hide the arithmetic from the rewrite rules). It
//!    runs before `cholesky` only by convention: they touch different
//!    operations, and `cholesky`'s result must stay last so it is never
//!    rewritten.
//! 8. `cholesky`: `A⁻¹ @ b` with a PositiveDefinite `A` becomes
//!    `solve[cholesky](A, b)`, when the checker did not already rewrite it
//!    (`B = inv(A)` and then `B @ b`, which only the IR sees through);
//! 9. `transposes`: `transpose(A) @ B` becomes `matmul[ta](A, B)`: the
//!    transpose is not materialized ([`crate::linalg`]). Exact;
//! 10. `structure`: Gram products (`Aᵀ @ A`, one triangle mirrored), Diagonal
//!     scaling and the transpose of a Symmetric matrix ([`crate::linalg`]).
//!     Exact. It runs after `transposes`, which is what turns `Aᵀ @ A` into
//!     a product of one value by its own transpose;
//! 11. `algorithms`: records the algorithm class the policy chooses for
//!      each `matmul` ([`crate::algorithm`]). The default policy chooses
//!      nothing, so the pass changes nothing until a kernel registers one.
//!
//! The rewrites that change rounding (the order of an `@` chain, solves
//! instead of inverses, `Q⁻¹` as `Qᵀ`) are **not** passes: the checker
//! decides them, because the tree-walker must make the same choice, and
//! lowering emits the result (`sky_types::chain`). Everything here is exact.
//!
//! The manager verifies the module before the first pass and after every
//! pass ([`default_pipeline`] leaves that at the manager's default, on in
//! debug builds; callers turn it on with [`PassManager::verify_each`]).

use sky_egraph::EgraphPass;
use sky_ir::{Changed, Module, Pass, PassCx, PassManager};

use crate::algorithm::{algorithms, default_policy, MatMulPolicy};
use crate::rewrite::Rewrite;
use crate::{
    cholesky, contracts, dce, fold_with, fuse, powers, specialize, structure, transposes, FoldMode,
};

/// The pass names of [`default_pipeline`], in order: what
/// `sky check --emit-ir=after:<name>` accepts.
pub const DEFAULT_PASSES: [&str; 11] = [
    "specialize",
    "fold",
    "egraph",
    "contracts",
    "dce",
    "powers",
    "fuse",
    "cholesky",
    "transposes",
    "structure",
    "algorithms",
];

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

/// Specialization ([`specialize`]) as a pass named `"specialize"`.
#[derive(Clone, Copy, Default, Debug)]
pub struct SpecializePass;

impl Pass for SpecializePass {
    fn name(&self) -> &'static str {
        "specialize"
    }

    fn run(&self, m: &mut Module, cx: &mut PassCx) -> Changed {
        run_reporting(m, cx, specialize)
    }
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

/// `x^n` as multiplications ([`powers`]) as a pass named `"powers"`.
#[derive(Clone, Copy, Default, Debug)]
pub struct PowersPass;

impl Pass for PowersPass {
    fn name(&self) -> &'static str {
        "powers"
    }

    fn run(&self, m: &mut Module, cx: &mut PassCx) -> Changed {
        run_reporting(m, cx, powers)
    }
}

/// Transposes folded into `matmul[ta]` ([`transposes`]) as a pass named
/// `"transposes"`.
#[derive(Clone, Copy, Default, Debug)]
pub struct TransposesPass;

impl Pass for TransposesPass {
    fn name(&self) -> &'static str {
        "transposes"
    }

    fn run(&self, m: &mut Module, cx: &mut PassCx) -> Changed {
        run_reporting(m, cx, transposes)
    }
}

/// Structured products ([`structure`]) as a pass named `"structure"`.
#[derive(Clone, Copy, Default, Debug)]
pub struct StructurePass;

impl Pass for StructurePass {
    fn name(&self) -> &'static str {
        "structure"
    }

    fn run(&self, m: &mut Module, cx: &mut PassCx) -> Changed {
        run_reporting(m, cx, structure)
    }
}

/// The algorithm choice of every `matmul` ([`algorithms`]) as a pass named
/// `"algorithms"`, by a [`MatMulPolicy`] (the default one chooses nothing).
#[derive(Clone, Copy, Debug)]
pub struct AlgorithmPass {
    pub matmul: MatMulPolicy,
}

impl Default for AlgorithmPass {
    fn default() -> Self {
        AlgorithmPass {
            matmul: default_policy,
        }
    }
}

impl Pass for AlgorithmPass {
    fn name(&self) -> &'static str {
        "algorithms"
    }

    fn run(&self, m: &mut Module, cx: &mut PassCx) -> Changed {
        run_reporting(m, cx, |m| algorithms(m, self.matmul))
    }
}

/// Appends the default passes (see the module documentation) to `manager`.
pub fn default_passes(manager: PassManager) -> PassManager {
    default_passes_without_specialize(manager.with(SpecializePass))
}

/// [`default_passes`] without `specialize`: the pipeline as it was before
/// specialization, for tests that check what another pass does on its own
/// and for comparing behaviour with and without specializing.
pub fn default_passes_without_specialize(manager: PassManager) -> PassManager {
    manager
        .with(FoldPass::default())
        .with(EgraphPass::default())
        .with(ContractsPass)
        .with(DcePass)
        .with(PowersPass)
        .with(FusePass)
        .with(CholeskyPass)
        .with(TransposesPass)
        .with(StructurePass)
        .with(AlgorithmPass::default())
}

/// The default pipeline in a [`PassManager::new`] manager.
pub fn default_pipeline() -> PassManager {
    default_passes(PassManager::new())
}
