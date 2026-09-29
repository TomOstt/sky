//! Identity-aware rewriting of SIR with e-graphs (`docs/design/ir.md`,
//! "Rewrites and the e-graph"; `ROADMAP.md` ideas 5 and 99).
//!
//! The pure expression DAGs of a block are loaded into an `egg` e-graph,
//! saturated with a set of rules that each carry a stable id, a class and a
//! justification, and the cheapest equivalent form is written back with
//! the original types and spans. What makes the rewriting *identity-aware*
//! is that a rule's side conditions are questions about SIR facts: `Aᵀ → A`
//! fires only when `Symmetric` is among the tags of `A`'s type; `exp(log x)
//! → x` only when `x` is proven positive; `x + 0 → x` only when `x` is
//! provably not `-0.0`.
//!
//! # The numerical contract
//!
//! [`RewriteContract`] says what a rewrite may change. **`Reproducible`**
//! (the default, the design's `@reproducible`) demands that the rewritten
//! program computes bit-identical results and *fails at exactly the same
//! point with the same message*. **`Fast`** (`@fast`) allows rules that are
//! only valid over the reals: they may change the last bits of a result,
//! and they may remove an operation that could have failed (`log(exp(x))`
//! never overflows once it is `x`).
//!
//! Rules are classified ([`RuleClass`]): *exact* rules are bit-identical
//! for all inputs, *conditional* rules are bit-identical given a fact the
//! analysis proved, and *real-only* rules are enabled only under `Fast`.
//! Under both contracts:
//!
//! * effectful ops (`print`, `call`, `if`, `while`) and ops that can fail
//!   for a reason the e-graph does not model are never moved, duplicated or
//!   removed, and no op moves across them (see `convert`);
//! * a rewrite never *introduces* an operation that can fail.
//!
//! # Facts from the exactness analysis
//!
//! Besides what the e-graph derives from the shape of a term, the facts of
//! a class include what `sky_ir::exact` proved about the `f64` the
//! program's operation computes: strictly positive, nonzero, and *cannot
//! fail* although its kind can (`x / 2` and `x * 0.5` for any finite `x`,
//! `sqrt(2)`, `log(3)`). They come from the operations as written, so they
//! are used only under `Reproducible`, whose rewrites are bit-identical.
//! `sqrt(2) - sqrt(2)` then becomes `0` (ALG-003 needs the operand to be
//! unable to fail), and `sqrt(2) / sqrt(2)` becomes `1` (ALG-010 needs it
//! nonzero). The public API is unchanged; only more rules fire.
//!
//! Additionally, under `Reproducible`:
//!
//! * a rule may remove an operation only if it provably cannot fail, and a
//!   rule that stops evaluating a subterm requires that subterm to be
//!   unable to fail;
//! * operations that can fail keep their relative order, so the first
//!   failure is the same one; if the extracted form would reorder them the
//!   block is left alone.
//!
//! # As a pass
//!
//! [`EgraphPass`] is the crate as a [`sky_ir::Pass`] named `"egraph"`. Each
//! applied rule is a [`Rewrite`] (this crate's record, with the rule's
//! justification as one string), and [`Rewrite::explain`] is the single place
//! that turns it into a `sky_ir::Explanation` through `PassCx::explain`. The
//! rule ids and texts are unchanged.

#![forbid(unsafe_code)]

mod analysis;
mod convert;
mod lang;
mod rules;

use std::time::Duration;

use sky_diag::Span;
use sky_ir::{Changed, Function, Module, Pass, PassCx};

pub use analysis::CostModel;
pub use rules::{rule_table, RuleClass, RuleInfo};

/// What a rewrite is allowed to change. See the crate documentation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum RewriteContract {
    /// Bit-identical results and identical failures (`@reproducible`).
    #[default]
    Reproducible,
    /// Rules valid over the reals are allowed (`@fast`).
    Fast,
}

/// Settings for one run.
#[derive(Clone, Debug)]
pub struct RewriteConfig {
    pub contract: RewriteContract,
    /// Saturation stops after this many rounds of rule application.
    pub iter_limit: usize,
    /// ... or when the e-graph has this many e-nodes.
    pub node_limit: usize,
    /// ... or after this long. `None` (the default) keeps the result
    /// independent of the machine's speed: only the two limits above bound
    /// the work.
    pub time_limit: Option<Duration>,
    /// A segment with more expression ops than this is left alone.
    pub max_segment_ops: usize,
    /// If set, only the rules with these ids run (and only if the contract
    /// enables them).
    pub only_rules: Option<Vec<String>>,
}

impl Default for RewriteConfig {
    fn default() -> Self {
        RewriteConfig {
            contract: RewriteContract::Reproducible,
            iter_limit: 12,
            node_limit: 20_000,
            time_limit: None,
            max_segment_ops: 20_000,
            only_rules: None,
        }
    }
}

impl RewriteConfig {
    pub fn with_contract(contract: RewriteContract) -> Self {
        RewriteConfig {
            contract,
            ..RewriteConfig::default()
        }
    }
}

/// One applied rule, in the shape of the pass manager's explanation:
/// `{pass: "egraph", rule, span, before, after, because}`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Rewrite {
    /// Always `"egraph"`.
    pub pass: &'static str,
    /// The rule's stable id: `TRIG-001`, `ALG-003`, ...
    pub rule: &'static str,
    /// The source the rewritten expression came from: the join of the
    /// spans of the operations merged.
    pub span: Span,
    /// The expression before, printed from the program's own form.
    pub before: String,
    /// What it became.
    pub after: String,
    /// The rule's justification and the contract that allowed it.
    pub because: String,
}

/// Why saturation of a segment stopped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stop {
    /// No rule found anything new.
    Saturated,
    IterationLimit,
    NodeLimit,
    TimeLimit,
    Other,
}

/// What happened to one segment (a run of pure expression ops).
#[derive(Clone, Debug)]
pub struct SegmentReport {
    pub expr_ops: usize,
    pub iterations: usize,
    /// E-nodes in the e-graph when saturation ended.
    pub egraph_nodes: usize,
    pub stop: Option<Stop>,
    /// Why the segment was left as it was, if a rewrite was found but not
    /// applied (or the segment was too large).
    pub skipped: Option<&'static str>,
    /// Rewrites written back.
    pub applied: usize,
}

/// The result of [`rewrite_function_report`].
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub rewrites: Vec<Rewrite>,
    pub segments: Vec<SegmentReport>,
}

/// Rewrites every block of `func`; returns what was applied.
pub fn rewrite_function(func: &mut Function, cfg: &RewriteConfig) -> Vec<Rewrite> {
    rewrite_function_report(func, cfg).rewrites
}

/// Like [`rewrite_function`], with per-segment saturation statistics.
pub fn rewrite_function_report(func: &mut Function, cfg: &RewriteConfig) -> Report {
    let (rewrites, segments) = convert::rewrite_function(func, cfg);
    Report { rewrites, segments }
}

/// Rewrites every function of `module`.
pub fn rewrite_module(module: &mut Module, cfg: &RewriteConfig) -> Vec<Rewrite> {
    module
        .functions
        .iter_mut()
        .flat_map(|f| rewrite_function(f, cfg))
        .collect()
}

impl Rewrite {
    /// Reports the rewrite through the pass manager's collector: the one
    /// place that converts this crate's record into a
    /// [`sky_ir::Explanation`]. The pass name in the explanation is the
    /// running pass's, so it is `"egraph"` when run through [`EgraphPass`].
    pub fn explain(&self, cx: &mut PassCx) {
        debug_assert_eq!(self.pass, cx.pass_name(), "explained by a different pass");
        cx.explain(
            self.rule,
            self.span,
            self.before.clone(),
            self.after.clone(),
            vec![self.because.clone()],
        );
    }
}

/// E-graph rewriting as a [`Pass`]: [`rewrite_module`] with a fixed
/// [`RewriteConfig`]. The default is the `Reproducible` contract (exact and
/// conditional rules only), so the program computes the same bits and fails
/// at the same operation.
#[derive(Clone, Debug, Default)]
pub struct EgraphPass {
    pub config: RewriteConfig,
}

impl EgraphPass {
    /// A pass with the default limits and the given contract.
    pub fn new(contract: RewriteContract) -> Self {
        EgraphPass {
            config: RewriteConfig::with_contract(contract),
        }
    }
}

impl Pass for EgraphPass {
    fn name(&self) -> &'static str {
        "egraph"
    }

    fn run(&self, m: &mut Module, cx: &mut PassCx) -> Changed {
        let before = m.clone();
        for rw in rewrite_module(m, &self.config) {
            rw.explain(cx);
        }
        Changed::from(*m != before)
    }
}
