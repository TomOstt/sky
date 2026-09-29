//! The rewrite rules.
//!
//! Every rule has a stable id, a class, a one-line justification, and a
//! matcher and a builder written in Rust so that side conditions can ask
//! the analysis (types, tags, sign, failure) instead of being encoded in a
//! pattern. The matcher finds bindings; the builder makes the right-hand
//! side; the applier (below) checks that the two sides have the same
//! dimension and shape before it unions them, so a rule bug can at worst
//! miss an optimization, never change a type.
//!
//! # Classes and the numerical contract
//!
//! * [`RuleClass::Exact`]: the two sides are bit-identical in IEEE
//!   floating point for every input, and the removed operations cannot
//!   fail once the kept subterms have been computed. Always enabled.
//! * [`RuleClass::Conditional`]: bit-identical *given a proven fact* the
//!   matcher checks against the analysis (a tag, a sign, "cannot fail").
//!   Always enabled; without the fact it does not fire.
//! * [`RuleClass::RealOnly`]: equal over the reals, not in floating point
//!   (rounding differs) and/or it removes an operation that could have
//!   failed (`exp(log x)` with `x <= 0`, an overflow, a singular matrix).
//!   Enabled only under [`RewriteContract::Fast`].
//!
//! A rule that drops a subterm (`x - x → 0` drops `x`, the Pythagorean
//! identity drops `x`) must not hide a failure of that subterm, so under
//! [`RewriteContract::Reproducible`] the matcher requires the subterm to
//! be `infallible` in the analysis.

use egg::{Applier, Id, PatternAst, Rewrite, SearchMatches, Searcher, Subst, Symbol, Var};
use sky_ir::MathFn;
use sky_types::Tag;
use sky_units::{Dimension, Rational};

use crate::analysis::{add_node, lit_node, render, Firing, Graph, Sign, SkyAnalysis};
use crate::lang::{Lit, LitVal, Op, SkyNode};
use crate::RewriteContract;

/// How a rule relates to floating point. See the module documentation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RuleClass {
    Exact,
    Conditional,
    RealOnly,
}

/// A rule's public description.
#[derive(Clone, Copy, Debug)]
pub struct RuleInfo {
    pub id: &'static str,
    pub class: RuleClass,
    /// The rule in words: what it needs and why it is valid.
    pub justification: &'static str,
    /// Left and right side, with `{0}`, `{1}` standing for matched terms.
    pub before: &'static str,
    pub after: &'static str,
}

type Binds = Vec<Id>;
type MatchFn = fn(&Graph, Id, RewriteContract) -> Vec<Binds>;
type BuildFn = fn(&mut Graph, Id, &mut Binds) -> Option<Id>;

struct Rule {
    info: RuleInfo,
    matcher: MatchFn,
    build: BuildFn,
}

/// Every rule, in id order.
pub fn rule_table() -> Vec<RuleInfo> {
    RULES.iter().map(|r| r.info).collect()
}

/// The rules a contract enables (optionally restricted to `only`).
pub(crate) fn rewrites(
    contract: RewriteContract,
    only: Option<&[String]>,
) -> Vec<Rewrite<SkyNode, SkyAnalysis>> {
    RULES
        .iter()
        .filter(|r| contract == RewriteContract::Fast || r.info.class != RuleClass::RealOnly)
        .filter(|r| only.is_none_or(|ids| ids.iter().any(|i| i == r.info.id)))
        .map(|r| {
            Rewrite::new(
                r.info.id,
                RuleSearcher {
                    matcher: r.matcher,
                    contract,
                },
                RuleApplier { rule: r },
            )
            .expect("rules bind no pattern variables")
        })
        .collect()
}

struct RuleSearcher {
    matcher: MatchFn,
    contract: RewriteContract,
}

impl Searcher<SkyNode, SkyAnalysis> for RuleSearcher {
    fn search_eclass_with_limit(
        &self,
        eg: &Graph,
        eclass: Id,
        limit: usize,
    ) -> Option<SearchMatches<'_, SkyNode>> {
        let found = (self.matcher)(eg, eclass, self.contract);
        let substs: Vec<Subst> = found
            .into_iter()
            .take(limit)
            .map(|binds| {
                let mut s = Subst::with_capacity(binds.len());
                for (i, id) in binds.into_iter().enumerate() {
                    s.insert(Var::from_u32(i as u32), id);
                }
                s
            })
            .collect();
        (!substs.is_empty()).then_some(SearchMatches {
            eclass,
            substs,
            ast: None,
        })
    }

    fn vars(&self) -> Vec<Var> {
        Vec::new()
    }
}

struct RuleApplier {
    rule: &'static Rule,
}

// `RULES` is a static, so `&'static Rule` is `Send + Sync` (fn pointers and
// plain data).
impl Applier<SkyNode, SkyAnalysis> for RuleApplier {
    fn apply_one(
        &self,
        eg: &mut Graph,
        matched: Id,
        subst: &Subst,
        _pattern: Option<&PatternAst<SkyNode>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        let rule = self.rule;
        let root = eg.find(matched);
        let mut binds: Binds = (0u32..)
            .map_while(|i| subst.get(Var::from_u32(i)).copied())
            .collect();
        let Some(rhs) = (rule.build)(eg, root, &mut binds) else {
            return Vec::new();
        };
        let rhs = eg.find(rhs);
        if rhs == root {
            return Vec::new();
        }
        // Both sides must have the same dimension and shape.
        let same = match (&eg[root].data.ty, &eg[rhs].data.ty) {
            (Some(a), Some(b)) => a.dim == b.dim && a.shape == b.shape,
            _ => false,
        };
        if !same {
            return Vec::new();
        }
        if eg.analysis.seen.insert((rule.info.id, root)) {
            let fill = |template: &str, eg: &Graph| {
                let mut out = template.to_string();
                for (i, b) in binds.iter().enumerate() {
                    out = out.replace(&format!("{{{i}}}"), &render(eg, *b));
                }
                out
            };
            let firing = Firing {
                rule: rule.info.id,
                root,
                span: eg[root].data.span,
                before: fill(rule.info.before, eg),
                after: fill(rule.info.after, eg),
                because: rule.info.justification.to_string(),
                rhs_nodes: eg[rhs].nodes.clone(),
                root_nodes: eg[root].nodes.clone(),
            };
            eg.analysis.firings.push(firing);
        }
        eg.union(root, rhs);
        vec![rhs]
    }
}

// ---------------------------------------------------------------------
// Matching helpers
// ---------------------------------------------------------------------

fn nodes(eg: &Graph, id: Id) -> Vec<SkyNode> {
    eg[id].nodes.clone()
}

fn value(eg: &Graph, id: Id) -> Option<f64> {
    eg[id].data.value
}

fn is_scalar(eg: &Graph, id: Id) -> bool {
    eg[id].data.ty.as_ref().is_some_and(|t| t.shape.rank() == 0)
}

fn is_dimensionless(eg: &Graph, id: Id) -> bool {
    eg[id]
        .data
        .ty
        .as_ref()
        .is_some_and(|t| t.dim.is_dimensionless())
}

/// A dimensionless constant equal to 1.
fn is_one(eg: &Graph, id: Id) -> bool {
    value(eg, id) == Some(1.0) && is_dimensionless(eg, id) && is_scalar(eg, id)
}

fn has_tag(eg: &Graph, id: Id, tag: Tag) -> bool {
    eg[id]
        .data
        .ty
        .as_ref()
        .is_some_and(|t| t.tags.contains(tag))
}

/// The arguments of the nodes `op(...)` in a class.
fn args_of(eg: &Graph, id: Id, op: Op) -> Vec<Vec<Id>> {
    eg[id]
        .nodes
        .iter()
        .filter(|n| n.op == op)
        .map(|n| n.args.clone())
        .collect()
}

/// The bases `t` with `id = t^2` (`PowConst 2` or `t * t`).
fn squares_of(eg: &Graph, id: Id) -> Vec<Id> {
    let mut out = Vec::new();
    for n in &eg[id].nodes {
        match n.op {
            Op::PowC if eg[n.args[1]].data.rat == Some(Rational::integer(2)) => {
                out.push(eg.find(n.args[0]));
            }
            Op::Mul if eg.find(n.args[0]) == eg.find(n.args[1]) => {
                out.push(eg.find(n.args[0]));
            }
            _ => {}
        }
    }
    out
}

fn math_args(eg: &Graph, id: Id, f: MathFn) -> Vec<Id> {
    args_of(eg, id, Op::math(f))
        .into_iter()
        .map(|a| eg.find(a[0]))
        .collect()
}

/// May the rule drop `id`'s evaluation? Yes if it cannot fail, or if the
/// contract lets rewrites remove failures.
fn droppable(eg: &Graph, id: Id, contract: RewriteContract) -> bool {
    contract == RewriteContract::Fast || eg[id].data.infallible
}

// ---------------------------------------------------------------------
// The rules
// ---------------------------------------------------------------------

fn keep_first(_: &mut Graph, _: Id, b: &mut Binds) -> Option<Id> {
    b.first().copied()
}

fn match_pythagoras(eg: &Graph, root: Id, contract: RewriteContract) -> Vec<Binds> {
    let mut out = Vec::new();
    for n in nodes(eg, root).iter().filter(|n| n.op == Op::Add) {
        let (a, b) = (n.args[0], n.args[1]);
        for (p, q) in [(a, b), (b, a)] {
            for sp in squares_of(eg, p) {
                for x in math_args(eg, sp, MathFn::Sin) {
                    let cos_sq = squares_of(eg, q)
                        .into_iter()
                        .any(|cq| math_args(eg, cq, MathFn::Cos).contains(&x));
                    if cos_sq && droppable(eg, x, contract) {
                        out.push(vec![x]);
                    }
                }
            }
        }
    }
    out
}

fn build_one(eg: &mut Graph, root: Id, _: &mut Binds) -> Option<Id> {
    let span = eg[root].data.span;
    let node = lit_node(1.0, Dimension::DIMENSIONLESS);
    Some(add_node(eg, node, None, span))
}

fn match_mul_one(eg: &Graph, root: Id, _: RewriteContract) -> Vec<Binds> {
    let mut out = Vec::new();
    for n in nodes(eg, root).iter().filter(|n| n.op == Op::Mul) {
        let (a, b) = (n.args[0], n.args[1]);
        if is_one(eg, b) {
            out.push(vec![a]);
        }
        if is_one(eg, a) {
            out.push(vec![b]);
        }
    }
    out
}

fn match_add_zero(eg: &Graph, root: Id, _: RewriteContract) -> Vec<Binds> {
    let mut out = Vec::new();
    for n in nodes(eg, root).iter().filter(|n| n.op == Op::Add) {
        let (a, b) = (n.args[0], n.args[1]);
        for (x, z) in [(a, b), (b, a)] {
            let Some(zero) = value(eg, z).filter(|v| *v == 0.0) else {
                continue;
            };
            // `x + (-0.0)` is exactly `x`. `x + (+0.0)` is too, except that
            // `-0.0 + 0.0` is `+0.0`, so `x` must be provably not `-0.0`.
            if zero.is_sign_negative() || eg[x].data.sign >= Sign::NonNeg {
                out.push(vec![x]);
            }
        }
    }
    out
}

fn match_sub_self(eg: &Graph, root: Id, contract: RewriteContract) -> Vec<Binds> {
    let mut out = Vec::new();
    for n in nodes(eg, root).iter().filter(|n| n.op == Op::Sub) {
        let (a, b) = (eg.find(n.args[0]), eg.find(n.args[1]));
        // Sky numbers are always finite, so `x - x` is exactly `+0.0` and
        // cannot fail. Evaluating `x` might, so it must be droppable.
        if a == b && is_scalar(eg, a) && droppable(eg, a, contract) {
            out.push(vec![a]);
        }
    }
    out
}

fn build_zero_like(eg: &mut Graph, root: Id, b: &mut Binds) -> Option<Id> {
    let dim = eg[*b.first()?].data.ty.as_ref()?.dim;
    let span = eg[root].data.span;
    Some(add_node(eg, lit_node(0.0, dim), None, span))
}

fn match_pow_pow(eg: &Graph, root: Id, _: RewriteContract) -> Vec<Binds> {
    let mut out = Vec::new();
    for n in nodes(eg, root).iter().filter(|n| n.op == Op::PowC) {
        let Some(b) = eg[n.args[1]].data.rat else {
            continue;
        };
        for inner in args_of(eg, n.args[0], Op::PowC) {
            let (x, ea) = (eg.find(inner[0]), inner[1]);
            let Some(a) = eg[ea].data.rat else { continue };
            if a.checked_mul(b).is_err() {
                continue;
            }
            // `(x^a)^b = x^(ab)` needs `b` an integer, or `x > 0`: for
            // `x = -2`, `(x^2)^(1/2) = 2` but `x^1 = -2`.
            if b.is_integer() || eg[x].data.sign == Sign::Pos {
                out.push(vec![x, ea, n.args[1]]);
            }
        }
    }
    out
}

fn build_pow_pow(eg: &mut Graph, root: Id, b: &mut Binds) -> Option<Id> {
    let (x, a, c) = (
        *b.first()?,
        eg[*b.get(1)?].data.rat?,
        eg[*b.get(2)?].data.rat?,
    );
    let ab = a.checked_mul(c).ok()?;
    let span = eg[root].data.span;
    let e = add_node(
        eg,
        SkyNode::leaf(Op::Lit(Lit::new(LitVal::Rat(ab), Dimension::DIMENSIONLESS))),
        None,
        span,
    );
    b.push(e);
    Some(add_node(eg, SkyNode::new(Op::PowC, vec![x, e]), None, span))
}

fn match_sqrt_square(eg: &Graph, root: Id, _: RewriteContract) -> Vec<Binds> {
    let mut out = Vec::new();
    for arg in math_args(eg, root, MathFn::Sqrt) {
        for x in squares_of(eg, arg) {
            out.push(vec![x]);
        }
    }
    out
}

fn build_abs(eg: &mut Graph, root: Id, b: &mut Binds) -> Option<Id> {
    let span = eg[root].data.span;
    let node = SkyNode::new(Op::math(MathFn::Abs), vec![*b.first()?]);
    Some(add_node(eg, node, None, span))
}

fn match_div_one(eg: &Graph, root: Id, _: RewriteContract) -> Vec<Binds> {
    nodes(eg, root)
        .iter()
        .filter(|n| n.op == Op::Div && is_one(eg, n.args[1]))
        .map(|n| vec![n.args[0]])
        .collect()
}

fn match_neg_neg(eg: &Graph, root: Id, _: RewriteContract) -> Vec<Binds> {
    let mut out = Vec::new();
    for n in nodes(eg, root).iter().filter(|n| n.op == Op::Neg) {
        for inner in args_of(eg, n.args[0], Op::Neg) {
            out.push(vec![inner[0]]);
        }
    }
    out
}

fn match_sub_zero(eg: &Graph, root: Id, _: RewriteContract) -> Vec<Binds> {
    nodes(eg, root)
        .iter()
        .filter(|n| n.op == Op::Sub)
        .filter(|n| value(eg, n.args[1]).is_some_and(|v| v == 0.0 && v.is_sign_positive()))
        .map(|n| vec![n.args[0]])
        .collect()
}

fn match_pow_one(eg: &Graph, root: Id, _: RewriteContract) -> Vec<Binds> {
    nodes(eg, root)
        .iter()
        .filter(|n| n.op == Op::PowC && eg[n.args[1]].data.rat == Some(Rational::ONE))
        .map(|n| vec![n.args[0]])
        .collect()
}

fn match_exp_log(eg: &Graph, root: Id, _: RewriteContract) -> Vec<Binds> {
    let mut out = Vec::new();
    for arg in math_args(eg, root, MathFn::Exp) {
        for x in math_args(eg, arg, MathFn::Log) {
            // `log` needs `x > 0`, and the identity does not hold for
            // `x <= 0` (the original fails, the rewrite would not).
            if eg[x].data.sign == Sign::Pos {
                out.push(vec![x]);
            }
        }
    }
    out
}

fn match_log_exp(eg: &Graph, root: Id, _: RewriteContract) -> Vec<Binds> {
    let mut out = Vec::new();
    for arg in math_args(eg, root, MathFn::Log) {
        for x in math_args(eg, arg, MathFn::Exp) {
            out.push(vec![x]);
        }
    }
    out
}

fn match_transpose_symmetric(eg: &Graph, root: Id, _: RewriteContract) -> Vec<Binds> {
    nodes(eg, root)
        .iter()
        .filter(|n| n.op == Op::Transpose && has_tag(eg, n.args[0], Tag::Symmetric))
        .map(|n| vec![n.args[0]])
        .collect()
}

fn match_transpose_twice(eg: &Graph, root: Id, _: RewriteContract) -> Vec<Binds> {
    let mut out = Vec::new();
    for n in nodes(eg, root).iter().filter(|n| n.op == Op::Transpose) {
        for inner in args_of(eg, n.args[0], Op::Transpose) {
            out.push(vec![inner[0]]);
        }
    }
    out
}

fn match_inverse_twice(eg: &Graph, root: Id, _: RewriteContract) -> Vec<Binds> {
    let mut out = Vec::new();
    for n in nodes(eg, root).iter().filter(|n| n.op == Op::Inv) {
        for inner in args_of(eg, n.args[0], Op::Inv) {
            out.push(vec![inner[0]]);
        }
    }
    out
}

macro_rules! rule {
    ($id:literal, $class:ident, $before:literal => $after:literal, $why:literal, $m:expr, $b:expr) => {
        Rule {
            info: RuleInfo {
                id: $id,
                class: RuleClass::$class,
                justification: $why,
                before: $before,
                after: $after,
            },
            matcher: $m,
            build: $b,
        }
    };
}

static RULES: &[Rule] = &[
    rule!(
        "ALG-001", Exact, "{0} * 1" => "{0}",
        "multiplying a finite float by 1 gives it back bit for bit, and cannot overflow",
        match_mul_one, keep_first
    ),
    rule!(
        "ALG-002", Conditional, "{0} + 0" => "{0}",
        "x + (-0.0) is exactly x; x + (+0.0) is exactly x unless x is -0.0, so x must be proven not -0.0 (a non-negative value)",
        match_add_zero, keep_first
    ),
    rule!(
        "ALG-003", Conditional, "{0} - {0}" => "0",
        "Sky numbers are finite, so x - x is exactly +0 and cannot fail; x itself must not be able to fail, since the rewrite stops evaluating it",
        match_sub_self, build_zero_like
    ),
    rule!(
        "ALG-004", RealOnly, "({0}^{1})^{2}" => "{0}^{3}",
        "(x^a)^b = x^(ab) over the reals when b is an integer or x > 0; the powers round differently and the original may fail where the rewrite does not",
        match_pow_pow, build_pow_pow
    ),
    rule!(
        "ALG-005", RealOnly, "sqrt({0}^2)" => "abs({0})",
        "sqrt(x^2) = |x| over the reals; x^2 can underflow or overflow in floating point",
        match_sqrt_square, build_abs
    ),
    rule!(
        "ALG-006", Exact, "{0} / 1" => "{0}",
        "dividing a finite float by 1 gives it back bit for bit, and cannot fail",
        match_div_one, keep_first
    ),
    rule!(
        "ALG-007", Exact, "-(-{0})" => "{0}",
        "negation only flips the sign bit, so applying it twice restores every bit",
        match_neg_neg, keep_first
    ),
    rule!(
        "ALG-008", Exact, "{0} - 0" => "{0}",
        "x - (+0.0) is exactly x for every x, including -0.0",
        match_sub_zero, keep_first
    ),
    rule!(
        "ALG-009", Exact, "{0}^1" => "{0}",
        "powi(x, 1) is x, and cannot overflow",
        match_pow_one, keep_first
    ),
    rule!(
        "EXP-001", RealOnly, "exp(log({0}))" => "{0}",
        "exp(log(x)) = x needs x > 0, which the analysis proved; rounding differs in floating point",
        match_exp_log, keep_first
    ),
    rule!(
        "EXP-002", RealOnly, "log(exp({0}))" => "{0}",
        "log(exp(x)) = x for every real x; exp can overflow in floating point and rounding differs",
        match_log_exp, keep_first
    ),
    rule!(
        "LA-001", Conditional, "{0}ᵀ" => "{0}",
        "Symmetric is among the tags of the operand, and a symmetric matrix equals its transpose",
        match_transpose_symmetric, keep_first
    ),
    rule!(
        "LA-002", Exact, "({0}ᵀ)ᵀ" => "{0}",
        "transposing twice restores every element",
        match_transpose_twice, keep_first
    ),
    rule!(
        "LA-003", RealOnly, "({0}⁻¹)⁻¹" => "{0}",
        "the inverse of the inverse is the matrix over the reals; in floating point it is only close, and the inverse fails on a singular matrix",
        match_inverse_twice, keep_first
    ),
    rule!(
        "TRIG-001", RealOnly, "sin({0})^2 + cos({0})^2" => "1",
        "the Pythagorean identity holds for every real x; in floating point the sum can differ from 1 in the last bit",
        match_pythagoras, build_one
    ),
];
