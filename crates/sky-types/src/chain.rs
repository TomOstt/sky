//! Matrix chains: the order of `A @ B @ C @ v`, and solving instead of
//! inverting, decided from shapes and tags.
//!
//! An unparenthesized `@` chain is one operation whose association the
//! compiler chooses (spec, "Algorithm choice"). [`plan_chain`] takes the
//! chain's factors, with their shapes, dimensions and tags, and picks the
//! cheapest way to compute it by the flop model of [`crate::flops`]:
//!
//! * **the order**, by the classic dynamic program over intervals
//!   (`A @ (B @ (C @ v))` instead of `((A @ B) @ C) @ v`: O(n²) instead of
//!   O(n³)); with symbolic shapes an order is chosen only if it is
//!   provably no more expensive for every size;
//! * **solves instead of inverses**: a factor `A⁻¹` (or `inv(A)`) next to
//!   the value `R` it multiplies becomes `solve(A, R)` (or `x @ A⁻¹` a
//!   transposed solve), by [`SolveMethod`]: Cholesky for a PositiveDefinite
//!   `A`, division for a Diagonal one, LU for any square matrix. `A⁻¹` is
//!   never formed;
//! * **`Q⁻¹` is `Qᵀ`** for an Orthogonal `Q` (exact by the tag's definition
//!   `QᵀQ = QQᵀ = I`; the SIR pass turns the transpose into `matmul[ta]`);
//! * **`A⁻¹ @ B⁻¹` is `(B @ A)⁻¹`**, when the flops say one inverse and one
//!   product beat two inverses (and a solve), which they do for square
//!   matrices.
//!
//! Parentheses the user wrote are respected: a parenthesized product is one
//! factor of the outer chain and is planned on its own. Only the chain of
//! unparenthesized `@` is reordered.
//!
//! None of these is bit-identical to the written order (the products and
//! solves round differently; the spec allows it, and the explanation says
//! so). The plan is recorded by the checker in [`crate::Checked::chains`]
//! keyed by the span of the whole chain, and every executor follows it: the
//! tree-walker evaluates the plan, lowering emits its operations. That is
//! what keeps the engines bit-identical to each other.
//!
//! # Evaluation
//!
//! Every executor does the same, in this order, so operands are evaluated
//! exactly once, left to right, whatever the plan does with them:
//!
//! 1. each factor, in source order: a factor the plan consumes as `A⁻¹`
//!    evaluates only `A` (the inverse is never formed); an Orthogonal
//!    `Q⁻¹` evaluates `Q` and transposes it; any other factor is evaluated
//!    as written;
//! 2. the tree, children first, each node one operation.

use sky_ast::ast::{BinaryOp, Expr, ExprKind};
use sky_diag::Span;
use sky_units::Dimension;

use crate::algo::SolveMethod;
use crate::flops::Flops;
use crate::shape::{Shape, ShapeDim};
use crate::tags::{Tag, TagSet};

/// The rule id of a chain rewrite (order, solves, transposes, products of
/// inverses), as `--explain` shows it.
pub const CHAIN_RULE: &str = "LA-CHAIN-001";

/// The most factors a chain is planned for; a longer chain is evaluated as
/// written.
pub const MAX_CHAIN_FACTORS: usize = 32;

/// The factors of the `@` chain `lhs @ rhs`, in source order: the operands
/// of the left-nested unparenthesized `@` in `lhs`, then `rhs`. A
/// parenthesized operand is one factor, whatever it contains.
pub fn chain_factors<'e>(lhs: &'e Expr, rhs: &'e Expr) -> Vec<&'e Expr> {
    let mut out = vec![rhs];
    let mut cur = lhs;
    loop {
        match &cur.kind {
            ExprKind::Binary {
                op: BinaryOp::MatMul,
                lhs,
                rhs,
                ..
            } => {
                out.push(rhs);
                cur = lhs;
            }
            _ => {
                out.push(cur);
                break;
            }
        }
    }
    out.reverse();
    out
}

/// The spans of the `@` expressions nested in the left spine of the chain
/// `lhs @ rhs` (not the chain's own): the sub-chains the planner subsumes.
pub fn nested_chain_spans(lhs: &Expr) -> Vec<Span> {
    let mut out = Vec::new();
    let mut cur = lhs;
    while let ExprKind::Binary {
        op: BinaryOp::MatMul,
        lhs,
        ..
    } = &cur.kind
    {
        out.push(cur.span);
        cur = lhs;
    }
    out
}

/// One factor of a chain, as the planner sees it.
#[derive(Clone, Debug)]
pub struct ChainFactor {
    /// The factor expression's span (parentheses included).
    pub span: Span,
    /// Its source text.
    pub text: String,
    /// The text to splice into a larger expression: `text`, parenthesized
    /// unless it is an atom.
    pub atom: String,
    /// The dimension of the factor's value.
    pub dim: Dimension,
    /// The shape of the factor's value (a vector or a matrix).
    pub shape: Shape,
    /// The closed tags of the factor's value.
    pub tags: TagSet,
    /// `Some` if the factor is `A⁻¹` or `inv(A)`.
    pub inverse: Option<InverseFactor>,
}

/// What the planner needs to know about the `A` of an `A⁻¹` factor.
#[derive(Clone, Debug)]
pub struct InverseFactor {
    /// `A`'s span.
    pub operand_span: Span,
    /// `A`'s source text, parenthesized unless it is an atom.
    pub operand_atom: String,
    /// `A`'s closed tags.
    pub tags: TagSet,
}

/// The factor as the plan records it, so an executor can check that it is
/// reading the same chain.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PlanFactor {
    pub span: Span,
    /// `A`'s span if the factor is `A⁻¹`.
    pub inverse_operand: Option<Span>,
}

/// How a leaf (one factor) is evaluated.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LeafForm {
    /// As written (an `A⁻¹` factor computes the inverse).
    Plain,
    /// The factor is an `A⁻¹` a parent solve consumes: only `A` is
    /// evaluated (the parent node names the factor).
    Deferred,
    /// The factor is `Q⁻¹` for an Orthogonal `Q`: `Q` is evaluated and
    /// transposed.
    Transposed,
}

/// One node of the plan's tree. Each internal node is one operation.
#[derive(Clone, PartialEq, Debug)]
pub enum Node {
    /// A factor.
    Leaf { factor: usize, form: LeafForm },
    /// `lhs @ rhs`.
    Mul { lhs: Box<Node>, rhs: Box<Node> },
    /// `A⁻¹ @ rhs`, where `A⁻¹` is the factor `inv`, by `method`.
    Solve {
        method: SolveMethod,
        inv: usize,
        rhs: Box<Node>,
    },
    /// `lhs @ A⁻¹`, where `A⁻¹` is the factor `inv`, by `method` (a
    /// transposed solve).
    SolveRight {
        method: SolveMethod,
        lhs: Box<Node>,
        inv: usize,
    },
    /// `A⁻¹ @ B⁻¹` as `(B @ A)⁻¹`: the factors `left` (`A⁻¹`) and `right`
    /// (`B⁻¹`), consumed as their operands.
    InvOfProduct { left: usize, right: usize },
}

impl Node {
    /// The first and last factor the node covers.
    pub fn range(&self) -> (usize, usize) {
        match self {
            Node::Leaf { factor, .. } => (*factor, *factor),
            Node::Mul { lhs, rhs } => (lhs.range().0, rhs.range().1),
            Node::Solve { inv, rhs, .. } => (*inv, rhs.range().1),
            Node::SolveRight { lhs, inv, .. } => (lhs.range().0, *inv),
            Node::InvOfProduct { left, right } => (*left, *right),
        }
    }

    /// Whether every product in the tree multiplies by a single factor on
    /// the right, which is what the written left-to-right order does.
    pub fn is_left_associated(&self) -> bool {
        match self {
            Node::Leaf { .. } | Node::InvOfProduct { .. } => true,
            Node::Mul { lhs, rhs } => single(rhs) && lhs.is_left_associated(),
            Node::Solve { rhs, .. } => single(rhs) && rhs.is_left_associated(),
            Node::SolveRight { lhs, .. } => lhs.is_left_associated(),
        }
    }
}

fn single(n: &Node) -> bool {
    let (lo, hi) = n.range();
    lo == hi
}

/// A planned chain: what the checker records for the whole `@` expression,
/// and what every executor follows.
#[derive(Clone, PartialEq, Debug)]
pub struct ChainPlan {
    pub factors: Vec<PlanFactor>,
    pub root: Node,
}

impl ChainPlan {
    /// The span a node covers: from its first factor to its last.
    pub fn span_of(&self, node: &Node) -> Span {
        let (lo, hi) = node.range();
        self.factors[lo].span.join(self.factors[hi].span)
    }

    /// How each factor is to be evaluated (step 1 of the module's
    /// "Evaluation"): a factor consumed as `A⁻¹` is [`LeafForm::Deferred`].
    pub fn factor_forms(&self) -> Vec<LeafForm> {
        let mut forms = vec![LeafForm::Plain; self.factors.len()];
        fn walk(n: &Node, forms: &mut [LeafForm]) {
            match n {
                Node::Leaf { factor, form } => forms[*factor] = *form,
                Node::Mul { lhs, rhs } => {
                    walk(lhs, forms);
                    walk(rhs, forms);
                }
                Node::Solve { inv, rhs, .. } => {
                    forms[*inv] = LeafForm::Deferred;
                    walk(rhs, forms);
                }
                Node::SolveRight { lhs, inv, .. } => {
                    walk(lhs, forms);
                    forms[*inv] = LeafForm::Deferred;
                }
                Node::InvOfProduct { left, right } => {
                    forms[*left] = LeafForm::Deferred;
                    forms[*right] = LeafForm::Deferred;
                }
            }
        }
        walk(&self.root, &mut forms);
        forms
    }
}

/// The result of planning: the plan and what to say about it.
#[derive(Clone, Debug)]
pub struct ChainDecision {
    pub plan: ChainPlan,
    /// The cost of the written order (inverses formed).
    pub before: Flops,
    /// The cost of the plan.
    pub after: Flops,
    /// Whether the plan multiplies in a different order than the written
    /// left to right.
    pub reassociated: bool,
    /// The plan as source-like text: `A @ (B @ (C @ v))`,
    /// `solve[lu](A, b)`.
    pub text: String,
    /// One line per solve, transposition and product of inverses the plan
    /// uses, saying why.
    pub reasons: Vec<String>,
    /// How the result can differ from the written form's, one clause for
    /// each kind of change the plan makes (never empty).
    pub rounding: Vec<String>,
}

/// How an `A⁻¹` factor is used when it meets a value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Choice {
    /// `A⁻¹` is `Aᵀ` (Orthogonal).
    Transpose,
    /// `A⁻¹ @ R` is a solve by this method.
    Solve(SolveMethod),
}

/// The method for `A⁻¹` next to a value, from `A`'s tags, in this order:
/// Orthogonal (a transpose), Diagonal (division), PositiveDefinite
/// (Cholesky), else LU. `tags` are closed.
fn choice_for(tags: TagSet) -> Choice {
    let tags = tags.closure();
    if tags.contains(Tag::Orthogonal) {
        Choice::Transpose
    } else if tags.contains(Tag::Diagonal) {
        Choice::Solve(SolveMethod::Diagonal)
    } else if tags.contains(Tag::PositiveDefinite) {
        Choice::Solve(SolveMethod::Cholesky)
    } else {
        Choice::Solve(SolveMethod::Lu)
    }
}

/// The method the planner picks for `A⁻¹ @ b` with `A`'s tags `tags`, or
/// `None` where `A⁻¹` is a transpose (Orthogonal). The one place the
/// decision is made, shared with the SIR passes.
pub fn solve_method_for(tags: TagSet) -> Option<SolveMethod> {
    match choice_for(tags) {
        Choice::Solve(m) => Some(m),
        Choice::Transpose => None,
    }
}

/// Why a method was chosen, for `--explain`.
pub fn method_reason(method: SolveMethod, operand: &str) -> String {
    match method {
        SolveMethod::Cholesky => format!(
            "{operand} is PositiveDefinite: factor {operand} = L·Lᵀ (L lower-triangular), then two triangular solves"
        ),
        SolveMethod::Diagonal => format!(
            "{operand} is Diagonal: x = b / diag({operand}), one division per entry"
        ),
        SolveMethod::Lu => format!(
            "{operand} is a general square matrix: LU with partial pivoting ({operand} = P·L·U), then two triangular solves"
        ),
    }
}

/// A candidate for an interval: its cost and its tree.
#[derive(Clone)]
struct Cand {
    cost: Flops,
    node: Node,
}

struct Planner<'a> {
    factors: &'a [ChainFactor],
    /// Row and column count of factor `t` as a matrix (a first vector is a
    /// row, a last vector a column).
    rows: Vec<ShapeDim>,
    cols: Vec<ShapeDim>,
    choices: Vec<Option<Choice>>,
}

impl Planner<'_> {
    fn leaf(&self, t: usize) -> Cand {
        let form = match self.choices[t] {
            None => LeafForm::Plain,
            Some(Choice::Transpose) => LeafForm::Transposed,
            Some(Choice::Solve(_)) => LeafForm::Deferred,
        };
        Cand {
            cost: Flops::zero(),
            node: Node::Leaf { factor: t, form },
        }
    }

    fn deferred(n: &Node) -> Option<usize> {
        match n {
            Node::Leaf {
                factor,
                form: LeafForm::Deferred,
            } => Some(*factor),
            _ => None,
        }
    }

    fn method(&self, t: usize) -> SolveMethod {
        match self.choices[t] {
            Some(Choice::Solve(m)) => m,
            _ => SolveMethod::Lu,
        }
    }

    /// The ways to combine an interval `[i..=k]` and `[k+1..=j]`, cheapest
    /// not yet chosen; the cost includes both children.
    fn combine(&self, i: usize, k: usize, j: usize, l: &Cand, r: &Cand) -> Vec<Cand> {
        let both = l.cost.plus(&r.cost);
        let plain_matmul =
            || both.plus(&Flops::matmul(&self.rows[i], &self.cols[k], &self.cols[j]));
        match (Self::deferred(&l.node), Self::deferred(&r.node)) {
            // `A⁻¹ @ B⁻¹`: one product and one inverse, or one inverse and a
            // solve against it (either side).
            (Some(a), Some(b)) => {
                let n = &self.rows[a];
                let product_then_inverse = Cand {
                    cost: Flops::matmul(n, n, n).plus(&Flops::inverse(n)),
                    node: Node::InvOfProduct { left: a, right: b },
                };
                let solve_against_inverse = Cand {
                    cost: Flops::inverse(n).plus(&Flops::solve(self.method(a), n, n)),
                    node: Node::Solve {
                        method: self.method(a),
                        inv: a,
                        rhs: Box::new(Node::Leaf {
                            factor: b,
                            form: LeafForm::Plain,
                        }),
                    },
                };
                let solve_right_against_inverse = Cand {
                    cost: Flops::inverse(n).plus(&Flops::solve(self.method(b), n, n)),
                    node: Node::SolveRight {
                        method: self.method(b),
                        lhs: Box::new(Node::Leaf {
                            factor: a,
                            form: LeafForm::Plain,
                        }),
                        inv: b,
                    },
                };
                vec![
                    product_then_inverse,
                    solve_against_inverse,
                    solve_right_against_inverse,
                ]
            }
            (Some(a), None) => {
                let m = self.method(a);
                vec![Cand {
                    cost: both.plus(&Flops::solve(m, &self.rows[a], &self.cols[j])),
                    node: Node::Solve {
                        method: m,
                        inv: a,
                        rhs: Box::new(r.node.clone()),
                    },
                }]
            }
            (None, Some(b)) => {
                let m = self.method(b);
                vec![Cand {
                    cost: both.plus(&Flops::solve(m, &self.rows[b], &self.rows[i])),
                    node: Node::SolveRight {
                        method: m,
                        lhs: Box::new(l.node.clone()),
                        inv: b,
                    },
                }]
            }
            (None, None) => vec![Cand {
                cost: plain_matmul(),
                node: Node::Mul {
                    lhs: Box::new(l.node.clone()),
                    rhs: Box::new(r.node.clone()),
                },
            }],
        }
    }

    /// The first candidate, replaced by a later one only when that is
    /// provably cheaper.
    fn pick(cands: Vec<Cand>) -> Option<Cand> {
        let mut it = cands.into_iter();
        let mut best = it.next()?;
        for c in it {
            if c.cost.provably_less(&best.cost) {
                best = c;
            }
        }
        Some(best)
    }

    /// A `Cand` for an interval `[i..=j]` after combining its children.
    /// The interval of a single factor is a leaf; a leaf that stays a
    /// deferred inverse has no cost of its own.
    fn best_of(&self, i: usize, j: usize, table: &[Vec<Option<Cand>>]) -> Option<Cand> {
        if i == j {
            return Some(self.leaf(i));
        }
        // Split at the last factor first: the written order wins ties.
        let mut cands = Vec::new();
        for k in (i..j).rev() {
            let l = if i == k {
                self.leaf(i)
            } else {
                table[i][k].clone()?
            };
            let r = if k + 1 == j {
                self.leaf(j)
            } else {
                table[k + 1][j].clone()?
            };
            // A deferred leaf is only a leaf: the split's node consumes it.
            cands.extend(self.combine(i, k, j, &l, &r).into_iter().filter(|c| {
                // A candidate for an interval must not leave a
                // deferred inverse as its whole value.
                !matches!(
                    c.node,
                    Node::Leaf {
                        form: LeafForm::Deferred,
                        ..
                    }
                )
            }));
        }
        // Among all candidates, with the both-deferred alternatives, pick
        // the cheapest provably; keep the written split otherwise.
        Self::pick(cands)
    }

    /// The best plan for the whole chain, by the interval dynamic program.
    fn optimal(&self) -> Option<Cand> {
        let n = self.factors.len();
        let mut table: Vec<Vec<Option<Cand>>> = vec![vec![None; n]; n];
        for len in 2..=n {
            for i in 0..=n - len {
                let j = i + len - 1;
                table[i][j] = self.best_of(i, j, &table);
            }
        }
        table[0][n - 1].clone()
    }

    /// The written left-to-right association, with `A⁻¹` factors converted
    /// where they meet a value (a plan for a chain with a vector in the
    /// middle, or the fallback when no order is provably better).
    fn left_to_right(&self) -> Option<Cand> {
        let n = self.factors.len();
        let mut acc = self.leaf(0);
        for j in 1..n {
            let r = self.leaf(j);
            let cands = self.combine(0, j - 1, j, &acc, &r);
            acc = Self::pick(cands)?;
        }
        Some(acc)
    }

    /// The cost of the chain as written: every `A⁻¹` formed, the products
    /// left to right.
    fn written_cost(&self) -> Flops {
        let mut cost = Flops::zero();
        for (t, f) in self.factors.iter().enumerate() {
            if f.inverse.is_some() {
                cost = cost.plus(&Flops::inverse(&self.rows[t]));
            }
        }
        for j in 1..self.factors.len() {
            cost = cost.plus(&Flops::matmul(
                &self.rows[0],
                &self.cols[j - 1],
                &self.cols[j],
            ));
        }
        cost
    }
}

/// Plans the chain of `factors` (in source order), or `None` if the written
/// evaluation is what the plan would be (nothing to reorder, solve or
/// transpose), if the planner cannot prove a plan cheaper for every size,
/// or if the chain is not one it understands (a factor that is not a vector
/// or matrix, shapes that do not chain).
pub fn plan_chain(factors: &[ChainFactor]) -> Option<ChainDecision> {
    let n = factors.len();
    if !(2..=MAX_CHAIN_FACTORS).contains(&n) {
        return None;
    }
    let mut rows = Vec::with_capacity(n);
    let mut cols = Vec::with_capacity(n);
    let mut fixed = false;
    for (t, f) in factors.iter().enumerate() {
        match f.shape.dims() {
            [r, c] => {
                rows.push(r.clone());
                cols.push(c.clone());
            }
            // A vector is a row on the left of a product and a column on
            // the right; in the middle it has no orientation, so the chain
            // cannot be reordered.
            [len] if t == 0 => {
                rows.push(ShapeDim::Known(1));
                cols.push(len.clone());
            }
            [len] if t == n - 1 => {
                rows.push(len.clone());
                cols.push(ShapeDim::Known(1));
            }
            [len] => {
                fixed = true;
                rows.push(len.clone());
                cols.push(len.clone());
            }
            _ => return None,
        }
        // An inverse is of a square matrix.
        if f.inverse.is_some() && rows[t] != cols[t] {
            return None;
        }
    }
    if (0..n - 1).any(|t| cols[t] != rows[t + 1]) && !fixed {
        return None;
    }
    let choices: Vec<Option<Choice>> = factors
        .iter()
        .map(|f| f.inverse.as_ref().map(|inv| choice_for(inv.tags)))
        .collect();
    let planner = Planner {
        factors,
        rows,
        cols,
        choices,
    };
    let before = planner.written_cost();
    let attempts = if fixed {
        vec![planner.left_to_right()]
    } else {
        vec![planner.optimal(), planner.left_to_right()]
    };
    for cand in attempts.into_iter().flatten() {
        let after = cand.cost.clone();
        if !after.provably_less(&before) {
            continue;
        }
        let plan = ChainPlan {
            factors: factors
                .iter()
                .map(|f| PlanFactor {
                    span: f.span,
                    inverse_operand: f.inverse.as_ref().map(|i| i.operand_span),
                })
                .collect(),
            root: cand.node,
        };
        if covered_by_cholesky_site(&plan, factors) {
            return None;
        }
        let reassociated = !plan.root.is_left_associated();
        let text = describe(&plan.root, factors, true);
        let reasons = reasons(&plan.root, factors);
        let rounding = rounding(&plan.root, reassociated);
        return Some(ChainDecision {
            plan,
            before,
            after,
            reassociated,
            text,
            reasons,
            rounding,
        });
    }
    None
}

/// Whether the plan is exactly what the checker's Cholesky rewrite of the
/// chain's first product already does: `inv(A) @ b` for a PositiveDefinite
/// `A`, left to right, and nothing else. Then the older, per-site rewrite
/// (with its own explanation) stands and no chain plan is recorded.
fn covered_by_cholesky_site(plan: &ChainPlan, factors: &[ChainFactor]) -> bool {
    let mut node = &plan.root;
    loop {
        match node {
            Node::Leaf { .. } => return false,
            Node::Mul { lhs, rhs } => {
                if !matches!(
                    **rhs,
                    Node::Leaf {
                        form: LeafForm::Plain,
                        ..
                    }
                ) || factors[rhs.range().0].inverse.is_some()
                {
                    return false;
                }
                node = lhs;
            }
            Node::Solve { method, inv, rhs } => {
                return *method == SolveMethod::Cholesky
                    && *inv == 0
                    && matches!(**rhs, Node::Leaf { form: LeafForm::Plain, factor } if factors[factor].inverse.is_none());
            }
            Node::SolveRight { .. } | Node::InvOfProduct { .. } => return false,
        }
    }
}

/// The plan as text. `top` is whether the text is the whole expression
/// (a product on the right is parenthesized when it is not).
fn describe(n: &Node, factors: &[ChainFactor], top: bool) -> String {
    let group = |s: String| if top { s } else { format!("({s})") };
    match n {
        Node::Leaf { factor, form } => {
            let f = &factors[*factor];
            match (form, &f.inverse) {
                (LeafForm::Transposed, Some(inv)) => format!("{}ᵀ", inv.operand_atom),
                (LeafForm::Deferred, Some(inv)) => format!("{}⁻¹", inv.operand_atom),
                _ => f.atom.clone(),
            }
        }
        Node::Mul { lhs, rhs } => {
            // `@` associates to the left: only a right operand that is
            // itself a product needs parentheses.
            let l = describe(lhs, factors, true);
            let r = match **rhs {
                Node::Mul { .. } => describe(rhs, factors, false),
                _ => describe(rhs, factors, true),
            };
            group(format!("{l} @ {r}"))
        }
        Node::Solve { method, inv, rhs } => {
            let a = &factors[*inv]
                .inverse
                .as_ref()
                .map_or(String::new(), |i| i.operand_atom.clone());
            format!(
                "solve[{}]({}, {})",
                method.mnemonic(),
                a,
                describe(rhs, factors, true)
            )
        }
        Node::SolveRight { method, lhs, inv } => {
            let a = &factors[*inv]
                .inverse
                .as_ref()
                .map_or(String::new(), |i| i.operand_atom.clone());
            format!(
                "solve.right[{}]({}, {})",
                method.mnemonic(),
                describe(lhs, factors, true),
                a
            )
        }
        Node::InvOfProduct { left, right } => {
            let name = |t: usize| {
                factors[t]
                    .inverse
                    .as_ref()
                    .map_or(String::new(), |i| i.operand_atom.clone())
            };
            format!("({} @ {})⁻¹", name(*right), name(*left))
        }
    }
}

fn reasons(root: &Node, factors: &[ChainFactor]) -> Vec<String> {
    let mut out = Vec::new();
    fn walk(n: &Node, factors: &[ChainFactor], out: &mut Vec<String>) {
        let operand = |t: usize| {
            factors[t]
                .inverse
                .as_ref()
                .map_or(String::new(), |i| i.operand_atom.clone())
        };
        match n {
            Node::Leaf { factor, form } => {
                if *form == LeafForm::Transposed {
                    let q = operand(*factor);
                    out.push(format!(
                        "{q} is Orthogonal (Qᵀ·Q = Q·Qᵀ = I), so {q}⁻¹ is {q}ᵀ, exactly by the definition of the tag; the transpose is not materialized"
                    ));
                }
            }
            Node::Mul { lhs, rhs } => {
                walk(lhs, factors, out);
                walk(rhs, factors, out);
            }
            Node::Solve { method, inv, rhs } => {
                let a = operand(*inv);
                out.push(format!(
                    "{}; {a}⁻¹ is never formed",
                    method_reason(*method, &a)
                ));
                walk(rhs, factors, out);
            }
            Node::SolveRight { method, lhs, inv } => {
                let a = operand(*inv);
                out.push(format!(
                    "{}, applied transposed for `x @ {a}⁻¹`; {a}⁻¹ is never formed",
                    method_reason(*method, &a)
                ));
                walk(lhs, factors, out);
            }
            Node::InvOfProduct { left, right } => {
                let (a, b) = (operand(*left), operand(*right));
                out.push(format!(
                    "{a}⁻¹ @ {b}⁻¹ is ({b} @ {a})⁻¹: one product and one inverse instead of two inverses and a product"
                ));
            }
        }
    }
    walk(root, factors, &mut out);
    out
}

/// How the plan's result can differ from the written form's.
fn rounding(root: &Node, reassociated: bool) -> Vec<String> {
    fn kinds(n: &Node, solves: &mut bool, transposes: &mut bool, products: &mut bool) {
        match n {
            Node::Leaf { form, .. } => *transposes |= *form == LeafForm::Transposed,
            Node::Mul { lhs, rhs } => {
                kinds(lhs, solves, transposes, products);
                kinds(rhs, solves, transposes, products);
            }
            Node::Solve { rhs, .. } => {
                *solves = true;
                kinds(rhs, solves, transposes, products);
            }
            Node::SolveRight { lhs, .. } => {
                *solves = true;
                kinds(lhs, solves, transposes, products);
            }
            Node::InvOfProduct { .. } => *products = true,
        }
    }
    let (mut solves, mut transposes, mut products) = (false, false, false);
    kinds(root, &mut solves, &mut transposes, &mut products);
    let mut out = Vec::new();
    if reassociated {
        out.push(
            "products in another order round differently, within the error bound of any association"
                .to_string(),
        );
    }
    if solves {
        out.push(
            "solving rounds differently from forming the inverse and multiplying, and is at least as accurate"
                .to_string(),
        );
    }
    if transposes {
        out.push(
            "the transpose is exact, where a computed inverse carries rounding error, so the last digits can differ"
                .to_string(),
        );
    }
    if products {
        out.push(
            "one inverse of a product rounds differently from two inverses and a product"
                .to_string(),
        );
    }
    out
}

/// The shape of `a @ b` for vectors and matrices whose inner dimensions
/// already agree: `Matrix[m,k] @ Matrix[k,n]` is `Matrix[m,n]`, a matrix
/// and a vector give a vector, two vectors a scalar. `None` for anything
/// else (a scalar, a tensor).
pub fn product_shape(a: &Shape, b: &Shape) -> Option<Shape> {
    match (a.dims(), b.dims()) {
        ([m, _], [_, n]) => Some(Shape::matrix(m.clone(), n.clone())),
        ([m, _], [_]) => Some(Shape::vector(m.clone())),
        ([_], [_, n]) => Some(Shape::vector(n.clone())),
        ([_], [_]) => Some(Shape::scalar()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::ShapeUnifier;

    fn factor(name: &str, shape: Shape, tags: TagSet) -> ChainFactor {
        ChainFactor {
            span: Span::new(0, 0),
            text: name.to_string(),
            atom: name.to_string(),
            dim: Dimension::DIMENSIONLESS,
            shape,
            tags,
            inverse: None,
        }
    }

    fn inv_factor(name: &str, shape: Shape, tags: TagSet) -> ChainFactor {
        let mut f = factor(&format!("inv({name})"), shape, tags);
        f.atom = f.text.clone();
        f.inverse = Some(InverseFactor {
            operand_span: Span::new(0, 0),
            operand_atom: name.to_string(),
            tags,
        });
        f
    }

    fn sq(n: impl Into<ShapeDim> + Clone) -> Shape {
        Shape::matrix(n.clone(), n)
    }

    #[test]
    fn a_matrix_chain_ending_in_a_vector_goes_right_to_left() {
        let n = ShapeDim::Known(500);
        let f = vec![
            factor("A", sq(n.clone()), TagSet::EMPTY),
            factor("B", sq(n.clone()), TagSet::EMPTY),
            factor("C", sq(n.clone()), TagSet::EMPTY),
            factor("v", Shape::vector(n), TagSet::EMPTY),
        ];
        let d = plan_chain(&f).expect("a cheaper order exists");
        assert_eq!(d.text, "A @ (B @ (C @ v))");
        assert!(d.reassociated);
        // ((A B) C) v: 2·500³·2 + 2·500²; A (B (C v)): 3·2·500².
        assert_eq!(d.before.value(), Some(500_000_000 + 500_000));
        assert_eq!(d.after.value(), Some(1_500_000));
    }

    #[test]
    fn symbolic_sizes_are_compared_for_every_size() {
        let mut u = ShapeUnifier::new();
        let n = ShapeDim::Var(u.fresh("N"));
        let f = vec![
            factor("A", sq(n.clone()), TagSet::EMPTY),
            factor("B", sq(n.clone()), TagSet::EMPTY),
            factor("v", Shape::vector(n), TagSet::EMPTY),
        ];
        let d = plan_chain(&f).expect("provably cheaper");
        assert_eq!(d.text, "A @ (B @ v)");
        assert_eq!(d.before.to_string(), "2·N³ + 2·N²");
        assert_eq!(d.after.to_string(), "4·N²");
    }

    #[test]
    fn the_written_order_stays_when_it_is_best_or_unclear() {
        // (1x100 @ 100x100) @ 100x100: already left to right.
        let f = vec![
            factor("u", Shape::vector(100u64), TagSet::EMPTY),
            factor("A", sq(100u64), TagSet::EMPTY),
            factor("B", sq(100u64), TagSet::EMPTY),
        ];
        assert!(plan_chain(&f).is_none());
        // N x N @ N x M @ M x K: which order wins depends on N, M and K.
        let mut u = ShapeUnifier::new();
        let (n, m, k) = (
            ShapeDim::Var(u.fresh("N")),
            ShapeDim::Var(u.fresh("M")),
            ShapeDim::Var(u.fresh("K")),
        );
        let f = vec![
            factor("A", Shape::matrix(n.clone(), n.clone()), TagSet::EMPTY),
            factor("B", Shape::matrix(n, m.clone()), TagSet::EMPTY),
            factor("C", Shape::matrix(m, k), TagSet::EMPTY),
        ];
        assert!(plan_chain(&f).is_none());
    }

    #[test]
    fn known_sizes_find_the_classic_optimum() {
        // CLRS: 10x100, 100x5, 5x50: (A B) C costs 7500 multiplies,
        // A (B C) 75000.
        let f = vec![
            factor("A", Shape::matrix(10u64, 100u64), TagSet::EMPTY),
            factor("B", Shape::matrix(100u64, 5u64), TagSet::EMPTY),
            factor("C", Shape::matrix(5u64, 50u64), TagSet::EMPTY),
        ];
        // Already optimal as written.
        assert!(plan_chain(&f).is_none());
        let f = vec![
            factor("A", Shape::matrix(50u64, 5u64), TagSet::EMPTY),
            factor("B", Shape::matrix(5u64, 100u64), TagSet::EMPTY),
            factor("C", Shape::matrix(100u64, 10u64), TagSet::EMPTY),
        ];
        let d = plan_chain(&f).expect("A (B C) is cheaper here");
        assert_eq!(d.text, "A @ (B @ C)");
    }

    #[test]
    fn an_inverse_next_to_a_vector_is_a_solve() {
        let n = ShapeDim::Known(100);
        let f = vec![
            inv_factor("A", sq(n.clone()), TagSet::EMPTY),
            factor("b", Shape::vector(n), TagSet::EMPTY),
        ];
        let d = plan_chain(&f).unwrap();
        assert_eq!(d.text, "solve[lu](A, b)");
        assert!(!d.reassociated);
        // inverse 4n³ + product 2n² against 2/3 n³ + 2n².
        assert_eq!(d.before.value(), Some(4_020_000));
        assert_eq!(d.after.value(), Some(686_667));
    }

    #[test]
    fn the_method_follows_the_tags() {
        let n = ShapeDim::Known(8);
        let b = factor("b", Shape::vector(n.clone()), TagSet::EMPTY);
        let text = |tags: TagSet| {
            plan_chain(&[inv_factor("A", sq(n.clone()), tags), b.clone()]).map(|d| d.text)
        };
        assert_eq!(
            text(TagSet::of(Tag::Diagonal)).unwrap(),
            "solve[diagonal](A, b)"
        );
        assert_eq!(text(TagSet::of(Tag::Symmetric)).unwrap(), "solve[lu](A, b)");
        assert_eq!(text(TagSet::of(Tag::Orthogonal)).unwrap(), "Aᵀ @ b");
        // PositiveDefinite is the checker's own Cholesky rewrite.
        assert!(text(TagSet::of(Tag::PositiveDefinite)).is_none());
    }

    #[test]
    fn a_vector_on_the_left_of_an_inverse_is_a_transposed_solve() {
        let n = ShapeDim::Known(50);
        let f = vec![
            factor("x", Shape::vector(n.clone()), TagSet::EMPTY),
            inv_factor("A", sq(n), TagSet::EMPTY),
        ];
        let d = plan_chain(&f).unwrap();
        assert_eq!(d.text, "solve.right[lu](x, A)");
    }

    #[test]
    fn a_product_of_inverses_is_the_inverse_of_the_reversed_product() {
        let n = ShapeDim::Known(50);
        let f = vec![
            inv_factor("A", sq(n.clone()), TagSet::EMPTY),
            inv_factor("B", sq(n), TagSet::EMPTY),
        ];
        let d = plan_chain(&f).unwrap();
        assert_eq!(d.text, "(B @ A)⁻¹");
        assert!(d.after.provably_less(&d.before));
    }

    #[test]
    fn inverses_and_a_vector_solve_right_to_left() {
        let n = ShapeDim::Known(50);
        let f = vec![
            inv_factor("A", sq(n.clone()), TagSet::EMPTY),
            inv_factor("B", sq(n.clone()), TagSet::EMPTY),
            factor("b", Shape::vector(n), TagSet::EMPTY),
        ];
        let d = plan_chain(&f).unwrap();
        assert_eq!(d.text, "solve[lu](A, solve[lu](B, b))");
        assert!(d.reassociated);
    }

    #[test]
    fn a_vector_in_the_middle_is_never_reordered() {
        let n = ShapeDim::Known(50);
        let f = vec![
            factor("A", sq(n.clone()), TagSet::EMPTY),
            factor("v", Shape::vector(n.clone()), TagSet::EMPTY),
            factor("w", Shape::vector(n), TagSet::EMPTY),
        ];
        assert!(plan_chain(&f).is_none());
    }
}
