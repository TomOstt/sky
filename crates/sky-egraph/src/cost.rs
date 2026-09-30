//! The extraction cost: the static cost model of `sky_ir::cost` applied to
//! e-nodes.
//!
//! # Where the prices come from
//!
//! An e-node's price is not a table of this crate's own. It is *asked of
//! `sky_ir::cost`*: the node is made into a one-operation function whose
//! parameters have the operand types and whose result has the node's type,
//! and [`sky_ir::cost::analyze`] says what that operation costs, as
//! polynomials in the symbolic sizes: flops, bytes moved (read plus
//! written) and allocations. So a `matmul` of an `(n, k)` and a `(k, m)`
//! matrix costs `2·n·k·m` flops, a scalar `exp` costs 20, an elementwise
//! array `+` costs one flop per entry and moves three arrays, and a
//! transpose is a copy, exactly as the performance contracts count them. A
//! change to the model changes the e-graph's choices with it.
//!
//! # One scalar
//!
//! Extraction needs a total order, and polynomials in symbolic sizes are
//! only partly ordered, so a price is weighed into one number:
//!
//! ```text
//! weighted = 8·flops + bytes + 8·ALLOC_FLOPS·allocations      (eighths of a flop)
//! ```
//!
//! that is, a flop counts 1, moving one `f64` counts 1 (`8` bytes at `1/8`
//! each: the model of a memory-bound loop, where an entry read or written
//! costs about what an add on it does), and creating an array counts
//! [`ALLOC_FLOPS`] (64), the rough cost of an allocator call. Scalars live in
//! registers and have no bytes or allocations, so a scalar operation is its
//! flops. Big array operations therefore dominate small scalar ones, and
//! creating a temporary array is not free even when no flop is saved.
//!
//! **Symbolic sizes.** A shape with a size variable (`Vector[N]`) is
//! evaluated with every variable equal to [`DEFAULT_SIZE`], 64: large enough
//! that an `O(N³)` form loses to an `O(N²)` one and an array pass loses to a
//! scalar op, small enough that constants still matter. Where two forms can
//! be compared without a default, the explanation says so: it compares the
//! polynomials by dominance (`sky_types::poly`: `p <= q` for *all* sizes at
//! least 1 when every coefficient of `q - p` in `1 + y` is non-negative) and
//! reports "for every size" when that holds for flops, bytes and
//! allocations at once.
//!
//! **Tie-break.** The cost of an extracted tree is the sum over its nodes (a
//! subterm used twice is counted twice, as before). When two forms have the
//! same weighted cost the extractor prefers the one with the smaller
//! *latency* count of this crate's earlier model (add 1, mul 2, div 8,
//! transcendental 24, matmul 16, ...), so scalar code whose prices tie in the
//! flop model (a `+` and a `*` are both one flop) keeps choosing the cheaper
//! instruction, and the choice never depends on node order for that reason.

use std::cell::RefCell;
use std::collections::HashMap;

use egg::{Id, Language};
use sky_diag::Span;
use sky_ir::cost::{analyze, Poly};
use sky_ir::{Function, FunctionBuilder, MathFn, Module, NumberType, OpKind, Type};
use sky_types::Shape;
use sky_units::Rational;

use crate::analysis::{derive_ty, Graph};
use crate::lang::{Op, SkyNode};

/// The size every symbolic dimension takes when a price is turned into a
/// number.
pub const DEFAULT_SIZE: f64 = 64.0;

/// The price of creating an array, in flops.
pub const ALLOC_FLOPS: u64 = 64;

/// What one operation costs: the three measures the weight uses, as
/// polynomials in the sizes.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct OpCost {
    pub flops: Poly,
    /// Bytes read plus bytes written.
    pub bytes: Poly,
    pub allocs: Poly,
}

impl OpCost {
    pub fn add(&self, other: &OpCost) -> Option<OpCost> {
        Some(OpCost {
            flops: self.flops.checked_add(&other.flops)?,
            bytes: self.bytes.checked_add(&other.bytes)?,
            allocs: self.allocs.checked_add(&other.allocs)?,
        })
    }

    /// The weighted scalar at `n` (every size variable equal to `n`), in
    /// eighths of a flop.
    pub fn weighted_at(&self, n: f64) -> u64 {
        let w = 8.0 * self.flops.weight_at(n)
            + self.bytes.weight_at(n)
            + 8.0 * ALLOC_FLOPS as f64 * self.allocs.weight_at(n);
        if w.is_finite() && w > 0.0 {
            w.round().min(u64::MAX as f64) as u64
        } else {
            0
        }
    }

    /// The weighted scalar at the default size.
    pub fn weighted(&self) -> u64 {
        self.weighted_at(DEFAULT_SIZE)
    }

    /// `self <= other` in all three measures for every size at least 1.
    pub fn dominated_by(&self, other: &OpCost) -> bool {
        self.flops.le_for_all_sizes(&other.flops)
            && self.bytes.le_for_all_sizes(&other.bytes)
            && self.allocs.le_for_all_sizes(&other.allocs)
    }
}

thread_local! {
    static PROBES: RefCell<HashMap<String, Option<OpCost>>> = RefCell::new(HashMap::new());
}

/// The cost `sky_ir::cost` gives to one operation of kind `kind` on operands
/// of these types producing `result`, or `None` if the model cannot state it
/// (an operand of unknown shape, a polynomial too large).
pub fn op_cost(kind: &OpKind, operands: &[NumberType], result: &NumberType) -> Option<OpCost> {
    // Only shapes matter to a price, so only shapes are in the key.
    let key = format!(
        "{kind:?}|{:?}|{:?}",
        operands
            .iter()
            .map(|t| t.shape.dims().to_vec())
            .collect::<Vec<_>>(),
        result.shape.dims()
    );
    if let Some(hit) = PROBES.with(|p| p.borrow().get(&key).cloned()) {
        return hit;
    }
    let cost = probe(kind, operands, result);
    PROBES.with(|p| p.borrow_mut().insert(key, cost.clone()));
    cost
}

fn probe(kind: &OpKind, operands: &[NumberType], result: &NumberType) -> Option<OpCost> {
    let span = Span::new(0, 1);
    let params = operands.iter().cloned().map(Type::Number).collect();
    let mut b = FunctionBuilder::new("probe", params, Vec::new(), span);
    let args = b.params();
    b.op1(kind.clone(), args, Type::Number(result.clone()), span);
    let f: Function = b.finish();
    let module = Module {
        functions: vec![f],
        ..Module::default()
    };
    let analysis = analyze(&module);
    let c = &analysis.get("probe")?.cost;
    let bounded = [&c.flops, &c.bytes_read, &c.bytes_written, &c.allocations]
        .iter()
        .all(|a| a.unbounded.is_empty());
    bounded.then_some(())?;
    Some(OpCost {
        flops: c.flops.hi.clone(),
        bytes: c.bytes_read.hi.checked_add(&c.bytes_written.hi)?,
        allocs: c.allocations.hi.clone(),
    })
}

/// The SIR operation an e-node stands for, if it is arithmetic (constants,
/// leaves and `Roots` cost nothing).
fn op_kind(op: Op, exponent: Option<Rational>) -> Option<OpKind> {
    Some(match op {
        Op::Lit(_) | Op::Leaf(_) | Op::Roots => return None,
        Op::Add => OpKind::Add,
        Op::Sub => OpKind::Sub,
        Op::Mul => OpKind::Mul,
        Op::Div => OpKind::Div,
        Op::Neg => OpKind::Neg,
        Op::Pow => OpKind::Pow,
        Op::Inv => OpKind::Inverse,
        Op::MatMul => OpKind::MatMul,
        Op::Transpose => OpKind::Transpose,
        Op::Math(_) => OpKind::Math(op.math_fn()?),
        Op::PowC => OpKind::PowConst {
            exponent: exponent?,
        },
    })
}

/// The price of `node`, given the analysis of its operands' e-classes.
/// `None` for a node with no price (a constant or a leaf) or one the model
/// cannot state; the caller treats the second like a scalar operation.
pub fn node_cost(eg: &Graph, node: &SkyNode, exponents: &HashMap<Id, Rational>) -> Option<OpCost> {
    let exponent = (node.op == Op::PowC)
        .then(|| node.args.get(1).and_then(|e| exponents.get(e)).copied())
        .flatten();
    let kind = op_kind(node.op, exponent)?;
    // A `PowConst`'s exponent is an attribute, not an operand.
    let value_args: &[Id] = if node.op == Op::PowC {
        &node.args[..1]
    } else {
        &node.args
    };
    let dimensionless = || NumberType::scalar(sky_units::Dimension::DIMENSIONLESS);
    let operands: Vec<NumberType> = value_args
        .iter()
        .map(|c| eg[*c].data.ty.clone().unwrap_or_else(dimensionless))
        .collect();
    let infos: Vec<&crate::analysis::Info> = node.args.iter().map(|c| &eg[*c].data).collect();
    let result = derive_ty(node.op, &infos)
        .or_else(|| operands.first().cloned())
        .unwrap_or_else(dimensionless);
    op_cost(&kind, &operands, &result)
}

/// The cost of a matrix product of these operand types, in flops: what
/// `sky_ir::cost` says a `matmul` of them costs.
pub fn matmul_cost(a: &NumberType, b: &NumberType, result: &NumberType) -> Option<OpCost> {
    op_cost(&OpKind::MatMul, &[a.clone(), b.clone()], result)
}

/// The result type of `a @ b`: `(n, k) @ (k, m)` is `(n, m)`, a vector
/// operand loses its axis, and two vectors give a number. `None` if the
/// inner sizes do not match or the dimensions overflow.
pub fn matmul_ty(a: &NumberType, b: &NumberType) -> Option<NumberType> {
    let (x, y) = (a.shape.dims(), b.shape.dims());
    let dims = match (x, y) {
        ([n, k1], [k2, m]) if k1 == k2 => vec![n.clone(), m.clone()],
        ([n, k1], [k2]) if k1 == k2 => vec![n.clone()],
        ([k1], [k2, m]) if k1 == k2 => vec![m.clone()],
        ([k1], [k2]) if k1 == k2 => Vec::new(),
        _ => return None,
    };
    Some(NumberType {
        dim: a.dim.checked_mul(b.dim).ok()?,
        shape: Shape::new(dims),
        tags: sky_types::TagSet::EMPTY,
        repr: sky_ir::Repr::Unchosen,
    })
}

// ---------------------------------------------------------------------
// The extraction cost function
// ---------------------------------------------------------------------

/// The cost of an extracted term: the weighted cost of the static cost model
/// first, and the latency count of the earlier model to break ties. Ordered
/// lexicographically.
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug, Default)]
pub struct Cost {
    pub weighted: u64,
    pub latency: u64,
}

impl Cost {
    fn plus(self, o: Cost) -> Cost {
        Cost {
            weighted: self.weighted.saturating_add(o.weighted),
            latency: self.latency.saturating_add(o.latency),
        }
    }
}

/// The extraction cost function. See the module documentation.
pub struct CostModel<'a> {
    eg: &'a Graph,
    exponents: HashMap<Id, Rational>,
    memo: HashMap<SkyNode, Cost>,
}

impl<'a> CostModel<'a> {
    pub fn new(eg: &'a Graph) -> Self {
        let exponents = eg
            .classes()
            .filter_map(|c| c.data.rat.map(|q| (c.id, q)))
            .collect();
        CostModel {
            eg,
            exponents,
            memo: HashMap::new(),
        }
    }

    fn own(&mut self, node: &SkyNode) -> Cost {
        if let Some(c) = self.memo.get(node) {
            return *c;
        }
        let latency = latency(node, &self.exponents);
        let weighted = match node.op {
            Op::Lit(_) | Op::Leaf(_) | Op::Roots => 0,
            _ => match node_cost(self.eg, node, &self.exponents) {
                Some(c) => c.weighted(),
                // The model could not price it: a scalar operation of the
                // old table, in eighths of a flop.
                None => 8 * latency.min(20),
            },
        };
        let c = Cost { weighted, latency };
        self.memo.insert(node.clone(), c);
        c
    }
}

impl egg::CostFunction<SkyNode> for CostModel<'_> {
    type Cost = Cost;

    fn cost<C>(&mut self, node: &SkyNode, mut costs: C) -> Cost
    where
        C: FnMut(Id) -> Cost,
    {
        let own = self.own(node);
        node.children()
            .iter()
            .fold(own, |acc, c| acc.plus(costs(*c)))
    }
}

/// Instruction-latency weights of the earlier cost model, kept to break ties.
///
/// | operation | weight |
/// |---|---|
/// | constant, leaf, `roots` | 0 |
/// | add, sub, neg, abs, min, max, floor, ceil, round, transpose | 1 |
/// | mul | 2 |
/// | `x^n`, integer `n` | 2 per multiplication of square-and-multiply, plus 8 if `n < 0` |
/// | `x^r`, other `r` | 30 |
/// | div, inverse | 8 |
/// | sqrt | 6 |
/// | hypot | 10 |
/// | cbrt | 20 |
/// | exp, log, sin, cos, ... | 24 |
/// | atan2 | 28 |
/// | `pow` (computed exponent) | 32 |
/// | matmul | 16 |
fn latency(node: &SkyNode, exponents: &HashMap<Id, Rational>) -> u64 {
    match node.op {
        Op::Lit(_) | Op::Leaf(_) | Op::Roots => 0,
        Op::Add | Op::Sub | Op::Neg | Op::Transpose => 1,
        Op::Mul => 2,
        Op::Div | Op::Inv => 8,
        Op::MatMul => 16,
        Op::Pow => 32,
        Op::PowC => match node.args.get(1).and_then(|e| exponents.get(e)) {
            Some(q) if q.is_integer() => pow_int_latency(q.numer()),
            _ => 30,
        },
        Op::Math(_) => match node.op.math_fn() {
            Some(
                MathFn::Abs
                | MathFn::Min
                | MathFn::Max
                | MathFn::Floor
                | MathFn::Ceil
                | MathFn::Round,
            ) => 1,
            Some(MathFn::Sqrt) => 6,
            Some(MathFn::Hypot) => 10,
            Some(MathFn::Cbrt) => 20,
            Some(MathFn::Atan2) => 28,
            _ => 24,
        },
    }
}

fn pow_int_latency(n: i64) -> u64 {
    let m = n.unsigned_abs();
    if m <= 1 {
        return if n < 0 { 8 } else { 0 };
    }
    let squarings = u64::from(63 - m.leading_zeros());
    let extra = u64::from(m.count_ones()) - 1;
    2 * (squarings + extra) + if n < 0 { 8 } else { 0 }
}

// ---------------------------------------------------------------------
// Costs of whole terms, for explanations
// ---------------------------------------------------------------------

/// The tree cost of the term rooted at `root`, choosing each class's node
/// with `choose`: the sum of the node prices, a shared subterm counted once
/// per use. `None` if the term has more than `limit` distinct classes or a
/// polynomial overflows.
pub fn term_cost(
    eg: &Graph,
    root: Id,
    choose: &dyn Fn(Id) -> SkyNode,
    limit: usize,
) -> Option<OpCost> {
    let exponents: HashMap<Id, Rational> = eg
        .classes()
        .filter_map(|c| c.data.rat.map(|q| (c.id, q)))
        .collect();
    let mut done: HashMap<Id, OpCost> = HashMap::new();
    let mut stack = vec![(eg.find(root), false)];
    while let Some((c, expanded)) = stack.pop() {
        if done.contains_key(&c) {
            continue;
        }
        let node = choose(c);
        if !expanded {
            if done.len() + stack.len() > limit {
                return None;
            }
            stack.push((c, true));
            for k in node.children() {
                let k = eg.find(*k);
                if !done.contains_key(&k) {
                    stack.push((k, false));
                }
            }
            continue;
        }
        let mut total = node_cost(eg, &node, &exponents).unwrap_or_default();
        for k in node.children() {
            total = total.add(done.get(&eg.find(*k))?)?;
        }
        done.insert(c, total);
    }
    done.remove(&eg.find(root))
}

/// A polynomial as `2·N³ + 2·N²`, highest degree first.
pub fn show_poly(p: &Poly) -> String {
    let terms = p.terms();
    if terms.is_empty() {
        return "0".into();
    }
    let mut out = String::new();
    for (i, (m, c)) in terms.iter().enumerate() {
        out.push_str(if i == 0 { "" } else { " + " });
        let vars: Vec<String> = m
            .powers()
            .map(|(name, e)| format!("{name}{}", superscript(e)))
            .collect();
        let coef = if c.is_integer() {
            format!("{}", c.numer())
        } else {
            format!("{}/{}", c.numer(), c.denom())
        };
        if vars.is_empty() {
            out.push_str(&coef);
        } else if coef == "1" {
            out.push_str(&vars.join("·"));
        } else {
            out.push_str(&format!("{coef}·{}", vars.join("·")));
        }
    }
    out
}

/// The tree cost of the program's own term at `expr[root]`: what
/// [`term_cost`] gives the term before any rewrite. `eg_of` maps each
/// `expr` index to its e-class (for the operand types). `None` past `limit`
/// nodes or on overflow.
pub fn program_cost(
    eg: &Graph,
    expr: &egg::RecExpr<SkyNode>,
    eg_of: &[Id],
    root: usize,
    limit: usize,
) -> Option<OpCost> {
    if root >= limit {
        return None;
    }
    let exponents: HashMap<Id, Rational> = eg
        .classes()
        .filter_map(|c| c.data.rat.map(|q| (c.id, q)))
        .collect();
    let mut costs: Vec<OpCost> = Vec::with_capacity(root + 1);
    for n in expr.as_ref().iter().take(root + 1) {
        let mapped = n.clone().map_children(|c| eg_of[usize::from(c)]);
        let mut total = node_cost(eg, &mapped, &exponents).unwrap_or_default();
        for k in n.children() {
            total = total.add(&costs[usize::from(*k)])?;
        }
        costs.push(total);
    }
    costs.pop()
}

/// The sentence of an `EG-COST-001` explanation: the two forms and what the
/// static cost model says each costs, as polynomials.
pub fn explain_choice(after: &str, before: &str, new: &OpCost, old: &OpCost) -> String {
    let mut out = format!(
        "chose `{after}` over `{before}`: {} vs {} flops",
        show_poly(&new.flops),
        show_poly(&old.flops)
    );
    if new.bytes != old.bytes {
        out.push_str(&format!(
            ", {} vs {} bytes",
            show_poly(&new.bytes),
            show_poly(&old.bytes)
        ));
    }
    if new.allocs != old.allocs {
        out.push_str(&format!(
            ", {} vs {} allocations",
            show_poly(&new.allocs),
            show_poly(&old.allocs)
        ));
    }
    let sized = [
        &new.flops,
        &new.bytes,
        &new.allocs,
        &old.flops,
        &old.bytes,
        &old.allocs,
    ]
    .iter()
    .any(|p| !p.vars().is_empty());
    let eighths = |w: u64| format!("{}", w as f64 / 8.0);
    if !sized {
        out.push_str(&format!(
            " (weighted cost {} vs {})",
            eighths(new.weighted()),
            eighths(old.weighted())
        ));
    } else if new.dominated_by(old) {
        out.push_str(" (cheaper for every size)");
    } else {
        out.push_str(&format!(
            " (at the default size N={DEFAULT_SIZE}: weighted cost {} vs {})",
            eighths(new.weighted()),
            eighths(old.weighted())
        ));
    }
    out
}

fn superscript(e: u32) -> String {
    if e == 1 {
        return String::new();
    }
    e.to_string()
        .chars()
        .map(|d| match d {
            '0' => '⁰',
            '1' => '¹',
            '2' => '²',
            '3' => '³',
            '4' => '⁴',
            '5' => '⁵',
            '6' => '⁶',
            '7' => '⁷',
            '8' => '⁸',
            _ => '⁹',
        })
        .collect()
}
