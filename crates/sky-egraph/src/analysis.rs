//! What the e-graph knows about each e-class: its SIR type (dimension,
//! shape, tags), whether it is a known constant, whether evaluating it can
//! fail, its sign, and where it came from in the source.
//!
//! This is the "facts" side of identity-aware rewriting: a rule's side
//! condition is a question to this analysis. The analysis only ever holds
//! facts that are true of *every* member of a class, and members of a class
//! are equal, so facts are merged by combining what either side proves.

use std::collections::{HashMap, HashSet};

use egg::{Analysis, DidMerge, EGraph, Id};
use sky_diag::Span;
use sky_ir::{MathFn, NumberType, Repr};
use sky_types::{preserved_by_inverse, preserved_by_transpose, Shape, TagSet};
use sky_units::{Dimension, Rational};

use crate::lang::{Lit, LitVal, Op, SkyNode};
use crate::RewriteContract;

pub type Graph = EGraph<SkyNode, SkyAnalysis>;

type Pending = Option<(Option<NumberType>, Option<Span>, Option<NodeFacts>)>;

/// What is known about the sign of a value. Ordered by strength.
///
/// `NonNeg` means the `f64` is `>= +0.0` and is never `-0.0` (so adding
/// `+0.0` to it is exact). `Pos` means *positive over the reals*: the `f64`
/// may still have underflowed to `+0.0`, so only rules that are real-only
/// anyway may rely on it. `Pos` implies `NonNeg`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Sign {
    Unknown,
    NonNeg,
    Pos,
}

/// The analysis data of an e-class.
#[derive(Clone, PartialEq, Debug)]
pub struct Info {
    /// The SIR type of every member; `None` if it could not be computed
    /// (a dimension exponent overflow, a matrix product).
    pub ty: Option<NumberType>,
    /// The constant's value, if the class contains a constant.
    pub value: Option<f64>,
    /// The constant's value as a rational, if it was written as one.
    pub rat: Option<Rational>,
    /// Evaluating the class's cheapest form can never fail, given that the
    /// leaves are values already computed. Used to decide when a rewrite
    /// may drop a subterm without changing failure behaviour.
    pub infallible: bool,
    pub sign: Sign,
    /// The class is provably not zero as an `f64` (a nonzero constant, or a
    /// fact from the exactness analysis). `Sign::Pos` does not imply it: a
    /// positive real can underflow to `+0.0`.
    pub nonzero: bool,
    /// The join of the spans of the operations merged into the class.
    pub span: Option<Span>,
    /// The first node inserted into the class and its insertion number; it
    /// is the form nearest to what the program wrote and is what
    /// explanations print.
    pub orig: (u64, SkyNode),
}

/// What is known about a leaf.
#[derive(Clone, Debug)]
pub struct LeafInfo {
    pub ty: NumberType,
    pub span: Option<Span>,
    /// Printed in explanations.
    pub name: String,
}

/// What the exactness analysis (`sky_ir::exact`) proved about the operation
/// a node was made from, on top of what the e-graph derives from the node's
/// shape. Every field is a fact about the `f64` the interpreter computes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct NodeFacts {
    /// The result is strictly greater than zero.
    pub positive: bool,
    /// The result is not zero.
    pub nonzero: bool,
    /// The operation cannot fail with these operands (division by a nonzero
    /// constant, `sqrt` of a positive one, `x * 0.5`, ...), although its
    /// kind can.
    pub infallible: bool,
}

/// A rule application that changed the e-graph, kept for explanations.
#[derive(Clone, Debug)]
pub struct Firing {
    pub rule: &'static str,
    pub root: Id,
    pub span: Option<Span>,
    pub before: String,
    pub after: String,
    pub because: String,
    /// The nodes of the right-hand side's class when the rule fired. The
    /// rule shows up in the output iff the extracted form of `root` is one
    /// of them, or a rule that does rewrote the class this one produced.
    pub rhs_nodes: Vec<SkyNode>,
    /// The nodes of the class the rule matched in, when it fired. A rule
    /// whose right-hand side is what another rule then rewrote is part of
    /// the same derivation.
    pub root_nodes: Vec<SkyNode>,
}

/// The e-class analysis and the shared state the rules use.
pub struct SkyAnalysis {
    pub contract: RewriteContract,
    pub leaves: Vec<LeafInfo>,
    /// Type, span and analysis facts for the next node added (see
    /// [`add_node`]).
    pending: Pending,
    seq: u64,
    /// Span of each node as added, with its insertion number.
    pub node_spans: HashMap<SkyNode, (u64, Span)>,
    pub firings: Vec<Firing>,
    pub seen: HashSet<(&'static str, Id)>,
    /// Pairs `(new, root)`: an operation of class `new` that a rule made
    /// below `root` fails, if it fails, exactly when and as an original
    /// operation of class `root` did (`Aᵀ + Bᵀ` becomes `(A + B)ᵀ`: the new
    /// `A + B` overflows in the same entries the old `+` did). The
    /// write-back check that keeps the order of failing operations reads
    /// them; without one, a rule that makes a new failing operation makes
    /// the block be left alone.
    pub fail_alias: Vec<(Id, Id)>,
}

impl SkyAnalysis {
    pub fn new(contract: RewriteContract, leaves: Vec<LeafInfo>) -> Self {
        SkyAnalysis {
            contract,
            leaves,
            pending: None,
            seq: 0,
            node_spans: HashMap::new(),
            firings: Vec::new(),
            seen: HashSet::new(),
            fail_alias: Vec::new(),
        }
    }
}

/// Adds `node` with an explicit type (for nodes that come from the IR,
/// whose types the checker computed) and span. A node that already exists
/// keeps its class and its data.
pub fn add_node(eg: &mut Graph, node: SkyNode, ty: Option<NumberType>, span: Option<Span>) -> Id {
    add_node_with_facts(eg, node, ty, span, None)
}

/// Like [`add_node`], for a node made from an operation of the program, with
/// what the exactness analysis proved about it.
pub fn add_node_with_facts(
    eg: &mut Graph,
    node: SkyNode,
    ty: Option<NumberType>,
    span: Option<Span>,
    facts: Option<NodeFacts>,
) -> Id {
    eg.analysis.pending = Some((ty, span, facts));
    let id = eg.add(node);
    eg.analysis.pending = None;
    id
}

impl Analysis<SkyNode> for SkyAnalysis {
    type Data = Info;

    fn make(eg: &mut Graph, node: &SkyNode, _id: Id) -> Info {
        let (given_ty, span, facts) = eg.analysis.pending.take().unwrap_or((None, None, None));
        let facts = facts.unwrap_or_default();
        let seq = eg.analysis.seq;
        eg.analysis.seq += 1;
        if let Some(s) = span {
            eg.analysis
                .node_spans
                .entry(node.clone())
                .or_insert((seq, s));
        }
        let kids: Vec<&Info> = node.args.iter().map(|c| &eg[*c].data).collect();

        let ty = given_ty.or_else(|| match node.op {
            Op::Leaf(i) => eg.analysis.leaves.get(i as usize).map(|l| l.ty.clone()),
            op => derive_ty(op, &kids),
        });
        let (value, rat) = match node.op {
            Op::Lit(l) => (
                Some(l.val.to_f64()),
                match l.val {
                    LitVal::Rat(q) => Some(q),
                    LitVal::Float(_) => None,
                },
            ),
            _ => (None, None),
        };
        let all_kids = kids.iter().all(|k| k.infallible);
        let infallible = match node.op {
            Op::Lit(_) | Op::Leaf(_) => true,
            Op::Roots | Op::Neg | Op::Transpose => all_kids,
            Op::Math(_) => {
                (node.op.math_fn().is_some_and(|f| !f.can_fail()) || facts.infallible) && all_kids
            }
            // Arithmetic can overflow, divide by zero or leave the reals,
            // unless the analysis proved it cannot with these operands.
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::PowC | Op::Pow | Op::Inv | Op::MatMul => {
                facts.infallible && all_kids
            }
        };
        let mut sign = derive_sign(node.op, &kids, value);
        if facts.positive {
            sign = sign.max(Sign::Pos);
        }
        let nonzero = facts.nonzero
            || match node.op {
                Op::Lit(_) => value.is_some_and(|v| v != 0.0),
                Op::Neg => kids.first().is_some_and(|k| k.nonzero),
                _ => false,
            };
        Info {
            ty,
            value,
            rat,
            infallible,
            sign,
            nonzero,
            span,
            orig: (seq, node.clone()),
        }
    }

    fn merge(&mut self, a: &mut Info, b: Info) -> DidMerge {
        let before = a.clone();
        let b_orig = b.clone();
        a.ty = match (a.ty.take(), b.ty) {
            (Some(mut x), Some(y)) => {
                // Equal values share every guaranteed property.
                x.tags = x.tags.meet(y.tags);
                debug_assert!(x.dim == y.dim && x.shape == y.shape, "merged unlike types");
                Some(x)
            }
            (x, y) => x.or(y),
        };
        a.value = a.value.or(b.value);
        a.rat = a.rat.or(b.rat);
        a.infallible |= b.infallible;
        a.sign = a.sign.max(b.sign);
        a.nonzero |= b.nonzero;
        a.span = match (a.span, b.span) {
            (Some(x), Some(y)) => Some(x.join(y)),
            (x, y) => x.or(y),
        };
        if b.orig.0 < a.orig.0 {
            a.orig = b.orig;
        }
        DidMerge(*a != before, *a != b_orig)
    }
}

pub(crate) fn derive_ty(op: Op, kids: &[&Info]) -> Option<NumberType> {
    let t = |i: usize| kids.get(i).and_then(|k| k.ty.as_ref());
    let bare = |dim: Dimension, shape: Shape, tags: TagSet| NumberType {
        dim,
        shape,
        tags,
        repr: Repr::Unchosen,
    };
    let scalar = |x: &NumberType| x.shape.rank() == 0;
    match op {
        Op::Lit(l) => Some(NumberType::scalar(l.dim())),
        Op::Leaf(_) | Op::Roots => None,
        Op::MatMul => crate::cost::matmul_ty(t(0)?, t(1)?),
        Op::Add | Op::Sub => {
            let (a, b) = (t(0)?, t(1)?);
            (a.dim == b.dim).then(|| bare(a.dim, a.shape.clone(), TagSet::EMPTY))
        }
        Op::Mul => {
            let (a, b) = (t(0)?, t(1)?);
            let dim = a.dim.checked_mul(b.dim).ok()?;
            let shape = if scalar(a) {
                b.shape.clone()
            } else if scalar(b) || a.shape == b.shape {
                // A number scales the array; two arrays of one shape
                // multiply entry by entry.
                a.shape.clone()
            } else {
                return None;
            };
            Some(bare(dim, shape, TagSet::EMPTY))
        }
        Op::Div => {
            let (a, b) = (t(0)?, t(1)?);
            let dim = a.dim.checked_div(b.dim).ok()?;
            (scalar(b) || a.shape == b.shape).then(|| bare(dim, a.shape.clone(), TagSet::EMPTY))
        }
        Op::Neg => {
            let a = t(0)?;
            Some(bare(a.dim, a.shape.clone(), TagSet::EMPTY))
        }
        Op::PowC => {
            let a = t(0)?;
            let dim = a.dim.pow_rat(kids.get(1)?.rat?).ok()?;
            Some(bare(dim, a.shape.clone(), TagSet::EMPTY))
        }
        Op::Pow => {
            let (a, b) = (t(0)?, t(1)?);
            (a.dim.is_dimensionless() && b.dim.is_dimensionless())
                .then(|| bare(a.dim, a.shape.clone(), TagSet::EMPTY))
        }
        Op::Inv => {
            let a = t(0)?;
            let dim = a.dim.inverse().ok()?;
            Some(bare(dim, a.shape.clone(), preserved_by_inverse(a.tags)))
        }
        Op::Transpose => {
            let a = t(0)?;
            let mut dims = a.shape.dims().to_vec();
            if dims.len() == 2 {
                dims.reverse();
            }
            Some(bare(
                a.dim,
                Shape::new(dims),
                preserved_by_transpose(a.tags),
            ))
        }
        Op::Math(_) => {
            let f = op.math_fn()?;
            let a = t(0)?;
            if !scalar(a) {
                return None;
            }
            let root = |n: i64| a.dim.pow_rat(Rational::new(1, n)).ok();
            let dim = match f {
                MathFn::Sqrt => root(2)?,
                MathFn::Cbrt => root(3)?,
                MathFn::Abs | MathFn::Floor | MathFn::Ceil | MathFn::Round => a.dim,
                MathFn::Sign => Dimension::DIMENSIONLESS,
                MathFn::Hypot | MathFn::Min | MathFn::Max => {
                    let b = t(1)?;
                    if b.dim != a.dim || !scalar(b) {
                        return None;
                    }
                    a.dim
                }
                MathFn::Atan2 => {
                    let b = t(1)?;
                    if b.dim != a.dim || !scalar(b) {
                        return None;
                    }
                    Dimension::DIMENSIONLESS
                }
                _ => {
                    if !a.dim.is_dimensionless() {
                        return None;
                    }
                    Dimension::DIMENSIONLESS
                }
            };
            Some(bare(dim, Shape::scalar(), TagSet::EMPTY))
        }
    }
}

fn derive_sign(op: Op, kids: &[&Info], value: Option<f64>) -> Sign {
    let s = |i: usize| kids.get(i).map_or(Sign::Unknown, |k| k.sign);
    let known = |x: Sign| x >= Sign::NonNeg;
    match op {
        Op::Lit(_) => match value {
            Some(v) if v > 0.0 => Sign::Pos,
            Some(v) if v == 0.0 && v.is_sign_positive() => Sign::NonNeg,
            _ => Sign::Unknown,
        },
        Op::Add => match (s(0), s(1)) {
            (Sign::Pos, b) | (b, Sign::Pos) if known(b) => Sign::Pos,
            (Sign::NonNeg, Sign::NonNeg) => Sign::NonNeg,
            _ => Sign::Unknown,
        },
        Op::Mul => match (s(0), s(1)) {
            (Sign::Pos, Sign::Pos) => Sign::Pos,
            (a, b) if known(a) && known(b) => Sign::NonNeg,
            _ => Sign::Unknown,
        },
        Op::Div => match (s(0), s(1)) {
            (Sign::Pos, Sign::Pos) => Sign::Pos,
            (a, b) if known(a) && known(b) => Sign::NonNeg,
            _ => Sign::Unknown,
        },
        Op::PowC => {
            let even = kids
                .get(1)
                .and_then(|k| k.rat)
                .is_some_and(|q| q.is_integer() && q.numer() % 2 == 0);
            match s(0) {
                Sign::Pos => Sign::Pos,
                b if known(b) => Sign::NonNeg,
                _ if even => Sign::NonNeg,
                _ => Sign::Unknown,
            }
        }
        Op::Math(_) => match op.math_fn() {
            Some(MathFn::Abs | MathFn::Hypot | MathFn::Round) => Sign::NonNeg,
            Some(MathFn::Exp | MathFn::Cosh) => Sign::Pos,
            Some(MathFn::Sqrt | MathFn::Cbrt | MathFn::Ceil) => s(0),
            Some(MathFn::Floor) => {
                if known(s(0)) {
                    Sign::NonNeg
                } else {
                    Sign::Unknown
                }
            }
            Some(MathFn::Min | MathFn::Max) => match (s(0), s(1)) {
                (Sign::Pos, Sign::Pos) => Sign::Pos,
                (a, b) if known(a) && known(b) => Sign::NonNeg,
                _ => Sign::Unknown,
            },
            _ => Sign::Unknown,
        },
        _ => Sign::Unknown,
    }
}

// ---------------------------------------------------------------------
// Explanations: printing terms
// ---------------------------------------------------------------------

/// A term rendered in Sky-like infix from the first-inserted node of each
/// class, so it reads like the program. Deep terms are cut with `…`.
pub fn render(eg: &Graph, id: Id) -> String {
    render_with(eg, id, &|c| eg[c].data.orig.1.clone())
}

/// [`render`] of the term in which `choose(id)` is the node of `id`: a class
/// of the e-graph (the extracted term: the best node of the class) or an
/// index of a `RecExpr` (the program's own term), as the caller's ids mean.
pub fn render_with(eg: &Graph, id: Id, choose: &dyn Fn(Id) -> SkyNode) -> String {
    render_at(eg, choose, id, 0, 5)
}

fn render_at(eg: &Graph, choose: &dyn Fn(Id) -> SkyNode, id: Id, ctx: u8, depth: u32) -> String {
    let node = &choose(id);
    if depth == 0 && !node.args.is_empty() {
        return "…".into();
    }
    let arg = |i: usize, p: u8| render_at(eg, choose, node.args[i], p, depth - 1);
    let (text, prec) = match node.op {
        Op::Lit(l) => {
            let n = match l.val {
                LitVal::Float(bits) => format!("{}", f64::from_bits(bits)),
                LitVal::Rat(q) if q.is_integer() => format!("{}", q.numer()),
                LitVal::Rat(q) => format!("{}/{}", q.numer(), q.denom()),
            };
            let n = if n.contains('/') { format!("({n})") } else { n };
            let dim = l.dim();
            if dim.is_dimensionless() {
                (n, 9)
            } else {
                (format!("{n} [{dim}]"), 0)
            }
        }
        Op::Leaf(i) => (
            eg.analysis
                .leaves
                .get(i as usize)
                .map_or_else(|| format!("%{i}"), |l| l.name.clone()),
            9,
        ),
        Op::Add => (format!("{} + {}", arg(0, 1), arg(1, 1)), 1),
        Op::Sub => (format!("{} - {}", arg(0, 1), arg(1, 2)), 1),
        Op::Mul => (format!("{} * {}", arg(0, 2), arg(1, 2)), 2),
        Op::Div => (format!("{} / {}", arg(0, 2), arg(1, 3)), 2),
        Op::Neg => (format!("-{}", arg(0, 3)), 3),
        Op::PowC | Op::Pow => (format!("{}^{}", arg(0, 5), arg(1, 5)), 4),
        Op::Inv => (format!("{}⁻¹", arg(0, 5)), 4),
        // Always parenthesized when nested, so `(A @ B) @ v` and `A @ (B @ v)`
        // are told apart (an unparenthesized chain is planned by the checker).
        Op::MatMul => (format!("{} @ {}", arg(0, 3), arg(1, 3)), 2),
        Op::Transpose => (format!("{}ᵀ", arg(0, 5)), 4),
        Op::Math(_) => {
            let name = node.op.math_fn().map_or("?", |f| f.name());
            let args: Vec<String> = (0..node.args.len()).map(|i| arg(i, 0)).collect();
            (format!("{name}({})", args.join(", ")), 9)
        }
        Op::Roots => ("roots".into(), 9),
    };
    if prec < ctx {
        format!("({text})")
    } else {
        text
    }
}

/// The literal `val` with dimension `dim`, as a node.
pub fn lit_node(x: f64, dim: Dimension) -> SkyNode {
    SkyNode::leaf(Op::Lit(Lit::float(x, dim)))
}
