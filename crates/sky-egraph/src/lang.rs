//! The e-graph language: SIR's pure expression operations, constants, and
//! opaque leaves.
//!
//! One node type, [`SkyNode`], is an [`Op`] with its child e-classes. A
//! constant carries its dimension (so `1` and `1 m` never share an
//! e-class), a leaf is an index into the leaf table of the segment being
//! rewritten (a value the rewriter does not look into: a parameter, a
//! result of an effectful op, a value of unknown type), and `Roots` is a
//! synthetic node that ties the roots of a segment's expression DAG
//! together so one `RecExpr` and one extraction cover all of them.

use egg::{Id, Language};
use sky_ir::MathFn;
use sky_units::{Dimension, Rational};

/// A constant's value, exactly as the IR spells it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum LitVal {
    /// `ConstFloat`: the bits of an `f64`.
    Float(u64),
    /// `ConstRational`: converted to `f64` as `numer as f64 / denom as f64`
    /// when a number is needed, exactly as the interpreter does.
    Rat(Rational),
}

impl LitVal {
    /// The number the interpreter computes for this constant.
    pub fn to_f64(self) -> f64 {
        match self {
            LitVal::Float(bits) => f64::from_bits(bits),
            LitVal::Rat(q) => q.numer() as f64 / q.denom() as f64,
        }
    }
}

/// A typed constant. `Dimension` has no `Ord`, so the exponents are kept
/// and the dimension rebuilt on demand.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Lit {
    pub val: LitVal,
    exps: [Rational; 7],
}

impl Lit {
    pub fn new(val: LitVal, dim: Dimension) -> Lit {
        Lit {
            val,
            exps: dim.exponents(),
        }
    }

    pub fn float(x: f64, dim: Dimension) -> Lit {
        Lit::new(LitVal::Float(x.to_bits()), dim)
    }

    pub fn dim(&self) -> Dimension {
        Dimension::from_exponents(self.exps)
    }
}

/// The operation of a node.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum Op {
    Lit(Lit),
    /// An opaque value: index into the segment's leaf table.
    Leaf(u32),
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    /// `PowConst`: children `[base, exponent]`, where the exponent is a
    /// dimensionless rational `Lit`, so rules can read and compute it.
    PowC,
    /// `Pow` with a computed exponent: `[base, exponent]`.
    Pow,
    /// `Inverse`.
    Inv,
    MatMul,
    Transpose,
    /// Index into [`MathFn::ALL`] (`MathFn` has no `Ord`).
    Math(u8),
    /// The roots of the segment, in program order. Never rewritten.
    Roots,
}

impl Op {
    pub fn math(f: MathFn) -> Op {
        let i = MathFn::ALL
            .iter()
            .position(|g| *g == f)
            .expect("MathFn::ALL lists every function");
        Op::Math(i as u8)
    }

    pub fn math_fn(self) -> Option<MathFn> {
        match self {
            Op::Math(i) => MathFn::ALL.get(i as usize).copied(),
            _ => None,
        }
    }
}

/// An e-node.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct SkyNode {
    pub op: Op,
    pub args: Vec<Id>,
}

impl SkyNode {
    pub fn new(op: Op, args: Vec<Id>) -> SkyNode {
        SkyNode { op, args }
    }

    pub fn leaf(op: Op) -> SkyNode {
        SkyNode::new(op, Vec::new())
    }
}

impl Language for SkyNode {
    type Discriminant = Op;

    fn discriminant(&self) -> Op {
        self.op
    }

    fn matches(&self, other: &Self) -> bool {
        self.op == other.op && self.args.len() == other.args.len()
    }

    fn children(&self) -> &[Id] {
        &self.args
    }

    fn children_mut(&mut self) -> &mut [Id] {
        &mut self.args
    }
}
