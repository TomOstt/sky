//! Choosing the algorithm of a `matmul` from its operand shapes: the
//! framework the kernel and tuning agents plug into.
//!
//! `matmul[alg=...]` records an [`Algorithm`] on the operation. Every
//! algorithm must give **exactly the reference result** (see
//! `sky_types::Algorithm`): so the choice, made from shapes alone, can
//! never change a printed digit, and a pass can make it freely and explain
//! it. What a shape class is *for* (an unrolled kernel for tiny fixed sizes,
//! a blocked one for large ones, a vector kernel for `matrix @ vector`) is
//! for the kernels; this pass only asks a **policy** and records the answer.
//!
//! A [`MatMulPolicy`] is a function from what is known about a product to an
//! [`Algorithm`]. The default policy, [`default_policy`], answers
//! [`Algorithm::Auto`] for everything, because nothing that needs a shape
//! class exists yet: no algorithm here beats the reference loop in the
//! reference interpreter. (The one structural algorithm that is exact and
//! wins, `syrk` for `Aᵀ @ A`, is chosen by `structure`, where the operand
//! being the same value is visible.) A kernel crate registers its own:
//!
//! ```ignore
//! let pass = AlgorithmPass { matmul: |p| match p.class {
//!     ShapeClass::MatVec => Algorithm::MatVec,
//!     ShapeClass::Gemm if p.small() => Algorithm::Gemm,
//!     _ => Algorithm::Auto,
//! }};
//! ```
//!
//! and puts it in its pipeline in place of the default one.

use sky_ir::{Function, Module, OpId, OpKind, Type};
use sky_types::flops::Flops;
use sky_types::shape::ShapeDim;
use sky_types::Algorithm;

use crate::rewrite::Rewrite;
use crate::util::reachable_ops;

pub const ALGORITHM_RULE: &str = "ALG-MATMUL-001";

/// The class of a product's operand shapes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShapeClass {
    /// `vector @ vector`.
    Dot,
    /// `matrix @ vector`.
    MatVec,
    /// `vector @ matrix`.
    VecMat,
    /// `matrix @ matrix`.
    Gemm,
}

/// What a policy knows about a product: its class, its sizes where known,
/// its transposition flags.
#[derive(Clone, PartialEq, Debug)]
pub struct MatMulShape {
    pub class: ShapeClass,
    /// Rows of the left operand (after its transpose flag), 1 for a vector.
    pub m: ShapeDim,
    /// The inner size.
    pub k: ShapeDim,
    /// Columns of the right operand, 1 for a vector.
    pub n: ShapeDim,
    pub ta: bool,
    pub tb: bool,
}

impl MatMulShape {
    /// The number of multiply-adds, if all three sizes are known.
    pub fn work(&self) -> Option<u64> {
        match (&self.m, &self.k, &self.n) {
            (ShapeDim::Known(m), ShapeDim::Known(k), ShapeDim::Known(n)) => {
                m.checked_mul(*k)?.checked_mul(*n)
            }
            _ => None,
        }
    }

    /// Whether every size is known and at most `limit` (a fixed-size
    /// kernel can be fully unrolled).
    pub fn all_at_most(&self, limit: u64) -> bool {
        [&self.m, &self.k, &self.n]
            .iter()
            .all(|d| matches!(d, ShapeDim::Known(s) if *s <= limit))
    }
}

/// Chooses an algorithm for a product.
pub type MatMulPolicy = fn(&MatMulShape) -> Algorithm;

/// The default policy: leave the choice to the engine.
pub fn default_policy(_: &MatMulShape) -> Algorithm {
    Algorithm::Auto
}

/// The shape class of an algorithm, if it names one (`Auto` and `Syrk` do
/// not).
fn class_of(a: Algorithm) -> Option<ShapeClass> {
    match a {
        Algorithm::Dot => Some(ShapeClass::Dot),
        Algorithm::MatVec => Some(ShapeClass::MatVec),
        Algorithm::VecMat => Some(ShapeClass::VecMat),
        Algorithm::Gemm => Some(ShapeClass::Gemm),
        Algorithm::Auto | Algorithm::Syrk => None,
    }
}

/// Records the policy's choice on every plain `matmul` and every
/// `matmul[...]` whose algorithm is still `auto`. A choice that does not fit
/// the operands' class is ignored (a policy cannot make the verifier fail).
pub fn algorithms(m: &mut Module, policy: MatMulPolicy) -> Vec<Rewrite> {
    let mut out = Vec::new();
    for f in &mut m.functions {
        for op in reachable_ops(f) {
            if f.block(f.op(op).parent).ops.contains(&op) {
                if let Some(rw) = choose(f, op, policy) {
                    out.push(rw);
                }
            }
        }
    }
    out
}

fn choose(f: &mut Function, id: OpId, policy: MatMulPolicy) -> Option<Rewrite> {
    let op = f.op(id).clone();
    let (ta, tb) = match op.kind {
        OpKind::MatMul => (false, false),
        OpKind::MatMulEx {
            ta,
            tb,
            algorithm: Algorithm::Auto,
        } => (ta, tb),
        _ => return None,
    };
    let &[a, b] = &op.operands[..] else {
        return None;
    };
    let (Type::Number(x), Type::Number(y)) = (f.ty(a).clone(), f.ty(b).clone()) else {
        return None;
    };
    let dims = |t: &sky_ir::NumberType, flip: bool| -> Option<(ShapeDim, ShapeDim)> {
        match t.shape.dims() {
            [r, c] if flip => Some((c.clone(), r.clone())),
            [r, c] => Some((r.clone(), c.clone())),
            _ => None,
        }
    };
    let (class, m, k, n) = match (x.shape.rank(), y.shape.rank()) {
        (1, 1) => {
            let len = x.shape.dims()[0].clone();
            (ShapeClass::Dot, ShapeDim::Known(1), len, ShapeDim::Known(1))
        }
        (2, 1) => {
            let (m, k) = dims(&x, ta)?;
            (ShapeClass::MatVec, m, k, ShapeDim::Known(1))
        }
        (1, 2) => {
            let (k, n) = dims(&y, tb)?;
            (ShapeClass::VecMat, ShapeDim::Known(1), k, n)
        }
        (2, 2) => {
            let (m, k) = dims(&x, ta)?;
            let (_, n) = dims(&y, tb)?;
            (ShapeClass::Gemm, m, k, n)
        }
        _ => return None,
    };
    let shape = MatMulShape {
        class,
        m,
        k,
        n,
        ta,
        tb,
    };
    let chosen = policy(&shape);
    if chosen == Algorithm::Auto || class_of(chosen) != Some(class) {
        return None;
    }
    f.ops[id.index()].kind = OpKind::MatMulEx {
        ta,
        tb,
        algorithm: chosen,
    };
    let flops = Flops::matmul(&shape.m, &shape.k, &shape.n);
    let named = |v| f.value(v).name.clone();
    let (an, bn) = (
        named(a).unwrap_or_else(|| "A".to_string()),
        named(b).unwrap_or_else(|| "B".to_string()),
    );
    Some(Rewrite {
        pass: "algorithms",
        rule: ALGORITHM_RULE,
        span: op.span,
        before: format!("{an} @ {bn}"),
        after: format!("matmul[alg={}]({an}, {bn})", chosen.mnemonic()),
        because: vec![
            format!(
                "the operands are {} and {}: a {:?} product of {}×{} by {}×{}",
                x.shape, y.shape, class, shape.m, shape.k, shape.k, shape.n
            ),
            format!(
                "the algorithm `{}` was chosen from the shapes alone; every algorithm computes exactly the reference result (index-order sums from +0.0), so no digit changes",
                chosen.mnemonic()
            ),
            format!("{flops} flops either way"),
        ],
    })
}
