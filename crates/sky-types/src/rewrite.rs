//! The one v0.1 rewrite: `A⁻¹ @ b` to a Cholesky solve when `A` is
//! positive definite.
//!
//! [`cholesky_rewrite`] is a pure decision over types. It never evaluates
//! anything and never changes the caller's unifier. No other solver
//! selection exists in v0.1 (spec: out of scope).

use std::fmt;

use crate::rules::{inverse, matmul, TensorType};
use crate::shape::{Shape, ShapeUnifier};
use crate::tags::Tag;

/// The syntactic form of the left operand of `@`, as the checker sees it.
#[derive(Clone, Copy, Debug)]
pub enum MatmulLhs<'a> {
    /// The explicit inverse applied to an operand (`A⁻¹`, or `inv(A)`). This
    /// is the only form eligible for the rewrite.
    Inverse(&'a TensorType),
    /// A power such as `A^-1`. Never rewritten, even when the exponent is -1.
    Power {
        /// The base's type.
        base: &'a TensorType,
        /// The exponent.
        exponent: i64,
    },
    /// Any other operand (its type as written).
    Value(&'a TensorType),
}

/// A rewrite plan for `sky check --explain` and, later, the interpreter.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Rewrite {
    /// Replace `A⁻¹ @ b` with a Cholesky solve: factor `A = L·Lᵀ`, then
    /// solve `L·y = b` and `Lᵀ·x = y`.
    CholeskySolve {
        /// `A`'s shape (square).
        a: Shape,
        /// `b`'s shape (a vector, or a matrix of right-hand sides).
        b: Shape,
        /// The result shape, the same as `b`'s.
        result: Shape,
        /// Human-readable explanation for `sky check --explain`.
        explanation: String,
    },
}

impl Rewrite {
    /// The human-readable explanation.
    pub fn explanation(&self) -> &str {
        match self {
            Rewrite::CholeskySolve { explanation, .. } => explanation,
        }
    }
}

impl fmt::Display for Rewrite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.explanation())
    }
}

/// Decides whether `lhs @ rhs` becomes a Cholesky solve. It does when
///
/// 1. `lhs` is [`MatmulLhs::Inverse`] of some `A` (not a power, not a value),
/// 2. `A`'s tags imply [`Tag::PositiveDefinite`] (after closure), and
/// 3. `A⁻¹` and `A⁻¹ @ rhs` both type-check: `A` is square and `rhs` is
///    `Vector[N]` or `Matrix[N,K]` with the same `N`.
///
/// Otherwise it returns `None`. Shapes are checked on a scratch copy of `u`,
/// so the caller's unifier is never changed. Operand names aren't known at
/// this layer, so the explanation calls them `A` and `b`.
pub fn cholesky_rewrite(lhs: MatmulLhs<'_>, rhs: &TensorType, u: &ShapeUnifier) -> Option<Rewrite> {
    let MatmulLhs::Inverse(a) = lhs else {
        return None;
    };
    if !a.tags().closure().contains(Tag::PositiveDefinite) {
        return None;
    }
    let mut scratch = u.clone();
    let a_inv = inverse(&mut scratch, a).ok()?;
    let out = matmul(&mut scratch, &a_inv, rhs).ok()?;
    let (a_shape, b_shape, result) = (
        scratch.apply(a.shape()),
        scratch.apply(rhs.shape()),
        scratch.apply(out.shape()),
    );
    let explanation = format!(
        "rewrite `A⁻¹ @ b` to a Cholesky solve: A is {a_shape} and PositiveDefinite, \
         b is {b_shape}. Factor A = L·Lᵀ (L lower-triangular), solve L·y = b, \
         then Lᵀ·x = y. x = A⁻¹·b is {result}, and A⁻¹ is never formed."
    );
    Some(Rewrite::CholeskySolve {
        a: a_shape,
        b: b_shape,
        result,
        explanation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tags::TagSet;

    fn tagged(u: &mut ShapeUnifier, shape: Shape, t: Tag) -> TensorType {
        TensorType::tagged(u, shape, TagSet::of(t)).unwrap()
    }

    #[test]
    fn pd_inverse_times_vector_is_rewritten() {
        let mut u = ShapeUnifier::new();
        let a = tagged(&mut u, Shape::matrix(3, 3), Tag::PositiveDefinite);
        let b = TensorType::untagged(Shape::vector(3));
        let r = cholesky_rewrite(MatmulLhs::Inverse(&a), &b, &u).unwrap();
        let Rewrite::CholeskySolve {
            a: sa,
            b: sb,
            result,
            ..
        } = &r;
        assert_eq!(
            (sa, sb, result),
            (&Shape::matrix(3, 3), &Shape::vector(3), &Shape::vector(3))
        );
        assert!(r.explanation().contains("Cholesky"));
    }

    #[test]
    fn symbolic_shapes_are_resolved_on_a_scratch_copy() {
        let mut u = ShapeUnifier::new();
        let n = u.fresh("N");
        let k = u.fresh("K");
        let a = tagged(&mut u, Shape::matrix(n.clone(), n), Tag::PositiveDefinite);
        let b = TensorType::untagged(Shape::matrix(4, k.clone()));
        let before = u.clone();
        let r = cholesky_rewrite(MatmulLhs::Inverse(&a), &b, &u).unwrap();
        let Rewrite::CholeskySolve {
            a: sa,
            b: sb,
            result,
            ..
        } = r;
        assert_eq!(sa, Shape::matrix(4, 4));
        assert_eq!(sb, Shape::matrix(4, k.clone()));
        assert_eq!(result, Shape::matrix(4, k));
        // N = 4 was solved on a scratch copy only.
        assert_eq!(u, before);
    }

    #[test]
    fn no_rewrite_without_pd_inverse() {
        let mut u = ShapeUnifier::new();
        let pd = tagged(&mut u, Shape::matrix(3, 3), Tag::PositiveDefinite);
        let sym = tagged(&mut u, Shape::matrix(3, 3), Tag::Symmetric);
        let diag = tagged(&mut u, Shape::matrix(3, 3), Tag::Diagonal);
        let b = TensorType::untagged(Shape::vector(3));
        assert_eq!(cholesky_rewrite(MatmulLhs::Inverse(&sym), &b, &u), None);
        assert_eq!(cholesky_rewrite(MatmulLhs::Inverse(&diag), &b, &u), None);
        assert_eq!(cholesky_rewrite(MatmulLhs::Value(&pd), &b, &u), None);
        let pow = MatmulLhs::Power {
            base: &pd,
            exponent: -1,
        };
        assert_eq!(cholesky_rewrite(pow, &b, &u), None);
    }

    #[test]
    fn no_rewrite_when_shapes_do_not_fit() {
        let mut u = ShapeUnifier::new();
        let pd = tagged(&mut u, Shape::matrix(3, 3), Tag::PositiveDefinite);
        let wrong = TensorType::untagged(Shape::vector(4));
        assert_eq!(cholesky_rewrite(MatmulLhs::Inverse(&pd), &wrong, &u), None);
        assert_eq!(
            cholesky_rewrite(MatmulLhs::Inverse(&pd), &TensorType::scalar(), &u),
            None
        );
        let t3 = TensorType::untagged(Shape::new(vec![3.into(), 3.into(), 3.into()]));
        assert_eq!(cholesky_rewrite(MatmulLhs::Inverse(&pd), &t3, &u), None);
    }
}
