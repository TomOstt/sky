//! Typing rules for `@`, elementwise operators, transpose, and inverse.
//!
//! Every rule takes the operands' [`TensorType`]s and returns the result
//! type or a [`ShapeError`]. Rules that constrain shapes do so through a
//! [`ShapeUnifier`] and leave it unchanged when they fail.

use std::fmt;

use crate::shape::{Shape, ShapeError, ShapeErrorKind, ShapeUnifier};
use crate::tag_ops::{tag_result, TagOp};
use crate::tags::{matmul_tags, preserved_by_inverse, preserved_by_transpose, TagSet};

/// The static type of an array value: a shape plus guaranteed property tags.
///
/// Invariant: the tag set is closed, and a non-empty tag set only appears on
/// a rank-2 shape whose row and column counts have been unified.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TensorType {
    shape: Shape,
    tags: TagSet,
}

impl TensorType {
    /// A type with no tags.
    pub fn untagged(shape: Shape) -> Self {
        TensorType {
            shape,
            tags: TagSet::EMPTY,
        }
    }

    /// The rank-0 type.
    pub fn scalar() -> Self {
        TensorType::untagged(Shape::scalar())
    }

    /// A type with tags (closed automatically). Non-empty tags need a
    /// rank-2 shape, and its row and column counts are unified: every tag
    /// implies a square matrix. On error the unifier is unchanged.
    pub fn tagged(u: &mut ShapeUnifier, shape: Shape, tags: TagSet) -> Result<Self, ShapeError> {
        let tags = tags.closure();
        if tags.is_empty() {
            return Ok(TensorType::untagged(shape));
        }
        let square = match shape.dims() {
            [r, c] => u.unify_dim(r, c).is_ok(),
            _ => false,
        };
        if !square {
            return Err(ShapeErrorKind::TagsNeedSquareMatrix { shape, tags }.into());
        }
        Ok(TensorType { shape, tags })
    }

    /// The shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The closed set of guaranteed tags.
    pub fn tags(&self) -> TagSet {
        self.tags
    }
}

impl fmt::Display for TensorType {
    /// `Matrix[3,3] {Symmetric, PositiveDefinite}`, or just the shape if untagged.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.shape)?;
        if !self.tags.is_empty() {
            write!(f, " {}", self.tags)?;
        }
        Ok(())
    }
}

impl ShapeUnifier {
    /// `t` with its shape's solved variables replaced.
    pub fn apply_type(&self, t: &TensorType) -> TensorType {
        TensorType {
            shape: self.apply(&t.shape),
            tags: t.tags,
        }
    }
}

/// `lhs @ rhs`. Operands must be vectors or matrices (no batching or broadcasting):
///
/// | lhs           | rhs           | result        |
/// |---------------|---------------|---------------|
/// | `Matrix[M,K]` | `Matrix[K,N]` | `Matrix[M,N]` |
/// | `Matrix[M,K]` | `Vector[K]`   | `Vector[M]`   |
/// | `Vector[K]`   | `Matrix[K,N]` | `Vector[N]`   |
/// | `Vector[K]`   | `Vector[K]`   | `Scalar`      |
///
/// The two `K`s are unified. Tags follow [`matmul_tags`] for matrix @ matrix;
/// any result involving a vector is untagged.
pub fn matmul(
    u: &mut ShapeUnifier,
    lhs: &TensorType,
    rhs: &TensorType,
) -> Result<TensorType, ShapeError> {
    let (l, r) = (&lhs.shape, &rhs.shape);
    let (k_left, k_right, out) = match (l.dims(), r.dims()) {
        ([m, k], [k2, n]) => (k, k2, vec![m.clone(), n.clone()]),
        ([m, k], [k2]) => (k, k2, vec![m.clone()]),
        ([k], [k2, n]) => (k, k2, vec![n.clone()]),
        ([k], [k2]) => (k, k2, Vec::new()),
        _ => {
            return Err(ShapeErrorKind::MatmulRank {
                lhs: l.clone(),
                rhs: r.clone(),
            }
            .into())
        }
    };
    u.unify_dim(k_left, k_right).map_err(|c| -> ShapeError {
        ShapeErrorKind::MatmulInner {
            lhs: l.clone(),
            rhs: r.clone(),
            left: c.expected,
            right: c.found,
        }
        .into()
    })?;
    let tags = if l.rank() == 2 && r.rank() == 2 {
        matmul_tags(lhs.tags, rhs.tags)
    } else {
        TagSet::EMPTY
    };
    Ok(TensorType {
        shape: Shape::new(out),
        tags,
    })
}

/// An elementwise operator (`+`, `-`, …): both shapes must be equal (no
/// broadcasting in v0.1). The result has the left operand's shape and no
/// tags. That is conservative: some tags survive some operators (a sum of
/// symmetric matrices is symmetric), but v0.1 claims none. The tags are
/// [`tag_result`]`(`[`TagOp::Elementwise`]`, ..)`.
///
/// On error the unifier is unchanged; on success the bindings are committed,
/// so a caller with a further check to make (the checker compares element
/// dimensions afterwards) must pass a copy and keep it only if that passes.
pub fn elementwise(
    u: &mut ShapeUnifier,
    lhs: &TensorType,
    rhs: &TensorType,
) -> Result<TensorType, ShapeError> {
    u.unify(&lhs.shape, &rhs.shape)?;
    // `tag_result` never adds tags, so any tag here would come from a tagged
    // (hence square) operand of this same shape: the invariant holds.
    Ok(TensorType {
        shape: lhs.shape.clone(),
        tags: tag_result(TagOp::Elementwise, &[lhs.tags, rhs.tags]),
    })
}

/// `Aᵀ`. A scalar is its own transpose. `Matrix[M,N]` becomes `Matrix[N,M]`
/// with tags from [`preserved_by_transpose`]. Vectors and higher ranks are an
/// error: v0.1 has no row/column-vector distinction and no axis permutation.
pub fn transpose(t: &TensorType) -> Result<TensorType, ShapeError> {
    match t.shape.dims() {
        [] => Ok(TensorType::scalar()),
        [m, n] => Ok(TensorType {
            shape: Shape::matrix(n.clone(), m.clone()),
            tags: preserved_by_transpose(t.tags),
        }),
        _ => Err(ShapeErrorKind::TransposeRank {
            shape: t.shape.clone(),
        }
        .into()),
    }
}

/// `A⁻¹`.
///
/// * Rank 0: the reciprocal, a scalar with no tags.
/// * Rank 2: the row and column counts must unify (a square matrix). The
///   result is `Matrix[N,M]` (equal to `Matrix[M,M]` after unification), with
///   tags from [`preserved_by_inverse`].
/// * Rank 1: `cannot invert a vector`. Rank 3 or more: `cannot invert a rank-N tensor`.
///
/// On error the unifier is unchanged.
pub fn inverse(u: &mut ShapeUnifier, t: &TensorType) -> Result<TensorType, ShapeError> {
    match t.shape.dims() {
        [] => Ok(TensorType::scalar()),
        [m, n] => {
            u.unify_dim(m, n).map_err(|c| -> ShapeError {
                ShapeErrorKind::NotSquare {
                    shape: t.shape.clone(),
                    rows: c.expected,
                    cols: c.found,
                }
                .into()
            })?;
            Ok(TensorType {
                shape: Shape::matrix(n.clone(), m.clone()),
                tags: preserved_by_inverse(t.tags),
            })
        }
        [_] => Err(ShapeErrorKind::InvertVector {
            shape: t.shape.clone(),
        }
        .into()),
        _ => Err(ShapeErrorKind::InvertTensor {
            shape: t.shape.clone(),
        }
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::ShapeDim;
    use crate::tags::Tag;

    fn ty(shape: Shape) -> TensorType {
        TensorType::untagged(shape)
    }

    #[test]
    fn matmul_known() {
        let mut u = ShapeUnifier::new();
        let out = matmul(&mut u, &ty(Shape::matrix(3, 4)), &ty(Shape::matrix(4, 8))).unwrap();
        assert_eq!(out.shape(), &Shape::matrix(3, 8));
    }

    #[test]
    fn matmul_mismatch_is_reported_and_leaves_state() {
        let mut u = ShapeUnifier::new();
        let before = u.clone();
        let err = matmul(&mut u, &ty(Shape::matrix(3, 4)), &ty(Shape::matrix(5, 8))).unwrap_err();
        assert_eq!(
            err.to_string(),
            "`@` inner dimensions differ: 4 vs 5 in Matrix[3,4] @ Matrix[5,8]"
        );
        assert_eq!(u, before);
    }

    #[test]
    fn matmul_vector_cases() {
        let mut u = ShapeUnifier::new();
        let k = u.fresh("K");
        let a = ty(Shape::matrix(3, k.clone()));
        let out = matmul(&mut u, &a, &ty(Shape::vector(5))).unwrap();
        assert_eq!(out.shape(), &Shape::vector(3));
        assert_eq!(u.resolve(&k), ShapeDim::Known(5));
        let row = matmul(&mut u, &ty(Shape::vector(3)), &ty(Shape::matrix(3, 7))).unwrap();
        assert_eq!(row.shape(), &Shape::vector(7));
        let dot = matmul(&mut u, &ty(Shape::vector(3)), &ty(Shape::vector(3))).unwrap();
        assert_eq!(dot.shape(), &Shape::scalar());
        let err = matmul(&mut u, &TensorType::scalar(), &ty(Shape::vector(3))).unwrap_err();
        assert_eq!(
            err.to_string(),
            "`@` needs vector or matrix operands, found Scalar @ Vector[3]"
        );
    }

    #[test]
    fn matmul_tags_are_conservative() {
        let mut u = ShapeUnifier::new();
        let sym =
            TensorType::tagged(&mut u, Shape::matrix(3, 3), TagSet::of(Tag::Symmetric)).unwrap();
        assert!(matmul(&mut u, &sym, &sym).unwrap().tags().is_empty());
        let d = TensorType::tagged(&mut u, Shape::matrix(3, 3), TagSet::of(Tag::Diagonal)).unwrap();
        let dd = matmul(&mut u, &d, &d).unwrap();
        assert!(dd.tags().contains(Tag::Diagonal) && dd.tags().contains(Tag::Symmetric));
        let v = matmul(&mut u, &d, &ty(Shape::vector(3))).unwrap();
        assert!(v.tags().is_empty());
    }

    #[test]
    fn elementwise_requires_equal_shapes() {
        let mut u = ShapeUnifier::new();
        let n = u.fresh("N");
        let a = ty(Shape::matrix(n.clone(), 4));
        let out = elementwise(&mut u, &a, &ty(Shape::matrix(3, 4))).unwrap();
        assert_eq!(u.apply(out.shape()), Shape::matrix(3, 4));
        let before = u.clone();
        let err = elementwise(&mut u, &a, &ty(Shape::matrix(3, 5))).unwrap_err();
        assert!(matches!(
            err.kind(),
            ShapeErrorKind::DimMismatch { index: 1, .. }
        ));
        assert_eq!(u, before);
    }

    #[test]
    fn transpose_rules() {
        let mut u = ShapeUnifier::new();
        let t = transpose(&ty(Shape::matrix(3, 4))).unwrap();
        assert_eq!(t.shape(), &Shape::matrix(4, 3));
        let q =
            TensorType::tagged(&mut u, Shape::matrix(4, 4), TagSet::of(Tag::Orthogonal)).unwrap();
        assert_eq!(transpose(&q).unwrap().tags(), TagSet::of(Tag::Orthogonal));
        assert_eq!(transpose(&TensorType::scalar()), Ok(TensorType::scalar()));
        assert!(transpose(&ty(Shape::vector(3))).is_err());
    }

    #[test]
    fn inverse_by_rank() {
        let mut u = ShapeUnifier::new();
        assert_eq!(
            inverse(&mut u, &TensorType::scalar()),
            Ok(TensorType::scalar())
        );
        let pd = TensorType::tagged(
            &mut u,
            Shape::matrix(3, 3),
            TagSet::of(Tag::PositiveDefinite),
        )
        .unwrap();
        let inv = inverse(&mut u, &pd).unwrap();
        assert_eq!(inv.to_string(), "Matrix[3,3] {Symmetric, PositiveDefinite}");
        let err = inverse(&mut u, &ty(Shape::vector(3))).unwrap_err();
        assert_eq!(err.to_string(), "cannot invert a vector");
        let t3 = ty(Shape::new(vec![2.into(), 2.into(), 2.into()]));
        assert_eq!(
            inverse(&mut u, &t3).unwrap_err().to_string(),
            "cannot invert a rank-3 tensor"
        );
    }

    #[test]
    fn inverse_requires_square_and_solves_symbolic() {
        let mut u = ShapeUnifier::new();
        let before = u.clone();
        let err = inverse(&mut u, &ty(Shape::matrix(3, 4))).unwrap_err();
        assert_eq!(
            err.to_string(),
            "cannot invert a non-square matrix Matrix[3,4]: 3 rows, 4 columns"
        );
        assert_eq!(u, before);
        let n = u.fresh("N");
        let inv = inverse(&mut u, &ty(Shape::matrix(n.clone(), 6))).unwrap();
        assert_eq!(u.apply(inv.shape()), Shape::matrix(6, 6));
        assert_eq!(u.resolve(&n), ShapeDim::Known(6));
    }

    #[test]
    fn tags_need_square_matrix() {
        let mut u = ShapeUnifier::new();
        let err = TensorType::tagged(
            &mut u,
            Shape::matrix(3, 4),
            TagSet::of(Tag::PositiveDefinite),
        )
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            "matrix property tags {Symmetric, PositiveDefinite} need a square matrix, found Matrix[3,4]"
        );
        assert!(TensorType::tagged(&mut u, Shape::vector(3), TagSet::of(Tag::Diagonal)).is_err());
        let n = u.fresh("N");
        let m = u.fresh("M");
        TensorType::tagged(
            &mut u,
            Shape::matrix(n.clone(), m.clone()),
            TagSet::of(Tag::Symmetric),
        )
        .unwrap();
        assert_eq!(u.resolve(&n), u.resolve(&m));
    }
}
