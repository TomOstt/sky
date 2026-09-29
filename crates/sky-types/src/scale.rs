//! Tag rules for scaling by a scalar (`c * A`, `A * c`, `A / c`) and for
//! negation (`-A`). Like every tag rule, they are conservative: a tag is kept
//! only if it holds for every value of `c`.

use crate::tag_ops::{tag_result, TagOp};
use crate::tags::TagSet;

/// The tags of `c·A` (and `A / c`, which is `(1/c)·A`) for an arbitrary
/// real scalar `c`, given `A`'s tags:
///
/// * Symmetric is kept: `(cA)ᵀ = c·Aᵀ = cA`.
/// * Diagonal is kept: every off-diagonal entry is `c·0 = 0`.
/// * PositiveDefinite is dropped: for `c ≤ 0`, `xᵀ(cA)x = c·xᵀAx ≤ 0`.
///   (Symmetric still holds, from the closure.)
/// * Orthogonal is dropped: `(cQ)ᵀ(cQ) = c²·I`, which is `I` only for `c = ±1`.
///
/// Defined by [`tag_result`]`(`[`TagOp::Scale`]`, ..)`.
pub fn scaled_tags(tags: TagSet) -> TagSet {
    tag_result(TagOp::Scale, &[tags])
}

/// The tags of `-A`, given `A`'s tags:
///
/// * Symmetric is kept: `(-A)ᵀ = -Aᵀ = -A`.
/// * Diagonal is kept: `-0 = 0` off the diagonal.
/// * PositiveDefinite is dropped: `xᵀ(-A)x = -xᵀAx < 0`, so `-A` is negative
///   definite. (Symmetric still holds, from the closure.)
/// * Orthogonal is kept (unlike [`scaled_tags`]): `(-Q)ᵀ(-Q) = QᵀQ = I`.
///
/// Defined by [`tag_result`]`(`[`TagOp::Negate`]`, ..)`.
pub fn negated_tags(tags: TagSet) -> TagSet {
    tag_result(TagOp::Negate, &[tags])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tags::Tag;

    fn set(tags: &[Tag]) -> TagSet {
        TagSet::from_tags(tags.iter().copied())
    }

    #[test]
    fn scaling() {
        let pd = TagSet::of(Tag::PositiveDefinite);
        assert_eq!(scaled_tags(pd), TagSet::of(Tag::Symmetric));
        assert_eq!(scaled_tags(TagSet::of(Tag::Orthogonal)), TagSet::EMPTY);
        let d = TagSet::of(Tag::Diagonal);
        assert_eq!(scaled_tags(d), set(&[Tag::Symmetric, Tag::Diagonal]));
    }

    #[test]
    fn negation() {
        let pd = TagSet::of(Tag::PositiveDefinite);
        assert_eq!(negated_tags(pd), TagSet::of(Tag::Symmetric));
        let q = TagSet::of(Tag::Orthogonal);
        assert_eq!(negated_tags(q), q);
        let all = set(&Tag::ALL);
        assert_eq!(
            negated_tags(all),
            set(&[Tag::Symmetric, Tag::Diagonal, Tag::Orthogonal])
        );
    }

    #[test]
    fn results_are_closed_and_never_add_tags() {
        for bits in 0u8..16 {
            let s = TagSet::from_tags(
                Tag::ALL
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| bits & (1 << i) != 0)
                    .map(|(_, t)| t),
            );
            for out in [scaled_tags(s), negated_tags(s)] {
                assert!(out.is_closed());
                assert!(s.is_subtag_of(out));
            }
        }
    }
}
