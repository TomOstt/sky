//! Matrix-property tags and their subtyping lattice.
//!
//! # Lattice
//!
//! The implication edges (the spec names the tags but not the edges):
//!
//! * `PositiveDefinite ⇒ Symmetric`. v0.1 uses the real-symmetric definition:
//!   A is PD iff `A = Aᵀ` and `xᵀAx > 0` for all `x ≠ 0`.
//! * `Diagonal ⇒ Symmetric`: a square matrix with `aᵢⱼ = 0` for `i ≠ j`
//!   satisfies `aᵢⱼ = aⱼᵢ`.
//! * `Orthogonal` (`QᵀQ = QQᵀ = I`) is independent. Not every orthogonal matrix
//!   is symmetric (a 90° rotation), diagonal, or PD, and not every symmetric,
//!   diagonal, or PD matrix is orthogonal.
//!
//! A [`TagSet`] is any set of tags. Its [`closure`](TagSet::closure) adds
//! everything implied. A type whose closed tag set is larger is a *subtype*:
//! a `PositiveDefinite` matrix can be used wherever a `Symmetric` one is
//! expected. Ordered that way, [`meet`](TagSet::meet) (greatest lower bound)
//! is the union of the closed sets and [`join`](TagSet::join) (least upper
//! bound) is their intersection. Both results are closed again. The empty set
//! (no guarantees) is the top.
//!
//! Tags are only ever *guaranteed* properties. Every propagation rule below is
//! conservative: it never claims a tag that does not always hold.

use std::fmt;

use crate::tag_ops::{tag_result, TagOp};

/// A matrix property.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Tag {
    /// `A = Aᵀ`.
    Symmetric,
    /// Symmetric with `xᵀAx > 0` for all `x ≠ 0`.
    PositiveDefinite,
    /// `aᵢⱼ = 0` for `i ≠ j`.
    Diagonal,
    /// `QᵀQ = QQᵀ = I`.
    Orthogonal,
}

impl Tag {
    /// Every tag, in display order.
    pub const ALL: [Tag; 4] = [
        Tag::Symmetric,
        Tag::PositiveDefinite,
        Tag::Diagonal,
        Tag::Orthogonal,
    ];

    /// The source spelling, e.g. `"PositiveDefinite"`.
    pub const fn name(self) -> &'static str {
        match self {
            Tag::Symmetric => "Symmetric",
            Tag::PositiveDefinite => "PositiveDefinite",
            Tag::Diagonal => "Diagonal",
            Tag::Orthogonal => "Orthogonal",
        }
    }

    /// Looks a tag up by its exact source spelling.
    pub fn from_name(name: &str) -> Option<Tag> {
        Tag::ALL.into_iter().find(|t| t.name() == name)
    }

    const fn bit(self) -> u8 {
        1 << (self as u8)
    }

    /// The lattice edges: the tags this one implies directly.
    pub const fn implies_directly(self) -> TagSet {
        match self {
            Tag::PositiveDefinite | Tag::Diagonal => TagSet::of(Tag::Symmetric),
            Tag::Symmetric | Tag::Orthogonal => TagSet::EMPTY,
        }
    }

    /// `self <: other`: every matrix with `self` also has `other`
    /// (reflexive; e.g. `PositiveDefinite <: Symmetric`).
    pub fn is_subtag(self, other: Tag) -> bool {
        TagSet::of(self).closure().contains(other)
    }
}

impl fmt::Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A set of tags (not necessarily closed; see [`TagSet::closure`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct TagSet {
    bits: u8,
}

impl TagSet {
    /// No tags: no guarantees (the top of the lattice).
    pub const EMPTY: TagSet = TagSet { bits: 0 };

    /// The set containing just `t` (not closed).
    pub const fn of(t: Tag) -> TagSet {
        TagSet { bits: t.bit() }
    }

    /// The set of the given tags, exactly as listed (not closed).
    pub fn from_tags(tags: impl IntoIterator<Item = Tag>) -> TagSet {
        tags.into_iter().fold(TagSet::EMPTY, |s, t| s.with(t))
    }

    /// `self ∪ {t}` (not closed).
    pub const fn with(self, t: Tag) -> TagSet {
        TagSet {
            bits: self.bits | t.bit(),
        }
    }

    /// Literal membership. Use `closure().contains(t)` to ask whether `t` is implied.
    pub const fn contains(self, t: Tag) -> bool {
        self.bits & t.bit() != 0
    }

    /// True if there are no tags.
    pub const fn is_empty(self) -> bool {
        self.bits == 0
    }

    /// The tags in display order.
    pub fn iter(self) -> impl Iterator<Item = Tag> {
        Tag::ALL.into_iter().filter(move |&t| self.contains(t))
    }

    const fn union(self, other: TagSet) -> TagSet {
        TagSet {
            bits: self.bits | other.bits,
        }
    }

    const fn intersection(self, other: TagSet) -> TagSet {
        TagSet {
            bits: self.bits & other.bits,
        }
    }

    const fn is_superset(self, other: TagSet) -> bool {
        self.bits & other.bits == other.bits
    }

    /// The smallest superset closed under the implication edges.
    pub fn closure(self) -> TagSet {
        let mut s = self;
        loop {
            let next = s.iter().fold(s, |acc, t| acc.union(t.implies_directly()));
            if next == s {
                return s;
            }
            s = next;
        }
    }

    /// True if the set already contains everything it implies.
    pub fn is_closed(self) -> bool {
        self.closure() == self
    }

    /// `self <: other`: a matrix guaranteed `self` is guaranteed `other`,
    /// i.e. `closure(self) ⊇ closure(other)`. A partial order on closed sets.
    pub fn is_subtag_of(self, other: TagSet) -> bool {
        self.closure().is_superset(other.closure())
    }

    /// Greatest lower bound: the tags guaranteed by having both,
    /// `closure(self) ∪ closure(other)`.
    pub fn meet(self, other: TagSet) -> TagSet {
        self.closure().union(other.closure())
    }

    /// Least upper bound: the tags guaranteed by having either,
    /// `closure(self) ∩ closure(other)`.
    pub fn join(self, other: TagSet) -> TagSet {
        self.closure().intersection(other.closure())
    }
}

impl fmt::Display for TagSet {
    /// `{Symmetric, PositiveDefinite}`, or `{}`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("{")?;
        for (i, t) in self.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            f.write_str(t.name())?;
        }
        f.write_str("}")
    }
}

/// The tags `A⁻¹` is guaranteed to have, given `A`'s tags, assuming the
/// inverse exists (typing `A⁻¹` does not prove invertibility; PD and
/// orthogonal matrices are always invertible, symmetric and diagonal ones
/// need not be). Every tag is preserved:
///
/// * Symmetric: `(A⁻¹)ᵀ = (Aᵀ)⁻¹ = A⁻¹`.
/// * PositiveDefinite: `A⁻¹` is symmetric (above), and its eigenvalues are
///   `1/λᵢ > 0`. Equivalently `xᵀA⁻¹x = yᵀAy > 0` with `y = A⁻¹x ≠ 0`.
/// * Diagonal: `diag(d₁…dₙ)⁻¹ = diag(1/d₁…1/dₙ)`, which exists iff every `dᵢ ≠ 0`.
/// * Orthogonal: `Q⁻¹ = Qᵀ`, and `(Qᵀ)ᵀQᵀ = QQᵀ = I`.
///
/// Defined by [`tag_result`]`(`[`TagOp::Inverse`]`, ..)`.
pub fn preserved_by_inverse(tags: TagSet) -> TagSet {
    tag_result(TagOp::Inverse, &[tags])
}

/// The tags `Aᵀ` is guaranteed to have, given `A`'s tags. Every tag is preserved:
///
/// * Symmetric, PositiveDefinite, Diagonal: all imply `Aᵀ = A`, so `Aᵀ` is
///   the same matrix with the same properties.
/// * Orthogonal: `(Qᵀ)ᵀQᵀ = QQᵀ = I` and `Qᵀ(Qᵀ)ᵀ = QᵀQ = I`.
///
/// Defined by [`tag_result`]`(`[`TagOp::Transpose`]`, ..)`.
pub fn preserved_by_transpose(tags: TagSet) -> TagSet {
    tag_result(TagOp::Transpose, &[tags])
}

/// The tags `A @ B` is guaranteed to have (both matrices). Conservative:
///
/// * Diagonal if both are Diagonal: `(D₁D₂)ᵢⱼ = Σₖ d₁ᵢₖ d₂ₖⱼ` is zero unless `i = k = j`.
///   (Symmetric then follows by closure.)
/// * Orthogonal if both are Orthogonal: `(Q₁Q₂)ᵀ(Q₁Q₂) = Q₂ᵀQ₁ᵀQ₁Q₂ = I`.
/// * Nothing else. Symmetric is *not* preserved: `(AB)ᵀ = BᵀAᵀ = BA ≠ AB`
///   in general. Nor is PositiveDefinite, since the product of two PD matrices
///   need not be symmetric.
///
/// Defined by [`tag_result`]`(`[`TagOp::MatMul`]`, ..)`.
pub fn matmul_tags(a: TagSet, b: TagSet) -> TagSet {
    tag_result(TagOp::MatMul, &[a, b])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(tags: &[Tag]) -> TagSet {
        TagSet::from_tags(tags.iter().copied())
    }

    #[test]
    fn edges() {
        assert!(Tag::PositiveDefinite.is_subtag(Tag::Symmetric));
        assert!(Tag::Diagonal.is_subtag(Tag::Symmetric));
        assert!(!Tag::Symmetric.is_subtag(Tag::PositiveDefinite));
        assert!(!Tag::Orthogonal.is_subtag(Tag::Symmetric));
        assert!(!Tag::Symmetric.is_subtag(Tag::Orthogonal));
        assert!(!Tag::PositiveDefinite.is_subtag(Tag::Diagonal));
        for t in Tag::ALL {
            assert!(t.is_subtag(t));
        }
    }

    #[test]
    fn closure_and_display() {
        let pd = TagSet::of(Tag::PositiveDefinite);
        assert!(!pd.is_closed());
        assert_eq!(pd.closure(), set(&[Tag::Symmetric, Tag::PositiveDefinite]));
        assert_eq!(pd.closure().to_string(), "{Symmetric, PositiveDefinite}");
        assert_eq!(TagSet::EMPTY.to_string(), "{}");
        assert!(TagSet::of(Tag::Orthogonal).is_closed());
    }

    #[test]
    fn meet_and_join() {
        let pd = TagSet::of(Tag::PositiveDefinite);
        let diag = TagSet::of(Tag::Diagonal);
        assert_eq!(
            pd.meet(diag),
            set(&[Tag::Symmetric, Tag::PositiveDefinite, Tag::Diagonal])
        );
        assert_eq!(pd.join(diag), TagSet::of(Tag::Symmetric));
        assert_eq!(pd.join(TagSet::of(Tag::Orthogonal)), TagSet::EMPTY);
        assert!(pd.is_subtag_of(TagSet::of(Tag::Symmetric)));
        assert!(pd.is_subtag_of(TagSet::EMPTY));
    }

    #[test]
    fn propagation() {
        let all = set(&Tag::ALL);
        assert_eq!(preserved_by_inverse(all), all);
        assert_eq!(preserved_by_transpose(all), all);
        assert_eq!(
            preserved_by_inverse(TagSet::of(Tag::PositiveDefinite)),
            set(&[Tag::Symmetric, Tag::PositiveDefinite])
        );
        let sym = TagSet::of(Tag::Symmetric);
        assert_eq!(matmul_tags(sym, sym), TagSet::EMPTY);
        let pd = TagSet::of(Tag::PositiveDefinite);
        assert_eq!(matmul_tags(pd, pd), TagSet::EMPTY);
        let d = TagSet::of(Tag::Diagonal);
        assert_eq!(matmul_tags(d, d), set(&[Tag::Symmetric, Tag::Diagonal]));
        let q = TagSet::of(Tag::Orthogonal);
        assert_eq!(matmul_tags(q, q), q);
        assert_eq!(matmul_tags(d, q), TagSet::EMPTY);
    }

    #[test]
    fn names_round_trip() {
        for t in Tag::ALL {
            assert_eq!(Tag::from_name(t.name()), Some(t));
        }
        assert_eq!(Tag::from_name("symmetric"), None);
    }
}
