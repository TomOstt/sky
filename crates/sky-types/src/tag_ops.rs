//! The single source of truth for how operations propagate matrix-property
//! tags: [`TagOp`], [`tag_result`], and [`needs_mirroring`].
//!
//! Every per-operation rule in this crate ([`preserved_by_inverse`],
//! [`preserved_by_transpose`], [`matmul_tags`], [`scaled_tags`],
//! [`negated_tags`], and the tag-dropping in [`elementwise`]) is a thin
//! wrapper over [`tag_result`]. The interpreter reads [`needs_mirroring`] to
//! decide when a result tagged Symmetric must be computed as one triangle and
//! mirrored, so that it is bitwise symmetric.
//!
//! [`preserved_by_inverse`]: crate::tags::preserved_by_inverse
//! [`preserved_by_transpose`]: crate::tags::preserved_by_transpose
//! [`matmul_tags`]: crate::tags::matmul_tags
//! [`scaled_tags`]: crate::scale::scaled_tags
//! [`negated_tags`]: crate::scale::negated_tags
//! [`elementwise`]: crate::rules::elementwise

use std::fmt;

use crate::tags::{Tag, TagSet};

/// An operation that produces a tensor from tagged operands.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TagOp {
    /// `A⁻¹` or `inv(A)`. One operand.
    Inverse,
    /// `Aᵀ`. One operand.
    Transpose,
    /// `c * A`, `A * c`, `A / c` for a scalar `c`. One operand (the tensor).
    Scale,
    /// `-A`. One operand.
    Negate,
    /// `A @ B` between two matrices. Two operands.
    MatMul,
    /// `A + B`, `A - B`, ... between two tensors of equal shape. Two operands.
    Elementwise,
}

impl TagOp {
    /// Every operation, for exhaustive tests.
    pub const ALL: [TagOp; 6] = [
        TagOp::Inverse,
        TagOp::Transpose,
        TagOp::Scale,
        TagOp::Negate,
        TagOp::MatMul,
        TagOp::Elementwise,
    ];

    /// How many tagged operands the operation takes (1 or 2).
    pub const fn arity(self) -> usize {
        match self {
            TagOp::Inverse | TagOp::Transpose | TagOp::Scale | TagOp::Negate => 1,
            TagOp::MatMul | TagOp::Elementwise => 2,
        }
    }
}

impl fmt::Display for TagOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            TagOp::Inverse => "inverse",
            TagOp::Transpose => "transpose",
            TagOp::Scale => "scale",
            TagOp::Negate => "negate",
            TagOp::MatMul => "matmul",
            TagOp::Elementwise => "elementwise",
        })
    }
}

/// The tags the result of `op` is guaranteed to have, given its operands'
/// tags (`inputs[0]`, and `inputs[1]` for binary ops). A missing operand
/// counts as untagged and extra operands are ignored, so a wrong-length slice
/// can only lose tags, never invent them. The result is closed.
///
/// Every rule is conservative: a tag is claimed only if it always holds.
///
/// * **Inverse** keeps every tag, assuming the inverse exists:
///   Symmetric since `(A⁻¹)ᵀ = (Aᵀ)⁻¹ = A⁻¹`; PositiveDefinite since the
///   eigenvalues become `1/λᵢ > 0`; Diagonal since
///   `diag(dᵢ)⁻¹ = diag(1/dᵢ)`; Orthogonal since `Q⁻¹ = Qᵀ` is orthogonal.
/// * **Transpose** keeps every tag: Symmetric, PositiveDefinite and Diagonal
///   all mean `Aᵀ = A`, and `(Qᵀ)ᵀQᵀ = QQᵀ = I`.
/// * **Scale** keeps Symmetric (`(cA)ᵀ = cA`) and Diagonal (`c·0 = 0`), and
///   drops PositiveDefinite (`c ≤ 0` breaks it) and Orthogonal
///   (`(cQ)ᵀ(cQ) = c²I`).
/// * **Negate** keeps Symmetric, Diagonal and Orthogonal
///   (`(-Q)ᵀ(-Q) = QᵀQ = I`), and drops PositiveDefinite (`-A` is negative
///   definite).
/// * **MatMul** keeps Diagonal if both are Diagonal (so Symmetric by closure)
///   and Orthogonal if both are Orthogonal (`(Q₁Q₂)ᵀQ₁Q₂ = I`). Nothing
///   else: `(AB)ᵀ = BA ≠ AB` in general.
/// * **Elementwise** claims nothing (`P - P = 0` is not PD).
pub fn tag_result(op: TagOp, inputs: &[TagSet]) -> TagSet {
    let a = inputs.first().copied().unwrap_or(TagSet::EMPTY).closure();
    let b = inputs.get(1).copied().unwrap_or(TagSet::EMPTY).closure();
    let kept = |set: TagSet, rule: fn(Tag) -> bool| {
        TagSet::from_tags(set.iter().filter(|&t| rule(t))).closure()
    };
    match op {
        TagOp::Inverse | TagOp::Transpose => a,
        TagOp::Scale => kept(a, |t| matches!(t, Tag::Symmetric | Tag::Diagonal)),
        TagOp::Negate => kept(a, |t| {
            matches!(t, Tag::Symmetric | Tag::Diagonal | Tag::Orthogonal)
        }),
        TagOp::MatMul => kept(a.join(b), |t| matches!(t, Tag::Diagonal | Tag::Orthogonal)),
        TagOp::Elementwise => TagSet::EMPTY,
    }
}

/// Whether the interpreter must compute a result that [`tag_result`] tags
/// Symmetric (or Diagonal) as one triangle and mirror it, to make it bitwise
/// symmetric. True only for [`TagOp::Inverse`].
///
/// The other ops that can produce Symmetric are exact on a bitwise-symmetric
/// input, so their output is bitwise symmetric without mirroring:
///
/// * **Transpose** only moves entries; no arithmetic is done.
/// * **Negate** flips each entry's sign bit, which is exact, so
///   `-aᵢⱼ` and `-aⱼᵢ` have the same bits when `aᵢⱼ` and `aⱼᵢ` do.
/// * **Scale** computes each entry as one IEEE-754 multiplication (or
///   division) `c·aᵢⱼ`. It is correctly rounded and deterministic, so equal
///   inputs `aᵢⱼ = aⱼᵢ` give equal outputs. (The interpreter must compute
///   every entry the same way, e.g. no fused or reassociated forms that
///   differ by position.)
/// * **MatMul** gives Symmetric only through Diagonal @ Diagonal. That is
///   the product of the two diagonals, with every off-diagonal entry exactly
///   zero. (The interpreter must compute it that way, not as a general
///   product whose `0·∞` terms could produce NaN off the diagonal.)
///
/// **Inverse** is not exact: LU, Cholesky and friends compute `(A⁻¹)ᵢⱼ` and
/// `(A⁻¹)ⱼᵢ` through different sequences of roundings, so a symmetric input
/// gives a result that is symmetric only up to rounding. **Elementwise**
/// never produces a tag, so it never needs mirroring either.
pub const fn needs_mirroring(op: TagOp) -> bool {
    matches!(op, TagOp::Inverse)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every closed tag set.
    fn closed_sets() -> Vec<TagSet> {
        let mut sets: Vec<TagSet> = (0u8..16)
            .map(|bits| {
                TagSet::from_tags(
                    Tag::ALL
                        .into_iter()
                        .enumerate()
                        .filter(|(i, _)| bits & (1 << i) != 0)
                        .map(|(_, t)| t),
                )
            })
            .filter(|s| s.is_closed())
            .collect();
        sets.dedup();
        sets
    }

    /// Per `needs_mirroring`'s doc comment: is `op` exact on a
    /// bitwise-symmetric input, for a result with these tags?
    fn exact(op: TagOp, out: TagSet) -> bool {
        match op {
            TagOp::Transpose | TagOp::Scale | TagOp::Negate => true,
            // Only Diagonal @ Diagonal, computed as a product of diagonals.
            TagOp::MatMul => out.contains(Tag::Diagonal),
            TagOp::Inverse | TagOp::Elementwise => false,
        }
    }

    #[test]
    fn every_symmetric_result_is_exact_or_mirrored() {
        let sets = closed_sets();
        // Of the 16 subsets, the 6 containing PositiveDefinite or Diagonal but
        // not Symmetric are not closed.
        assert_eq!(sets.len(), 10, "{sets:?}");
        let mut symmetric_results = 0;
        for op in TagOp::ALL {
            let inputs: Vec<Vec<TagSet>> = match op.arity() {
                1 => sets.iter().map(|&a| vec![a]).collect(),
                _ => sets
                    .iter()
                    .flat_map(|&a| sets.iter().map(move |&b| vec![a, b]))
                    .collect(),
            };
            for input in inputs {
                let out = tag_result(op, &input);
                assert!(out.is_closed(), "{op} {input:?}");
                if out.contains(Tag::Symmetric) {
                    symmetric_results += 1;
                    assert!(
                        exact(op, out) || needs_mirroring(op),
                        "{op} on {input:?} gives {out} but is neither exact nor mirrored"
                    );
                }
            }
        }
        assert!(symmetric_results > 0);
    }

    #[test]
    fn only_inverse_needs_mirroring() {
        for op in TagOp::ALL {
            assert_eq!(needs_mirroring(op), op == TagOp::Inverse, "{op}");
        }
    }

    #[test]
    fn results_never_add_tags() {
        let sets = closed_sets();
        for op in TagOp::ALL {
            for &a in &sets {
                for &b in &sets {
                    let out = tag_result(op, &[a, b]);
                    let operands = if op.arity() == 1 { a } else { a.join(b) };
                    assert!(operands.is_subtag_of(out), "{op} {a} {b} gives {out}");
                }
            }
        }
    }

    #[test]
    fn missing_operands_only_lose_tags() {
        let pd = TagSet::of(Tag::PositiveDefinite);
        assert_eq!(tag_result(TagOp::Inverse, &[]), TagSet::EMPTY);
        assert_eq!(tag_result(TagOp::MatMul, &[pd]), TagSet::EMPTY);
    }

    #[test]
    fn the_named_rules_are_wrappers() {
        use crate::scale::{negated_tags, scaled_tags};
        use crate::tags::{matmul_tags, preserved_by_inverse, preserved_by_transpose};
        let sets = closed_sets();
        for &a in &sets {
            assert_eq!(preserved_by_inverse(a), tag_result(TagOp::Inverse, &[a]));
            assert_eq!(
                preserved_by_transpose(a),
                tag_result(TagOp::Transpose, &[a])
            );
            assert_eq!(scaled_tags(a), tag_result(TagOp::Scale, &[a]));
            assert_eq!(negated_tags(a), tag_result(TagOp::Negate, &[a]));
            for &b in &sets {
                assert_eq!(matmul_tags(a, b), tag_result(TagOp::MatMul, &[a, b]));
            }
        }
    }
}
