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
    /// `A * B` between two arrays of one shape, entry by entry (the
    /// Hadamard product, not `@`). Two operands.
    Product,
    /// `A / B` between two arrays of one shape, entry by entry. Two
    /// operands (numerator, denominator).
    Quotient,
    /// A math function of one array, or of an array and a number
    /// (`sqrt(A)`, `min(A, 1)`), entry by entry. One operand (the array).
    Map1,
    /// A math function of two arrays of one shape (`atan2(A, B)`), entry by
    /// entry. Two operands.
    Map2,
}

impl TagOp {
    /// Every operation, for exhaustive tests.
    pub const ALL: [TagOp; 10] = [
        TagOp::Inverse,
        TagOp::Transpose,
        TagOp::Scale,
        TagOp::Negate,
        TagOp::MatMul,
        TagOp::Elementwise,
        TagOp::Product,
        TagOp::Quotient,
        TagOp::Map1,
        TagOp::Map2,
    ];

    /// How many tagged operands the operation takes (1 or 2).
    pub const fn arity(self) -> usize {
        match self {
            TagOp::Inverse | TagOp::Transpose | TagOp::Scale | TagOp::Negate | TagOp::Map1 => 1,
            TagOp::MatMul | TagOp::Elementwise | TagOp::Product | TagOp::Quotient | TagOp::Map2 => {
                2
            }
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
            TagOp::Product => "product",
            TagOp::Quotient => "quotient",
            TagOp::Map1 => "map",
            TagOp::Map2 => "map2",
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
/// * **Elementwise** (`+`, `-`) claims nothing (`P - P = 0` is not PD).
/// * **Product** (`A * B`, entry by entry) keeps Symmetric if both operands
///   are (the entry at `(i, j)` and at `(j, i)` are one function of equal
///   inputs), and Diagonal if either is (an off-diagonal entry is `0 · b`,
///   which is `±0` for a finite `b`). PositiveDefinite is dropped: the
///   Schur product theorem gives it over the reals, but the entries are
///   rounded, and a product of nearly singular matrices can lose it (a
///   tag the interpreter would then find broken at run time). Orthogonal is
///   dropped (`Q ∘ Q` is not orthogonal).
/// * **Quotient** (`A / B`) keeps Symmetric if both are, and Diagonal if
///   the *numerator* is (`0 / b` is `±0`; a zero `b` fails at run time, so no
///   result exists to be wrong). A Diagonal denominator says nothing: its
///   off-diagonal entries are zero, and dividing by them is an error.
/// * **Map1** (`sqrt(A)`, `exp(A)`, ...) keeps Symmetric (`f` of equal
///   entries is equal) and nothing else: `exp` and `cos` send the zeros of a
///   Diagonal matrix to 1, `log` fails on them, and whether `f(0) = 0` is a
///   property of each function this rule does not consult.
/// * **Map2** (`atan2(A, B)`, `min(A, B)`) keeps Symmetric if both are.
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
        // `join` is the tags both operands have, `meet` the tags either has.
        TagOp::Product => {
            let symmetric = kept(a.join(b), |t| t == Tag::Symmetric);
            let diagonal = kept(a.meet(b), |t| t == Tag::Diagonal);
            TagSet::from_tags(symmetric.iter().chain(diagonal.iter())).closure()
        }
        TagOp::Quotient => {
            let symmetric = kept(a.join(b), |t| t == Tag::Symmetric);
            let diagonal = kept(a, |t| t == Tag::Diagonal);
            TagSet::from_tags(symmetric.iter().chain(diagonal.iter())).closure()
        }
        TagOp::Map1 => kept(a, |t| t == Tag::Symmetric),
        TagOp::Map2 => kept(a.join(b), |t| t == Tag::Symmetric),
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
/// never produces a tag, so it never needs mirroring either. **Product**,
/// **Quotient**, **Map1** and **Map2** compute every entry with one
/// deterministic function of the entries at that position, so equal inputs
/// `aᵢⱼ = aⱼᵢ` (and `bᵢⱼ = bⱼᵢ`) give equal outputs.
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
            TagOp::Transpose
            | TagOp::Scale
            | TagOp::Negate
            | TagOp::Product
            | TagOp::Quotient
            | TagOp::Map1
            | TagOp::Map2 => true,
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
                    // What guarantees a result tag: `a` alone (one operand,
                    // or the numerator of a quotient), either operand (a
                    // product's Diagonal, so the tags of both), or both.
                    let operands = match op {
                        _ if op.arity() == 1 => a,
                        TagOp::Product => a.meet(b),
                        TagOp::Quotient => a,
                        _ => a.join(b),
                    };
                    assert!(operands.is_subtag_of(out), "{op} {a} {b} gives {out}");
                }
            }
        }
    }

    #[test]
    fn entry_by_entry_rules_are_exact_and_conservative() {
        let set = |tags: &[Tag]| TagSet::from_tags(tags.iter().copied()).closure();
        let sym = set(&[Tag::Symmetric]);
        let diag = set(&[Tag::Diagonal]);
        let pd = set(&[Tag::PositiveDefinite]);
        let orth = set(&[Tag::Orthogonal]);
        let none = TagSet::EMPTY;
        let p = |a, b| tag_result(TagOp::Product, &[a, b]);
        let q = |a, b| tag_result(TagOp::Quotient, &[a, b]);
        // Symmetric needs both operands; a product keeps Diagonal from either.
        assert_eq!(p(sym, sym), sym);
        assert_eq!(p(sym, none), none);
        assert_eq!(p(diag, none), diag);
        assert_eq!(p(none, diag), diag);
        assert_eq!(p(diag, sym), diag);
        // A quotient keeps Diagonal from the numerator only.
        assert_eq!(q(diag, none), diag);
        assert_eq!(q(none, diag), none);
        assert_eq!(q(sym, sym), sym);
        assert_eq!(q(sym, none), none);
        // PositiveDefinite and Orthogonal never survive (Symmetric, which
        // PositiveDefinite implies, does when both have it).
        assert_eq!(p(pd, pd), sym);
        assert_eq!(q(pd, pd), sym);
        assert_eq!(p(orth, orth), none);
        assert_eq!(q(orth, orth), none);
        // A function of one array keeps Symmetric only; of two, if both are.
        let m1 = |a| tag_result(TagOp::Map1, &[a]);
        let m2 = |a, b| tag_result(TagOp::Map2, &[a, b]);
        assert_eq!(m1(sym), sym);
        assert_eq!(m1(diag), sym);
        assert_eq!(m1(pd), sym);
        assert_eq!(m1(orth), none);
        assert_eq!(m2(sym, sym), sym);
        assert_eq!(m2(sym, none), none);
        assert_eq!(m2(diag, pd), sym);
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
