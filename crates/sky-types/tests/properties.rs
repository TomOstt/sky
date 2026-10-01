//! Property tests: lattice laws for `TagSet`, and soundness and
//! transactionality of shape unification and the shape rules.

use proptest::prelude::*;
use sky_types::{
    elementwise, inverse, matmul, matmul_tags, preserved_by_inverse, preserved_by_transpose, Shape,
    ShapeDim, ShapeUnifier, ShapeVar, Tag, TagSet, TensorType,
};

fn tag_set() -> impl Strategy<Value = TagSet> {
    (0u8..16).prop_map(|bits| {
        TagSet::from_tags(
            Tag::ALL
                .into_iter()
                .enumerate()
                .filter(move |(i, _)| bits & (1 << i) != 0)
                .map(|(_, t)| t),
        )
    })
}

/// A dim spec: a known size 0..4 or one of four variables.
fn dim_spec() -> impl Strategy<Value = Result<u64, usize>> {
    prop_oneof![(0u64..4).prop_map(Ok), (0usize..4).prop_map(Err)]
}

fn shape_spec(max_rank: usize) -> impl Strategy<Value = Vec<Result<u64, usize>>> {
    prop::collection::vec(dim_spec(), 0..=max_rank)
}

fn build(spec: &[Result<u64, usize>], vars: &[ShapeVar]) -> Shape {
    Shape::new(
        spec.iter()
            .map(|d| match d {
                Ok(n) => ShapeDim::Known(*n),
                Err(i) => ShapeDim::Var(vars[*i].clone()),
            })
            .collect(),
    )
}

fn setup() -> (ShapeUnifier, Vec<ShapeVar>) {
    let mut u = ShapeUnifier::new();
    let vars = ["A", "B", "C", "D"].map(|n| u.fresh(n)).to_vec();
    (u, vars)
}

proptest! {
    #[test]
    fn closure_is_idempotent_and_extensive(s in tag_set()) {
        let c = s.closure();
        prop_assert_eq!(c.closure(), c);
        prop_assert!(c.is_closed());
        for t in s.iter() {
            prop_assert!(c.contains(t));
        }
    }

    #[test]
    fn closure_is_monotone(a in tag_set(), b in tag_set()) {
        let union = TagSet::from_tags(a.iter().chain(b.iter()));
        prop_assert!(union.closure().is_subtag_of(a.closure()));
    }

    #[test]
    fn subtag_is_a_partial_order(a in tag_set(), b in tag_set(), c in tag_set()) {
        prop_assert!(a.is_subtag_of(a));
        if a.is_subtag_of(b) && b.is_subtag_of(a) {
            prop_assert_eq!(a.closure(), b.closure());
        }
        if a.is_subtag_of(b) && b.is_subtag_of(c) {
            prop_assert!(a.is_subtag_of(c));
        }
    }

    #[test]
    fn meet_and_join_are_commutative_and_associative(a in tag_set(), b in tag_set(), c in tag_set()) {
        prop_assert_eq!(a.meet(b), b.meet(a));
        prop_assert_eq!(a.join(b), b.join(a));
        prop_assert_eq!(a.meet(b).meet(c), a.meet(b.meet(c)));
        prop_assert_eq!(a.join(b).join(c), a.join(b.join(c)));
    }

    #[test]
    fn meet_and_join_are_absorptive(a in tag_set(), b in tag_set()) {
        prop_assert_eq!(a.meet(a.join(b)), a.closure());
        prop_assert_eq!(a.join(a.meet(b)), a.closure());
    }

    #[test]
    fn meet_and_join_are_bounds(a in tag_set(), b in tag_set(), c in tag_set()) {
        let (m, j) = (a.meet(b), a.join(b));
        prop_assert!(m.is_closed() && j.is_closed());
        prop_assert!(m.is_subtag_of(a) && m.is_subtag_of(b));
        prop_assert!(a.is_subtag_of(j) && b.is_subtag_of(j));
        // Greatest lower bound and least upper bound.
        if c.is_subtag_of(a) && c.is_subtag_of(b) {
            prop_assert!(c.is_subtag_of(m));
        }
        if a.is_subtag_of(c) && b.is_subtag_of(c) {
            prop_assert!(j.is_subtag_of(c));
        }
    }

    /// Propagated tags are closed and never exceed what the operands guarantee.
    #[test]
    fn propagation_is_closed_and_conservative(a in tag_set(), b in tag_set()) {
        for out in [preserved_by_inverse(a), preserved_by_transpose(a)] {
            prop_assert!(out.is_closed());
            prop_assert!(a.is_subtag_of(out));
        }
        let m = matmul_tags(a, b);
        prop_assert!(m.is_closed());
        // matmul never claims a tag that either operand lacks ...
        prop_assert!(a.join(b).is_subtag_of(m));
        // ... never claims PositiveDefinite ...
        prop_assert!(!m.contains(Tag::PositiveDefinite));
        // ... and claims Symmetric only as a consequence of Diagonal.
        prop_assert!(!m.contains(Tag::Symmetric) || m.contains(Tag::Diagonal));

        // Elementwise ops (`+`, `-`, ...) keep only Symmetric and Diagonal,
        // and only when both operands carry them: e.g. `P - P` is the zero
        // matrix, not PD, and `Q + Q` is not Orthogonal.
        let mut u = ShapeUnifier::new();
        let sq = Shape::matrix(3u64, 3u64);
        let ta = TensorType::tagged(&mut u, sq.clone(), a).unwrap();
        let tb = TensorType::tagged(&mut u, sq, b).unwrap();
        for (l, r) in [(&ta, &tb), (&tb, &ta), (&ta, &ta)] {
            let e = elementwise(&mut u, l, r).unwrap();
            prop_assert!(e.tags().is_closed());
            prop_assert!(l.tags().join(r.tags()).is_subtag_of(e.tags()));
            let both = l.tags().join(r.tags());
            for t in e.tags().iter() {
                prop_assert!(
                    matches!(t, Tag::Symmetric | Tag::Diagonal) && both.contains(t),
                    "elementwise kept {}",
                    e.tags()
                );
            }
        }
    }

    /// After any sequence of unifications, every successful one still holds,
    /// and failures leave the unifier unchanged.
    #[test]
    fn shape_unify_is_sound_and_transactional(
        eqs in prop::collection::vec((shape_spec(3), shape_spec(3)), 1..8)
    ) {
        let (mut u, vars) = setup();
        let mut solved = Vec::new();
        for (l, r) in &eqs {
            let (l, r) = (build(l, &vars), build(r, &vars));
            let before = u.clone();
            match u.unify(&l, &r) {
                Ok(()) => {
                    prop_assert_eq!(u.apply(&l), u.apply(&r));
                    solved.push((l, r));
                }
                Err(_) => prop_assert_eq!(&u, &before),
            }
        }
        for (l, r) in &solved {
            prop_assert_eq!(u.apply(l), u.apply(r));
        }
    }

    #[test]
    fn unify_is_reflexive_and_rank_checked(a in shape_spec(3), b in shape_spec(3)) {
        let (mut u, vars) = setup();
        let (a, b) = (build(&a, &vars), build(&b, &vars));
        prop_assert!(u.unify(&a, &a).is_ok());
        if a.rank() != b.rank() {
            let rank_mismatch = matches!(
                u.unify(&a, &b).map_err(|e| e.into_kind()),
                Err(sky_types::ShapeErrorKind::RankMismatch { .. })
            );
            prop_assert!(rank_mismatch);
        }
    }

    /// matmul and inverse either succeed with a sound result or leave the
    /// unifier unchanged.
    #[test]
    fn rules_are_transactional(a in shape_spec(3), b in shape_spec(3)) {
        let (mut u, vars) = setup();
        let (a, b) = (build(&a, &vars), build(&b, &vars));
        let (ta, tb) = (TensorType::untagged(a.clone()), TensorType::untagged(b.clone()));
        let before = u.clone();
        match matmul(&mut u, &ta, &tb) {
            Ok(_) => {
                let (la, lb) = (u.apply(&a), u.apply(&b));
                let k_left = &la.dims()[la.rank() - 1];
                prop_assert_eq!(k_left, &lb.dims()[0]);
            }
            Err(_) => prop_assert_eq!(&u, &before),
        }
        let before = u.clone();
        match inverse(&mut u, &ta) {
            Ok(inv) => {
                let s = u.apply(inv.shape());
                prop_assert!(s.rank() == 0 || (s.rank() == 2 && s.dims()[0] == s.dims()[1]));
            }
            Err(_) => prop_assert_eq!(&u, &before),
        }
    }
}
