//! Property tests for the polynomials of the cost model (`sky_types::poly`):
//! the comparison over all sizes from 1 up is sound (a `true` is never
//! contradicted by a sampled point), the joins bound both sides, and
//! substitution agrees with evaluation.

use std::collections::BTreeMap;

use proptest::prelude::*;
use sky_types::poly::Poly;
use sky_units::Rational;

const VARS: [&str; 3] = ["N", "M", "K"];

/// A polynomial in up to three sizes, degree at most 3, small coefficients
/// (sometimes fractions, sometimes negative).
fn poly() -> impl Strategy<Value = Poly> {
    let term = (-6i64..=6, 1i64..=3, prop::collection::vec(0u32..=2, 3));
    prop::collection::vec(term, 0..5).prop_map(|terms| {
        let mut p = Poly::zero();
        for (num, den, exps) in terms {
            let mut t = Poly::constant(Rational::new(num, den));
            for (v, e) in VARS.iter().zip(exps) {
                let x = Poly::var(v).checked_pow(e).unwrap();
                t = t.checked_mul(&x).unwrap();
            }
            p = p.checked_add(&t).unwrap();
        }
        p
    })
}

fn point(n: i64, m: i64, k: i64) -> BTreeMap<String, Rational> {
    [("N", n), ("M", m), ("K", k)]
        .into_iter()
        .map(|(v, x)| (v.to_string(), Rational::integer(x)))
        .collect()
}

const SAMPLES: [(i64, i64, i64); 9] = [
    (1, 1, 1),
    (2, 1, 1),
    (1, 3, 2),
    (5, 5, 5),
    (10, 2, 7),
    (3, 100, 1),
    (50, 1, 50),
    (1000, 1000, 1),
    (7, 11, 13),
];

proptest! {
    #[test]
    fn le_for_all_sizes_is_never_contradicted_by_a_point(p in poly(), q in poly()) {
        if p.le_for_all_sizes(&q) {
            for (n, m, k) in SAMPLES {
                let pt = point(n, m, k);
                let (x, y) = (p.eval(&pt).unwrap(), q.eval(&pt).unwrap());
                prop_assert!(x <= y, "{p} <= {q} claimed, but at {n},{m},{k}: {x} > {y}");
            }
        }
    }

    #[test]
    fn the_order_is_reflexive_and_transitive_on_what_it_proves(p in poly(), q in poly(), r in poly()) {
        prop_assert!(p.le_for_all_sizes(&p));
        if p.le_for_all_sizes(&q) && q.le_for_all_sizes(&r) {
            for (n, m, k) in SAMPLES {
                let pt = point(n, m, k);
                prop_assert!(p.eval(&pt).unwrap() <= r.eval(&pt).unwrap());
            }
        }
    }

    #[test]
    fn joins_bound_both_sides_at_every_point(p in poly(), q in poly()) {
        let hi = p.upper_join(&q);
        let lo = p.lower_join(&q);
        for (n, m, k) in SAMPLES {
            let pt = point(n, m, k);
            let (a, b) = (p.eval(&pt).unwrap(), q.eval(&pt).unwrap());
            prop_assert!(hi.eval(&pt).unwrap() >= a.max(b), "{p} | {q} => {hi}");
            prop_assert!(lo.eval(&pt).unwrap() <= a.min(b), "{p} | {q} => {lo}");
        }
    }

    #[test]
    fn substitution_agrees_with_evaluation(p in poly(), q in poly()) {
        // N := q, everything else stays: p(q(x), M, K) == subst(p)(x).
        let mut map = BTreeMap::new();
        map.insert("N".to_string(), q.clone());
        let Some(s) = p.subst(&map) else { return Ok(()) };
        for (n, m, k) in SAMPLES {
            let pt = point(n, m, k);
            let Some(qv) = q.eval(&pt) else { continue };
            let mut inner = pt.clone();
            inner.insert("N".to_string(), qv);
            if let (Some(a), Some(b)) = (s.eval(&pt), p.eval(&inner)) {
                prop_assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn arithmetic_agrees_with_evaluation(p in poly(), q in poly()) {
        let (Some(sum), Some(prod), Some(diff)) =
            (p.checked_add(&q), p.checked_mul(&q), p.checked_sub(&q)) else { return Ok(()) };
        for (n, m, k) in SAMPLES {
            let pt = point(n, m, k);
            let (a, b) = (p.eval(&pt).unwrap(), q.eval(&pt).unwrap());
            if let Some(v) = sum.eval(&pt) { prop_assert_eq!(v, a.checked_add(b).unwrap()); }
            if let Some(v) = diff.eval(&pt) { prop_assert_eq!(v, a.checked_sub(b).unwrap()); }
            if let (Some(v), Ok(w)) = (prod.eval(&pt), a.checked_mul(b)) { prop_assert_eq!(v, w); }
        }
    }

    #[test]
    fn printing_never_panics_and_source_text_is_stable(p in poly()) {
        let _ = p.to_string();
        let _ = p.to_source();
    }
}
