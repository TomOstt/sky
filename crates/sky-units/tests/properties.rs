//! Property tests: group laws for `Dimension`, field laws for `Rational`,
//! and soundness of `DimUnifier`.

use proptest::prelude::*;
use sky_units::{DimTerm, DimUnifier, Dimension, Rational};

fn rational() -> impl Strategy<Value = Rational> {
    (-6i64..=6, 1i64..=4).prop_map(|(n, d)| Rational::new(n, d))
}

fn nonzero_rational() -> impl Strategy<Value = Rational> {
    rational().prop_filter("non-zero", |r| !r.is_zero())
}

fn dimension() -> impl Strategy<Value = Dimension> {
    prop::array::uniform7(rational()).prop_map(Dimension::from_exponents)
}

/// A term over three variables: known part plus one exponent per variable.
#[derive(Clone, Debug)]
struct TermSpec {
    known: Dimension,
    exps: [Rational; 3],
}

fn term_spec() -> impl Strategy<Value = TermSpec> {
    (
        dimension(),
        prop::array::uniform3(prop_oneof![Just(Rational::ZERO), rational()]),
    )
        .prop_map(|(known, exps)| TermSpec { known, exps })
}

fn build(spec: &TermSpec, vars: &[DimTerm; 3]) -> DimTerm {
    spec.exps
        .iter()
        .zip(vars)
        .fold(DimTerm::known(spec.known), |t, (&e, v)| t * v.pow_rat(e))
}

proptest! {
    #[test]
    fn rational_normalization(n in -50i64..=50, d in 1i64..=50, k in 1i64..=20) {
        let r = Rational::new(n, d);
        prop_assert_eq!(Rational::new(n * k, d * k), r);
        prop_assert_eq!(Rational::new(-n, -d), r);
        prop_assert!(r.denom() > 0);
    }

    #[test]
    fn rational_field_laws(a in rational(), b in rational(), c in rational()) {
        prop_assert_eq!((a + b) + c, a + (b + c));
        prop_assert_eq!(a * (b + c), a * b + a * c);
        prop_assert_eq!(a - a, Rational::ZERO);
        if !b.is_zero() {
            prop_assert_eq!((a / b) * b, a);
        }
    }

    #[test]
    fn mul_is_associative(a in dimension(), b in dimension(), c in dimension()) {
        prop_assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn mul_is_commutative(a in dimension(), b in dimension()) {
        prop_assert_eq!(a * b, b * a);
    }

    #[test]
    fn dimensionless_is_identity(a in dimension()) {
        prop_assert_eq!(a * Dimension::DIMENSIONLESS, a);
        prop_assert_eq!(Dimension::DIMENSIONLESS * a, a);
    }

    #[test]
    fn inverse_cancels(a in dimension()) {
        prop_assert!((a * a.inverse()).is_dimensionless());
        prop_assert_eq!(a.inverse().inverse(), a);
    }

    #[test]
    fn div_is_mul_by_inverse(a in dimension(), b in dimension()) {
        prop_assert_eq!(a / b, a * b.inverse());
    }

    #[test]
    fn pow_distributes_over_mul(a in dimension(), b in dimension(), n in -5i64..=5, r in rational()) {
        prop_assert_eq!((a * b).pow_int(n), a.pow_int(n) * b.pow_int(n));
        prop_assert_eq!((a * b).pow_rat(r), a.pow_rat(r) * b.pow_rat(r));
    }

    #[test]
    fn pow_int_composes(a in dimension(), m in -5i64..=5, n in -5i64..=5) {
        prop_assert_eq!(a.pow_int(m).pow_int(n), a.pow_int(m * n));
        prop_assert_eq!(a.pow_int(m) * a.pow_int(n), a.pow_int(m + n));
    }

    #[test]
    fn rational_pow_round_trips(a in dimension(), r in nonzero_rational()) {
        prop_assert_eq!(a.pow_rat(r).pow_rat(r.recip()), a);
    }

    #[test]
    fn unify_is_reflexive(spec in term_spec()) {
        let mut u = DimUnifier::new();
        let vars = [u.fresh("a"), u.fresh("b"), u.fresh("c")].map(DimTerm::var);
        let t = build(&spec, &vars);
        prop_assert!(u.unify(&t, &t).is_ok());
    }

    /// After any sequence of unifications, every equation that succeeded
    /// still holds under the final substitution, and failures change nothing.
    #[test]
    fn unify_is_sound(eqs in prop::collection::vec((term_spec(), term_spec()), 1..6)) {
        let mut u = DimUnifier::new();
        let vars = [u.fresh("a"), u.fresh("b"), u.fresh("c")].map(DimTerm::var);
        let mut solved = Vec::new();
        for (l, r) in &eqs {
            let (l, r) = (build(l, &vars), build(r, &vars));
            let before = u.clone();
            match u.unify(&l, &r) {
                Ok(()) => {
                    prop_assert_eq!(u.apply(&l), u.apply(&r));
                    solved.push((l, r));
                }
                Err(e) => {
                    prop_assert_eq!(&u, &before);
                    prop_assert_ne!(&e.expected, &e.found);
                }
            }
        }
        for (l, r) in &solved {
            prop_assert_eq!(u.apply(l), u.apply(r));
        }
    }

    /// `α^k · d1 = d2` with k ≠ 0 always has the unique solution (d2/d1)^(1/k).
    #[test]
    fn single_variable_equations_are_solved(
        k in nonzero_rational(), d1 in dimension(), d2 in dimension()
    ) {
        let mut u = DimUnifier::new();
        let a = u.fresh("α");
        let lhs = DimTerm::var(a.clone()).pow_rat(k) * DimTerm::known(d1);
        prop_assert!(u.unify(&lhs, &DimTerm::known(d2)).is_ok());
        prop_assert_eq!(u.resolve(&a), DimTerm::known((d2 / d1).pow_rat(k.recip())));
    }
}
