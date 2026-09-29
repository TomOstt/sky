//! Property tests: group laws for `Dimension`, field laws for `Rational`,
//! soundness of `DimUnifier`, and overflow safety (huge exponents yield
//! `Err`, never a panic).

use proptest::prelude::*;
use sky_units::{DimError, DimTerm, DimUnifier, Dimension, Rational, UnifyError};

fn rational() -> impl Strategy<Value = Rational> {
    (-6i64..=6, 1i64..=4).prop_map(|(n, d)| Rational::new(n, d))
}

fn nonzero_rational() -> impl Strategy<Value = Rational> {
    rational().prop_filter("non-zero", |r| !r.is_zero())
}

fn dimension() -> impl Strategy<Value = Dimension> {
    prop::array::uniform7(rational()).prop_map(Dimension::from_exponents)
}

/// Numerators clustered at the i64 extremes, plus small and arbitrary values.
fn huge_numer() -> impl Strategy<Value = i64> {
    prop_oneof![
        (i64::MAX - 8)..=i64::MAX,
        i64::MIN..=(i64::MIN + 8),
        -8i64..=8,
        any::<i64>(),
    ]
}

/// Positive denominators clustered near i64::MAX, plus small and arbitrary ones.
fn huge_denom() -> impl Strategy<Value = i64> {
    prop_oneof![1i64..=8, (i64::MAX - 8)..=i64::MAX, 1i64..=i64::MAX]
}

fn huge_rational() -> impl Strategy<Value = Rational> {
    (huge_numer(), huge_denom()).prop_map(|(n, d)| Rational::new(n, d))
}

fn huge_dimension() -> impl Strategy<Value = Dimension> {
    prop::array::uniform7(prop_oneof![rational(), huge_rational()])
        .prop_map(Dimension::from_exponents)
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Independent reference: the exact value of `num/den` (den > 0) as a
/// `Rational`, or `None` if its lowest-terms form does not fit in i64.
fn exact(num: i128, den: i128) -> Option<Rational> {
    assert!(den > 0);
    let g = gcd(num.unsigned_abs(), den.unsigned_abs()) as i128;
    let (n, d) = (i64::try_from(num / g).ok()?, i64::try_from(den / g).ok()?);
    Some(Rational::new(n, d))
}

fn wide(r: Rational) -> (i128, i128) {
    (i128::from(r.numer()), i128::from(r.denom()))
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

fn huge_term_spec() -> impl Strategy<Value = TermSpec> {
    (
        huge_dimension(),
        prop::array::uniform3(prop_oneof![
            Just(Rational::ZERO),
            rational(),
            huge_rational()
        ]),
    )
        .prop_map(|(known, exps)| TermSpec { known, exps })
}

fn build(spec: &TermSpec, vars: &[DimTerm; 3]) -> Result<DimTerm, DimError> {
    let mut t = DimTerm::known(spec.known);
    for (&e, v) in spec.exps.iter().zip(vars) {
        t = t.mul(&v.pow_rat(e)?)?;
    }
    Ok(t)
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
        prop_assert_eq!(a.checked_mul(b)?.checked_mul(c), a.checked_mul(b.checked_mul(c)?));
    }

    #[test]
    fn mul_is_commutative(a in dimension(), b in dimension()) {
        prop_assert_eq!(a.checked_mul(b), b.checked_mul(a));
    }

    #[test]
    fn dimensionless_is_identity(a in dimension()) {
        prop_assert_eq!(a.checked_mul(Dimension::DIMENSIONLESS), Ok(a));
        prop_assert_eq!(Dimension::DIMENSIONLESS.checked_mul(a), Ok(a));
    }

    #[test]
    fn inverse_cancels(a in dimension()) {
        prop_assert!(a.checked_mul(a.inverse()?)?.is_dimensionless());
        prop_assert_eq!(a.inverse()?.inverse(), Ok(a));
    }

    #[test]
    fn div_is_mul_by_inverse(a in dimension(), b in dimension()) {
        prop_assert_eq!(a.checked_div(b), a.checked_mul(b.inverse()?));
    }

    #[test]
    fn pow_distributes_over_mul(a in dimension(), b in dimension(), n in -5i64..=5, r in rational()) {
        let ab = a.checked_mul(b)?;
        prop_assert_eq!(ab.pow_int(n), a.pow_int(n)?.checked_mul(b.pow_int(n)?));
        prop_assert_eq!(ab.pow_rat(r), a.pow_rat(r)?.checked_mul(b.pow_rat(r)?));
    }

    #[test]
    fn pow_int_composes(a in dimension(), m in -5i64..=5, n in -5i64..=5) {
        prop_assert_eq!(a.pow_int(m)?.pow_int(n), a.pow_int(m * n));
        prop_assert_eq!(a.pow_int(m)?.checked_mul(a.pow_int(n)?), a.pow_int(m + n));
    }

    #[test]
    fn rational_pow_round_trips(a in dimension(), r in nonzero_rational()) {
        prop_assert_eq!(a.pow_rat(r)?.pow_rat(r.recip()), Ok(a));
    }

    #[test]
    fn unify_is_reflexive(spec in term_spec()) {
        let mut u = DimUnifier::new();
        let vars = [u.fresh("a"), u.fresh("b"), u.fresh("c")].map(DimTerm::var);
        let t = build(&spec, &vars)?;
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
            let (l, r) = (build(l, &vars)?, build(r, &vars)?);
            let before = u.clone();
            match u.unify(&l, &r) {
                Ok(()) => {
                    prop_assert_eq!(u.apply(&l), u.apply(&r));
                    solved.push((l, r));
                }
                Err(UnifyError::Mismatch(m)) => {
                    prop_assert_eq!(&u, &before);
                    prop_assert_ne!(&m.expected, &m.found);
                }
                Err(e @ UnifyError::Overflow(_)) => prop_assert!(false, "small case overflowed: {}", e),
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
        let lhs = DimTerm::var(a.clone()).pow_rat(k)?.mul(&DimTerm::known(d1))?;
        prop_assert!(u.unify(&lhs, &DimTerm::known(d2)).is_ok());
        let want = d2.checked_div(d1)?.pow_rat(k.recip())?;
        prop_assert_eq!(u.resolve(&a), Ok(DimTerm::known(want)));
    }

    // ---- Overflow safety: huge exponents give exact results or Err, never a panic.

    #[test]
    fn huge_pow_rat_is_exact_or_err(e in huge_rational(), r in huge_rational()) {
        let d = Dimension::LENGTH.pow_rat(e)?;
        let ((a, b), (c, dd)) = (wide(e), wide(r));
        match exact(a * c, b * dd) {
            Some(x) => prop_assert_eq!(d.pow_rat(r), Dimension::LENGTH.pow_rat(x)),
            None => prop_assert_eq!(d.pow_rat(r), Err(DimError::ExponentOverflow)),
        }
        // The same through the term API.
        let t = DimTerm::known(d).pow_rat(r);
        prop_assert_eq!(t.map(|t| t.as_dimension()), d.pow_rat(r).map(Some));
    }

    #[test]
    fn huge_pow_int_is_exact_or_err(e in huge_rational(), n in huge_numer()) {
        let d = Dimension::TIME.pow_rat(e)?;
        let (a, b) = wide(e);
        match exact(a * i128::from(n), b) {
            Some(x) => prop_assert_eq!(d.pow_int(n), Dimension::TIME.pow_rat(x)),
            None => prop_assert_eq!(d.pow_int(n), Err(DimError::ExponentOverflow)),
        }
    }

    #[test]
    fn huge_mul_div_inverse_are_exact_or_err(e1 in huge_rational(), e2 in huge_rational()) {
        let (x, y) = (Dimension::MASS.pow_rat(e1)?, Dimension::MASS.pow_rat(e2)?);
        let ((a, b), (c, d)) = (wide(e1), wide(e2));
        let expect = |r: Option<Rational>| match r {
            Some(r) => Dimension::MASS.pow_rat(r),
            None => Err(DimError::ExponentOverflow),
        };
        prop_assert_eq!(x.checked_mul(y), expect(exact(a * d + c * b, b * d)));
        prop_assert_eq!(x.checked_div(y), expect(exact(a * d - c * b, b * d)));
        prop_assert_eq!(x.inverse(), expect(exact(-a, b)));
        let (tx, ty) = (DimTerm::known(x), DimTerm::known(y));
        prop_assert_eq!(tx.mul(&ty).map(|t| t.as_dimension()), x.checked_mul(y).map(Some));
        prop_assert_eq!(tx.div(&ty).map(|t| t.as_dimension()), x.checked_div(y).map(Some));
    }

    /// `m^(i64::MAX/2)` raised to any power: exact when it fits, Err otherwise.
    #[test]
    fn half_max_meter_powers(n in -4i64..=4) {
        let big = Dimension::LENGTH.pow_rat(Rational::new(i64::MAX, 2))?;
        let r = big.pow_int(n);
        match n {
            -2..=2 => prop_assert!(r.is_ok()),
            _ => prop_assert_eq!(r, Err(DimError::ExponentOverflow)),
        }
    }

    /// With huge exponents, unification never panics, keeps solved equations
    /// true, and leaves the unifier unchanged on any error.
    #[test]
    fn huge_unify_never_panics_and_is_transactional(
        eqs in prop::collection::vec((huge_term_spec(), huge_term_spec()), 1..6)
    ) {
        let mut u = DimUnifier::new();
        let vars = [u.fresh("a"), u.fresh("b"), u.fresh("c")].map(DimTerm::var);
        for (l, r) in &eqs {
            let (Ok(l), Ok(r)) = (build(l, &vars), build(r, &vars)) else { continue };
            let before = u.clone();
            match u.unify(&l, &r) {
                Ok(()) => {
                    // Applying can itself overflow for extreme solutions;
                    // when it doesn't, both sides must agree.
                    if let (Ok(al), Ok(ar)) = (u.apply(&l), u.apply(&r)) {
                        prop_assert_eq!(al, ar);
                    }
                }
                Err(_) => prop_assert_eq!(&u, &before),
            }
        }
    }
}
