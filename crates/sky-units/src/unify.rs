//! Unification of dimensions containing dimension variables.
//!
//! A [`DimTerm`] is a monomial: a known [`Dimension`] times a product of
//! dimension variables raised to rational powers (`α²·L·T⁻¹`). Because the
//! exponents are rational, dimension terms form a vector space over ℚ, so an
//! equation `s = t` is the linear equation `s / t = 1`. [`DimUnifier`] solves
//! it by eliminating one variable at a time (Kennedy-style unification for
//! abelian groups; over ℚ every non-zero coefficient is invertible, so no
//! integer side conditions arise). The substitution is kept idempotent: bound
//! variables never appear on the right-hand side of another binding.
//!
//! Unification compares dimensions only, never unit scale factors.
//!
//! Every operation here returns `Result`: exponent overflow is reported as
//! [`DimError::ExponentOverflow`] (or [`UnifyError::Overflow`]) and never
//! panics. [`DimTerm`] deliberately has no arithmetic operator impls.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::dimension::{exponent_overflow, write_factors, write_power, DimError, Dimension};
use crate::rational::{Rational, RationalError};

/// A dimension variable. Identity is the numeric id; the name is for display.
#[derive(Clone, Debug)]
pub struct DimVar {
    id: u64,
    name: Arc<str>,
}

impl DimVar {
    /// The variable's unique id within its [`DimUnifier`].
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The display name given at creation.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl PartialEq for DimVar {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for DimVar {}

impl Hash for DimVar {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Ord for DimVar {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for DimVar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for DimVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

/// A known dimension times a product of dimension variables to rational powers.
///
/// Invariant: no variable is stored with a zero exponent, so derived equality
/// is equality of terms.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DimTerm {
    vars: BTreeMap<DimVar, Rational>,
    known: Dimension,
}

impl DimTerm {
    /// A term with no variables.
    pub fn known(d: Dimension) -> Self {
        DimTerm {
            vars: BTreeMap::new(),
            known: d,
        }
    }

    /// The term consisting of a single variable.
    pub fn var(v: DimVar) -> Self {
        let mut vars = BTreeMap::new();
        vars.insert(v, Rational::ONE);
        DimTerm {
            vars,
            known: Dimension::DIMENSIONLESS,
        }
    }

    /// The variable-free factor.
    pub fn known_part(&self) -> Dimension {
        self.known
    }

    /// The variables with their (non-zero) exponents, ordered by id.
    pub fn vars(&self) -> impl Iterator<Item = (&DimVar, Rational)> {
        self.vars.iter().map(|(v, &e)| (v, e))
    }

    /// The exponent of `v` in this term (zero if absent).
    pub fn var_exponent(&self, v: &DimVar) -> Rational {
        self.vars.get(v).copied().unwrap_or(Rational::ZERO)
    }

    /// True if the term contains no variables.
    pub fn is_ground(&self) -> bool {
        self.vars.is_empty()
    }

    /// The dimension, if the term contains no variables.
    pub fn as_dimension(&self) -> Option<Dimension> {
        self.is_ground().then_some(self.known)
    }

    /// `self · rhs`.
    pub fn mul(&self, rhs: &DimTerm) -> Result<DimTerm, DimError> {
        let mut out = self.clone();
        for (v, &e) in &rhs.vars {
            out.mul_var(v, e)?;
        }
        out.known = self.known.checked_mul(rhs.known)?;
        Ok(out)
    }

    /// `self / rhs`.
    pub fn div(&self, rhs: &DimTerm) -> Result<DimTerm, DimError> {
        // Subtract exponents directly rather than multiplying by
        // `rhs.inverse()`: negating an `i64::MIN` exponent overflows even
        // when the quotient itself fits.
        let mut out = self.clone();
        for (v, &e) in &rhs.vars {
            out.add_var_exponent(v, e, Rational::checked_sub)?;
        }
        out.known = self.known.checked_div(rhs.known)?;
        Ok(out)
    }

    /// `self ^ r`.
    pub fn pow_rat(&self, r: Rational) -> Result<DimTerm, DimError> {
        if r.is_zero() {
            return Ok(DimTerm::known(Dimension::DIMENSIONLESS));
        }
        let mut vars = BTreeMap::new();
        for (v, &e) in &self.vars {
            // Both factors are non-zero, so the product is too.
            vars.insert(v.clone(), e.checked_mul(r).map_err(exponent_overflow)?);
        }
        Ok(DimTerm {
            vars,
            known: self.known.pow_rat(r)?,
        })
    }

    /// `self ^ n`.
    pub fn pow_int(&self, n: i64) -> Result<DimTerm, DimError> {
        self.pow_rat(Rational::integer(n))
    }

    /// `self ^ -1`.
    pub fn inverse(&self) -> Result<DimTerm, DimError> {
        self.pow_rat(Rational::integer(-1))
    }

    /// The term with `v` removed (its factor set to 1).
    pub fn without(&self, v: &DimVar) -> Self {
        let mut t = self.clone();
        t.vars.remove(v);
        t
    }

    fn mul_var(&mut self, v: &DimVar, e: Rational) -> Result<(), DimError> {
        self.add_var_exponent(v, e, Rational::checked_add)
    }

    /// Replaces `v`'s exponent `x` with `op(x, e)`, dropping it if zero.
    fn add_var_exponent(
        &mut self,
        v: &DimVar,
        e: Rational,
        op: fn(Rational, Rational) -> Result<Rational, RationalError>,
    ) -> Result<(), DimError> {
        let sum = op(self.var_exponent(v), e).map_err(exponent_overflow)?;
        if sum.is_zero() {
            self.vars.remove(v);
        } else {
            self.vars.insert(v.clone(), sum);
        }
        Ok(())
    }

    /// A diagnostic description: `length (L)` for named ground dimensions,
    /// otherwise the plain rendering (`L·M`, `α·L`).
    pub fn describe(&self) -> String {
        match self.as_dimension().and_then(|d| d.name()) {
            Some(name) => format!("{name} ({self})"),
            None => self.to_string(),
        }
    }
}

impl From<Dimension> for DimTerm {
    fn from(d: Dimension) -> Self {
        DimTerm::known(d)
    }
}

impl From<DimVar> for DimTerm {
    fn from(v: DimVar) -> Self {
        DimTerm::var(v)
    }
}

impl fmt::Display for DimTerm {
    /// Variables first (by id), then base dimensions: `α²·L·T⁻¹`, or `1`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (v, &e) in &self.vars {
            if !first {
                f.write_str("·")?;
            }
            first = false;
            write_power(f, v.name(), e)?;
        }
        let wrote = write_factors(f, &self.known, first)?;
        if first && !wrote {
            f.write_str("1")?;
        }
        Ok(())
    }
}

/// Two dimension terms that cannot be made equal.
///
/// Both sides are reported after applying the unifier's current substitution.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DimMismatch {
    /// The expected side (first argument to [`DimUnifier::unify`]).
    pub expected: DimTerm,
    /// The found side (second argument to [`DimUnifier::unify`]).
    pub found: DimTerm,
}

impl fmt::Display for DimMismatch {
    /// e.g. `cannot unify length (L) with temperature (Θ)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cannot unify {} with {}",
            self.expected.describe(),
            self.found.describe()
        )
    }
}

impl std::error::Error for DimMismatch {}

/// Why [`DimUnifier::unify`] failed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum UnifyError {
    /// The two sides have different dimensions.
    Mismatch(DimMismatch),
    /// Solving would need an exponent that does not fit in `i64`.
    Overflow(DimError),
}

impl fmt::Display for UnifyError {
    /// The mismatch message, or `dimension exponent too large`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnifyError::Mismatch(m) => m.fmt(f),
            UnifyError::Overflow(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for UnifyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            UnifyError::Mismatch(m) => Some(m),
            UnifyError::Overflow(e) => Some(e),
        }
    }
}

impl From<DimMismatch> for UnifyError {
    fn from(m: DimMismatch) -> Self {
        UnifyError::Mismatch(m)
    }
}

impl From<DimError> for UnifyError {
    fn from(e: DimError) -> Self {
        UnifyError::Overflow(e)
    }
}

/// Creates dimension variables and solves equations between dimension terms.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct DimUnifier {
    next_id: u64,
    /// Idempotent: right-hand sides mention only unbound variables.
    subst: BTreeMap<DimVar, DimTerm>,
}

impl DimUnifier {
    /// An empty unifier.
    pub fn new() -> Self {
        Self::default()
    }

    /// A fresh, unbound dimension variable with a display name.
    pub fn fresh(&mut self, name: impl Into<Arc<str>>) -> DimVar {
        let id = self.next_id;
        // Not reachable from exponent arithmetic: exhausting u64 ids would
        // take 2^64 calls to `fresh`.
        self.next_id = self
            .next_id
            .checked_add(1)
            .expect("dimension variable ids exhausted");
        DimVar {
            id,
            name: name.into(),
        }
    }

    /// `t` with every bound variable replaced by its solution.
    pub fn apply(&self, t: &DimTerm) -> Result<DimTerm, DimError> {
        let mut out = DimTerm::known(t.known);
        for (v, &e) in &t.vars {
            match self.subst.get(v) {
                Some(sol) => out = out.mul(&sol.pow_rat(e)?)?,
                None => out.mul_var(v, e)?,
            }
        }
        Ok(out)
    }

    /// The current solution for `v` (just `v` itself if it is unbound).
    pub fn resolve(&self, v: &DimVar) -> Result<DimTerm, DimError> {
        self.apply(&DimTerm::var(v.clone()))
    }

    /// Makes `expected` and `found` equal, extending the substitution, or
    /// reports why they cannot be. Never panics. On any error, including
    /// overflow, the unifier is left unchanged.
    pub fn unify(&mut self, expected: &DimTerm, found: &DimTerm) -> Result<(), UnifyError> {
        let a = self.apply(expected)?;
        let b = self.apply(found)?;
        // a = b  ⇔  a / b = 1.
        let residual = a.div(&b)?;
        let Some((v, c)) = residual.vars().next().map(|(v, c)| (v.clone(), c)) else {
            return if residual.known.is_dimensionless() {
                Ok(())
            } else {
                Err(UnifyError::Mismatch(DimMismatch {
                    expected: a,
                    found: b,
                }))
            };
        };
        // v^c · rest = 1  ⇒  v = rest^(-1/c). `rest` mentions no bound
        // variables (both sides were applied) and does not mention v.
        let power = c
            .checked_recip()
            .and_then(Rational::checked_neg)
            .map_err(exponent_overflow)?;
        let solution = residual.without(&v).pow_rat(power)?;
        // Build the new substitution completely before committing, so an
        // overflow part-way through leaves `self` untouched.
        let mut next = BTreeMap::new();
        for (w, rhs) in &self.subst {
            let e = rhs.var_exponent(&v);
            let rhs = if e.is_zero() {
                rhs.clone()
            } else {
                rhs.without(&v).mul(&solution.pow_rat(e)?)?
            };
            next.insert(w.clone(), rhs);
        }
        next.insert(v, solution);
        self.subst = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(d: Dimension) -> DimTerm {
        DimTerm::known(d)
    }

    fn v(x: &DimVar) -> DimTerm {
        DimTerm::var(x.clone())
    }

    #[test]
    fn ground_equal_and_unequal() {
        let mut u = DimUnifier::new();
        assert!(u
            .unify(&k(Dimension::LENGTH), &k(Dimension::LENGTH))
            .is_ok());
        let err = u
            .unify(&k(Dimension::LENGTH), &k(Dimension::TEMPERATURE))
            .unwrap_err();
        assert!(matches!(err, UnifyError::Mismatch(_)));
        assert_eq!(
            err.to_string(),
            "cannot unify length (L) with temperature (Θ)"
        );
    }

    #[test]
    fn solves_linear_variable() {
        // α·L = L·T  ⇒  α = T
        let mut u = DimUnifier::new();
        let a = u.fresh("α");
        let lhs = v(&a).mul(&k(Dimension::LENGTH)).unwrap();
        let rhs = k(Dimension::LENGTH * Dimension::TIME);
        u.unify(&lhs, &rhs).unwrap();
        assert_eq!(u.resolve(&a), Ok(k(Dimension::TIME)));
    }

    #[test]
    fn solves_squared_variable() {
        // α² = L²  ⇒  α = L
        let mut u = DimUnifier::new();
        let a = u.fresh("α");
        let l2 = Dimension::LENGTH.pow_int(2).unwrap();
        u.unify(&v(&a).pow_int(2).unwrap(), &k(l2)).unwrap();
        assert_eq!(u.resolve(&a), Ok(k(Dimension::LENGTH)));
    }

    #[test]
    fn solves_rational_exponent() {
        // α² = L  ⇒  α = L^(1/2)
        let mut u = DimUnifier::new();
        let a = u.fresh("α");
        u.unify(&v(&a).pow_int(2).unwrap(), &k(Dimension::LENGTH))
            .unwrap();
        assert_eq!(u.resolve(&a).unwrap().to_string(), "L^(1/2)");
    }

    #[test]
    fn variable_chains_stay_idempotent() {
        let mut u = DimUnifier::new();
        let a = u.fresh("α");
        let b = u.fresh("β");
        u.unify(&v(&a), &v(&b)).unwrap();
        u.unify(&v(&b), &k(Dimension::MASS)).unwrap();
        assert_eq!(u.resolve(&a), Ok(k(Dimension::MASS)));
        assert_eq!(u.resolve(&b), Ok(k(Dimension::MASS)));
    }

    #[test]
    fn inconsistent_self_reference_fails_and_leaves_state() {
        // α = α·L has no solution.
        let mut u = DimUnifier::new();
        let a = u.fresh("α");
        let before = u.clone();
        let err = u
            .unify(&v(&a), &v(&a).mul(&k(Dimension::LENGTH)).unwrap())
            .unwrap_err();
        assert_eq!(err.to_string(), "cannot unify α with α·L");
        assert_eq!(u, before);
    }

    #[test]
    fn same_variable_on_both_sides_cancels() {
        // α·L = α·L is trivially fine and binds nothing.
        let mut u = DimUnifier::new();
        let a = u.fresh("α");
        let t = v(&a).mul(&k(Dimension::LENGTH)).unwrap();
        u.unify(&t, &t).unwrap();
        assert_eq!(u.resolve(&a), Ok(v(&a)));
    }

    #[test]
    fn display_of_terms() {
        let mut u = DimUnifier::new();
        let a = u.fresh("α");
        let t = v(&a)
            .pow_int(2)
            .unwrap()
            .mul(&k(Dimension::VELOCITY))
            .unwrap();
        assert_eq!(t.to_string(), "α²·L·T⁻¹");
        assert_eq!(k(Dimension::DIMENSIONLESS).to_string(), "1");
        assert_eq!(k(Dimension::ENERGY).describe(), "energy (L²·M·T⁻²)");
    }

    /// Regression (Eng Lead review of #2): `m^(9223372036854775807/2)` squared
    /// is exact; cubed is an `Err`, never a panic.
    #[test]
    fn term_half_max_exponent_regression() {
        let m = k(crate::lookup("m").unwrap().dim);
        let big = m.pow_rat(Rational::new(i64::MAX, 2)).unwrap();
        let max = k(Dimension::LENGTH.pow_int(i64::MAX).unwrap());
        assert_eq!(big.pow_int(2), Ok(max.clone()));
        assert_eq!(big.mul(&big), Ok(max.clone()));
        assert_eq!(big.pow_int(3), Err(DimError::ExponentOverflow));
        assert_eq!(max.mul(&m), Err(DimError::ExponentOverflow));
        let min = k(Dimension::LENGTH.pow_int(i64::MIN).unwrap());
        assert_eq!(min.inverse(), Err(DimError::ExponentOverflow));
        assert_eq!(m.div(&min), Err(DimError::ExponentOverflow));
    }

    /// Found by `huge_mul_div_inverse_are_exact_or_err`: dividing by a term
    /// whose exponent is `i64::MIN` must not go through its inverse.
    #[test]
    fn div_by_min_exponent_is_exact_when_quotient_fits() {
        let mut u = DimUnifier::new();
        let a = u.fresh("α");
        let min = Rational::integer(i64::MIN);
        let x = v(&a)
            .pow_rat(min)
            .unwrap()
            .mul(&k(Dimension::LENGTH.pow_rat(min).unwrap()))
            .unwrap();
        assert_eq!(x.div(&x), Ok(k(Dimension::DIMENSIONLESS)));
        assert_eq!(x.inverse(), Err(DimError::ExponentOverflow));
    }

    #[test]
    fn variable_exponent_overflow_is_an_error() {
        let mut u = DimUnifier::new();
        let a = u.fresh("α");
        let big = v(&a).pow_int(i64::MAX).unwrap();
        assert_eq!(big.pow_int(2), Err(DimError::ExponentOverflow));
        assert_eq!(big.mul(&v(&a)), Err(DimError::ExponentOverflow));
    }

    #[test]
    fn unify_overflow_is_an_error_and_leaves_state() {
        // α^(1/MAX) = L²  ⇒  α = L^(2·MAX), which does not fit.
        let mut u = DimUnifier::new();
        let a = u.fresh("α");
        let b = u.fresh("β");
        u.unify(&v(&b), &v(&a).pow_int(2).unwrap()).unwrap();
        let before = u.clone();
        let lhs = v(&a).pow_rat(Rational::new(1, i64::MAX)).unwrap();
        let l2 = k(Dimension::LENGTH.pow_int(2).unwrap());
        let err = u.unify(&lhs, &l2).unwrap_err();
        assert_eq!(err, UnifyError::Overflow(DimError::ExponentOverflow));
        assert_eq!(err.to_string(), "dimension exponent too large");
        assert_eq!(u, before);
    }

    #[test]
    fn overflow_while_rewriting_existing_bindings_leaves_state() {
        // α = β^MAX binds α (lowest id). Solving β = L² then fits for β
        // itself, but rewriting α's binding would need α = L^(2·MAX).
        let mut u = DimUnifier::new();
        let a = u.fresh("α");
        let b = u.fresh("β");
        u.unify(&v(&a), &v(&b).pow_int(i64::MAX).unwrap()).unwrap();
        assert_eq!(u.resolve(&b), Ok(v(&b)));
        let before = u.clone();
        let l2 = k(Dimension::LENGTH.pow_int(2).unwrap());
        let err = u.unify(&v(&b), &l2).unwrap_err();
        assert_eq!(err, UnifyError::Overflow(DimError::ExponentOverflow));
        assert_eq!(u, before);
        u.unify(&v(&b), &k(Dimension::LENGTH)).unwrap();
        assert_eq!(
            u.resolve(&a),
            Ok(k(Dimension::LENGTH.pow_int(i64::MAX).unwrap()))
        );
    }
}
