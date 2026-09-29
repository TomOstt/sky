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

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Div, Mul};
use std::sync::Arc;

use crate::dimension::{write_factors, write_power, Dimension};
use crate::rational::Rational;

/// A dimension variable. Identity is the numeric id; the name is for display.
#[derive(Clone, Debug)]
pub struct DimVar {
    id: u32,
    name: Arc<str>,
}

impl DimVar {
    /// The variable's unique id within its [`DimUnifier`].
    pub fn id(&self) -> u32 {
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

    /// `self ^ r`. Panics on exponent overflow.
    pub fn pow_rat(&self, r: Rational) -> Self {
        if r.is_zero() {
            return DimTerm::known(Dimension::DIMENSIONLESS);
        }
        DimTerm {
            vars: self.vars.iter().map(|(v, &e)| (v.clone(), e * r)).collect(),
            known: self.known.pow_rat(r),
        }
    }

    /// `self ^ n`. Panics on exponent overflow.
    pub fn pow_int(&self, n: i64) -> Self {
        self.pow_rat(Rational::integer(n))
    }

    /// `self ^ -1`.
    pub fn inverse(&self) -> Self {
        self.pow_int(-1)
    }

    /// The term with `v` removed (its factor set to 1).
    pub fn without(&self, v: &DimVar) -> Self {
        let mut t = self.clone();
        t.vars.remove(v);
        t
    }

    fn mul_var(&mut self, v: &DimVar, e: Rational) {
        let sum = self.var_exponent(v) + e;
        if sum.is_zero() {
            self.vars.remove(v);
        } else {
            self.vars.insert(v.clone(), sum);
        }
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

impl Mul for DimTerm {
    type Output = DimTerm;
    fn mul(mut self, rhs: Self) -> Self {
        for (v, e) in &rhs.vars {
            self.mul_var(v, *e);
        }
        self.known = self.known * rhs.known;
        self
    }
}

impl Div for DimTerm {
    type Output = DimTerm;
    // Division is multiplication by the inverse, which clippy flags as suspicious.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse()
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

/// Creates dimension variables and solves equations between dimension terms.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct DimUnifier {
    next_id: u32,
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
        self.next_id = self
            .next_id
            .checked_add(1)
            .expect("too many dimension variables");
        DimVar {
            id,
            name: name.into(),
        }
    }

    /// `t` with every bound variable replaced by its solution.
    pub fn apply(&self, t: &DimTerm) -> DimTerm {
        let mut out = DimTerm::known(t.known);
        for (v, &e) in &t.vars {
            match self.subst.get(v) {
                Some(sol) => out = out * sol.pow_rat(e),
                None => out.mul_var(v, e),
            }
        }
        out
    }

    /// The current solution for `v` (just `v` itself if it is unbound).
    pub fn resolve(&self, v: &DimVar) -> DimTerm {
        self.apply(&DimTerm::var(v.clone()))
    }

    /// Makes `expected` and `found` equal, extending the substitution, or
    /// reports why they cannot be. On error the unifier is left unchanged.
    ///
    /// Panics only if an exponent overflows `i64`.
    pub fn unify(&mut self, expected: &DimTerm, found: &DimTerm) -> Result<(), DimMismatch> {
        let a = self.apply(expected);
        let b = self.apply(found);
        // a = b  ⇔  a / b = 1.
        let residual = a.clone() / b.clone();
        let Some((v, c)) = residual.vars().next().map(|(v, c)| (v.clone(), c)) else {
            return if residual.known.is_dimensionless() {
                Ok(())
            } else {
                Err(DimMismatch {
                    expected: a,
                    found: b,
                })
            };
        };
        // v^c · rest = 1  ⇒  v = rest^(-1/c). `rest` mentions no bound
        // variables (both sides were applied) and does not mention v.
        let solution = residual.without(&v).pow_rat(-c.recip());
        for rhs in self.subst.values_mut() {
            let e = rhs.var_exponent(&v);
            if !e.is_zero() {
                *rhs = rhs.without(&v) * solution.pow_rat(e);
            }
        }
        self.subst.insert(v, solution);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(d: Dimension) -> DimTerm {
        DimTerm::known(d)
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
        let lhs = DimTerm::var(a.clone()) * k(Dimension::LENGTH);
        let rhs = k(Dimension::LENGTH * Dimension::TIME);
        u.unify(&lhs, &rhs).unwrap();
        assert_eq!(u.resolve(&a), k(Dimension::TIME));
    }

    #[test]
    fn solves_squared_variable() {
        // α² = L²  ⇒  α = L
        let mut u = DimUnifier::new();
        let a = u.fresh("α");
        u.unify(
            &DimTerm::var(a.clone()).pow_int(2),
            &k(Dimension::LENGTH.pow_int(2)),
        )
        .unwrap();
        assert_eq!(u.resolve(&a), k(Dimension::LENGTH));
    }

    #[test]
    fn solves_rational_exponent() {
        // α² = L  ⇒  α = L^(1/2)
        let mut u = DimUnifier::new();
        let a = u.fresh("α");
        u.unify(&DimTerm::var(a.clone()).pow_int(2), &k(Dimension::LENGTH))
            .unwrap();
        assert_eq!(u.resolve(&a).to_string(), "L^(1/2)");
    }

    #[test]
    fn variable_chains_stay_idempotent() {
        let mut u = DimUnifier::new();
        let a = u.fresh("α");
        let b = u.fresh("β");
        u.unify(&DimTerm::var(a.clone()), &DimTerm::var(b.clone()))
            .unwrap();
        u.unify(&DimTerm::var(b.clone()), &k(Dimension::MASS))
            .unwrap();
        assert_eq!(u.resolve(&a), k(Dimension::MASS));
        assert_eq!(u.resolve(&b), k(Dimension::MASS));
    }

    #[test]
    fn inconsistent_self_reference_fails_and_leaves_state() {
        // α = α·L has no solution.
        let mut u = DimUnifier::new();
        let a = u.fresh("α");
        let before = u.clone();
        let err = u
            .unify(
                &DimTerm::var(a.clone()),
                &(DimTerm::var(a) * k(Dimension::LENGTH)),
            )
            .unwrap_err();
        assert_eq!(err.to_string(), "cannot unify α with α·L");
        assert_eq!(u, before);
    }

    #[test]
    fn same_variable_on_both_sides_cancels() {
        // α·L = α·L is trivially fine and binds nothing.
        let mut u = DimUnifier::new();
        let a = u.fresh("α");
        let t = DimTerm::var(a.clone()) * k(Dimension::LENGTH);
        u.unify(&t, &t).unwrap();
        assert_eq!(u.resolve(&a), DimTerm::var(a));
    }

    #[test]
    fn display_of_terms() {
        let mut u = DimUnifier::new();
        let a = u.fresh("α");
        let t = DimTerm::var(a).pow_int(2) * k(Dimension::VELOCITY);
        assert_eq!(t.to_string(), "α²·L·T⁻¹");
        assert_eq!(k(Dimension::DIMENSIONLESS).to_string(), "1");
        assert_eq!(k(Dimension::ENERGY).describe(), "energy (L²·M·T⁻²)");
    }
}
