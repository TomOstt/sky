//! Physical dimensions as exponent vectors over the seven SI base dimensions.

use std::fmt;
use std::ops::{Div, Mul};

use crate::rational::{Rational, RationalError};

/// One of the seven SI base dimensions, in SI order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum BaseDim {
    /// Length, L.
    Length,
    /// Mass, M.
    Mass,
    /// Time, T.
    Time,
    /// Electric current, I.
    Current,
    /// Thermodynamic temperature, Θ.
    Temperature,
    /// Amount of substance, N.
    Amount,
    /// Luminous intensity, J.
    Luminosity,
}

impl BaseDim {
    /// All base dimensions, in the order used by [`Dimension`] and its `Display`.
    pub const ALL: [BaseDim; 7] = [
        BaseDim::Length,
        BaseDim::Mass,
        BaseDim::Time,
        BaseDim::Current,
        BaseDim::Temperature,
        BaseDim::Amount,
        BaseDim::Luminosity,
    ];

    /// The SI dimension symbol.
    pub const fn symbol(self) -> &'static str {
        match self {
            BaseDim::Length => "L",
            BaseDim::Mass => "M",
            BaseDim::Time => "T",
            BaseDim::Current => "I",
            BaseDim::Temperature => "Θ",
            BaseDim::Amount => "N",
            BaseDim::Luminosity => "J",
        }
    }

    /// Looks a base dimension up by its SI symbol.
    pub fn from_symbol(s: &str) -> Option<BaseDim> {
        BaseDim::ALL.into_iter().find(|b| b.symbol() == s)
    }

    const fn index(self) -> usize {
        self as usize
    }
}

/// A physical dimension: a rational exponent for each SI base dimension.
///
/// Dimensions form an abelian group under multiplication, with
/// [`Dimension::DIMENSIONLESS`] as the identity.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Dimension {
    exps: [Rational; 7],
}

impl Dimension {
    /// The identity: every exponent zero.
    pub const DIMENSIONLESS: Dimension = Dimension::from_ints([0; 7]);
    /// L.
    pub const LENGTH: Dimension = Dimension::from_ints([1, 0, 0, 0, 0, 0, 0]);
    /// M.
    pub const MASS: Dimension = Dimension::from_ints([0, 1, 0, 0, 0, 0, 0]);
    /// T.
    pub const TIME: Dimension = Dimension::from_ints([0, 0, 1, 0, 0, 0, 0]);
    /// I.
    pub const CURRENT: Dimension = Dimension::from_ints([0, 0, 0, 1, 0, 0, 0]);
    /// Θ.
    pub const TEMPERATURE: Dimension = Dimension::from_ints([0, 0, 0, 0, 1, 0, 0]);
    /// N.
    pub const AMOUNT: Dimension = Dimension::from_ints([0, 0, 0, 0, 0, 1, 0]);
    /// J.
    pub const LUMINOSITY: Dimension = Dimension::from_ints([0, 0, 0, 0, 0, 0, 1]);
    /// L²·M·T⁻².
    pub const ENERGY: Dimension = Dimension::from_ints([2, 1, -2, 0, 0, 0, 0]);
    /// T⁻¹.
    pub const FREQUENCY: Dimension = Dimension::from_ints([0, 0, -1, 0, 0, 0, 0]);
    /// L·T⁻¹.
    pub const VELOCITY: Dimension = Dimension::from_ints([1, 0, -1, 0, 0, 0, 0]);

    /// A dimension with integer exponents, in [`BaseDim::ALL`] order.
    pub const fn from_ints(exps: [i64; 7]) -> Self {
        let mut out = [Rational::ZERO; 7];
        let mut i = 0;
        while i < 7 {
            out[i] = Rational::integer(exps[i]);
            i += 1;
        }
        Dimension { exps: out }
    }

    /// A dimension with the given exponents, in [`BaseDim::ALL`] order.
    pub const fn from_exponents(exps: [Rational; 7]) -> Self {
        Dimension { exps }
    }

    /// The dimension of a single base dimension, e.g. `base(Length) == L`.
    pub const fn base(b: BaseDim) -> Self {
        let mut exps = [0; 7];
        exps[b.index()] = 1;
        Dimension::from_ints(exps)
    }

    /// The exponent of one base dimension.
    pub const fn exponent(&self, b: BaseDim) -> Rational {
        self.exps[b.index()]
    }

    /// All exponents, in [`BaseDim::ALL`] order.
    pub const fn exponents(&self) -> [Rational; 7] {
        self.exps
    }

    /// True if every exponent is zero.
    pub fn is_dimensionless(&self) -> bool {
        self.exps.iter().all(|e| e.is_zero())
    }

    fn try_map(
        self,
        f: impl Fn(usize, Rational) -> Result<Rational, RationalError>,
    ) -> Result<Self, RationalError> {
        let mut out = [Rational::ZERO; 7];
        for (i, e) in self.exps.into_iter().enumerate() {
            out[i] = f(i, e)?;
        }
        Ok(Dimension { exps: out })
    }

    /// `self · rhs`, or an error if an exponent overflows.
    pub fn checked_mul(self, rhs: Self) -> Result<Self, RationalError> {
        self.try_map(|i, e| e.checked_add(rhs.exps[i]))
    }

    /// `self / rhs`, or an error if an exponent overflows.
    pub fn checked_div(self, rhs: Self) -> Result<Self, RationalError> {
        self.try_map(|i, e| e.checked_sub(rhs.exps[i]))
    }

    /// `self ^ r` for a rational power, or an error if an exponent overflows.
    pub fn checked_pow_rat(self, r: Rational) -> Result<Self, RationalError> {
        self.try_map(|_, e| e.checked_mul(r))
    }

    /// `self ^ n`. Panics on exponent overflow.
    pub fn pow_int(self, n: i64) -> Self {
        self.pow_rat(Rational::integer(n))
    }

    /// `self ^ r`, e.g. `L.pow_rat(1/2) == L^(1/2)`. Panics on exponent overflow.
    pub fn pow_rat(self, r: Rational) -> Self {
        expect_exp(self.checked_pow_rat(r))
    }

    /// The multiplicative inverse, `self ^ -1`.
    pub fn inverse(self) -> Self {
        self.pow_int(-1)
    }

    /// A human name for common dimensions, used in diagnostics
    /// (e.g. `"length"`, `"energy"`, `"dimensionless"`).
    pub fn name(&self) -> Option<&'static str> {
        NAMES.iter().find(|(d, _)| d == self).map(|&(_, name)| name)
    }
}

static NAMES: &[(Dimension, &str)] = &[
    (Dimension::DIMENSIONLESS, "dimensionless"),
    (Dimension::LENGTH, "length"),
    (Dimension::MASS, "mass"),
    (Dimension::TIME, "time"),
    (Dimension::CURRENT, "electric current"),
    (Dimension::TEMPERATURE, "temperature"),
    (Dimension::AMOUNT, "amount of substance"),
    (Dimension::LUMINOSITY, "luminous intensity"),
    (Dimension::ENERGY, "energy"),
    (Dimension::FREQUENCY, "frequency"),
    (Dimension::VELOCITY, "velocity"),
];

fn expect_exp(r: Result<Dimension, RationalError>) -> Dimension {
    match r {
        Ok(d) => d,
        Err(e) => panic!("dimension exponent: {e}"),
    }
}

impl Mul for Dimension {
    type Output = Dimension;
    fn mul(self, rhs: Self) -> Self {
        expect_exp(self.checked_mul(rhs))
    }
}

impl Div for Dimension {
    type Output = Dimension;
    fn div(self, rhs: Self) -> Self {
        expect_exp(self.checked_div(rhs))
    }
}

fn superscript(n: i64) -> String {
    n.to_string()
        .chars()
        .map(|c| match c {
            '-' => '⁻',
            '0' => '⁰',
            '1' => '¹',
            '2' => '²',
            '3' => '³',
            '4' => '⁴',
            '5' => '⁵',
            '6' => '⁶',
            '7' => '⁷',
            '8' => '⁸',
            '9' => '⁹',
            other => other,
        })
        .collect()
}

/// Writes `symbol` raised to `exp`: `L`, `T⁻²`, or `L^(1/2)`.
pub(crate) fn write_power(f: &mut fmt::Formatter<'_>, symbol: &str, exp: Rational) -> fmt::Result {
    if exp == Rational::ONE {
        f.write_str(symbol)
    } else if exp.is_integer() {
        write!(f, "{symbol}{}", superscript(exp.numer()))
    } else {
        write!(f, "{symbol}^({exp})")
    }
}

/// Writes the non-zero base factors of `d` joined by `·`, returning whether
/// anything was written. `first` says whether a separator is needed.
pub(crate) fn write_factors(
    f: &mut fmt::Formatter<'_>,
    d: &Dimension,
    mut first: bool,
) -> Result<bool, fmt::Error> {
    for b in BaseDim::ALL {
        let e = d.exponent(b);
        if e.is_zero() {
            continue;
        }
        if !first {
            f.write_str("·")?;
        }
        first = false;
        write_power(f, b.symbol(), e)?;
    }
    Ok(!first)
}

impl fmt::Display for Dimension {
    /// SI order with `·` separators, e.g. `L²·M·T⁻²`, `L^(1/2)`, or `1`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !write_factors(f, self, true)? {
            f.write_str("1")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let accel = Dimension::LENGTH / Dimension::TIME.pow_int(2);
        assert_eq!(accel.to_string(), "L·T⁻²");
        assert_eq!(Dimension::ENERGY.to_string(), "L²·M·T⁻²");
        assert_eq!(Dimension::DIMENSIONLESS.to_string(), "1");
        assert_eq!(
            Dimension::LENGTH.pow_rat(Rational::new(1, 2)).to_string(),
            "L^(1/2)"
        );
        assert_eq!(
            Dimension::TIME.pow_rat(Rational::new(-3, 2)).to_string(),
            "T^(-3/2)"
        );
        assert_eq!(Dimension::TEMPERATURE.pow_int(10).to_string(), "Θ¹⁰");
    }

    #[test]
    fn names() {
        assert_eq!(Dimension::LENGTH.name(), Some("length"));
        assert_eq!(Dimension::TEMPERATURE.name(), Some("temperature"));
        assert_eq!(
            (Dimension::MASS * Dimension::VELOCITY.pow_int(2)).name(),
            Some("energy")
        );
        assert_eq!(Dimension::DIMENSIONLESS.name(), Some("dimensionless"));
        assert_eq!((Dimension::LENGTH * Dimension::MASS).name(), None);
    }

    #[test]
    fn group_basics() {
        let v = Dimension::VELOCITY;
        assert!((v * v.inverse()).is_dimensionless());
        assert_eq!(v / v, Dimension::DIMENSIONLESS);
        assert_eq!(Dimension::base(BaseDim::Current), Dimension::CURRENT);
        assert_eq!(BaseDim::from_symbol("Θ"), Some(BaseDim::Temperature));
        let sqrt_l = Dimension::LENGTH.pow_rat(Rational::new(1, 2));
        assert_eq!(sqrt_l * sqrt_l, Dimension::LENGTH);
    }

    #[test]
    fn overflow_is_reported() {
        let huge = Dimension::LENGTH.pow_int(i64::MAX);
        assert!(huge.checked_mul(Dimension::LENGTH).is_err());
    }
}
