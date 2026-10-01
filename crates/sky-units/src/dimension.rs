//! Physical dimensions as exponent vectors over the seven SI base dimensions.

use std::fmt;
use std::ops::{Div, Mul};

use crate::rational::{Rational, RationalError};

/// An error from dimension arithmetic.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DimError {
    /// An exponent's numerator or denominator no longer fits in `i64`.
    ExponentOverflow,
}

impl fmt::Display for DimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DimError::ExponentOverflow => f.write_str("dimension exponent too large"),
        }
    }
}

impl std::error::Error for DimError {}

/// Maps an exponent-arithmetic failure to [`DimError`]. Dimension arithmetic
/// only adds, subtracts, and multiplies exponents (and takes reciprocals of
/// non-zero ones), so the only reachable failure is overflow.
pub(crate) fn exponent_overflow(_: RationalError) -> DimError {
    DimError::ExponentOverflow
}

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

    /// The symbol a dimension is shown with in messages: the SI symbol,
    /// except `mol` for amount of substance and `cd` for luminous
    /// intensity. Their SI symbols `N` and `J` read as the newton and the
    /// joule, so `L⁻³·T⁻¹·N` would look like a force; `L⁻³·T⁻¹·mol` cannot.
    pub const fn display_symbol(self) -> &'static str {
        match self {
            BaseDim::Amount => "mol",
            BaseDim::Luminosity => "cd",
            other => other.symbol(),
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
///
/// Checker code must use the fallible methods ([`checked_mul`](Self::checked_mul),
/// [`checked_div`](Self::checked_div), [`pow_int`](Self::pow_int),
/// [`pow_rat`](Self::pow_rat), [`inverse`](Self::inverse)), which return
/// [`DimError::ExponentOverflow`] instead of panicking. The `*` and `/`
/// operators panic on overflow and exist only for constants and tests.
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
    /// L·M·T⁻² (the newton, `N`; not the base dimension N, amount of substance).
    pub const FORCE: Dimension = Dimension::from_ints([1, 1, -2, 0, 0, 0, 0]);
    /// L⁻¹·M·T⁻² (the pascal, `Pa`).
    pub const PRESSURE: Dimension = Dimension::from_ints([-1, 1, -2, 0, 0, 0, 0]);
    /// L²·M·T⁻³ (the watt, `W`).
    pub const POWER: Dimension = Dimension::from_ints([2, 1, -3, 0, 0, 0, 0]);
    /// L³ (the litre, `L`, is a volume; not the base dimension L, length).
    pub const VOLUME: Dimension = Dimension::from_ints([3, 0, 0, 0, 0, 0, 0]);
    /// T·I (the coulomb, `C`).
    pub const CHARGE: Dimension = Dimension::from_ints([0, 0, 1, 1, 0, 0, 0]);
    /// L²·M·T⁻³·I⁻¹ (the volt, `V`).
    pub const VOLTAGE: Dimension = Dimension::from_ints([2, 1, -3, -1, 0, 0, 0]);
    /// L²·M·T⁻³·I⁻² (the ohm, `Ω`).
    pub const RESISTANCE: Dimension = Dimension::from_ints([2, 1, -3, -2, 0, 0, 0]);
    /// L⁻²·M⁻¹·T³·I² (the siemens, `S`).
    pub const CONDUCTANCE: Dimension = Dimension::from_ints([-2, -1, 3, 2, 0, 0, 0]);
    /// L⁻²·M⁻¹·T⁴·I² (the farad, `F`).
    pub const CAPACITANCE: Dimension = Dimension::from_ints([-2, -1, 4, 2, 0, 0, 0]);
    /// L²·M·T⁻²·I⁻¹ (the weber, `Wb`).
    pub const MAGNETIC_FLUX: Dimension = Dimension::from_ints([2, 1, -2, -1, 0, 0, 0]);
    /// M·T⁻²·I⁻¹ (the tesla, `T`; not the base dimension T, time).
    pub const MAGNETIC_FLUX_DENSITY: Dimension = Dimension::from_ints([0, 1, -2, -1, 0, 0, 0]);
    /// L²·M·T⁻²·I⁻² (the henry, `H`).
    pub const INDUCTANCE: Dimension = Dimension::from_ints([2, 1, -2, -2, 0, 0, 0]);

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
    ) -> Result<Self, DimError> {
        let mut out = [Rational::ZERO; 7];
        for (i, e) in self.exps.into_iter().enumerate() {
            out[i] = f(i, e).map_err(exponent_overflow)?;
        }
        Ok(Dimension { exps: out })
    }

    /// `self · rhs`.
    pub fn checked_mul(self, rhs: Self) -> Result<Self, DimError> {
        self.try_map(|i, e| e.checked_add(rhs.exps[i]))
    }

    /// `self / rhs`.
    pub fn checked_div(self, rhs: Self) -> Result<Self, DimError> {
        self.try_map(|i, e| e.checked_sub(rhs.exps[i]))
    }

    /// `self ^ n`.
    pub fn pow_int(self, n: i64) -> Result<Self, DimError> {
        self.pow_rat(Rational::integer(n))
    }

    /// `self ^ r`, e.g. `L.pow_rat(1/2) == Ok(L^(1/2))`.
    pub fn pow_rat(self, r: Rational) -> Result<Self, DimError> {
        self.try_map(|_, e| e.checked_mul(r))
    }

    /// The multiplicative inverse, `self ^ -1`.
    pub fn inverse(self) -> Result<Self, DimError> {
        self.try_map(|_, e| e.checked_neg())
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
    (Dimension::FORCE, "force"),
    (Dimension::PRESSURE, "pressure"),
    (Dimension::POWER, "power"),
    (Dimension::VOLUME, "volume"),
    (Dimension::CHARGE, "electric charge"),
    (Dimension::VOLTAGE, "voltage"),
    (Dimension::RESISTANCE, "electrical resistance"),
    (Dimension::CONDUCTANCE, "electrical conductance"),
    (Dimension::CAPACITANCE, "capacitance"),
    (Dimension::MAGNETIC_FLUX, "magnetic flux"),
    (Dimension::MAGNETIC_FLUX_DENSITY, "magnetic flux density"),
    (Dimension::INDUCTANCE, "inductance"),
];

fn expect_exp(r: Result<Dimension, DimError>) -> Dimension {
    match r {
        Ok(d) => d,
        Err(e) => panic!("{e}"),
    }
}

/// Panics on exponent overflow; for constants and tests only.
/// Checker code uses [`Dimension::checked_mul`].
impl Mul for Dimension {
    type Output = Dimension;
    fn mul(self, rhs: Self) -> Self {
        expect_exp(self.checked_mul(rhs))
    }
}

/// Panics on exponent overflow; for constants and tests only.
/// Checker code uses [`Dimension::checked_div`].
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
        write_power(f, b.display_symbol(), e)?;
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
        let accel = Dimension::LENGTH / Dimension::TIME.pow_int(2).unwrap();
        assert_eq!(accel.to_string(), "L·T⁻²");
        assert_eq!(Dimension::ENERGY.to_string(), "L²·M·T⁻²");
        assert_eq!(Dimension::DIMENSIONLESS.to_string(), "1");
        let half = Rational::new(1, 2);
        assert_eq!(
            Dimension::LENGTH.pow_rat(half).unwrap().to_string(),
            "L^(1/2)"
        );
        assert_eq!(
            Dimension::TIME
                .pow_rat(Rational::new(-3, 2))
                .unwrap()
                .to_string(),
            "T^(-3/2)"
        );
        assert_eq!(
            Dimension::TEMPERATURE.pow_int(10).unwrap().to_string(),
            "Θ¹⁰"
        );
    }

    #[test]
    fn names() {
        assert_eq!(Dimension::LENGTH.name(), Some("length"));
        assert_eq!(Dimension::TEMPERATURE.name(), Some("temperature"));
        let v2 = Dimension::VELOCITY.pow_int(2).unwrap();
        assert_eq!(
            Dimension::MASS.checked_mul(v2).unwrap().name(),
            Some("energy")
        );
        assert_eq!(Dimension::DIMENSIONLESS.name(), Some("dimensionless"));
        assert_eq!((Dimension::LENGTH * Dimension::MASS).name(), None);
    }

    #[test]
    fn group_basics() {
        let v = Dimension::VELOCITY;
        let inv = v.inverse().unwrap();
        assert!(v.checked_mul(inv).unwrap().is_dimensionless());
        assert_eq!(v.checked_div(v), Ok(Dimension::DIMENSIONLESS));
        assert_eq!(Dimension::base(BaseDim::Current), Dimension::CURRENT);
        assert_eq!(BaseDim::from_symbol("Θ"), Some(BaseDim::Temperature));
        let sqrt_l = Dimension::LENGTH.pow_rat(Rational::new(1, 2)).unwrap();
        assert_eq!(sqrt_l.checked_mul(sqrt_l), Ok(Dimension::LENGTH));
    }

    #[test]
    fn overflow_is_an_error() {
        let huge = Dimension::LENGTH.pow_int(i64::MAX).unwrap();
        assert_eq!(
            huge.checked_mul(Dimension::LENGTH),
            Err(DimError::ExponentOverflow)
        );
        assert_eq!(huge.pow_int(2), Err(DimError::ExponentOverflow));
        let min = Dimension::LENGTH.pow_int(i64::MIN).unwrap();
        assert_eq!(min.inverse(), Err(DimError::ExponentOverflow));
        assert_eq!(
            min.checked_div(Dimension::LENGTH),
            Err(DimError::ExponentOverflow)
        );
        // Denominators overflow too: 1/MAX + 1/(MAX-1) needs a denominator > i64::MAX.
        let a = Dimension::LENGTH
            .pow_rat(Rational::new(1, i64::MAX))
            .unwrap();
        let b = Dimension::LENGTH
            .pow_rat(Rational::new(1, i64::MAX - 1))
            .unwrap();
        assert_eq!(a.checked_mul(b), Err(DimError::ExponentOverflow));
        assert_eq!(
            DimError::ExponentOverflow.to_string(),
            "dimension exponent too large"
        );
    }

    /// Regression (Eng Lead review of #2): `m^(9223372036854775807/2)` squared.
    /// The exact result, `m^9223372036854775807`, fits in `i64`, so squaring
    /// returns it exactly; any further growth is an `Err`, never a panic.
    #[test]
    fn half_max_exponent_squared_regression() {
        let m = crate::lookup("m").unwrap().dim;
        let half_max = Rational::new(i64::MAX, 2);
        let big = m.pow_rat(half_max).unwrap();
        let max = Dimension::LENGTH.pow_int(i64::MAX).unwrap();
        assert_eq!(big.pow_int(2), Ok(max));
        assert_eq!(big.checked_mul(big), Ok(max));
        assert_eq!(big.pow_int(3), Err(DimError::ExponentOverflow));
        assert_eq!(big.pow_int(4), Err(DimError::ExponentOverflow));
        assert_eq!(max.pow_int(2), Err(DimError::ExponentOverflow));
        assert_eq!(max.checked_mul(m), Err(DimError::ExponentOverflow));
        assert_eq!(
            big.pow_rat(Rational::new(3, 1)),
            Err(DimError::ExponentOverflow)
        );
    }

    #[test]
    #[should_panic(expected = "dimension exponent too large")]
    fn operators_panic_and_are_for_constants_only() {
        let _ = Dimension::LENGTH.pow_int(i64::MAX).unwrap() * Dimension::LENGTH;
    }
}
