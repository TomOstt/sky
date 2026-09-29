//! A small exact rational number used for dimension exponents.
//!
//! Values are always normalized: the denominator is positive and
//! `gcd(|num|, den) == 1`, so derived equality and hashing are structural.
//! Arithmetic is computed in `i128` and checked on the way back to `i64`;
//! it never wraps. The `checked_*` methods return an error, and the operator
//! impls panic on overflow.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// An exact rational number `num / den` with `den > 0`, in lowest terms.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Rational {
    num: i64,
    den: i64,
}

/// Why a rational operation could not produce a value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RationalError {
    /// A zero denominator, or division by zero.
    DivisionByZero,
    /// The normalized result does not fit in `i64`.
    Overflow,
}

impl fmt::Display for RationalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RationalError::DivisionByZero => f.write_str("rational division by zero"),
            RationalError::Overflow => f.write_str("rational arithmetic overflow"),
        }
    }
}

impl std::error::Error for RationalError {}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

impl Rational {
    /// Zero.
    pub const ZERO: Rational = Rational { num: 0, den: 1 };
    /// One.
    pub const ONE: Rational = Rational { num: 1, den: 1 };

    /// The integer `n`.
    pub const fn integer(n: i64) -> Self {
        Rational { num: n, den: 1 }
    }

    /// `num / den`, normalized. Panics if `den == 0`.
    pub fn new(num: i64, den: i64) -> Self {
        match Self::checked_new(num, den) {
            Ok(r) => r,
            Err(e) => panic!("Rational::new({num}, {den}): {e}"),
        }
    }

    /// `num / den`, normalized, or an error for a zero denominator or overflow
    /// (e.g. `i64::MIN / -1`).
    pub fn checked_new(num: i64, den: i64) -> Result<Self, RationalError> {
        Self::from_i128(i128::from(num), i128::from(den))
    }

    fn from_i128(num: i128, den: i128) -> Result<Self, RationalError> {
        if den == 0 {
            return Err(RationalError::DivisionByZero);
        }
        let g = gcd(num.unsigned_abs(), den.unsigned_abs());
        // g >= 1 because den != 0, and g <= |den| < 2^127, so the cast is lossless.
        let g = g as i128;
        let (mut n, mut d) = (num / g, den / g);
        if d < 0 {
            n = n.checked_neg().ok_or(RationalError::Overflow)?;
            d = d.checked_neg().ok_or(RationalError::Overflow)?;
        }
        Ok(Rational {
            num: i64::try_from(n).map_err(|_| RationalError::Overflow)?,
            den: i64::try_from(d).map_err(|_| RationalError::Overflow)?,
        })
    }

    /// The numerator (carries the sign).
    pub const fn numer(self) -> i64 {
        self.num
    }

    /// The denominator (always positive).
    pub const fn denom(self) -> i64 {
        self.den
    }

    /// True if the denominator is 1.
    pub const fn is_integer(self) -> bool {
        self.den == 1
    }

    /// True if the value is zero.
    pub const fn is_zero(self) -> bool {
        self.num == 0
    }

    /// Checked addition.
    pub fn checked_add(self, rhs: Self) -> Result<Self, RationalError> {
        let (a, b, c, d) = self.wide(rhs);
        let n = (a * d).checked_add(c * b).ok_or(RationalError::Overflow)?;
        Self::from_i128(n, b * d)
    }

    /// Checked subtraction.
    pub fn checked_sub(self, rhs: Self) -> Result<Self, RationalError> {
        self.checked_add(rhs.checked_neg()?)
    }

    /// Checked multiplication.
    pub fn checked_mul(self, rhs: Self) -> Result<Self, RationalError> {
        let (a, b, c, d) = self.wide(rhs);
        Self::from_i128(a * c, b * d)
    }

    /// Checked division; dividing by zero is an error.
    pub fn checked_div(self, rhs: Self) -> Result<Self, RationalError> {
        let (a, b, c, d) = self.wide(rhs);
        Self::from_i128(a * d, b * c)
    }

    /// Checked negation (fails only for a numerator of `i64::MIN`).
    pub fn checked_neg(self) -> Result<Self, RationalError> {
        Self::from_i128(-i128::from(self.num), i128::from(self.den))
    }

    /// Checked reciprocal; the reciprocal of zero is an error.
    pub fn checked_recip(self) -> Result<Self, RationalError> {
        Self::from_i128(i128::from(self.den), i128::from(self.num))
    }

    /// The reciprocal. Panics on zero.
    pub fn recip(self) -> Self {
        unwrap_or_panic(self.checked_recip())
    }

    fn wide(self, rhs: Self) -> (i128, i128, i128, i128) {
        (
            i128::from(self.num),
            i128::from(self.den),
            i128::from(rhs.num),
            i128::from(rhs.den),
        )
    }
}

fn unwrap_or_panic(r: Result<Rational, RationalError>) -> Rational {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{e}"),
    }
}

impl From<i64> for Rational {
    fn from(n: i64) -> Self {
        Rational::integer(n)
    }
}

impl Add for Rational {
    type Output = Rational;
    fn add(self, rhs: Self) -> Self {
        unwrap_or_panic(self.checked_add(rhs))
    }
}

impl Sub for Rational {
    type Output = Rational;
    fn sub(self, rhs: Self) -> Self {
        unwrap_or_panic(self.checked_sub(rhs))
    }
}

impl Mul for Rational {
    type Output = Rational;
    fn mul(self, rhs: Self) -> Self {
        unwrap_or_panic(self.checked_mul(rhs))
    }
}

impl Div for Rational {
    type Output = Rational;
    fn div(self, rhs: Self) -> Self {
        unwrap_or_panic(self.checked_div(rhs))
    }
}

impl Neg for Rational {
    type Output = Rational;
    fn neg(self) -> Self {
        unwrap_or_panic(self.checked_neg())
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> Ordering {
        // Denominators are positive, so cross-multiplying preserves order.
        let (a, b, c, d) = self.wide(*other);
        (a * d).cmp(&(c * b))
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_sign_and_gcd() {
        let r = Rational::new(6, -8);
        assert_eq!((r.numer(), r.denom()), (-3, 4));
        assert_eq!(Rational::new(0, -5), Rational::ZERO);
        assert_eq!(Rational::new(4, 2), Rational::integer(2));
    }

    #[test]
    fn arithmetic() {
        let h = Rational::new(1, 2);
        let t = Rational::new(1, 3);
        assert_eq!(h + t, Rational::new(5, 6));
        assert_eq!(h - t, Rational::new(1, 6));
        assert_eq!(h * t, Rational::new(1, 6));
        assert_eq!(h / t, Rational::new(3, 2));
        assert_eq!(-h, Rational::new(-1, 2));
        assert_eq!(t.recip(), Rational::integer(3));
        assert!(t < h);
    }

    #[test]
    fn display() {
        assert_eq!(Rational::new(-1, 2).to_string(), "-1/2");
        assert_eq!(Rational::integer(3).to_string(), "3");
    }

    #[test]
    fn zero_denominator_is_an_error() {
        assert_eq!(
            Rational::checked_new(1, 0),
            Err(RationalError::DivisionByZero)
        );
        assert_eq!(
            Rational::ONE.checked_div(Rational::ZERO),
            Err(RationalError::DivisionByZero)
        );
    }

    #[test]
    fn overflow_is_an_error_not_a_wrap() {
        let big = Rational::integer(i64::MAX);
        assert_eq!(big.checked_add(Rational::ONE), Err(RationalError::Overflow));
        assert_eq!(big.checked_mul(big), Err(RationalError::Overflow));
        assert_eq!(
            Rational::integer(i64::MIN).checked_neg(),
            Err(RationalError::Overflow)
        );
        assert_eq!(
            Rational::checked_new(i64::MIN, -1),
            Err(RationalError::Overflow)
        );
        // Large intermediates that normalize back into range are fine.
        assert_eq!(
            big.checked_mul(Rational::new(1, i64::MAX)),
            Ok(Rational::ONE)
        );
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn operator_overflow_panics() {
        let _ = Rational::integer(i64::MAX) + Rational::ONE;
    }
}
