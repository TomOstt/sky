//! The symbolic side of the exactness analysis: the real number an
//! expression denotes, when it has a small closed form.
//!
//! A [`Sym`] is a sum of at most [`MAX_TERMS`] terms `q · atom` with rational
//! coefficients `q`, where an [`Atom`] is `1`, `π`, `e` or `√n` for a
//! squarefree integer `n >= 2`:
//!
//! ```text
//! 1/2          3π/4        2√2        1/2 + √3        -√2 + 2√3 + 5
//! ```
//!
//! # Why a value equals another exactly when the forms are equal
//!
//! Terms are kept sorted by atom, equal atoms merged, zero coefficients
//! dropped, and `√n` always squarefree (`√8` is `2√2`). Two forms are equal
//! as reals exactly when they are equal as data, because the atoms are
//! linearly independent over the rationals: the square roots of distinct
//! squarefree integers are (a classical result), and `π` and `e` are
//! transcendental, so neither is a rational combination of algebraic
//! numbers. **`π` and `e` never appear in one form together**: whether `1`,
//! `π` and `e` are linearly independent over the rationals is an open
//! problem, so a sum like `π - e` is not representable, and equality of
//! forms stays a theorem.
//!
//! # What it is closed under
//!
//! Sums, differences and negation always (subject to the term cap and
//! `i64` coefficients). Products when one side is rational, or when neither
//! side has `π` or `e` (`√2 · √3 = √6`, `√2 · √2 = 2`, `(1 + √2)² = 3 + 2√2`).
//! Division by a rational or by a single `q·√n`. Powers by small integers.
//! `sqrt` of a rational. Anything else (`π²`, `1/π`, `√π`, `sin 1`) is not
//! representable, and the operation gives `None`: "not known exactly".
//! `None` is never wrong, only less informative.
//!
//! The forms are about the *real number*, not about `f64`: `π` here is the
//! irrational number, and `sin(π)` is exactly `0`. What the interpreter
//! computes for the same expression is a separate question, answered by
//! [`super::Range`]; see the module documentation of [`super`].

use std::cmp::Ordering;
use std::fmt;

use sky_units::Rational;

use super::range::{next_down, next_up};

/// The most terms a [`Sym`] keeps. A result with more is "not known".
pub const MAX_TERMS: usize = 4;

/// Largest numerator and denominator for which [`Sym::root`] factors the
/// radicand (trial division, so it must stay small).
const ROOT_LIMIT: u64 = 60_000;

/// Largest exponent [`Sym::powi`] expands.
const POWI_LIMIT: u32 = 16;

/// The irrational (or rational) numbers a term is a multiple of.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Atom {
    /// `1`: the rational part.
    One,
    Pi,
    E,
    /// `√n`, `n` squarefree and at least 2.
    Sqrt(u32),
}

/// A number with a closed form: see the module documentation.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Sym {
    /// Sorted by atom; no zero coefficient; at most [`MAX_TERMS`]; not both
    /// `Pi` and `E`.
    terms: Vec<(Atom, Rational)>,
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// `n = s² · t` with `t` squarefree, for `n >= 1`.
fn split_square(mut n: u64) -> (u64, u64) {
    let (mut s, mut t) = (1u64, 1u64);
    let mut p = 2u64;
    while p * p <= n {
        let mut k = 0;
        while n % p == 0 {
            n /= p;
            k += 1;
        }
        for _ in 0..k / 2 {
            s *= p;
        }
        if k % 2 == 1 {
            t *= p;
        }
        p += if p == 2 { 1 } else { 2 };
    }
    (s, t * n)
}

/// The integer square root of `n`, rounded down.
fn isqrt(n: u64) -> u64 {
    let mut r = (n as f64).sqrt() as u64;
    while r * r > n {
        r -= 1;
    }
    while (r + 1) * (r + 1) <= n {
        r += 1;
    }
    r
}

impl Sym {
    /// The exact rational `q`.
    pub fn rational(q: Rational) -> Sym {
        Sym {
            terms: if q.is_zero() {
                Vec::new()
            } else {
                vec![(Atom::One, q)]
            },
        }
    }

    pub fn integer(n: i64) -> Sym {
        Sym::rational(Rational::integer(n))
    }

    /// The number `π`.
    pub fn pi() -> Sym {
        Sym {
            terms: vec![(Atom::Pi, Rational::ONE)],
        }
    }

    /// Euler's number `e`.
    pub fn e() -> Sym {
        Sym {
            terms: vec![(Atom::E, Rational::ONE)],
        }
    }

    /// `√q` for a non-negative rational, when it is representable: `√(a/b)
    /// = √(ab) / b`, with `ab = s²t` and `t` squarefree. `None` for a
    /// negative `q`, and for radicands too large to factor cheaply that are
    /// not perfect squares.
    pub fn root(q: Rational) -> Option<Sym> {
        if q.numer() < 0 {
            return None;
        }
        let (a, b) = (q.numer().unsigned_abs(), q.denom().unsigned_abs());
        let (ra, rb) = (isqrt(a), isqrt(b));
        if ra * ra == a && rb * rb == b {
            return Some(Sym::rational(
                Rational::checked_new(ra as i64, rb as i64).ok()?,
            ));
        }
        if a > ROOT_LIMIT || b > ROOT_LIMIT {
            return None;
        }
        let (s, t) = split_square(a * b);
        let coeff = Rational::checked_new(s as i64, b as i64).ok()?;
        if t == 1 {
            return Some(Sym::rational(coeff));
        }
        Sym::from_terms(vec![(Atom::Sqrt(u32::try_from(t).ok()?), coeff)])
    }

    /// Normalizes: sorts, merges equal atoms, drops zeros, and gives `None`
    /// past the caps or on coefficient overflow.
    fn from_terms(mut v: Vec<(Atom, Rational)>) -> Option<Sym> {
        v.sort_by_key(|(a, _)| *a);
        let mut out: Vec<(Atom, Rational)> = Vec::with_capacity(v.len());
        for (a, q) in v {
            match out.last_mut() {
                Some((b, r)) if *b == a => *r = r.checked_add(q).ok()?,
                _ => out.push((a, q)),
            }
        }
        out.retain(|(_, q)| !q.is_zero());
        let has = |x: Atom| out.iter().any(|(a, _)| *a == x);
        if out.len() > MAX_TERMS || (has(Atom::Pi) && has(Atom::E)) {
            return None;
        }
        Some(Sym { terms: out })
    }

    /// The terms, sorted by atom.
    pub fn terms(&self) -> &[(Atom, Rational)] {
        &self.terms
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// The value, if it is rational.
    pub fn as_rational(&self) -> Option<Rational> {
        match self.terms[..] {
            [] => Some(Rational::ZERO),
            [(Atom::One, q)] => Some(q),
            _ => None,
        }
    }

    fn transcendental(&self) -> bool {
        self.terms
            .iter()
            .any(|(a, _)| matches!(a, Atom::Pi | Atom::E))
    }

    fn scale(&self, k: Rational) -> Option<Sym> {
        if k.is_zero() {
            return Some(Sym::rational(Rational::ZERO));
        }
        let terms = self
            .terms
            .iter()
            .map(|(a, q)| Some((*a, q.checked_mul(k).ok()?)))
            .collect::<Option<Vec<_>>>()?;
        Sym::from_terms(terms)
    }

    pub fn neg(&self) -> Option<Sym> {
        self.scale(Rational::integer(-1))
    }

    pub fn add(&self, o: &Sym) -> Option<Sym> {
        Sym::from_terms(self.terms.iter().chain(&o.terms).copied().collect())
    }

    pub fn sub(&self, o: &Sym) -> Option<Sym> {
        self.add(&o.neg()?)
    }

    pub fn mul(&self, o: &Sym) -> Option<Sym> {
        if let Some(q) = o.as_rational() {
            return self.scale(q);
        }
        if let Some(q) = self.as_rational() {
            return o.scale(q);
        }
        if self.transcendental() || o.transcendental() {
            return None;
        }
        let mut out = Vec::new();
        for (a, p) in &self.terms {
            for (b, q) in &o.terms {
                let c = p.checked_mul(*q).ok()?;
                out.push(match (*a, *b) {
                    (Atom::One, x) | (x, Atom::One) => (x, c),
                    (Atom::Sqrt(m), Atom::Sqrt(n)) => {
                        // √m·√n = g·√((m/g)(n/g)), g = gcd(m, n): the
                        // quotients are coprime and squarefree, so the
                        // product is squarefree too.
                        let g = gcd(u64::from(m), u64::from(n));
                        let rest = (u64::from(m) / g) * (u64::from(n) / g);
                        let c = c
                            .checked_mul(Rational::integer(i64::try_from(g).ok()?))
                            .ok()?;
                        if rest == 1 {
                            (Atom::One, c)
                        } else {
                            (Atom::Sqrt(u32::try_from(rest).ok()?), c)
                        }
                    }
                    _ => return None,
                });
            }
        }
        Sym::from_terms(out)
    }

    /// `self / o`: `o` must be a non-zero rational or a single `q·√n`.
    pub fn div(&self, o: &Sym) -> Option<Sym> {
        match o.terms[..] {
            [(Atom::One, q)] => self.scale(q.checked_recip().ok()?),
            // a / (q√n) = a·√n / (q·n)
            [(Atom::Sqrt(n), q)] => {
                let d = q.checked_mul(Rational::integer(i64::from(n))).ok()?;
                let inv = Sym::from_terms(vec![(Atom::Sqrt(n), d.checked_recip().ok()?)])?;
                self.mul(&inv)
            }
            _ => None,
        }
    }

    /// `self ^ n` for a small integer `n`.
    pub fn powi(&self, n: i32) -> Option<Sym> {
        if n.unsigned_abs() > POWI_LIMIT {
            return None;
        }
        let mut acc = Sym::integer(1);
        for _ in 0..n.unsigned_abs() {
            acc = acc.mul(self)?;
        }
        if n < 0 {
            Sym::integer(1).div(&acc)
        } else {
            Some(acc)
        }
    }

    /// `√self`, when `self` is a non-negative rational.
    pub fn sqrt(&self) -> Option<Sym> {
        Sym::root(self.as_rational()?)
    }

    /// A rigorous enclosure `[lo, hi]` of the real number, in `f64`. `None`
    /// only when a coefficient is too large for `f64` to bound sensibly.
    pub fn enclose(&self) -> Option<(f64, f64)> {
        let mut lo = 0.0f64;
        let mut hi = 0.0f64;
        for (atom, q) in &self.terms {
            // `numer as f64 / denom as f64` is within 1.5 ulps of the true
            // quotient; two steps of slack cover it.
            let c = q.numer() as f64 / q.denom() as f64;
            let (cl, ch) = (next_down(next_down(c)), next_up(next_up(c)));
            let (al, ah) = match atom {
                Atom::One => (1.0, 1.0),
                // π and e lie strictly between the f64 below and above.
                Atom::Pi => (std::f64::consts::PI, next_up(std::f64::consts::PI)),
                Atom::E => (std::f64::consts::E, next_up(std::f64::consts::E)),
                Atom::Sqrt(n) => {
                    let s = f64::from(*n).sqrt();
                    (next_down(s), next_up(s))
                }
            };
            let p = [cl * al, cl * ah, ch * al, ch * ah];
            let (pl, ph) = (
                p.into_iter().fold(f64::INFINITY, f64::min),
                p.into_iter().fold(f64::NEG_INFINITY, f64::max),
            );
            lo = next_down(lo + next_down(pl));
            hi = next_up(hi + next_up(ph));
        }
        (lo.is_finite() && hi.is_finite()).then_some((lo, hi))
    }

    /// The sign of the number, if it can be decided: exact for a rational,
    /// by the enclosure otherwise. The atoms are independent, so a non-zero
    /// form is never zero, but the enclosure may still straddle zero.
    pub fn sign(&self) -> Option<Ordering> {
        if self.is_zero() {
            return Some(Ordering::Equal);
        }
        if let Some(q) = self.as_rational() {
            return Some(q.numer().cmp(&0));
        }
        let (lo, hi) = self.enclose()?;
        if lo > 0.0 {
            Some(Ordering::Greater)
        } else if hi < 0.0 {
            Some(Ordering::Less)
        } else {
            None
        }
    }

    /// `|self|`, when the sign is decidable.
    pub fn abs(&self) -> Option<Sym> {
        match self.sign()? {
            Ordering::Less => self.neg(),
            _ => Some(self.clone()),
        }
    }
}

/// `q` followed by a symbol: `3π`, `π/4`, `-2√2`, `√3/2`.
fn term_text(atom: Atom, q: Rational) -> String {
    let symbol = match atom {
        Atom::One => return q.to_string(),
        Atom::Pi => "π".to_string(),
        Atom::E => "e".to_string(),
        Atom::Sqrt(n) => format!("√{n}"),
    };
    let (n, d) = (q.numer(), q.denom());
    let head = match n {
        1 => symbol,
        -1 => format!("-{symbol}"),
        _ => format!("{n}{symbol}"),
    };
    if d == 1 {
        head
    } else {
        format!("{head}/{d}")
    }
}

impl fmt::Display for Sym {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return f.write_str("0");
        }
        for (i, (atom, q)) in self.terms.iter().enumerate() {
            let text = term_text(*atom, *q);
            if i == 0 {
                f.write_str(&text)?;
            } else if let Some(rest) = text.strip_prefix('-') {
                write!(f, " - {rest}")?;
            } else {
                write!(f, " + {text}")?;
            }
        }
        Ok(())
    }
}
