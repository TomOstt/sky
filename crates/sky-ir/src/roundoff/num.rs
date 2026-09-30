//! The abstract value of one scalar: a range of the computed `f64`, and the
//! bounds on how far it is from the exact real number.

use super::hp::Enc;
use super::{add_up, div_up, mul_up, one_minus, INF};
use crate::exact::range::{next_down, Range};

/// What is known about one scalar `f64` value.
///
/// Let `c` be the value the interpreter computes and `x` the real number
/// the same computation gives when every operation is exact (inputs and
/// literals are taken as the exact reals they hold). Then
///
/// * `c` is in [`Num::range`],
/// * `|c - x| <= abs`, and
/// * `|c - x| <= rel * |x|`,
///
/// where either bound may be `+inf` ("nothing proven"). The two are
/// independent claims about one error; [`Num::tighten`] derives each from
/// the other when the range allows, so a caller can read whichever it needs.
///
/// Errors are only ever *upper* bounds, except [`Num::lower_abs`] and
/// [`Num::lower_rel`]: when an enclosure of `x` is known ([`super::hp`]),
/// the error is also known to be at least that much, which is what lets the
/// analysis say a budget is violated.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Num {
    /// Every `f64` the interpreter can compute here.
    pub range: Range,
    /// `|computed - exact| <= abs`.
    pub abs: f64,
    /// `|computed - exact| <= rel * |exact|`.
    pub rel: f64,
    /// `|computed - exact| >= lower_abs`.
    pub lower_abs: f64,
    /// `|computed - exact| >= lower_rel * |exact|`, for every exact value
    /// the enclosure allows.
    pub lower_rel: f64,
    /// An enclosure of the exact value, while `range` is a single point.
    pub(crate) hp: Option<Enc>,
}

/// A relative error at or above this is not tracked: the first-order and
/// the interval formulas below need it well under 1.
const REL_CAP: f64 = 0.5;

/// The largest magnitude in `r`.
pub(crate) fn mag_max(r: Range) -> f64 {
    r.lo.abs().max(r.hi.abs())
}

/// The smallest magnitude in `r`.
pub(crate) fn mag_min(r: Range) -> f64 {
    if r.contains_zero() {
        0.0
    } else {
        r.lo.abs().min(r.hi.abs())
    }
}

/// What the sign of the *exact* value is known to be.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Sign {
    Pos,
    Neg,
    NonNeg,
    NonPos,
    Unknown,
}

impl Sign {
    pub(crate) fn flip(self) -> Sign {
        match self {
            Sign::Pos => Sign::Neg,
            Sign::Neg => Sign::Pos,
            Sign::NonNeg => Sign::NonPos,
            Sign::NonPos => Sign::NonNeg,
            Sign::Unknown => Sign::Unknown,
        }
    }

    pub(crate) fn non_negative(self) -> bool {
        matches!(self, Sign::Pos | Sign::NonNeg)
    }

    pub(crate) fn non_positive(self) -> bool {
        matches!(self, Sign::Neg | Sign::NonPos)
    }
}

impl Num {
    /// A finite number about which nothing is known, with an error nothing
    /// bounds: what an unmodelled operation gives.
    pub fn unknown() -> Num {
        Num {
            range: Range::FINITE,
            abs: INF,
            rel: INF,
            lower_abs: 0.0,
            lower_rel: 0.0,
            hp: None,
        }
    }

    /// A value the computation does not round: an input or a literal, in
    /// `range`. Its error is zero by definition.
    pub fn exact(range: Range) -> Num {
        let hp = range.as_point().and_then(Enc::point);
        Num {
            range,
            abs: 0.0,
            rel: 0.0,
            lower_abs: 0.0,
            lower_rel: 0.0,
            hp,
        }
    }

    /// The exact value, to about 27 digits, when an enclosure is known.
    pub fn exact_value(&self) -> Option<f64> {
        self.hp.map(|e| e.value())
    }

    /// The same range, with no bound on the error: the value is a number
    /// the computation reached in a way the analysis cannot follow.
    pub fn unbounded(self) -> Num {
        Num {
            abs: INF,
            rel: INF,
            lower_abs: 0.0,
            lower_rel: 0.0,
            hp: None,
            ..self
        }
    }

    /// The error is zero: the `f64` is the exact value.
    pub fn is_exact(&self) -> bool {
        self.abs == 0.0
    }

    /// Whether some bound on the error is proven.
    pub fn is_bounded(&self) -> bool {
        self.abs.is_finite() || self.rel.is_finite()
    }

    /// The sign of the exact value, as far as the range and the error say.
    pub(crate) fn sign(&self) -> Sign {
        let r = self.range;
        if self.abs == 0.0 {
            return if r.lo > 0.0 {
                Sign::Pos
            } else if r.lo >= 0.0 {
                Sign::NonNeg
            } else if r.hi < 0.0 {
                Sign::Neg
            } else if r.hi <= 0.0 {
                Sign::NonPos
            } else {
                Sign::Unknown
            };
        }
        let h = self.exact_hull();
        if h.lo > 0.0 {
            return Sign::Pos;
        }
        if h.hi < 0.0 {
            return Sign::Neg;
        }
        // `c = x (1 + e)` with `|e| < 1` keeps the sign; `|c - x| <= abs`
        // with `abs < |c|` does too.
        let keeps = |m: f64| self.rel < 1.0 || self.abs < m;
        if r.lo > 0.0 && keeps(r.lo) {
            Sign::Pos
        } else if r.hi < 0.0 && keeps(-r.hi) {
            Sign::Neg
        } else {
            Sign::Unknown
        }
    }

    /// Every real number the exact value can be: the range, widened by the
    /// absolute error, and inside the enclosure when there is one.
    pub(crate) fn exact_hull(&self) -> Range {
        let mut h = if self.abs == 0.0 {
            self.range
        } else if self.abs.is_finite() {
            Range::new(
                next_down(self.range.lo - self.abs),
                crate::exact::range::next_up(self.range.hi + self.abs),
            )
        } else {
            Range::FINITE
        };
        if let Some(e) = self.hp {
            let (lo, hi) = e.interval();
            h = Range::new(h.lo.max(lo), h.hi.min(hi));
        }
        h
    }

    /// Derives each error bound from the others, and from the enclosure.
    /// Every step is sound; the result is never looser than the input.
    pub(crate) fn tighten(mut self) -> Num {
        if let (Some(h), Some(c)) = (self.hp, self.range.as_point()) {
            let (lower, upper) = h.error_of(c);
            let (m_lo, m_hi) = h.magnitude();
            self.abs = self.abs.min(upper);
            if m_lo > 0.0 {
                self.rel = self.rel.min(div_up(upper, m_lo));
            }
            self.lower_abs = self.lower_abs.max(lower);
            let lower_rel = if m_hi > 0.0 {
                next_down(lower / m_hi)
            } else if lower > 0.0 {
                INF
            } else {
                0.0
            };
            self.lower_rel = self.lower_rel.max(lower_rel);
        } else {
            self.hp = None;
        }
        // |c - x| <= r |x| and |x| <= |c| + |c - x|  give
        // |c - x| <= r |c| / (1 - r).
        if self.rel < REL_CAP {
            let big = mul_up(self.rel, mag_max(self.range));
            self.abs = self.abs.min(div_up(big, one_minus(self.rel)));
        }
        // |x| >= |c| - |c - x|.
        let m = mag_min(self.range);
        if self.abs < m {
            self.rel = self.rel.min(div_up(self.abs, next_down(m - self.abs)));
        }
        if self.rel >= REL_CAP {
            self.rel = INF;
        }
        if self.abs == 0.0 {
            self.rel = 0.0;
        }
        if self.rel == 0.0 {
            self.abs = 0.0;
        }
        self
    }

    /// The value after an `if` whose arm is not known: what holds whichever
    /// arm ran.
    pub fn join(&self, o: &Num) -> Num {
        Num {
            range: self.range.hull(o.range),
            abs: self.abs.max(o.abs),
            rel: self.rel.max(o.rel),
            lower_abs: self.lower_abs.min(o.lower_abs),
            lower_rel: self.lower_rel.min(o.lower_rel),
            hp: if self.hp == o.hp { self.hp } else { None },
        }
    }
}

/// The relative rounding error `rho` of one correctly rounded operation, as
/// a sum with a propagated relative error: `(p + rho) / (1 - rho)`, rounded
/// up. (`|fl - y| <= rho |fl|` and `|y - t| <= p |t|` give
/// `|fl - t| <= (p + rho)/(1 - rho) |t|`.)
pub(crate) fn compose_rel(p: f64, rho: f64) -> f64 {
    if !p.is_finite() {
        return INF;
    }
    div_up(add_up(p, rho), one_minus(rho))
}
