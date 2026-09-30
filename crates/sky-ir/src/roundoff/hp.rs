//! A rigorous high-precision enclosure of the exact real value of a
//! straight-line computation, in double-double arithmetic.
//!
//! The first-order bounds of the parent module are *upper* bounds on the
//! rounding error. They cannot say that an error is large, only that it is
//! not proven small. To prove a budget **violated** the analysis needs the
//! other side: a lower bound on the error, which needs the exact value. For
//! programs whose inputs are constants (most of a script) the exact value
//! can be enclosed to about 27 digits, far beyond the 16 of an `f64`, and
//! then the error of the computed `f64` is known to about 11 digits.
//!
//! An [`Enc`] is `hi + lo ± rad`: the exact real number lies in that
//! interval. `hi` and `lo` are the two halves of a double-double (`lo` at
//! most half an ulp of `hi`), `rad` is a rigorous radius, in `f64`, that
//! covers
//!
//! * the radius of the operands, propagated with interval rules, and
//! * the rounding of the double-double operation itself, which the
//!   algorithms of Joldes, Muller and Popescu (*Tight and rigorous error
//!   bounds for basic building blocks of double-word arithmetic*, 2017)
//!   bound by 3u² to 10u² of the result, `u = 2^-53` (about `1e-32`). The
//!   code claims [`OP_REL`] `= 2^-90` of the result (`8e-28`) instead, more
//!   than 10^4 times the proven bound, so a transcription slip in the
//!   algorithms would have to be large to matter, and the property tests in
//!   `sky-opt` compare against an independent evaluator.
//!
//! Every step that would round when computing `rad` itself is rounded *up*
//! (`next_up` after each operation). Values near the subnormal range, where
//! the double-double error terms underflow, and any operation the
//! algorithms do not cover, give `None`: no enclosure, so no claim.

use super::{add_up, div_up, mul_up};
use crate::exact::range::{next_down, next_up};

/// The relative error claimed for one double-double operation: `2^-90`.
pub const OP_REL: f64 = 8.077935669463161e-28;

/// Below this magnitude (and not zero) the low halves of a double-double
/// underflow, so nothing is claimed.
const TINY: f64 = 1e-250;

/// A few ulps, for turning a computed magnitude into a safe bound.
const PAD: f64 = 4.0 * f64::EPSILON;

/// The exact value lies in `[hi + lo - rad, hi + lo + rad]`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Enc {
    hi: f64,
    lo: f64,
    rad: f64,
}

type Dd = (f64, f64);

fn two_sum(a: f64, b: f64) -> Dd {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// `a + b` exactly, when `|a| >= |b|`.
fn fast_two_sum(a: f64, b: f64) -> Dd {
    let s = a + b;
    (s, b - (s - a))
}

fn two_prod(a: f64, b: f64) -> Dd {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

fn dd_neg(x: Dd) -> Dd {
    (-x.0, -x.1)
}

fn dd_add(x: Dd, y: Dd) -> Dd {
    let (sh, sl) = two_sum(x.0, y.0);
    let (th, tl) = two_sum(x.1, y.1);
    let c = sl + th;
    let (vh, vl) = fast_two_sum(sh, c);
    let w = tl + vl;
    fast_two_sum(vh, w)
}

fn dd_mul(x: Dd, y: Dd) -> Dd {
    let (ch, cl1) = two_prod(x.0, y.0);
    let tl0 = x.1 * y.1;
    let tl1 = x.0.mul_add(y.1, tl0);
    let cl2 = x.1.mul_add(y.0, tl1);
    fast_two_sum(ch, cl1 + cl2)
}

fn dd_mul_f(x: Dd, y: f64) -> Dd {
    let (ch, cl1) = two_prod(x.0, y);
    let cl2 = x.1 * y;
    fast_two_sum(ch, cl1 + cl2)
}

fn dd_div(x: Dd, y: Dd) -> Dd {
    let q1 = x.0 / y.0;
    let r = dd_add(x, dd_neg(dd_mul_f(y, q1)));
    let q2 = r.0 / y.0;
    let r = dd_add(r, dd_neg(dd_mul_f(y, q2)));
    let q3 = r.0 / y.0;
    let (a, b) = fast_two_sum(q1, q2);
    dd_add((a, b), (q3, 0.0))
}

/// `sqrt(x)` for `x >= 0`: the double square root, and one Newton step on
/// the exact residual.
fn dd_sqrt(x: Dd) -> Dd {
    if x.0 == 0.0 {
        return (0.0, 0.0);
    }
    let q1 = x.0.sqrt();
    let r = dd_add(x, dd_neg(two_prod(q1, q1)));
    fast_two_sum(q1, r.0 / (2.0 * q1))
}

fn tiny(x: f64) -> bool {
    x != 0.0 && x.abs() < TINY
}

impl Enc {
    /// The enclosure of the constant `x` (a point).
    pub fn point(x: f64) -> Option<Enc> {
        (x.is_finite() && !tiny(x)).then_some(Enc {
            hi: x,
            lo: 0.0,
            rad: 0.0,
        })
    }

    /// The result of an operation from its double-double value and the
    /// radius inherited from the operands (already rounded up); adds the
    /// operation's own error.
    fn make(v: Dd, inherited: f64) -> Option<Enc> {
        let (hi, lo) = v;
        if !hi.is_finite() || !lo.is_finite() || !inherited.is_finite() || tiny(hi) {
            return None;
        }
        let rad = add_up(inherited, mul_up(OP_REL, hi.abs()));
        rad.is_finite().then_some(Enc { hi, lo, rad })
    }

    fn dd(self) -> Dd {
        (self.hi, self.lo)
    }

    /// An upper bound on `|hi + lo|`.
    fn mag_up(self) -> f64 {
        mul_up(self.hi.abs(), 1.0 + PAD)
    }

    /// A lower bound on `|hi + lo|`.
    fn mag_lower(self) -> f64 {
        self.hi.abs() * (1.0 - PAD)
    }

    pub fn neg(self) -> Enc {
        Enc {
            hi: -self.hi,
            lo: -self.lo,
            rad: self.rad,
        }
    }

    pub fn add(self, o: Enc) -> Option<Enc> {
        if tiny(self.hi) || tiny(o.hi) {
            return None;
        }
        Enc::make(dd_add(self.dd(), o.dd()), add_up(self.rad, o.rad))
    }

    pub fn sub(self, o: Enc) -> Option<Enc> {
        self.add(o.neg())
    }

    pub fn mul(self, o: Enc) -> Option<Enc> {
        let v = dd_mul(self.dd(), o.dd());
        if v.0 == 0.0 && self.hi != 0.0 && o.hi != 0.0 {
            return None;
        }
        // |a| rb + |b| ra + ra rb
        let inherited = add_up(
            add_up(mul_up(self.mag_up(), o.rad), mul_up(o.mag_up(), self.rad)),
            mul_up(self.rad, o.rad),
        );
        Enc::make(v, inherited)
    }

    pub fn div(self, o: Enc) -> Option<Enc> {
        // The divisor's interval must stay well away from zero.
        let b = o.mag_lower();
        if o.hi == 0.0 || o.rad * 2.0 > b {
            return None;
        }
        let v = dd_div(self.dd(), o.dd());
        if v.0 == 0.0 && self.hi != 0.0 {
            return None;
        }
        let q = mul_up(v.0.abs(), 1.0 + PAD);
        // (|q| rb + ra) / (|b| - rb)
        let num = add_up(mul_up(q, o.rad), self.rad);
        let den = (b - o.rad) * (1.0 - PAD);
        Enc::make(v, div_up(num, den))
    }

    pub fn sqrt(self) -> Option<Enc> {
        // The exact operand must be non-negative.
        if self.hi < 0.0 || self.mag_lower() < self.rad {
            return None;
        }
        if self.hi == 0.0 {
            return (self.rad == 0.0).then_some(self);
        }
        let v = dd_sqrt(self.dd());
        // |sqrt(a) - sqrt(a')| <= min(r / sqrt(a), sqrt(r)) for a, a' >= 0.
        let inherited = if self.rad == 0.0 {
            0.0
        } else {
            let root = v.0 * (1.0 - PAD);
            div_up(self.rad, root).min(next_up(self.rad.sqrt()))
        };
        Enc::make(v, inherited)
    }

    /// `|self|`, when the sign of the exact value is known.
    pub fn abs(self) -> Option<Enc> {
        (self.mag_lower() >= self.rad).then(|| if self.hi < 0.0 { self.neg() } else { self })
    }

    /// `self ^ n` for a small integer `n`, by repeated multiplication.
    pub fn powi(self, n: i32) -> Option<Enc> {
        if n.unsigned_abs() > 64 {
            return None;
        }
        let one = Enc::point(1.0)?;
        let mut acc = one;
        for _ in 0..n.unsigned_abs() {
            acc = acc.mul(self)?;
        }
        if n < 0 {
            one.div(acc)
        } else {
            Some(acc)
        }
    }

    /// Bounds on the magnitude of the exact value: `(min, max)`.
    pub fn magnitude(self) -> (f64, f64) {
        let lo = (self.mag_lower() - self.rad).max(0.0);
        (lo, add_up(self.mag_up(), self.rad))
    }

    /// The centre, rounded to an `f64`.
    pub fn value(self) -> f64 {
        self.hi + self.lo
    }

    /// An interval of `f64`s that contains the exact value.
    pub fn interval(self) -> (f64, f64) {
        let lower = dd_add(self.dd(), (-self.rad, 0.0));
        let upper = dd_add(self.dd(), (self.rad, 0.0));
        (next_down(lower.0 + lower.1), next_up(upper.0 + upper.1))
    }

    /// Bounds on `|c - exact|` for the `f64` `c`: `(lower, upper)`, both
    /// rigorous.
    pub fn error_of(self, c: f64) -> (f64, f64) {
        let d = dd_add((c, 0.0), dd_neg(self.dd()));
        // `d` is `c - (hi + lo)` to about 1e-32 relative; its magnitude,
        // padded either way by 1e-12 of itself.
        let m = (d.0 + d.1).abs();
        let up = add_up(mul_up(m, 1.0 + 1e-12), self.rad);
        let down = (m * (1.0 - 1e-12) - self.rad).max(0.0);
        (down, up)
    }
}
