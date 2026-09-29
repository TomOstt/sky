//! The f64 side of the exactness analysis: an enclosure of the number the
//! reference interpreter computes.
//!
//! A [`Range`] `[lo, hi]` contains **every value the interpreter can
//! produce for the value, if it produces one at all**. Sky numbers are
//! always finite, so every range lies inside `[-MAX, MAX]`. The transfer
//! functions in this module take the ranges of the operands and return a
//! [`Step`]: the range of the result, and whether the operation *may fail*
//! for some operand values inside those ranges (division by zero,
//! overflow, a domain error). "May not fail" is the fact a pass needs to
//! treat a `Fail` operation as pure.
//!
//! # Why this is sound
//!
//! * `+ - * /` and `sqrt` are correctly rounded by IEEE 754, and rounding
//!   is monotone, so for operands inside their ranges the result lies
//!   between the results at the corners of the operand box. The corners are
//!   computed with the same `f64` operations the interpreter uses, so those
//!   transfer functions are exact and add no slack.
//! * `powi` is repeated multiplication, so its result is within a few ulps
//!   of the true power; the endpoints get a relative slack that grows with
//!   the exponent.
//! * The transcendental functions (`exp`, `log`, `sin`, ...) come from the
//!   platform's libm, which promises a small error but not correct
//!   rounding, and differs between platforms. Their endpoints get a
//!   relative slack of [`LIBM_REL`] (about 450 ulps) and an absolute slack
//!   of [`LIBM_ABS`] for results near the subnormal range. A range is
//!   therefore a little wider than necessary, never narrower.
//! * A function this module does not know (a variant added to `MathFn`
//!   later) gets the whole finite range and "may fail" if `can_fail`
//!   says so.
//!
//! The sign of zero is not tracked: `-0.0` and `0.0` are the same point of a
//! range. Nothing that uses ranges needs the difference.

use std::fmt;

use crate::op::MathFn;

/// Relative slack for results of libm functions.
pub const LIBM_REL: f64 = 1e-13;
/// Absolute slack for results of libm functions, for results so small that
/// relative error means nothing (a few subnormal ulps).
pub const LIBM_ABS: f64 = 1e-322;

/// The next `f64` above `x` (`x` itself for `+inf` and NaN).
pub fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    let b = x.to_bits();
    f64::from_bits(if x > 0.0 { b + 1 } else { b - 1 })
}

/// The next `f64` below `x`.
pub fn next_down(x: f64) -> f64 {
    -next_up(-x)
}

/// An interval of finite `f64`s, both ends included.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Range {
    pub lo: f64,
    pub hi: f64,
}

/// The result of a transfer function.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Step {
    /// Every value the operation can produce when it succeeds.
    pub range: Range,
    /// Whether some operands inside their ranges make it fail.
    pub may_fail: bool,
}

impl Range {
    /// Any finite number: what nothing is known about.
    pub const FINITE: Range = Range {
        lo: -f64::MAX,
        hi: f64::MAX,
    };

    /// Exactly `x` (which must be finite; anything else gives [`Range::FINITE`]).
    pub fn point(x: f64) -> Range {
        if x.is_finite() {
            Range { lo: x, hi: x }
        } else {
            Range::FINITE
        }
    }

    /// `[lo, hi]`, clamped into the finite numbers. A NaN end, or `lo > hi`,
    /// gives [`Range::FINITE`].
    pub fn new(lo: f64, hi: f64) -> Range {
        if lo.is_nan() || hi.is_nan() || lo > hi {
            return Range::FINITE;
        }
        Range {
            lo: lo.max(-f64::MAX),
            hi: hi.min(f64::MAX),
        }
    }

    /// The smallest range containing both.
    pub fn hull(self, other: Range) -> Range {
        Range {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    pub fn contains(self, x: f64) -> bool {
        self.lo <= x && x <= self.hi
    }

    pub fn contains_zero(self) -> bool {
        self.contains(0.0)
    }

    /// The only value in the range, if it is one.
    pub fn as_point(self) -> Option<f64> {
        (self.lo == self.hi).then_some(self.lo)
    }

    /// Everything strictly above zero.
    pub fn is_positive(self) -> bool {
        self.lo > 0.0
    }

    /// Everything strictly below zero.
    pub fn is_negative(self) -> bool {
        self.hi < 0.0
    }

    /// Zero is not in the range.
    pub fn is_nonzero(self) -> bool {
        self.is_positive() || self.is_negative()
    }

    /// The range is every finite number.
    pub fn is_unbounded(self) -> bool {
        self == Range::FINITE
    }

    fn magnitude(self) -> (f64, f64) {
        let (a, b) = (self.lo.abs(), self.hi.abs());
        let lo = if self.contains_zero() { 0.0 } else { a.min(b) };
        (lo, a.max(b))
    }
}

/// A number the way an explanation writes it: short, and exponent form for
/// the very large and very small.
pub fn num_text(x: f64) -> String {
    if x == 0.0 {
        "0".to_string()
    } else if x.abs() >= 1e15 || x.abs() < 1e-4 {
        format!("{x:e}")
    } else {
        format!("{x}")
    }
}

impl fmt::Display for Range {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_unbounded() {
            f.write_str("any finite number")
        } else if let Some(p) = self.as_point() {
            f.write_str(&num_text(p))
        } else if self.lo == -f64::MAX {
            write!(f, "at most {}", num_text(self.hi))
        } else if self.hi == f64::MAX {
            write!(f, "at least {}", num_text(self.lo))
        } else {
            write!(f, "between {} and {}", num_text(self.lo), num_text(self.hi))
        }
    }
}

impl Step {
    /// Nothing is known about the result, and the operation can fail.
    pub const UNKNOWN: Step = Step {
        range: Range::FINITE,
        may_fail: true,
    };

    fn ok(range: Range) -> Step {
        Step {
            range,
            may_fail: false,
        }
    }

    /// The step for an interval whose ends were computed with `f64`
    /// operations that overflow to infinity: the operation can overflow, and
    /// the values that do not are inside the clamped range.
    fn checked(lo: f64, hi: f64) -> Step {
        if lo.is_nan() || hi.is_nan() {
            return Step::UNKNOWN;
        }
        Step {
            range: Range::new(lo, hi),
            may_fail: lo.is_infinite() || hi.is_infinite(),
        }
    }
}

fn lower(v: f64) -> f64 {
    v - v.abs() * LIBM_REL - LIBM_ABS
}

fn upper(v: f64) -> f64 {
    v + v.abs() * LIBM_REL + LIBM_ABS
}

fn min4(v: [f64; 4]) -> f64 {
    v.into_iter().fold(f64::INFINITY, f64::min)
}

fn max4(v: [f64; 4]) -> f64 {
    v.into_iter().fold(f64::NEG_INFINITY, f64::max)
}

/// `a + b`.
pub fn add(a: Range, b: Range) -> Step {
    Step::checked(a.lo + b.lo, a.hi + b.hi)
}

/// `a - b`.
pub fn sub(a: Range, b: Range) -> Step {
    Step::checked(a.lo - b.hi, a.hi - b.lo)
}

/// `a * b`.
pub fn mul(a: Range, b: Range) -> Step {
    let p = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi];
    Step::checked(min4(p), max4(p))
}

/// `a / b`. Fails when `b` can be zero.
pub fn div(a: Range, b: Range) -> Step {
    if b.contains_zero() {
        return Step::UNKNOWN;
    }
    let q = [a.lo / b.lo, a.lo / b.hi, a.hi / b.lo, a.hi / b.hi];
    Step::checked(min4(q), max4(q))
}

/// `-a`.
pub fn neg(a: Range) -> Step {
    Step::ok(Range {
        lo: -a.hi,
        hi: -a.lo,
    })
}

/// `1 / a`.
pub fn recip(a: Range) -> Step {
    div(Range::point(1.0), a)
}

/// `a ^ e` for the constant exponent `e`, following the interpreter: `powi`
/// for an integer exponent of at most 64, `powf` otherwise.
pub fn pow_const(a: Range, e: f64) -> Step {
    if !e.is_finite() {
        return Step::UNKNOWN;
    }
    if e.fract() == 0.0 && e.abs() <= 64.0 {
        return powi(a, e as i32);
    }
    // `powf`: a negative base with a fractional exponent has no real value,
    // and so does `0 ^ negative`. Stay on the safe side: a strictly
    // positive base (or a non-negative one for a positive exponent), where
    // the function is monotone.
    let ok = a.is_positive() || (a.lo >= 0.0 && e > 0.0);
    if !ok {
        return Step::UNKNOWN;
    }
    let (p, q) = (a.lo.powf(e), a.hi.powf(e));
    let (lo, hi) = if e > 0.0 { (p, q) } else { (q, p) };
    let s = Step::checked(lower(lo), upper(hi));
    Step {
        range: Range::new(s.range.lo.max(0.0), s.range.hi),
        ..s
    }
}

fn powi(a: Range, n: i32) -> Step {
    if n == 0 {
        return Step::ok(Range::point(1.0));
    }
    if n < 0 && a.contains_zero() {
        return Step::UNKNOWN;
    }
    let k = f64::from(n.unsigned_abs()) + 2.0;
    let rel = k * f64::EPSILON * 2.0;
    let abs = k * 1e-322;
    let (p, q) = (a.lo.powi(n), a.hi.powi(n));
    let (lo, hi) = if !a.contains_zero() || (n > 0 && n % 2 != 0) {
        // Monotone on each side of zero (increasing for an odd positive
        // exponent everywhere): the extremes are at the ends.
        (p.min(q), p.max(q))
    } else {
        // An even positive exponent across zero: minimum 0.
        (0.0, p.max(q))
    };
    Step::checked(lo - lo.abs() * rel - abs, hi + hi.abs() * rel + abs)
}

/// A function of one argument, increasing everywhere it is defined.
fn increasing(a: Range, f: impl Fn(f64) -> f64) -> Step {
    Step::checked(lower(f(a.lo)), upper(f(a.hi)))
}

/// `f(args)` for a built-in math function.
pub fn math(f: MathFn, args: &[Range]) -> Step {
    if args.len() != f.arity() {
        return Step::UNKNOWN;
    }
    let a = args[0];
    match f {
        MathFn::Sqrt => {
            if a.hi < 0.0 {
                return Step::UNKNOWN;
            }
            Step {
                range: Range::new(a.lo.max(0.0).sqrt(), a.hi.sqrt()),
                may_fail: a.lo < 0.0,
            }
        }
        MathFn::Cbrt => Step::ok(increasing(a, f64::cbrt).range),
        MathFn::Abs => {
            let (lo, hi) = a.magnitude();
            Step::ok(Range { lo, hi })
        }
        MathFn::Exp => {
            let s = increasing(a, f64::exp);
            Step {
                range: Range::new(s.range.lo.max(0.0), s.range.hi),
                ..s
            }
        }
        MathFn::Log | MathFn::Log10 | MathFn::Log2 => {
            if a.hi <= 0.0 {
                return Step::UNKNOWN;
            }
            let g: fn(f64) -> f64 = match f {
                MathFn::Log => f64::ln,
                MathFn::Log10 => f64::log10,
                _ => f64::log2,
            };
            let mut s = increasing(Range::new(a.lo.max(5e-324), a.hi), g);
            s.may_fail = a.lo <= 0.0;
            s
        }
        MathFn::Sin | MathFn::Cos => {
            let g: fn(f64) -> f64 = if f == MathFn::Sin { f64::sin } else { f64::cos };
            match a.as_point() {
                Some(x) => {
                    let v = g(x);
                    Step::ok(Range::new(lower(v).max(-1.0), upper(v).min(1.0)))
                }
                None => Step::ok(Range { lo: -1.0, hi: 1.0 }),
            }
        }
        MathFn::Tan => match a.as_point() {
            Some(x) => {
                let v = x.tan();
                Step::ok(Range::new(lower(v), upper(v)))
            }
            // `tan` of a finite number is finite (below 2e16), but there
            // is no simple tight range.
            None => Step::ok(Range::FINITE),
        },
        MathFn::Asin | MathFn::Acos => {
            if a.hi < -1.0 || a.lo > 1.0 {
                return Step::UNKNOWN;
            }
            let inside = Range::new(a.lo.max(-1.0), a.hi.min(1.0));
            let mut s = if f == MathFn::Asin {
                increasing(inside, f64::asin)
            } else {
                // Decreasing: the ends swap.
                let (p, q) = (inside.hi.acos(), inside.lo.acos());
                Step::checked(lower(p), upper(q))
            };
            s.may_fail = a.lo < -1.0 || a.hi > 1.0;
            s
        }
        MathFn::Atan => increasing(a, f64::atan),
        MathFn::Atan2 => {
            let x = args[1];
            Step {
                range: Range {
                    lo: -3.1416,
                    hi: 3.1416,
                },
                // Only the zero vector fails.
                may_fail: a.contains_zero() && x.contains_zero(),
            }
        }
        MathFn::Hypot => {
            let ((l1, h1), (l2, h2)) = (a.magnitude(), args[1].magnitude());
            let s = Step::checked(lower(l1.hypot(l2)), upper(h1.hypot(h2)));
            Step {
                range: Range::new(s.range.lo.max(0.0), s.range.hi),
                ..s
            }
        }
        MathFn::Min => Step::ok(Range {
            lo: a.lo.min(args[1].lo),
            hi: a.hi.min(args[1].hi),
        }),
        MathFn::Max => Step::ok(Range {
            lo: a.lo.max(args[1].lo),
            hi: a.hi.max(args[1].hi),
        }),
        MathFn::Floor => Step::ok(Range {
            lo: a.lo.floor(),
            hi: a.hi.floor(),
        }),
        MathFn::Ceil => Step::ok(Range {
            lo: a.lo.ceil(),
            hi: a.hi.ceil(),
        }),
        MathFn::Round => Step::ok(Range {
            lo: a.lo.round(),
            hi: a.hi.round(),
        }),
        // A function added after this analysis was written: know nothing.
        #[allow(unreachable_patterns)]
        _ => Step {
            range: Range::FINITE,
            may_fail: f.can_fail(),
        },
    }
}
