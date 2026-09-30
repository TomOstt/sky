//! The error transfer function of each scalar operation.
//!
//! Each function takes the abstract values of the operands ([`Num`]) and
//! returns the result's: the range of the computed `f64` (the ranges of
//! `crate::exact::range`), and its error against the exact real result.
//!
//! # How an operation's error is built
//!
//! For `y = op(x_1, ...)` the computed result is `fl(op(c_1, ...))`, where
//! `c_k` are the computed operands and `x_k` the exact ones. Its distance to
//! the exact `t = op(x_1, ...)` is at most
//!
//! * the **propagated** error `|op(c) - op(x)|`, from the operands' errors,
//!   bounded through the operation's derivative (a Lipschitz constant over
//!   the range, in the absolute channel; a condition number, in the
//!   relative channel), plus
//! * the **rounding** error `|fl(op(c)) - op(c)|` of this one operation:
//!   `u |fl|` for a correctly rounded one (`u = 2^-53`), a few ulps for a
//!   math function, and exactly zero for `neg`, `abs`, `min`, `max`,
//!   `floor`, ... which round nothing.
//!
//! Bounds are *not* first-order expansions with the second-order terms
//! dropped: each is an inequality that holds for all sizes of the incoming
//! error (`(1+p)(1+u) - 1` and not `p + u`; a mean-value bound over the
//! whole interval the exact operand can be in, and not a derivative at one
//! point). So a bound is valid even when the error is large, and the
//! analysis is sound, not just accurate to first order.
//!
//! When all operands are single points, the rounding error of `+ - * / sqrt`
//! is computed exactly instead (an error-free transformation with `fma`), so
//! `1/4 + 1/2` is known to be exact and `0.1 + 0.2` to be off by exactly its
//! `5.55e-17`.
//!
//! # Assumptions
//!
//! * `+ - * / sqrt` are correctly rounded (IEEE 754, round to nearest).
//! * The platform's math library is within [`LIBM_ULPS`] ulps for the
//!   functions `sky run` calls (`Erf` is Sky's own, measured at 7 ulps, and
//!   is given 16). A library that is worse than that voids the bounds of
//!   the operations that use it, and no others.
//! * `powi` is repeated multiplication, within `(|n| + 2)` ulps.
//! * Failing executions (overflow, a domain error, a division by zero) are
//!   not analysed: a bound is about the runs that produce a number.

use sky_units::Rational;

use super::hp::Enc;
use super::num::{compose_rel, mag_max, mag_min, Num, Sign};
use super::{add_up, div_up, mul_up, one_minus, pad, INF};
use crate::exact::range::{self, next_down, next_up, Range};
use crate::op::{MathFn, OpKind};

/// `2^-53`, the unit roundoff of round-to-nearest `f64`.
pub const U: f64 = 1.1102230246251565e-16;
/// The absolute error of one rounding in the subnormal range: half the
/// smallest subnormal, rounded up to it.
const ETA: f64 = 5e-324;

/// `2/sqrt(pi)`, the largest slope of `erf`, rounded up.
fn erf_slope() -> f64 {
    next_up(std::f64::consts::FRAC_2_SQRT_PI)
}

/// How many ulps (`2^-52` relative) a math function is assumed to be off by.
pub const LIBM_ULPS: f64 = 4.0;

/// The ulps assumed for `f`.
pub fn libm_ulps(f: MathFn) -> f64 {
    match f {
        // Sky's own series and continued fraction; measured worst 7.
        MathFn::Erf => 16.0,
        // Sky's large-argument branch adds `ln 2` to `ln x`.
        MathFn::Asinh | MathFn::Acosh => 8.0,
        MathFn::Hypot => 2.0,
        _ => LIBM_ULPS,
    }
}

/// What the blame report needs to know about one operation.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct OpNote {
    /// The absolute rounding error this operation adds.
    pub round_abs: f64,
    /// How much an operand's absolute error is multiplied into the result's.
    pub lip: [f64; 2],
    /// The largest relative error among the operands.
    pub rel_in: f64,
}

impl OpNote {
    const NONE: OpNote = OpNote {
        round_abs: 0.0,
        lip: [0.0, 0.0],
        rel_in: 0.0,
    };

    pub(crate) fn join(self, o: OpNote) -> OpNote {
        OpNote {
            round_abs: self.round_abs.max(o.round_abs),
            lip: [self.lip[0].max(o.lip[0]), self.lip[1].max(o.lip[1])],
            rel_in: self.rel_in.max(o.rel_in),
        }
    }
}

pub(crate) struct Out {
    pub num: Num,
    pub note: OpNote,
}

/// The rounding an operation adds.
#[derive(Clone, Copy)]
enum Round {
    /// None: the result is a selection or an integer.
    Exact,
    /// A correctly rounded `*`, `/` or `sqrt`: `u`, and an absolute error
    /// when the result is subnormal.
    Correct,
    /// A correctly rounded `+` or `-`: `u`; subnormal results are exact.
    CorrectAdd,
    /// A multiplication or division by a power of two: nothing, but a
    /// subnormal result rounds.
    Scale,
    /// A library function or `powi`: this relative error.
    Rel(f64),
    /// A function Rust composes from others (`atanh`, `acosh`), whose
    /// error near a critical point is not relative to the result: this
    /// relative error, and this much absolute error on top.
    RelPlus(f64, f64),
}

/// The error the inputs' errors cause in the result (before its rounding).
#[derive(Clone, Copy)]
struct Prop {
    abs: f64,
    rel: f64,
}

const NO_PROP: Prop = Prop { abs: INF, rel: INF };

fn finish(
    prop: Prop,
    round: Round,
    range: Range,
    eft: Option<f64>,
    hp: Option<Enc>,
    lip: [f64; 2],
    rel_in: f64,
) -> Out {
    let big = mag_max(range);
    let small = mag_min(range);
    let (mut rho, underflow, extra) = match round {
        Round::Exact => (0.0, false, 0.0),
        Round::Correct => (U, true, 0.0),
        Round::CorrectAdd => (U, false, 0.0),
        Round::Scale => (0.0, true, 0.0),
        Round::Rel(r) => (r, true, 0.0),
        Round::RelPlus(r, e) => (r, true, e),
    };
    let mut rel_valid = (!underflow || small >= f64::MIN_POSITIVE) && extra == 0.0;
    let round_abs = match eft {
        Some(e) => {
            // The rounding error itself is known: no `u` bound needed.
            rel_valid = true;
            rho = match range.as_point() {
                Some(c) if c != 0.0 => div_up(e, c.abs()),
                _ if e == 0.0 => 0.0,
                _ => {
                    rel_valid = false;
                    0.0
                }
            };
            e
        }
        None => add_up(
            add_up(mul_up(rho, big), if underflow { ETA } else { 0.0 }),
            extra,
        ),
    };
    let abs = add_up(prop.abs, round_abs);
    let rel = if !rel_valid {
        INF
    } else if rho == 0.0 {
        prop.rel
    } else {
        compose_rel(prop.rel, rho)
    };
    let num = Num {
        range,
        abs,
        rel,
        lower_abs: 0.0,
        lower_rel: 0.0,
        hp,
    }
    .tighten();
    Out {
        num,
        note: OpNote {
            round_abs,
            lip,
            rel_in,
        },
    }
}

fn plain(num: Num) -> Out {
    Out {
        num,
        note: OpNote::NONE,
    }
}

fn is_pow2(x: f64) -> bool {
    x != 0.0 && x.is_finite() && x.to_bits() & ((1u64 << 52) - 1) == 0
}

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

fn rel_in(args: &[&Num]) -> f64 {
    args.iter().fold(0.0, |m, a| m.max(a.rel))
}

/// The result of `kind` on `args`, or `None` for an operation this analysis
/// does not model (its results are then unknown). `same` says the two
/// operands are the same value.
pub(crate) fn transfer(kind: &OpKind, args: &[&Num], same: bool) -> Option<Out> {
    Some(match (kind, args) {
        (OpKind::ConstFloat { bits }, []) => {
            let x = f64::from_bits(*bits);
            if !x.is_finite() {
                return None;
            }
            plain(Num::exact(Range::point(x)))
        }
        (OpKind::ConstRational(q), []) => plain(const_rational(*q)),
        (OpKind::Neg, [x]) => neg(x),
        (OpKind::Add, [x, y]) => add_sub(x, y, false, same),
        (OpKind::Sub, [x, y]) => add_sub(x, y, true, same),
        (OpKind::Mul, [x, y]) => mul(x, y),
        (OpKind::Div, [x, y]) => div(x, y, same),
        (OpKind::Inverse, [x]) => div(&Num::exact(Range::point(1.0)), x, false),
        (OpKind::PowConst { exponent }, [x]) => pow_const(x, *exponent),
        (OpKind::Pow, [x, y]) => pow(x, y),
        (OpKind::Math(f), _) if args.len() == f.arity() => math(*f, args),
        _ => return None,
    })
}

/// A rational constant, as `numer as f64 / denom as f64` (one rounding).
fn const_rational(q: Rational) -> Num {
    let (n, d) = (q.numer(), q.denom());
    let x = n as f64 / d as f64;
    let mut num = Num::exact(Range::point(x));
    let exact_int = |v: i64| v.unsigned_abs() < (1u64 << 53);
    if exact_int(n) && exact_int(d) {
        // `x` is `n / d` rounded once: the residual `x d - n` is exact.
        let r = x.mul_add(d as f64, -(n as f64)).abs();
        num.abs = div_up(r, d as f64);
        num.hp = Enc::point(n as f64).and_then(|n| n.div(Enc::point(d as f64)?));
        if x != 0.0 {
            num.rel = div_up(num.abs, x.abs());
        }
    } else {
        // Each of `n` and `d` may itself round on conversion.
        num.rel = 4.0 * U;
        num.abs = mul_up(num.rel, x.abs());
        num.hp = None;
    }
    if x == 0.0 {
        num.abs = 0.0;
        num.rel = 0.0;
    }
    num.tighten()
}

fn neg(x: &Num) -> Out {
    Out {
        num: Num {
            range: Range {
                lo: -x.range.hi,
                hi: -x.range.lo,
            },
            hp: x.hp.map(Enc::neg),
            ..*x
        },
        note: OpNote {
            round_abs: 0.0,
            lip: [1.0, 0.0],
            rel_in: x.rel,
        },
    }
}

fn add_sub(x: &Num, y: &Num, sub: bool, same: bool) -> Out {
    if sub && same {
        // `x - x` is exactly zero, whatever `x` is.
        return plain(Num::exact(Range::point(0.0)));
    }
    let step = if sub {
        range::sub(x.range, y.range)
    } else {
        range::add(x.range, y.range)
    };
    let sy = if sub { y.sign().flip() } else { y.sign() };
    let sx = x.sign();
    // Two terms of one sign: `|(x + y)| = |x| + |y|`, so the relative errors
    // do not grow. Otherwise the terms may cancel and only the absolute
    // channel carries over.
    let same_sign =
        (sx.non_negative() && sy.non_negative()) || (sx.non_positive() && sy.non_positive());
    let rel = if same_sign { x.rel.max(y.rel) } else { INF };
    let eft = match (x.range.as_point(), y.range.as_point()) {
        (Some(a), Some(b)) => {
            let (s, e) = two_sum(a, if sub { -b } else { b });
            s.is_finite().then_some(e.abs())
        }
        _ => None,
    };
    let hp = match (x.hp, y.hp) {
        (Some(a), Some(b)) => {
            if sub {
                a.sub(b)
            } else {
                a.add(b)
            }
        }
        _ => None,
    };
    finish(
        Prop {
            abs: add_up(x.abs, y.abs),
            rel,
        },
        Round::CorrectAdd,
        step.range,
        eft,
        hp,
        [1.0, 1.0],
        rel_in(&[x, y]),
    )
}

fn mul(x: &Num, y: &Num) -> Out {
    let step = range::mul(x.range, y.range);
    let rel = if x.rel.is_finite() && y.rel.is_finite() {
        add_up(add_up(x.rel, y.rel), mul_up(x.rel, y.rel))
    } else {
        INF
    };
    // |c_x c_y - x y| <= |c_x| |c_y - y| + |y| |c_x - x|
    let (mx, my) = (mag_max(x.range), mag_max(y.range));
    let abs = add_up(mul_up(mx, y.abs), mul_up(add_up(my, y.abs), x.abs));
    let (eft, round) = match (x.range.as_point(), y.range.as_point()) {
        (Some(a), Some(b)) => {
            let p = a * b;
            let e = a.mul_add(b, -p);
            // Below about 1e-290 the product's low part underflows.
            let ok = p.is_finite() && (p == 0.0 && (a == 0.0 || b == 0.0) || p.abs() > 1e-290);
            let round = if is_pow2(a) || is_pow2(b) {
                Round::Scale
            } else {
                Round::Correct
            };
            (ok.then_some(e.abs()), round)
        }
        (Some(a), _) | (_, Some(a)) if is_pow2(a) => (None, Round::Scale),
        _ => (None, Round::Correct),
    };
    let hp = match (x.hp, y.hp) {
        (Some(a), Some(b)) => a.mul(b),
        _ => None,
    };
    finish(
        Prop { abs, rel },
        round,
        step.range,
        eft,
        hp,
        [add_up(my, y.abs), mx],
        rel_in(&[x, y]),
    )
}

fn div(x: &Num, y: &Num, same: bool) -> Out {
    if same && !y.range.contains_zero() {
        return plain(Num::exact(Range::point(1.0)));
    }
    let step = range::div(x.range, y.range);
    let rel = if x.rel.is_finite() && y.rel.is_finite() {
        div_up(add_up(x.rel, y.rel), one_minus(y.rel))
    } else {
        INF
    };
    // |c_x / c_y - x / y| <= (|c_x| |c_y - y| + |c_y| |c_x - x|) / (|c_y| |y|)
    let dy = mag_min(y.range);
    let dyx = next_down(dy - y.abs);
    let (abs, lip) = if dy > 0.0 && dyx > 0.0 && x.abs.is_finite() && y.abs.is_finite() {
        let den = next_down(dy * dyx);
        let (mx, my) = (mag_max(x.range), mag_max(y.range));
        (
            div_up(add_up(mul_up(mx, y.abs), mul_up(my, x.abs)), den),
            [div_up(1.0, dyx), div_up(mx, den)],
        )
    } else {
        (INF, [INF, INF])
    };
    let (eft, round) = match (x.range.as_point(), y.range.as_point()) {
        (Some(a), Some(b)) if b != 0.0 => {
            let q = a / b;
            // `q b - a` is exact, and `q - a/b = (q b - a) / b`.
            let r = q.mul_add(b, -a);
            let ok = q.is_finite() && q.abs() > 1e-280 && a.abs() > 1e-280;
            (
                ok.then(|| div_up(r.abs(), b.abs())),
                if is_pow2(b) {
                    Round::Scale
                } else {
                    Round::Correct
                },
            )
        }
        (_, Some(b)) if is_pow2(b) => (None, Round::Scale),
        _ => (None, Round::Correct),
    };
    let hp = match (x.hp, y.hp) {
        (Some(a), Some(b)) => a.div(b),
        _ => None,
    };
    finish(
        Prop { abs, rel },
        round,
        step.range,
        eft,
        hp,
        lip,
        rel_in(&[x, y]),
    )
}

/// `x ^ e` for a constant exponent, as the interpreter computes it: `powi`
/// for an integer of at most 64, `powf` otherwise.
fn pow_const(x: &Num, exponent: Rational) -> Out {
    let e = exponent.numer() as f64 / exponent.denom() as f64;
    if !e.is_finite() {
        return plain(Num::unknown());
    }
    if e.fract() == 0.0 && e.abs() <= 64.0 {
        return pow_int(x, e as i32);
    }
    // The exponent the interpreter passes to `powf` is `n / d` rounded: the
    // exact `x ^ (n/d)` differs by a factor `x ^ delta`.
    let small = |v: i64| v.unsigned_abs() < (1u64 << 53);
    let exact = small(exponent.numer())
        && small(exponent.denom())
        && e.mul_add(exponent.denom() as f64, -(exponent.numer() as f64)) == 0.0;
    let delta = if exact {
        0.0
    } else if small(exponent.numer()) && small(exponent.denom()) {
        mul_up(U, e.abs())
    } else {
        mul_up(3.0 * U, e.abs())
    };
    pow_real(x, e, delta)
}

fn pow_int(x: &Num, n: i32) -> Out {
    match n {
        0 => return plain(Num::exact(Range::point(1.0))),
        1 => {
            return Out {
                num: *x,
                note: OpNote::NONE,
            }
        }
        2 => return mul(x, x),
        -1 => return div(&Num::exact(Range::point(1.0)), x, false),
        _ => {}
    }
    let step = range::pow_const(x.range, f64::from(n));
    let k = f64::from(n.unsigned_abs());
    let r = x.rel;
    // |(1+e)^n - 1| <= |n| |e| sup (1+xi)^(n-1), xi between 0 and e: the
    // sup is at 1 + r for n > 0 and at 1 - r for n < 0.
    let rel = if r.is_finite() {
        let sup = if n > 0 {
            pad((1.0 + r).powi(n - 1))
        } else {
            pad(one_minus(r).powi(n - 1))
        };
        mul_up(mul_up(k, r), sup)
    } else {
        INF
    };
    // n > 0: |c^n - x^n| <= n max(|c|, |x|)^(n-1) |c - x|.
    let (abs, l) = if n > 0 && x.abs.is_finite() {
        let l = mul_up(k, pad(add_up(mag_max(x.range), x.abs).powi(n - 1)));
        (mul_up(l, x.abs), l)
    } else {
        (INF, INF)
    };
    // `powi` is repeated multiplication: about one rounding per factor.
    let rho = (k + 2.0) * f64::EPSILON;
    let hp = x.hp.and_then(|h| h.powi(n));
    finish(
        Prop { abs, rel },
        Round::Rel(rho),
        step.range,
        None,
        hp,
        [l, 0.0],
        x.rel,
    )
}

/// `x ^ p` by `powf`, where `p` is the exponent's `f64`, `delta` away from
/// the exponent the program means.
fn pow_real(x: &Num, p: f64, delta: f64) -> Out {
    let step = range::pow_const(x.range, p);
    if x.sign() != Sign::Pos {
        return plain(Num {
            range: step.range,
            ..Num::unknown()
        });
    }
    let r = x.rel;
    let rel = if r.is_finite() {
        // 0 < p < 1 and p < 0 take the sup at 1 - r, p > 1 at 1 + r.
        let base = if p > 1.0 { 1.0 + r } else { 1.0 - r };
        let sup = pad(base.powf(p - 1.0));
        mul_up(mul_up(p.abs(), r), sup)
    } else {
        INF
    };
    // x ^ (p + delta) = x ^ p (1 + delta ln x + ...).
    let rel = if delta == 0.0 {
        rel
    } else {
        let (lo, hi) = hull(x);
        if lo > 0.0 && hi.is_finite() {
            let l = pad(lo.ln().abs().max(hi.ln().abs()));
            let extra = pad(mul_up(delta, l).exp_m1());
            add_up(add_up(rel, extra), mul_up(rel, extra))
        } else {
            INF
        }
    };
    finish(
        Prop { abs: INF, rel },
        Round::Rel(libm_ulps(MathFn::Exp) * f64::EPSILON),
        step.range,
        None,
        None,
        [INF, 0.0],
        x.rel,
    )
}

/// `x ^ y` with both computed (both dimensionless). Modelled only for a
/// base that is positive for sure.
fn pow(x: &Num, y: &Num) -> Out {
    let unknown = plain(Num::unknown());
    if x.sign() != Sign::Pos || !x.rel.is_finite() || !y.abs.is_finite() {
        return unknown;
    }
    let (r, ab) = (x.rel, y.abs);
    // The exact base is in [lo / (1 + r), hi / (1 - r)].
    let lo = x.range.lo / (1.0 + r);
    let hi = x.range.hi / one_minus(r);
    if !(lo > 0.0 && hi.is_finite()) {
        return unknown;
    }
    let l = pad(lo.ln().abs().max(hi.ln().abs()));
    let bmax = add_up(mag_max(y.range), ab);
    // |ln y| = |b ln a|: the change is at most |db| |ln a| + |b| |d ln a|.
    let delta = add_up(mul_up(l, ab), mul_up(bmax, div_up(r, one_minus(r))));
    let rel = if delta < 0.4 {
        mul_up(pad(delta.exp_m1()), 1.0)
    } else {
        INF
    };
    // The result's range: monotone in each argument for a positive base.
    let corners = [
        x.range.lo.powf(y.range.lo),
        x.range.lo.powf(y.range.hi),
        x.range.hi.powf(y.range.lo),
        x.range.hi.powf(y.range.hi),
    ];
    let lo_v = corners.iter().copied().fold(INF, f64::min);
    let hi_v = corners.iter().copied().fold(-INF, f64::max);
    let range = if lo_v.is_finite() && hi_v.is_finite() {
        Range::new(
            (lo_v - lo_v.abs() * range::LIBM_REL).max(0.0),
            hi_v + hi_v.abs() * range::LIBM_REL,
        )
    } else {
        Range::FINITE
    };
    finish(
        Prop { abs: INF, rel },
        // An integer exponent of at most 64 is `powi`, whatever the base:
        // (|n| + 2) ulps.
        Round::Rel(66.0 * f64::EPSILON),
        range,
        None,
        None,
        [INF, INF],
        rel_in(&[x, y]),
    )
}

/// `sqrt` of a non-negative exact value.
fn sqrt(x: &Num) -> Out {
    let step = range::math(MathFn::Sqrt, &[x.range]);
    let rel = if x.rel.is_finite() {
        // |sqrt(1+e) - 1| <= r / (2 sqrt(1-r)) for |e| <= r.
        let r = x.rel;
        div_up(r, 2.0 * next_down(pad_down((1.0 - r).sqrt())))
    } else {
        INF
    };
    // |sqrt(a) - sqrt(b)| <= sqrt(|a - b|).
    let abs = if x.abs.is_finite() {
        pad(x.abs.sqrt())
    } else {
        INF
    };
    // The residual `q q - a` is exact, and `q - sqrt(a) = r / (q + sqrt(a))`.
    let eft = x.range.as_point().and_then(|a| {
        let q = a.sqrt();
        (a > 1e-280 && q.is_finite()).then(|| {
            let r = q.mul_add(q, -a).abs();
            div_up(r, q * (1.0 - 4.0 * f64::EPSILON))
        })
    });
    let lip = match x.range.as_point() {
        Some(a) if a > 0.0 => [div_up(1.0, 2.0 * next_down(a.sqrt())), 0.0],
        _ => [INF, 0.0],
    };
    finish(
        Prop { abs, rel },
        Round::Correct,
        step.range,
        eft,
        x.hp.and_then(Enc::sqrt),
        lip,
        x.rel,
    )
}

/// The functions with one argument whose derivative is bounded by `l` on
/// the exact argument's hull: `|f(c) - f(x)| <= l |c - x|`.
fn lipschitz(x: &Num, l: f64) -> Prop {
    Prop {
        abs: mul_up(l, x.abs),
        rel: INF,
    }
}

/// `(lo, hi)` of every point between the computed argument and the exact
/// one (the segment a mean-value bound runs over), or infinite ends.
fn hull(x: &Num) -> (f64, f64) {
    if !x.abs.is_finite() {
        return (-INF, INF);
    }
    let h = x.exact_hull().hull(x.range);
    (h.lo, h.hi)
}

/// The largest magnitude the exact argument can have.
fn hull_mag(x: &Num) -> f64 {
    let (lo, hi) = hull(x);
    lo.abs().max(hi.abs())
}

fn math(f: MathFn, args: &[&Num]) -> Out {
    let x = args[0];
    if f == MathFn::Sqrt {
        // The exact operand must be non-negative for the run to succeed.
        return sqrt(x);
    }
    let ranges: Vec<Range> = args.iter().map(|a| a.range).collect();
    let step = math_range(f, &ranges);
    let ulp = libm_ulps(f) * f64::EPSILON;
    let lip1 = |l: f64| [l, 0.0];
    let (prop, round, lip, hp): (Prop, Round, [f64; 2], Option<Enc>) = match f {
        MathFn::Sqrt => unreachable!("handled above"),
        MathFn::Abs => {
            let hp = x.hp.and_then(Enc::abs);
            (
                Prop {
                    abs: x.abs,
                    rel: x.rel,
                },
                Round::Exact,
                lip1(1.0),
                hp,
            )
        }
        MathFn::Cbrt => {
            let rel = if x.rel.is_finite() {
                // (1 + e)^(1/3): the slope is largest at 1 - r.
                let r = x.rel;
                div_up(r, 3.0 * next_down(pad_down((1.0 - r).powf(2.0 / 3.0))))
            } else {
                INF
            };
            // |cbrt(a) - cbrt(b)| <= 2^(2/3) cbrt(|a - b|): the constant is
            // reached for a and b of opposite sign.
            let abs = if x.abs.is_finite() {
                mul_up(pad(x.abs.cbrt()), 1.587_401_051_968_199_8)
            } else {
                INF
            };
            (Prop { abs, rel }, Round::Rel(ulp), [INF, 0.0], None)
        }
        MathFn::Exp => {
            let rel = if x.abs.is_finite() {
                pad(x.abs.exp_m1())
            } else {
                INF
            };
            let l = pad(hull(x).1.exp());
            (Prop { abs: INF, rel }, Round::Rel(ulp), lip1(l), None)
        }
        MathFn::Log | MathFn::Log2 | MathFn::Log10 => {
            let scale = match f {
                MathFn::Log => 1.0,
                MathFn::Log2 => next_up(std::f64::consts::LOG2_E),
                _ => next_up(std::f64::consts::LOG10_E),
            };
            // ln(1 + e) is at most r / (1 - r) in size; or, on the hull,
            // 1 / lo in slope.
            let by_rel = if x.rel.is_finite() {
                mul_up(scale, div_up(x.rel, one_minus(x.rel)))
            } else {
                INF
            };
            let (hl, _) = hull(x);
            let by_abs = if hl > 0.0 && x.abs.is_finite() {
                mul_up(scale, div_up(x.abs, hl))
            } else {
                INF
            };
            let l = if hl > 0.0 { div_up(scale, hl) } else { INF };
            (
                Prop {
                    abs: by_rel.min(by_abs),
                    rel: INF,
                },
                Round::Rel(ulp),
                lip1(l),
                None,
            )
        }
        MathFn::Sin | MathFn::Cos => (lipschitz(x, 1.0), Round::Rel(ulp), lip1(1.0), None),
        MathFn::Atan => (lipschitz(x, 1.0), Round::Rel(ulp), lip1(1.0), None),
        MathFn::Tan => {
            let b = hull_mag(x);
            let l = if let (Some(c), true) = (x.range.as_point(), x.abs.is_finite()) {
                // tan(c + d) = (t + tan d) / (1 - t tan d) for |d| <= abs.
                let t = c.tan().abs();
                let td = x.abs.tan();
                let den = 1.0 - t * td;
                if x.abs < 0.5 && den > 0.1 {
                    let big = pad((t + td) / den);
                    pad(1.0 + big * big)
                } else {
                    INF
                }
            } else if b < 1.5 {
                let c = b.cos();
                pad(1.0 / (c * c))
            } else {
                INF
            };
            (lipschitz(x, l), Round::Rel(ulp), lip1(l), None)
        }
        MathFn::Asin | MathFn::Acos => {
            let b = hull_mag(x);
            let l = if b < 0.999_999 {
                pad(1.0 / next_down(lower_1_minus_sq(b).sqrt()))
            } else {
                INF
            };
            (lipschitz(x, l), Round::Rel(ulp), lip1(l), None)
        }
        MathFn::Atan2 => {
            // atan2(y, x): `x` (the first argument) is the y coordinate.
            let (y, xc) = (args[0], args[1]);
            let (hy, hx) = (hull(y), hull(xc));
            // The angle jumps by 2 pi across the negative x axis; the exact
            // point and the computed one must be on one side of it.
            let no_cross = hx.0 > 0.0 || hy.0 > 0.0 || hy.1 < 0.0;
            // The exact point is at least this far from the origin.
            let dy = (mag_min(y.range) - y.abs).max(0.0);
            let dx = (mag_min(xc.range) - xc.abs).max(0.0);
            let rmin = next_down(dx.hypot(dy) * (1.0 - 1e-14));
            let l = if rmin > 0.0 { div_up(1.0, rmin) } else { INF };
            let prop = if no_cross && y.abs.is_finite() && xc.abs.is_finite() {
                Prop {
                    abs: mul_up(l, add_up(y.abs, xc.abs)),
                    rel: INF,
                }
            } else {
                NO_PROP
            };
            (prop, Round::Rel(ulp), [l, l], None)
        }
        MathFn::Hypot => {
            let y = args[1];
            (
                Prop {
                    abs: add_up(x.abs, y.abs),
                    rel: x.rel.max(y.rel),
                },
                Round::Rel(ulp),
                [1.0, 1.0],
                None,
            )
        }
        MathFn::Min | MathFn::Max => {
            let y = args[1];
            (
                Prop {
                    abs: x.abs.max(y.abs),
                    rel: INF,
                },
                Round::Exact,
                [1.0, 1.0],
                None,
            )
        }
        MathFn::Floor | MathFn::Ceil | MathFn::Round => {
            let abs = if x.abs == 0.0 {
                0.0
            } else {
                add_up(x.abs, 1.0)
            };
            (Prop { abs, rel: INF }, Round::Exact, lip1(1.0), None)
        }
        MathFn::Expm1 => {
            let l = pad(hull(x).1.exp());
            (lipschitz(x, l), Round::Rel(ulp), lip1(l), None)
        }
        MathFn::Log1p => {
            let (hl, _) = hull(x);
            let l = if 1.0 + hl > 0.01 {
                div_up(1.0, next_down(1.0 + hl))
            } else {
                INF
            };
            (lipschitz(x, l), Round::Rel(ulp), lip1(l), None)
        }
        MathFn::Sinh | MathFn::Cosh => {
            let b = hull_mag(x);
            let l = pad(if f == MathFn::Sinh {
                b.cosh()
            } else {
                b.sinh()
            });
            (lipschitz(x, l), Round::Rel(ulp), lip1(l), None)
        }
        MathFn::Tanh | MathFn::Asinh => (lipschitz(x, 1.0), Round::Rel(ulp), lip1(1.0), None),
        MathFn::Acosh => {
            let (hl, _) = hull(x);
            let l = if hl > 1.0001 {
                pad(1.0 / next_down((((hl - 1.0) * (hl + 1.0)) * (1.0 - 1e-14)).sqrt()))
            } else {
                INF
            };
            // ln(x + sqrt(x - 1) sqrt(x + 1)): the sum rounds once in
            // absolute terms, whatever the (perhaps small) result is.
            (lipschitz(x, l), Round::RelPlus(ulp, 2.0 * U), lip1(l), None)
        }
        MathFn::Atanh => {
            let b = hull_mag(x);
            let l = if b < 0.999_999 {
                pad(1.0 / next_down(lower_1_minus_sq(b)))
            } else {
                INF
            };
            // 0.5 ln_1p(2x / (1 - x)): the argument carries a few `u` of
            // relative error, which for a negative x near -1 is a large
            // absolute error in `1 + arg`: the amplification is
            // 2|x| / (1 + x).
            let (hl, _) = hull(x);
            let round = if hl >= 0.0 {
                Round::Rel(ulp)
            } else if hl > -1.0 {
                let amp = div_up(mul_up(2.0, -hl), next_down(1.0 + hl));
                Round::RelPlus(ulp, mul_up(4.0 * U, amp))
            } else {
                Round::RelPlus(ulp, INF)
            };
            (lipschitz(x, l), round, lip1(l), None)
        }
        MathFn::Erf => (
            lipschitz(x, erf_slope()),
            Round::Rel(ulp),
            lip1(erf_slope()),
            None,
        ),
        MathFn::Sign => {
            let (hl, hh) = hull(x);
            let abs = if x.abs == 0.0 || hl > 0.0 || hh < 0.0 {
                0.0
            } else {
                2.0
            };
            (Prop { abs, rel: INF }, Round::Exact, lip1(0.0), None)
        }
    };
    finish(prop, round, step, None, hp, lip, rel_in(args))
}

/// A lower bound on `1 - b^2` for `0 <= b < 1`, without the cancellation
/// of `1 - b*b`.
fn lower_1_minus_sq(b: f64) -> f64 {
    (1.0 - b) * (1.0 + b) * (1.0 - 1e-14)
}

fn pad_down(x: f64) -> f64 {
    x * (1.0 - 1e-13)
}

/// A relative and an absolute slack around a library result, as
/// `crate::exact::range` gives one (`LIBM_REL`, about 450 ulps).
fn around(v: f64) -> (f64, f64) {
    let d = v.abs() * range::LIBM_REL + range::LIBM_ABS;
    (v - d, v + d)
}

/// The range of `f(args)`: `range::math`, and the functions it does not
/// know (added after it was written) for which "any finite number" would
/// make every bound useless.
fn math_range(f: MathFn, args: &[Range]) -> Range {
    let a = args[0];
    let increasing = |g: fn(f64) -> f64, lo_in: f64, hi_in: f64| {
        let (l, h) = (a.lo.max(lo_in), a.hi.min(hi_in));
        if l > h {
            return Range::FINITE;
        }
        Range::new(around(g(l)).0, around(g(h)).1)
    };
    let above_minus_one = next_up(-1.0);
    let below_one = next_down(1.0);
    match f {
        MathFn::Expm1 => increasing(f64::exp_m1, -INF, INF),
        MathFn::Log1p => increasing(f64::ln_1p, above_minus_one, INF),
        MathFn::Sinh => increasing(f64::sinh, -INF, INF),
        MathFn::Tanh => {
            let r = increasing(f64::tanh, -INF, INF);
            Range::new(r.lo.max(-1.0), r.hi.min(1.0))
        }
        MathFn::Asinh => increasing(f64::asinh, -INF, INF),
        MathFn::Acosh => increasing(f64::acosh, 1.0, INF),
        MathFn::Atanh => increasing(f64::atanh, above_minus_one, below_one),
        MathFn::Cosh => {
            let (lo, hi) = (mag_min(a), mag_max(a));
            Range::new(around(lo.cosh()).0.max(1.0 - 1e-13), around(hi.cosh()).1)
        }
        // The largest slope of `erf` is below 2, so no value exceeds one in
        // size, and its sign is the argument's.
        MathFn::Erf => Range::new(
            if a.lo >= 0.0 { 0.0 } else { -1.0 },
            if a.hi <= 0.0 { 0.0 } else { 1.0 },
        ),
        MathFn::Sign => Range::new(
            if a.lo >= 0.0 { 0.0 } else { -1.0 },
            if a.hi <= 0.0 { 0.0 } else { 1.0 },
        ),
        _ => range::math(f, args).range,
    }
}
