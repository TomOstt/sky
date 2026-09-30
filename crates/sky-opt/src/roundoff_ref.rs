//! A test-only high-precision reference for the roundoff analysis: a small
//! double-double (`hi + lo`, about 32 digits) evaluator for the scalar
//! operations of SIR.
//!
//! It is deliberately written separately from `sky_ir::roundoff::hp`, the
//! production enclosure (different algorithms where a choice exists: the
//! QD library's), and it computes the elementary functions too, so it also
//! measures the math library the analysis assumes a few ulps of. It returns
//! `None` where it does not model a case (out of domain, too large an
//! argument for its range reduction): the property test then skips the value.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dd {
    pub hi: f64,
    pub lo: f64,
}

pub fn dd(x: f64) -> Dd {
    Dd { hi: x, lo: 0.0 }
}

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

impl Dd {
    pub fn neg(self) -> Dd {
        Dd {
            hi: -self.hi,
            lo: -self.lo,
        }
    }

    pub fn abs(self) -> Dd {
        if self.hi < 0.0 {
            self.neg()
        } else {
            self
        }
    }

    pub fn add(self, o: Dd) -> Dd {
        let (s1, mut s2) = two_sum(self.hi, o.hi);
        let (t1, t2) = two_sum(self.lo, o.lo);
        s2 += t1;
        let (s1, mut s2) = quick_two_sum(s1, s2);
        s2 += t2;
        let (hi, lo) = quick_two_sum(s1, s2);
        Dd { hi, lo }
    }

    pub fn sub(self, o: Dd) -> Dd {
        self.add(o.neg())
    }

    pub fn mul(self, o: Dd) -> Dd {
        let (p1, mut p2) = two_prod(self.hi, o.hi);
        p2 += self.hi * o.lo + self.lo * o.hi;
        let (hi, lo) = quick_two_sum(p1, p2);
        Dd { hi, lo }
    }

    pub fn mul_f(self, f: f64) -> Dd {
        self.mul(dd(f))
    }

    pub fn div(self, o: Dd) -> Option<Dd> {
        if o.hi == 0.0 {
            return None;
        }
        let q1 = self.hi / o.hi;
        let r = self.sub(o.mul_f(q1));
        let q2 = r.hi / o.hi;
        let r = r.sub(o.mul_f(q2));
        let q3 = r.hi / o.hi;
        let (a, b) = quick_two_sum(q1, q2);
        Some(Dd { hi: a, lo: b }.add(dd(q3)))
    }

    pub fn sqrt(self) -> Option<Dd> {
        if self.hi < 0.0 {
            return None;
        }
        if self.hi == 0.0 {
            return Some(dd(0.0));
        }
        let x = 1.0 / self.hi.sqrt();
        let ax = self.hi * x;
        let d = self.sub(dd(ax).mul(dd(ax)));
        let corr = d.hi * (x * 0.5);
        Some(dd(ax).add(dd(corr)))
    }

    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    pub fn powi(self, n: i32) -> Option<Dd> {
        let mut acc = dd(1.0);
        for _ in 0..n.unsigned_abs() {
            acc = acc.mul(self);
        }
        if n < 0 {
            dd(1.0).div(acc)
        } else {
            Some(acc)
        }
    }

    fn floor(self) -> Dd {
        let f = self.hi.floor();
        if f == self.hi {
            let g = self.lo.floor();
            let (hi, lo) = quick_two_sum(f, g);
            Dd { hi, lo }
        } else {
            dd(f)
        }
    }
}

const LN2: Dd = Dd {
    hi: 6.931_471_805_599_453e-1,
    lo: 2.319_046_813_846_299_6e-17,
};
const LN10: Dd = Dd {
    hi: std::f64::consts::LN_10,
    lo: -2.170_756_223_382_249_4e-16,
};
const PI: Dd = Dd {
    hi: std::f64::consts::PI,
    lo: 1.224_646_799_147_353_2e-16,
};

fn ldexp(x: Dd, k: i32) -> Dd {
    let s = 2.0f64.powi(k);
    Dd {
        hi: x.hi * s,
        lo: x.lo * s,
    }
}

pub fn exp(x: Dd) -> Option<Dd> {
    if x.hi.abs() > 700.0 {
        return None;
    }
    let k = (x.hi / LN2.hi).round();
    let r = x.sub(LN2.mul_f(k));
    // r / 512: a power of two, exact.
    let r = ldexp(r, -9);
    let mut term = dd(1.0);
    let mut sum = dd(1.0);
    for n in 1..14 {
        term = term.mul(r).div(dd(f64::from(n)))?;
        sum = sum.add(term);
    }
    for _ in 0..9 {
        sum = sum.mul(sum);
    }
    Some(ldexp(sum, k as i32))
}

pub fn ln(x: Dd) -> Option<Dd> {
    if x.hi <= 0.0 {
        return None;
    }
    let mut y = dd(x.hi.ln());
    for _ in 0..2 {
        let e = exp(y)?;
        // Halley: y += 2 (x - e) / (x + e)
        y = y.add(x.sub(e).mul_f(2.0).div(x.add(e))?);
    }
    Some(y)
}

/// `(sin x, cos x)` for `|x| <= 1000`.
pub fn sin_cos(x: Dd) -> Option<(Dd, Dd)> {
    if x.hi.abs() > 1000.0 {
        return None;
    }
    let half_pi = ldexp(PI, -1);
    let k = (x.hi / half_pi.hi).round();
    let r = x.sub(half_pi.mul_f(k));
    let r2 = r.mul(r);
    // sin and cos series, |r| <= pi/4.
    let (mut s, mut c) = (r, dd(1.0));
    let (mut ts, mut tc) = (r, dd(1.0));
    for n in 1..20 {
        let m = f64::from(2 * n);
        tc = tc.mul(r2).div(dd(m * (m - 1.0)))?.neg();
        ts = ts.mul(r2).div(dd(m * (m + 1.0)))?.neg();
        c = c.add(tc);
        s = s.add(ts);
    }
    Some(match (k as i64).rem_euclid(4) {
        0 => (s, c),
        1 => (c, s.neg()),
        2 => (s.neg(), c.neg()),
        _ => (c.neg(), s),
    })
}

fn atan(x: Dd) -> Option<Dd> {
    let mut y = dd(x.hi.atan());
    for _ in 0..2 {
        let (s, c) = sin_cos(y)?;
        // Newton on sin y - x cos y.
        let f = s.sub(x.mul(c));
        let fp = c.add(x.mul(s));
        y = y.sub(f.div(fp)?);
    }
    Some(y)
}

pub fn atan2(y: Dd, x: Dd) -> Option<Dd> {
    if x.hi == 0.0 {
        return match y.hi {
            v if v > 0.0 => Some(ldexp(PI, -1)),
            v if v < 0.0 => Some(ldexp(PI, -1).neg()),
            _ => None,
        };
    }
    let a = atan(y.div(x)?)?;
    Some(if x.hi > 0.0 {
        a
    } else if y.hi >= 0.0 {
        a.add(PI)
    } else {
        a.sub(PI)
    })
}

fn cbrt(x: Dd) -> Option<Dd> {
    if x.hi == 0.0 {
        return Some(dd(0.0));
    }
    let neg = x.hi < 0.0;
    let a = x.abs();
    let mut y = dd(a.hi.cbrt());
    for _ in 0..2 {
        let f = y.mul(y).mul(y).sub(a);
        y = y.sub(f.div(y.mul(y).mul_f(3.0))?);
    }
    Some(if neg { y.neg() } else { y })
}

/// `f(args)` for an SIR math function, or `None` where the reference does
/// not model it.
pub fn math(f: sky_ir::MathFn, a: &[Dd]) -> Option<Dd> {
    use sky_ir::MathFn as M;
    let x = a[0];
    Some(match f {
        M::Sqrt => x.sqrt()?,
        M::Cbrt => cbrt(x)?,
        M::Abs => x.abs(),
        M::Exp => exp(x)?,
        M::Log => ln(x)?,
        M::Log2 => ln(x)?.div(LN2)?,
        M::Log10 => ln(x)?.div(LN10)?,
        M::Sin => sin_cos(x)?.0,
        M::Cos => sin_cos(x)?.1,
        M::Tan => {
            let (s, c) = sin_cos(x)?;
            s.div(c)?
        }
        M::Atan => atan(x)?,
        M::Atan2 => atan2(x, a[1])?,
        M::Asin => {
            if x.abs().hi > 1.0 {
                return None;
            }
            if x.abs().hi == 1.0 && x.abs().lo == 0.0 {
                let h = ldexp(PI, -1);
                return Some(if x.hi < 0.0 { h.neg() } else { h });
            }
            atan(x.div(dd(1.0).sub(x.mul(x)).sqrt()?)?)?
        }
        M::Acos => {
            let h = ldexp(PI, -1);
            h.sub(math(M::Asin, &[x])?)
        }
        M::Hypot => x.mul(x).add(a[1].mul(a[1])).sqrt()?,
        M::Min => {
            if x.hi < a[1].hi || (x.hi == a[1].hi && x.lo <= a[1].lo) {
                x
            } else {
                a[1]
            }
        }
        M::Max => {
            if x.hi > a[1].hi || (x.hi == a[1].hi && x.lo >= a[1].lo) {
                x
            } else {
                a[1]
            }
        }
        M::Floor => x.floor(),
        M::Ceil => x.neg().floor().neg(),
        // Half away from zero.
        M::Round => {
            let h = dd(0.5);
            if x.hi >= 0.0 {
                x.add(h).floor()
            } else {
                x.neg().add(h).floor().neg()
            }
        }
        M::Expm1 => {
            if x.hi.abs() < 1e-12 {
                return None;
            }
            exp(x)?.sub(dd(1.0))
        }
        M::Log1p => {
            if x.hi.abs() < 1e-12 {
                return None;
            }
            ln(dd(1.0).add(x))?
        }
        M::Sinh => {
            if x.hi.abs() < 1e-8 {
                return None;
            }
            let e = exp(x)?;
            e.sub(dd(1.0).div(e)?).mul_f(0.5)
        }
        M::Cosh => {
            let e = exp(x)?;
            e.add(dd(1.0).div(e)?).mul_f(0.5)
        }
        M::Tanh => {
            if x.hi.abs() < 1e-8 || x.hi.abs() > 300.0 {
                return None;
            }
            let e = exp(x.mul_f(2.0))?;
            e.sub(dd(1.0)).div(e.add(dd(1.0)))?
        }
        M::Asinh => {
            if x.hi.abs() < 1e-8 {
                return None;
            }
            let a = x.abs();
            let r = ln(a.add(a.mul(a).add(dd(1.0)).sqrt()?))?;
            if x.hi < 0.0 {
                r.neg()
            } else {
                r
            }
        }
        M::Acosh => {
            if x.hi < 1.0 + 1e-8 {
                return None;
            }
            ln(x.add(x.mul(x).sub(dd(1.0)).sqrt()?))?
        }
        M::Atanh => {
            if x.hi.abs() >= 1.0 || x.hi.abs() < 1e-8 {
                return None;
            }
            ln(dd(1.0).add(x).div(dd(1.0).sub(x))?)?.mul_f(0.5)
        }
        // Sky's own `erf` has no independent reference here.
        M::Erf => return None,
        M::Sign => dd(if x.hi == 0.0 && x.lo == 0.0 {
            0.0
        } else if x.hi > 0.0 {
            1.0
        } else {
            -1.0
        }),
    })
}
