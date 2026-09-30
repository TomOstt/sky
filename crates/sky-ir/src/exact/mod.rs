//! Exact and symbolic values: an analysis, not a change of meaning
//! (`docs/design/ir.md`, "Exact and symbolic values").
//!
//! SIR means what `sky run` computes: `f64`, one rounding per operation, in
//! the order written. This module never changes that. It *analyses* a
//! function and answers two different questions about every scalar value,
//! and keeps the answers apart:
//!
//! * [`Known::ideal`], the **ideal value**: the real number the expression
//!   denotes if every operation were exact, when it has a closed form (a
//!   [`Sym`]: a rational, or a sum of a few `q·π`, `q·e`, `q·√n`). `1/3 +
//!   1/6` is exactly `1/2`; `sqrt(8)` is `2√2`; `sqrt(2) * sqrt(2)` is `2`.
//!   It is used to *say things*: `--explain` text, and whether a folded
//!   float is exactly the value the program denotes or a rounding of it.
//!   **It is never used to decide a rewrite**, because a rewrite that is
//!   right for the real number can be wrong for the float (`sqrt(2) *
//!   sqrt(2) - 2` is exactly zero and `4.4e-16` in `f64`).
//! * [`Known::range`], the **enclosure**: an interval that contains every
//!   `f64` the interpreter can produce for the value ([`Range`]). Constants
//!   are points, unknown finite values (parameters) are the whole finite
//!   range, and every operation is followed with the same rounding the
//!   interpreter uses. This is what passes may *rely on*: a value whose
//!   range excludes zero is nonzero *as an `f64`*, a division whose divisor
//!   range excludes zero and whose quotient range fits cannot fail.
//!
//! [`analyze`] computes both for a function in one pass over its operations
//! (values are SSA, so one visit each; an `if` joins its two branches). The
//! result, an [`Analysis`], is a plain table: it changes nothing, so it is
//! computed again by whoever needs it, and a stale one is never used.
//!
//! # What it enables
//!
//! * [`Analysis::proof`]: an operation that can fail by kind (`div`, `add`,
//!   `sqrt`, `log`, ...) but provably cannot with these operands. `dce`
//!   then removes it when unused, and the e-graph treats it as droppable.
//!   `x * 0.5` and `x / 2` cannot fail for any finite `x`; `sqrt(2)` and
//!   `log(3)` cannot fail; `1 / x` still can.
//! * [`Known::is_positive`], [`Known::is_nonzero`]: facts for the e-graph's
//!   conditional rules (`x / x` needs `x` nonzero, `exp(log x)` needs `x >
//!   0`).
//! * [`Known::note`]: text for `--explain`.
//!
//! # Trust
//!
//! The analysis trusts the same things the other passes trust: a value of
//! type `Number` scalar is a finite number of that dimension (a parameter's
//! declared type is what callers pass), the verifier has checked each
//! operation's types, and a value of type `Unknown` is not analysed at all
//! (no range, no proof), because its operations check their operands at run
//! time. Values with a chosen representation other than `f64` are skipped
//! too, since their rounding is not `f64`'s.
//!
//! Static shapes are trusted the same way: an array value has the shape its
//! type says (as lowering already trusts it when it writes a constant for
//! `len`). That is what lets [`Proof::InBounds`] say that `v[1]` on a
//! `Vector[3]` cannot fail, and gives `len(v)` a value: the specialization
//! pass (`sky_opt::specialize`) makes shapes concrete, and this is where a
//! concrete shape pays off.

pub mod range;
mod sym;

use crate::ir::{Function, OpId, Repr, Type, ValueId};
use crate::op::{MathFn, OpKind, SliceAxis};
use sky_types::ShapeDim;
use sky_units::Rational;

pub use range::{Range, Step};
pub use sym::{Atom, Sym, MAX_TERMS};

use crate::verify::MAX_REGION_DEPTH;

/// What is known about one scalar value.
#[derive(Clone, PartialEq, Debug)]
pub struct Known {
    /// The real number the expression denotes, if it has a closed form.
    /// Never used to justify a rewrite; see the module documentation.
    pub ideal: Option<Sym>,
    /// Every `f64` the interpreter can compute for the value.
    pub range: Range,
}

impl Known {
    /// A finite number about which nothing else is known.
    pub fn finite() -> Known {
        Known {
            ideal: None,
            range: Range::FINITE,
        }
    }

    /// Strictly greater than zero, as an `f64`.
    pub fn is_positive(&self) -> bool {
        self.range.is_positive()
    }

    /// Strictly less than zero, as an `f64`.
    pub fn is_negative(&self) -> bool {
        self.range.is_negative()
    }

    /// Not zero, as an `f64`: a divisor with this fact cannot cause a
    /// division-by-zero failure.
    pub fn is_nonzero(&self) -> bool {
        self.range.is_nonzero()
    }

    /// The `f64` is exactly the ideal value: nothing was rounded. True for
    /// `1/3 + 1/6` (`0.5`), false for `1/3` and `sqrt(2)`.
    pub fn faithful(&self) -> bool {
        let (Some(q), Some(p)) = (
            self.ideal.as_ref().and_then(Sym::as_rational),
            self.range.as_point(),
        ) else {
            return false;
        };
        dyadic(p) == Some(q)
    }

    /// Both the ideal value and the range of an `if`'s two arms: what holds
    /// whichever arm runs.
    pub fn join(&self, other: &Known) -> Known {
        Known {
            ideal: match (&self.ideal, &other.ideal) {
                (Some(a), Some(b)) if a == b => Some(a.clone()),
                _ => None,
            },
            range: self.range.hull(other.range),
        }
    }

    /// One sentence for `--explain` about what the value is over the reals
    /// and how the `f64` relates to it. `None` when there is nothing worth
    /// saying: no closed form, a form too long to read, or an integer that
    /// the float has exactly.
    pub fn note(&self) -> Option<String> {
        let ideal = self.ideal.as_ref()?;
        let shown = ideal.to_string();
        if shown.chars().count() > 30 {
            return None;
        }
        let faithful = self.faithful();
        if faithful && ideal.as_rational().is_some_and(Rational::is_integer) {
            return None;
        }
        Some(if faithful {
            format!("over the reals the result is exactly {shown}, and the f64 result is exactly that value")
        } else if let Some(p) = self.range.as_point() {
            format!(
                "over the reals the result is exactly {shown}; the f64 result {} is that value rounded (`sky run` rounds at every step)",
                range::num_text(p)
            )
        } else {
            format!("over the reals the result is exactly {shown}")
        })
    }
}

/// The `f64` `x` as an exact rational, when the numerator and denominator
/// fit an `i64`. Every finite `f64` is a rational (`0.1` is `3602879701896397 /
/// 2⁵⁵`, not `1/10`); this is the value SIR assigns to a `const.float`.
pub fn dyadic(x: f64) -> Option<Rational> {
    if !x.is_finite() {
        return None;
    }
    if x == 0.0 {
        return Some(Rational::ZERO);
    }
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    let (mut m, mut e) = if exp == 0 {
        (frac, -1074)
    } else {
        (frac | (1u64 << 52), exp - 1075)
    };
    let tz = m.trailing_zeros();
    m >>= tz;
    e += tz as i32;
    let m = i64::try_from(m).ok()?;
    let m = if bits >> 63 == 1 { -m } else { m };
    if e >= 0 {
        if e > 62 {
            return None;
        }
        let n = i64::try_from(i128::from(m) << e).ok()?;
        Rational::checked_new(n, 1).ok()
    } else if -e <= 62 {
        Rational::checked_new(m, 1i64 << -e).ok()
    } else {
        None
    }
}

/// Why an operation that can fail by kind provably cannot here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Proof {
    /// `div` or `inverse`: the divisor is never zero and the quotient fits
    /// an `f64`.
    NonzeroDivisor,
    /// `add`, `sub` or `mul`: the result fits an `f64` for every operand
    /// value the ranges allow (no overflow).
    Bounded,
    /// A power or a math function: the operands are inside its domain, and
    /// the result fits an `f64`.
    Domain,
    /// `index` or `slice`: every position is a constant integer inside the
    /// static length of its axis. `len`: the operand is a typed array.
    /// (Static shapes are trusted like static dimensions: a typed value has
    /// the shape its type says. See "Trust".)
    InBounds,
}

/// The result of [`analyze`]: a table indexed by [`ValueId`] and [`OpId`].
#[derive(Clone, Debug)]
pub struct Analysis {
    known: Vec<Option<Known>>,
    proofs: Vec<Option<Proof>>,
}

/// Analyses every scalar `f64` value of `f`. See the module documentation.
pub fn analyze(f: &Function) -> Analysis {
    analyze_with_params(f, &[])
}

/// [`analyze`], with what is known about the function's parameters:
/// `params[i]` is the enclosure of every value parameter `i` can be called
/// with (`None`, or a parameter past the end, is any finite number, as in
/// [`analyze`]). Only sound where the caller has proved it for *every* call
/// of the function (see `sky_opt::contracts`, which takes it from the calls
/// in the module).
pub fn analyze_with_params(f: &Function, params: &[Option<Range>]) -> Analysis {
    let mut a = Analysis {
        known: vec![None; f.values.len()],
        proofs: vec![None; f.ops.len()],
    };
    let entry = f.param_values();
    for (&v, range) in entry.iter().zip(params) {
        if let (Some(r), true) = (range, is_scalar_f64(f, v)) {
            a.set(
                v,
                Known {
                    ideal: None,
                    range: *r,
                },
            );
        }
    }
    a.region(f, f.body, 0);
    a
}

/// A scalar number of a type the interpreter computes in `f64`.
fn is_scalar_f64(f: &Function, v: ValueId) -> bool {
    matches!(
        f.ty(v),
        Type::Number(n) if n.shape.rank() == 0 && matches!(n.repr, Repr::Unchosen | Repr::F64)
    )
}

impl Analysis {
    /// What is known about `v`; `None` for anything that is not a scalar
    /// `f64` value of known type.
    pub fn known(&self, v: ValueId) -> Option<&Known> {
        self.known.get(v.index())?.as_ref()
    }

    /// Why `op`, which can fail by its kind, provably cannot.
    pub fn proof(&self, op: OpId) -> Option<Proof> {
        self.proofs.get(op.index()).copied().flatten()
    }

    /// The proof in words, for explanations: what the operands and the
    /// result are known to be.
    pub fn why(&self, f: &Function, id: OpId) -> Option<String> {
        let proof = self.proof(id)?;
        let op = f.op(id);
        let text = |v: Option<&ValueId>| {
            v.and_then(|v| self.known(*v))
                .map_or_else(|| "unknown".to_string(), |k| k.range.to_string())
        };
        let result = text(op.results.first());
        Some(match (proof, &op.kind) {
            (Proof::NonzeroDivisor, _) => format!(
                "the divisor is {} (never zero) and the result is {result}, which fits an f64",
                text(op.operands.last())
            ),
            (Proof::Bounded, _) => format!(
                "the operands are {} and {}, so the result is {result} and cannot overflow",
                text(op.operands.first()),
                text(op.operands.get(1))
            ),
            (Proof::Domain, OpKind::Math(m)) => format!(
                "the argument is {}, inside the domain of `{}`, and the result is {result}, which fits an f64",
                text(op.operands.first()),
                m.name()
            ),
            (Proof::Domain, _) => format!(
                "the base is {}, so the power is defined, and the result is {result}, which fits an f64",
                text(op.operands.first())
            ),
            (Proof::InBounds, OpKind::Len) => {
                "the operand is an array, so it has a length".to_string()
            }
            (Proof::InBounds, _) => {
                "every position is a constant inside the static shape of the array".to_string()
            }
        })
    }

    fn set(&mut self, v: ValueId, k: Known) {
        self.known[v.index()] = Some(k);
    }

    /// Nothing known but "finite" for every scalar result of `op`.
    fn finite_results(&mut self, f: &Function, id: OpId) {
        for r in f.op(id).results.clone() {
            if is_scalar_f64(f, r) {
                self.set(r, Known::finite());
            }
        }
    }

    fn region(&mut self, f: &Function, r: crate::ir::RegionId, depth: usize) {
        if depth > MAX_REGION_DEPTH {
            return;
        }
        for &b in &f.region(r).blocks {
            for &arg in &f.block(b).args {
                // A parameter seeded by `analyze_with_params` keeps its fact.
                if is_scalar_f64(f, arg) && self.known(arg).is_none() {
                    self.set(arg, Known::finite());
                }
            }
            for &o in &f.block(b).ops {
                self.op(f, o, depth);
            }
        }
    }

    fn op(&mut self, f: &Function, id: OpId, depth: usize) {
        let op = f.op(id);
        for &r in &op.regions {
            self.region(f, r, depth + 1);
        }
        match &op.kind {
            OpKind::If => self.join_arms(f, id),
            OpKind::While | OpKind::Call { .. } => self.finite_results(f, id),
            OpKind::Index | OpKind::Slice { .. } | OpKind::Len => {
                self.finite_results(f, id);
                self.proofs[id.index()] = self.in_bounds(f, id).then_some(Proof::InBounds);
                if let Some(k) = self.static_length(f, id) {
                    self.set(op.results[0], k);
                }
            }
            _ => match self.transfer(f, id) {
                Some((known, proof)) => {
                    self.set(op.results[0], known);
                    self.proofs[id.index()] = proof;
                }
                None => self.finite_results(f, id),
            },
        }
    }

    /// `len` of an array whose first axis has a static length is that
    /// length, exactly.
    fn static_length(&self, f: &Function, id: OpId) -> Option<Known> {
        let op = f.op(id);
        let ([base], [res]) = (&op.operands[..], &op.results[..]) else {
            return None;
        };
        if !matches!(op.kind, OpKind::Len) || !is_scalar_f64(f, *res) {
            return None;
        }
        let Type::Number(n) = f.ty(*base) else {
            return None;
        };
        let ShapeDim::Known(len) = n.shape.dims().first()? else {
            return None;
        };
        let x = *len as f64;
        Some(Known {
            ideal: dyadic(x).map(Sym::rational),
            range: Range::point(x),
        })
    }

    /// Whether `index`, `slice` or `len` provably cannot fail: the operand
    /// is a typed array, and every position is a constant, dimensionless,
    /// non-negative integer inside the static length of its axis (a range
    /// `lo:hi` needs `lo < hi <= length`).
    fn in_bounds(&self, f: &Function, id: OpId) -> bool {
        let op = f.op(id);
        let Some(&base) = op.operands.first() else {
            return false;
        };
        let Type::Number(array) = f.ty(base) else {
            return false;
        };
        if array.shape.rank() == 0 {
            return false;
        }
        if matches!(op.kind, OpKind::Len) {
            return op.operands.len() == 1;
        }
        // The value of position operand `i`, if it is a constant.
        let position = |v: ValueId| -> Option<u64> {
            let dimensionless = matches!(f.ty(v), Type::Number(n) if n.dim.is_dimensionless());
            let x = self.known(v)?.range.as_point()?;
            (dimensionless && x >= 0.0 && x.fract() == 0.0 && x < 2f64.powi(53)).then_some(x as u64)
        };
        let length = |axis: usize| match array.shape.dims().get(axis)? {
            ShapeDim::Known(n) => Some(*n),
            ShapeDim::Var(_) => None,
        };
        match &op.kind {
            OpKind::Index => {
                let idx = &op.operands[1..];
                !idx.is_empty()
                    && idx.len() <= array.shape.rank()
                    && idx.iter().enumerate().all(|(axis, &v)| {
                        matches!((position(v), length(axis)), (Some(i), Some(n)) if i < n)
                    })
            }
            OpKind::Slice { axes } => {
                if axes.len() > array.shape.rank() {
                    return false;
                }
                let mut next = 1;
                axes.iter().enumerate().all(|(axis, item)| {
                    let Some(n) = length(axis) else { return false };
                    let mut take = |present: bool| -> Option<Option<u64>> {
                        if !present {
                            return Some(None);
                        }
                        let v = *op.operands.get(next)?;
                        next += 1;
                        Some(Some(position(v)?))
                    };
                    match item {
                        SliceAxis::At => matches!(take(true), Some(Some(i)) if i < n),
                        SliceAxis::Range { start, stop } => {
                            let (Some(lo), Some(hi)) = (take(*start), take(*stop)) else {
                                return false;
                            };
                            let (lo, hi) = (lo.unwrap_or(0), hi.unwrap_or(n));
                            lo < hi && hi <= n
                        }
                    }
                })
            }
            _ => false,
        }
    }

    /// An `if`'s result is what its arms yield, whichever runs.
    fn join_arms(&mut self, f: &Function, id: OpId) {
        let op = f.op(id);
        let yields: Vec<Vec<ValueId>> = op
            .regions
            .iter()
            .filter_map(|r| {
                let block = *f.region(*r).blocks.first()?;
                let last = *f.block(block).ops.last()?;
                let last = f.op(last);
                matches!(last.kind, OpKind::Yield).then(|| last.operands.clone())
            })
            .collect();
        for (i, &r) in op.results.iter().enumerate() {
            if !is_scalar_f64(f, r) {
                continue;
            }
            let arms: Option<Vec<&Known>> = yields
                .iter()
                .map(|y| y.get(i).and_then(|v| self.known(*v)))
                .collect();
            let joined = match arms.as_deref() {
                Some([a, b]) => a.join(b),
                _ => Known::finite(),
            };
            self.set(r, joined);
        }
    }

    /// The value of a modelled operation, and the proof that it cannot
    /// fail when it can by kind. `None`: not modelled, or an operand is
    /// not known.
    fn transfer(&self, f: &Function, id: OpId) -> Option<(Known, Option<Proof>)> {
        let op = f.op(id);
        let [res] = op.results[..] else { return None };
        if !is_scalar_f64(f, res) {
            return None;
        }
        let args: Vec<&Known> = op
            .operands
            .iter()
            .map(|v| self.known(*v))
            .collect::<Option<_>>()?;
        if !dimensions_agree(f, op) {
            return None;
        }
        let ideal = |i: usize| args[i].ideal.as_ref();
        let both = |g: fn(&Sym, &Sym) -> Option<Sym>| g(ideal(0)?, ideal(1)?);
        let r = |i: usize| args[i].range;
        let (step, ideal_value, proof) = match (&op.kind, args.len()) {
            (OpKind::ConstFloat { bits }, 0) => {
                let x = f64::from_bits(*bits);
                if !x.is_finite() {
                    return None;
                }
                let k = Known {
                    ideal: dyadic(x).map(Sym::rational),
                    range: Range::point(x),
                };
                return Some((k, None));
            }
            (OpKind::ConstRational(q), 0) => {
                let x = q.numer() as f64 / q.denom() as f64;
                let k = Known {
                    ideal: Some(Sym::rational(*q)),
                    range: Range::point(x),
                };
                return Some((k, None));
            }
            (OpKind::Add, 2) => (range::add(r(0), r(1)), both(Sym::add), Proof::Bounded),
            (OpKind::Sub, 2) => (range::sub(r(0), r(1)), both(Sym::sub), Proof::Bounded),
            (OpKind::Mul, 2) => (range::mul(r(0), r(1)), both(Sym::mul), Proof::Bounded),
            (OpKind::Div, 2) => (
                range::div(r(0), r(1)),
                both(Sym::div),
                Proof::NonzeroDivisor,
            ),
            (OpKind::Neg, 1) => (
                range::neg(r(0)),
                ideal(0).and_then(Sym::neg),
                Proof::Bounded,
            ),
            (OpKind::Inverse, 1) => (
                range::recip(r(0)),
                ideal(0).and_then(|x| Sym::integer(1).div(x)),
                Proof::NonzeroDivisor,
            ),
            (OpKind::PowConst { exponent }, 1) => {
                let e = exponent.numer() as f64 / exponent.denom() as f64;
                let value = if exponent.is_integer() {
                    i32::try_from(exponent.numer())
                        .ok()
                        .and_then(|n| ideal(0)?.powi(n))
                } else if *exponent == Rational::new(1, 2) {
                    ideal(0).and_then(Sym::sqrt)
                } else {
                    None
                };
                (range::pow_const(r(0), e), value, Proof::Domain)
            }
            (OpKind::Math(m), n) if n == m.arity() => {
                let ranges: Vec<Range> = args.iter().map(|k| k.range).collect();
                let ideals: Vec<Option<&Sym>> = args.iter().map(|k| k.ideal.as_ref()).collect();
                (
                    range::math(*m, &ranges),
                    ideal_math(*m, &ideals),
                    Proof::Domain,
                )
            }
            _ => return None,
        };
        let fallible = op
            .kind
            .properties()
            .effects
            .contains(crate::ir::Effect::Fail);
        let proof = (fallible && !step.may_fail).then_some(proof);
        let known = Known {
            // An ideal value only makes sense for an operation that
            // succeeds on the ideal operands.
            ideal: ideal_value,
            range: step.range,
        };
        Some((known, proof))
    }
}

/// The operand and result dimensions are the ones the interpreter's own
/// dimension rules give, so no dynamic dimension failure (`cannot add length
/// and dimensionless`, `dimension exponent too large`) can happen. The
/// verifier guarantees it for verified IR; a proof of "cannot fail" is only
/// as good as this check, so it is repeated here instead of assumed.
fn dimensions_agree(f: &Function, op: &crate::ir::OpData) -> bool {
    let dim = |v: ValueId| f.ty(v).as_number().map(|n| n.dim);
    let Some(want) = op.results.first().and_then(|r| dim(*r)) else {
        return false;
    };
    let dims: Option<Vec<_>> = op.operands.iter().map(|v| dim(*v)).collect();
    let Some(dims) = dims else { return false };
    match (&op.kind, &dims[..]) {
        (OpKind::ConstFloat { .. } | OpKind::ConstRational(_), []) => true,
        (OpKind::Add | OpKind::Sub, [a, b]) => a == b && *a == want,
        (OpKind::Mul, [a, b]) => a.checked_mul(*b) == Ok(want),
        (OpKind::Div, [a, b]) => a.checked_div(*b) == Ok(want),
        (OpKind::Neg, [a]) => *a == want,
        (OpKind::Inverse, [a]) => a.inverse() == Ok(want),
        (OpKind::PowConst { exponent }, [a]) => {
            if a.is_dimensionless() {
                want.is_dimensionless()
            } else {
                a.pow_rat(*exponent) == Ok(want)
            }
        }
        (OpKind::Math(m), d) => m.result_dim(d) == Ok(want),
        _ => false,
    }
}

/// `f` of ideal values, at the arguments where the result has a closed
/// form. `None` everywhere else, which is always safe.
fn ideal_math(f: MathFn, a: &[Option<&Sym>]) -> Option<Sym> {
    let x = (*a.first()?)?;
    let is = |s: &Sym, n: i64| s.as_rational() == Some(Rational::integer(n));
    let half = |k: i64| Sym::pi().mul(&Sym::rational(Rational::new(k, 2)));
    let quarter = |k: i64| Sym::pi().mul(&Sym::rational(Rational::new(k, 4)));
    match f {
        MathFn::Sqrt => x.sqrt(),
        MathFn::Abs => x.abs(),
        MathFn::Exp if is(x, 0) => Some(Sym::integer(1)),
        MathFn::Exp if is(x, 1) => Some(Sym::e()),
        MathFn::Log | MathFn::Log2 | MathFn::Log10 if is(x, 1) => Some(Sym::integer(0)),
        MathFn::Log if *x == Sym::e() => Some(Sym::integer(1)),
        MathFn::Sin | MathFn::Tan | MathFn::Atan | MathFn::Asin if is(x, 0) => {
            Some(Sym::integer(0))
        }
        MathFn::Cos if is(x, 0) => Some(Sym::integer(1)),
        MathFn::Atan if is(x, 1) => quarter(1),
        MathFn::Atan if is(x, -1) => quarter(-1),
        MathFn::Asin if is(x, 1) => half(1),
        MathFn::Asin if is(x, -1) => half(-1),
        MathFn::Acos if is(x, 1) => Some(Sym::integer(0)),
        MathFn::Acos if is(x, 0) => half(1),
        MathFn::Acos if is(x, -1) => Some(Sym::pi()),
        MathFn::Floor | MathFn::Ceil | MathFn::Round => {
            let q = x.as_rational()?;
            let (n, d) = (q.numer(), q.denom());
            let k = match f {
                MathFn::Floor => n.div_euclid(d),
                MathFn::Ceil => -(-n).div_euclid(d),
                // Half away from zero.
                _ => {
                    let up = (2 * i128::from(n.abs()) + i128::from(d)) / (2 * i128::from(d));
                    i64::try_from(up).ok()? * n.signum()
                }
            };
            Some(Sym::integer(k))
        }
        MathFn::Hypot => {
            let y = (*a.get(1)?)?;
            x.mul(x)?.add(&y.mul(y)?)?.sqrt()
        }
        MathFn::Min | MathFn::Max => {
            let y = (*a.get(1)?)?;
            let order = x.sub(y)?.sign()?;
            let pick_x = if f == MathFn::Min {
                order != std::cmp::Ordering::Greater
            } else {
                order != std::cmp::Ordering::Less
            };
            Some(if pick_x { x.clone() } else { y.clone() })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests;
