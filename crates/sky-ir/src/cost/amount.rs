//! The bounded quantities the cost model adds up.
//!
//! An [`Amount`] is one measure (flops, say) as an interval of polynomials in
//! the sizes: a lower bound `lo` and an upper bound `hi` that hold for every
//! run, over all sizes at least 1, plus the reasons the upper bound does not
//! exist when it does not. For straight-line code the two are equal (the
//! amount is *exact*); an `if` whose branches cost differently, or a loop
//! whose trip count is only bounded, opens the interval; a loop with no
//! known bound makes `hi` unbounded, and says where.
//!
//! Every operation here is total. A polynomial that would overflow the
//! representation gives `lo = 0` (always a valid lower bound: costs are not
//! negative) and an unbounded `hi` with the reason [`ReasonKind::TooLarge`].

use std::collections::BTreeMap;

use sky_diag::Span;
use sky_types::poly::Poly;

/// The most reasons an amount keeps.
const MAX_REASONS: usize = 8;

/// Why an upper bound does not exist.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ReasonKind {
    /// A loop whose trip count the model cannot bound.
    Loop,
    /// A call that closes a cycle in the call graph.
    Recursion { callee: String },
    /// An operation on a value whose shape the checker could not state
    /// (an unannotated parameter).
    UnknownShape,
    /// An array whose size is only known at run time (`zeros(n)`).
    UnknownSize,
    /// A size of a callee that its arguments do not fix.
    FreeSize { name: String },
    /// The cost is too large to represent.
    TooLarge,
}

/// Where and why a cost is unbounded.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Reason {
    /// The function the source `span` is in.
    pub function: String,
    pub span: Span,
    pub kind: ReasonKind,
}

/// Where a computation happens, for the reasons it may produce.
#[derive(Clone, Copy)]
pub struct At<'a> {
    pub function: &'a str,
    pub span: Span,
}

impl At<'_> {
    pub fn reason(&self, kind: ReasonKind) -> Reason {
        Reason {
            function: self.function.to_string(),
            span: self.span,
            kind,
        }
    }
}

/// How many times a loop body runs: a lower and an upper bound, as
/// polynomials in the sizes (equal when the count is exact).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Trips {
    pub lo: Poly,
    pub hi: Poly,
}

impl Trips {
    pub fn exact(p: Poly) -> Trips {
        Trips {
            lo: p.clone(),
            hi: p,
        }
    }
}

/// One measure: `lo <= actual <= hi` for every run and all sizes >= 1. `lo`
/// is non-negative for all sizes >= 1 (an invariant the operations keep, so
/// that a product of lower bounds is one).
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Amount {
    pub lo: Poly,
    pub hi: Poly,
    /// Non-empty means there is no upper bound (`hi` is then meaningless).
    pub unbounded: Vec<Reason>,
}

fn push_reasons(into: &mut Vec<Reason>, from: &[Reason]) {
    for r in from {
        if into.len() >= MAX_REASONS {
            return;
        }
        if !into.contains(r) {
            into.push(r.clone());
        }
    }
}

impl Amount {
    pub fn zero() -> Amount {
        Amount::default()
    }

    /// Exactly `p`.
    pub fn exact(p: Poly) -> Amount {
        Amount {
            lo: p.clone(),
            hi: p,
            unbounded: Vec::new(),
        }
    }

    /// No upper bound, for `reason`; the lower bound is 0.
    pub fn unbounded(reason: Reason) -> Amount {
        Amount {
            lo: Poly::zero(),
            hi: Poly::zero(),
            unbounded: vec![reason],
        }
    }

    /// Whether it is exactly zero.
    pub fn is_zero(&self) -> bool {
        self.unbounded.is_empty() && self.hi.is_zero() && self.lo.is_zero()
    }

    pub fn is_unbounded(&self) -> bool {
        !self.unbounded.is_empty()
    }

    /// The polynomial, if the amount is exactly known.
    pub fn exact_poly(&self) -> Option<&Poly> {
        (self.unbounded.is_empty() && self.lo == self.hi).then_some(&self.hi)
    }

    /// The amount with `hi` made unbounded for `kind`, when a polynomial
    /// operation overflowed.
    fn too_large(mut self, at: At<'_>) -> Amount {
        self.lo = Poly::zero();
        push_reasons(&mut self.unbounded, &[at.reason(ReasonKind::TooLarge)]);
        self
    }

    pub fn add(&self, other: &Amount, at: At<'_>) -> Amount {
        let mut unbounded = self.unbounded.clone();
        push_reasons(&mut unbounded, &other.unbounded);
        let hi = self.hi.checked_add(&other.hi);
        let lo = self.lo.checked_add(&other.lo);
        match (lo, hi) {
            (Some(lo), Some(hi)) => Amount { lo, hi, unbounded },
            _ => Amount {
                lo: Poly::zero(),
                hi: Poly::zero(),
                unbounded,
            }
            .too_large(at),
        }
    }

    /// The amount repeated `trips` times (a loop body).
    pub fn times(&self, trips: &Trips, at: At<'_>) -> Amount {
        // A zero body costs nothing however often it runs.
        if self.is_zero() {
            return Amount::zero();
        }
        let lo = self.lo.checked_mul(&trips.lo);
        let hi = self.hi.checked_mul(&trips.hi);
        match (lo, hi) {
            (Some(lo), Some(hi)) => Amount {
                lo,
                hi,
                unbounded: self.unbounded.clone(),
            },
            _ => Amount {
                lo: Poly::zero(),
                hi: Poly::zero(),
                unbounded: self.unbounded.clone(),
            }
            .too_large(at),
        }
    }

    /// The amount repeated a number of times nothing bounds. A zero body
    /// stays zero; anything else is unbounded, for `reason`, above the
    /// least it costs (zero runs).
    pub fn times_unbounded(&self, reason: Reason) -> Amount {
        if self.is_zero() {
            return Amount::zero();
        }
        let mut unbounded = self.unbounded.clone();
        push_reasons(&mut unbounded, &[reason]);
        Amount {
            lo: Poly::zero(),
            hi: Poly::zero(),
            unbounded,
        }
    }

    /// Whichever of two alternatives runs (the branches of an `if`): the
    /// least of the lower bounds and the greatest of the upper.
    pub fn join(&self, other: &Amount) -> Amount {
        let mut unbounded = self.unbounded.clone();
        push_reasons(&mut unbounded, &other.unbounded);
        // A lower bound must stay non-negative for every size (a product of
        // two of them is a lower bound only then), which the coefficient-wise
        // minimum need not be: zero is always a lower bound.
        let lo = self.lo.lower_join(&other.lo);
        Amount {
            lo: if lo.nonnegative_for_all_sizes() {
                lo
            } else {
                Poly::zero()
            },
            hi: self.hi.upper_join(&other.hi),
            unbounded,
        }
    }

    /// The amount in another function's sizes: each size in `map` becomes
    /// its polynomial. A size that `map` does not give (a size of the callee
    /// its arguments do not fix) makes the upper bound unbounded.
    pub fn subst(&self, map: &BTreeMap<String, Poly>, at: At<'_>) -> Amount {
        let mut unbounded = self.unbounded.clone();
        let free =
            |p: &Poly| -> Option<String> { p.vars().into_iter().find(|v| !map.contains_key(v)) };
        let lo_ok = free(&self.lo).is_none();
        let lo = if lo_ok {
            self.lo.subst(map).unwrap_or_default()
        } else {
            Poly::zero()
        };
        let hi = match free(&self.hi) {
            Some(name) => {
                push_reasons(&mut unbounded, &[at.reason(ReasonKind::FreeSize { name })]);
                Poly::zero()
            }
            None => match self.hi.subst(map) {
                Some(h) => h,
                None => {
                    push_reasons(&mut unbounded, &[at.reason(ReasonKind::TooLarge)]);
                    Poly::zero()
                }
            },
        };
        Amount { lo, hi, unbounded }
    }
}
