//! Uncertain numbers: `5.31 ± 0.02 kg` (spec "Uncertainty").
//!
//! # Model
//!
//! Uncertainty is a run-time property of a number, not part of its type: a
//! [`Value::Number`] is exact, and a [`Value::Uncertain`] is a number that
//! also carries an uncertainty. Units and shapes are the checker's business
//! and do not change; the uncertain value has the same dimension the exact
//! one would.
//!
//! The uncertainty is one standard uncertainty (1σ), propagated to first
//! order (linearly), with correlations tracked. The approach is that of
//! Python's `uncertainties` package (BSD-licensed; the idea, not the code):
//! an uncertain value is
//!
//! ```text
//! x = x₀ + Σᵢ cᵢ εᵢ
//! ```
//!
//! where `x₀` is the nominal value, the `εᵢ` are independent *error sources*
//! with zero mean and unit variance, and the `cᵢ` are the coefficients (in
//! the unit of `x`; a coefficient already includes the source's standard
//! deviation). Every `±` creates one fresh source. An operation `f(x, y)`
//! has the value `f(x₀, y₀)` and the coefficients
//! `∂f/∂x · cᵢ(x) + ∂f/∂y · cᵢ(y)` for every source `i`: the chain rule on
//! the linear part. This is what makes correlations come out right without
//! ever storing a covariance matrix:
//!
//! * `x - x` has no coefficient left (the sources cancel), so it is `0 ± 0`;
//! * `x * x` has coefficient `2 x₀ c` and `x * y` for independent `x`, `y`
//!   has the two independent contributions;
//! * the standard uncertainty is `σ = sqrt(Σ cᵢ²)` and the covariance of two
//!   values is `Σ cᵢ(x) cᵢ(y)`.
//!
//! A source's identity is a process-wide counter. Coefficients are kept
//! sorted by it and summed in that order, so a run gives the same bits every
//! time.
//!
//! # What this is not
//!
//! Linear propagation is exact only for linear functions, and a good
//! approximation where the uncertainty is small against the scale on which
//! the function bends: `sqrt(1 ± 0.5)` is not a meaningful `±` of a normal
//! distribution. Nothing here is Monte Carlo, and a function with an
//! infinite derivative at the nominal value (`sqrt` at 0) is a runtime error
//! rather than a made-up number.
//!
//! The math functions are in [`crate::uncertain_math`], printing in
//! [`crate::uncertain_format`]. Arrays are not uncertain yet.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use sky_ast::BinaryOp;
use sky_diag::Span;
use sky_types::a_dimension;
use sky_units::{Dimension, Rational};

use crate::arith::{arith, describe, equal, finite, order, overflow, pow, ArithOp, R};
use crate::{RuntimeError, Value};

/// The next error source's id. Only the order of ids matters.
static NEXT_SOURCE: AtomicU64 = AtomicU64::new(1);

/// One error source's contribution to a value: the coefficient `c` of the
/// unit-variance source `source`, in the unit of the value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Term {
    pub source: u64,
    pub coef: f64,
}

/// The linear part `Σ cᵢ εᵢ` of an uncertain value: its terms, sorted by
/// source, none of them zero, all finite. Shared cheaply between clones.
#[derive(Clone, Debug, PartialEq)]
pub struct Lin(Arc<[Term]>);

impl Lin {
    /// No error source: an exact value.
    pub fn exact() -> Lin {
        Lin(Arc::from(Vec::new()))
    }

    /// A fresh independent source with standard deviation `sigma` (finite,
    /// not negative). A zero sigma has no source at all.
    pub fn source(sigma: f64) -> Lin {
        if sigma == 0.0 {
            return Lin::exact();
        }
        let source = NEXT_SOURCE.fetch_add(1, Ordering::Relaxed);
        Lin(Arc::from(vec![Term {
            source,
            coef: sigma,
        }]))
    }

    pub fn terms(&self) -> &[Term] {
        &self.0
    }

    pub fn is_exact(&self) -> bool {
        self.0.is_empty()
    }

    /// `sqrt(Σ cᵢ²)`, scaled so neither the squares nor their sum overflow
    /// or underflow before the root.
    pub fn std_dev(&self) -> f64 {
        std_dev(&self.0)
    }

    /// The covariance of two values: `Σ cᵢ(x) cᵢ(y)` over their common
    /// sources.
    pub fn covariance(&self, other: &Lin) -> f64 {
        let (a, b) = (&self.0, &other.0);
        let (mut i, mut j, mut sum) = (0, 0, 0.0);
        while i < a.len() && j < b.len() {
            match a[i].source.cmp(&b[j].source) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    sum += a[i].coef * b[j].coef;
                    i += 1;
                    j += 1;
                }
            }
        }
        sum
    }

    /// `Σ wₖ · linₖ`: the linear part of `f(x₁, ...)` given the partial
    /// derivatives `wₖ = ∂f/∂xₖ` at the nominal point. A coefficient that
    /// does not fit an `f64` is an overflow error at `span`; a coefficient
    /// that cancels to exactly zero (`x - x`) is dropped.
    pub fn combine(parts: &[(&[Term], f64)], span: Span) -> R<Lin> {
        let mut all: Vec<Term> = Vec::new();
        for &(terms, w) in parts {
            if w == 0.0 {
                continue;
            }
            for t in terms {
                all.push(Term {
                    source: t.source,
                    coef: w * t.coef,
                });
            }
        }
        // Stable: two contributions to one source add in operand order.
        all.sort_by_key(|t| t.source);
        let mut out: Vec<Term> = Vec::with_capacity(all.len());
        for t in all {
            match out.last_mut() {
                Some(last) if last.source == t.source => last.coef += t.coef,
                _ => out.push(t),
            }
        }
        out.retain(|t| t.coef != 0.0);
        if out.iter().any(|t| !t.coef.is_finite()) {
            return Err(overflow(span));
        }
        Ok(Lin(Arc::from(out)))
    }
}

/// `sqrt(Σ cᵢ²)` of a slice of terms.
fn std_dev(terms: &[Term]) -> f64 {
    let scale = terms.iter().fold(0.0_f64, |m, t| m.max(t.coef.abs()));
    if scale == 0.0 {
        return 0.0;
    }
    let sum: f64 = terms.iter().map(|t| (t.coef / scale).powi(2)).sum();
    scale * sum.sqrt()
}

/// A number with an uncertainty: the nominal value in SI, its dimension,
/// and the linear part.
#[derive(Clone, Debug, PartialEq)]
pub struct Uncertain {
    pub value: f64,
    pub dim: Dimension,
    pub lin: Lin,
}

impl Uncertain {
    /// The standard uncertainty (1σ), in SI.
    pub fn std_dev(&self) -> f64 {
        self.lin.std_dev()
    }
}

impl Value {
    /// An uncertain number.
    pub fn uncertain(value: f64, dim: Dimension, lin: Lin) -> Value {
        Value::Uncertain(Uncertain { value, dim, lin })
    }

    /// Whether this is an uncertain number.
    pub fn is_uncertain(&self) -> bool {
        matches!(self, Value::Uncertain(_))
    }
}

/// A number, exact or uncertain, read through one shape.
#[derive(Clone, Copy)]
struct Num<'a> {
    value: f64,
    dim: Dimension,
    terms: &'a [Term],
}

fn num(v: &Value) -> Option<Num<'_>> {
    match v {
        Value::Number { value, dim } => Some(Num {
            value: *value,
            dim: *dim,
            terms: &[],
        }),
        Value::Uncertain(u) => Some(Num {
            value: u.value,
            dim: u.dim,
            terms: u.lin.terms(),
        }),
        _ => None,
    }
}

/// `v` without its uncertainty: what the exact-number rules see.
fn nominal(v: &Value) -> Value {
    match v {
        Value::Uncertain(u) => Value::number(u.value, u.dim),
        other => other.clone(),
    }
}

/// The exact result of an operation on nominal values, as a number.
fn exact(v: Value, whole: Span) -> R<(f64, Dimension)> {
    match v {
        Value::Number { value, dim } => Ok((value, dim)),
        other => Err(RuntimeError::new(
            format!("an uncertain number does not combine with {}", other.kind()),
            whole,
        )),
    }
}

/// The error for a derivative that is infinite where the uncertainty must
/// pass through it.
pub(crate) fn infinite_slope(what: &str, at: &str, whole: Span) -> RuntimeError {
    RuntimeError::new(
        format!("cannot propagate an uncertainty through {what} at {at}"),
        whole,
    )
    .with_label("the derivative is infinite here")
    .with_note("uncertainty propagation is linear (first order): it needs a finite slope at the nominal value")
}

/// `a op b` where at least one operand is uncertain, for `+ - * / ^` and the
/// six comparisons. The nominal result is the exact rules' (so every
/// dimension check, domain error and message is theirs); this adds the
/// linear part.
pub fn binary(
    op: BinaryOp,
    l: &Value,
    r: &Value,
    const_exp: impl FnOnce() -> Option<Rational>,
    whole: Span,
    lspan: Span,
    rspan: Span,
) -> R<Value> {
    let (ln, rn) = (nominal(l), nominal(r));
    match op {
        BinaryOp::Eq | BinaryOp::Ne => {
            let eq = equal(&ln, &rn, whole, lspan, rspan)?;
            Ok(Value::Bool(eq == (op == BinaryOp::Eq)))
        }
        BinaryOp::Lt => order("<", |a, b| a < b, &ln, &rn, whole, lspan, rspan).map(Value::Bool),
        BinaryOp::Le => order("<=", |a, b| a <= b, &ln, &rn, whole, lspan, rspan).map(Value::Bool),
        BinaryOp::Gt => order(">", |a, b| a > b, &ln, &rn, whole, lspan, rspan).map(Value::Bool),
        BinaryOp::Ge => order(">=", |a, b| a >= b, &ln, &rn, whole, lspan, rspan).map(Value::Bool),
        BinaryOp::MatMul => Err(RuntimeError::new(
            format!("cannot multiply matrices of {} and {}", l.kind(), r.kind()),
            whole,
        )),
        BinaryOp::Pow => {
            let result = pow(&ln, &rn, const_exp, whole, lspan, rspan)?;
            let (value, dim) = exact(result, whole)?;
            let (Some(a), Some(b)) = (num(l), num(r)) else {
                return Err(RuntimeError::new("cannot raise these values", whole));
            };
            // d(a^b)/da = b a^(b-1); d(a^b)/db = a^b ln a.
            let wa = if a.terms.is_empty() {
                0.0
            } else {
                pow_base_slope(a.value, b.value, whole)?
            };
            let wb = if b.terms.is_empty() {
                0.0
            } else {
                pow_exponent_slope(a.value, value, whole)?
            };
            let lin = Lin::combine(&[(a.terms, wa), (b.terms, wb)], whole)?;
            Ok(Value::uncertain(value, dim, lin))
        }
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
            let aop = match op {
                BinaryOp::Add => ArithOp::Add,
                BinaryOp::Sub => ArithOp::Sub,
                BinaryOp::Mul => ArithOp::Mul,
                _ => ArithOp::Div,
            };
            let result = arith(aop, &ln, &rn, whole, lspan, rspan)?;
            let (value, dim) = exact(result, whole)?;
            let (Some(a), Some(b)) = (num(l), num(r)) else {
                return Err(RuntimeError::new(
                    format!(
                        "cannot apply `{}` to {} and {}",
                        op.symbol(),
                        l.kind(),
                        r.kind()
                    ),
                    whole,
                ));
            };
            let (wa, wb) = match aop {
                ArithOp::Add => (1.0, 1.0),
                ArithOp::Sub => (1.0, -1.0),
                ArithOp::Mul => (b.value, a.value),
                // a/b: ∂/∂a = 1/b, ∂/∂b = -a/b² = -(a/b)/b. `b` is not zero
                // (the exact division succeeded).
                ArithOp::Div => (
                    finite(1.0 / b.value, whole)?,
                    finite(-(value / b.value), whole)?,
                ),
            };
            let lin = Lin::combine(&[(a.terms, wa), (b.terms, wb)], whole)?;
            Ok(Value::uncertain(value, dim, lin))
        }
    }
}

/// `∂(b^x)/∂b = x b^(x-1)`.
fn pow_base_slope(b: f64, x: f64, whole: Span) -> R<f64> {
    if x == 0.0 {
        return Ok(0.0);
    }
    if b == 0.0 {
        return if x > 1.0 {
            Ok(0.0)
        } else if x == 1.0 {
            Ok(1.0)
        } else {
            Err(infinite_slope("`^`", "a zero base", whole))
        };
    }
    finite(
        x * crate::arith::pow_scalar(b, x - 1.0, whole, whole)?,
        whole,
    )
}

/// `∂(b^x)/∂x = b^x ln b`, which needs a positive base.
fn pow_exponent_slope(b: f64, result: f64, whole: Span) -> R<f64> {
    if b <= 0.0 {
        return Err(
            RuntimeError::new("the exponent's uncertainty needs a positive base", whole)
                .with_label("`ln` of the base is not a real number")
                .with_note("the derivative of `b^x` with respect to `x` is `b^x ln b`"),
        );
    }
    finite(result * b.ln(), whole)
}

/// `-v` or `+v` of an uncertain number.
pub fn sign(negate: bool, u: &Uncertain, whole: Span) -> R<Value> {
    if !negate {
        return Ok(Value::Uncertain(u.clone()));
    }
    let lin = Lin::combine(&[(u.lin.terms(), -1.0)], whole)?;
    Ok(Value::uncertain(-u.value + 0.0, u.dim, lin))
}

/// `1/x` of an uncertain number (`x⁻¹`, `inv(x)`): `-1/x²`.
pub fn reciprocal(u: &Uncertain, whole: Span, operand: Span) -> R<Value> {
    let nom = crate::arith::reciprocal("inv", &Value::number(u.value, u.dim), whole, operand)?;
    let (value, dim) = exact(nom, whole)?;
    let slope = finite(-(value / u.value), whole)?;
    let lin = Lin::combine(&[(u.lin.terms(), slope)], whole)?;
    Ok(Value::uncertain(value, dim, lin))
}

/// `value ± sigma`: `value` (exact or itself uncertain) with an additional
/// independent standard uncertainty of `sigma`, which must be an exact
/// number of the same dimension and not negative. `whole`, `vspan` and
/// `sspan` are the spans of the whole expression and of its operands.
pub fn plus_minus(value: &Value, sigma: &Value, whole: Span, vspan: Span, sspan: Span) -> R<Value> {
    let Some(v) = num(value) else {
        return Err(RuntimeError::new(
            format!("`±` needs a number on its left, found {}", value.kind()),
            vspan,
        ));
    };
    let s = match sigma {
        Value::Number { value, dim } => (*value, *dim),
        Value::Uncertain(_) => {
            return Err(RuntimeError::new(
                "the uncertainty after `±` must be an exact number",
                sspan,
            )
            .with_label("this number is itself uncertain")
            .with_help("write its nominal value: `nominal(...)`"))
        }
        other => {
            return Err(RuntimeError::new(
                format!("`±` needs a number on its right, found {}", other.kind()),
                sspan,
            ))
        }
    };
    if s.1 != v.dim {
        return Err(RuntimeError::new(
            format!(
                "cannot use {} as the uncertainty of {}",
                a_dimension(&s.1),
                a_dimension(&v.dim)
            ),
            whole,
        )
        .with_label("mismatched dimensions")
        .with_secondary(vspan, describe(&v.dim))
        .with_secondary(sspan, describe(&s.1)));
    }
    if s.0 < 0.0 {
        return Err(
            RuntimeError::new("an uncertainty cannot be negative", sspan)
                .with_label("this is negative")
                .with_note("`±` takes a standard deviation, which is never negative")
                .with_help("write its magnitude, e.g. `x ± 0.1`"),
        );
    }
    let fresh = Lin::source(s.0);
    let lin = Lin::combine(&[(v.terms, 1.0), (fresh.terms(), 1.0)], whole)?;
    Ok(Value::uncertain(v.value, v.dim, lin))
}

/// `nominal(x)` and `uncertainty(x)`: the two parts of a number, exact or
/// not. `uncertainty` of an exact number is zero.
pub fn part(name: &str, v: &Value, call: Span, arg: Span) -> R<Value> {
    let Some(n) = num(v) else {
        return Err(
            RuntimeError::new(format!("`{name}` needs a number, found {}", v.kind()), arg)
                .with_label("not a number")
                .with_note("arrays cannot carry an uncertainty yet")
                .with_secondary(call, format!("in this call to `{name}`")),
        );
    };
    Ok(match name {
        "nominal" => Value::number(n.value, n.dim),
        _ => Value::number(std_dev(n.terms), n.dim),
    })
}

#[cfg(test)]
#[path = "uncertain_tests.rs"]
mod tests;
