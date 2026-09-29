//! Value-level rules shared by every executor of Sky.
//!
//! The tree-walking interpreter and the SIR reference interpreter
//! (`sky-ir-interp`) must agree on every number, every error message and
//! every limit, so the rules that don't depend on how a program is
//! represented live here once: the finite checks, `powi`/`powf` selection,
//! the reciprocal, dynamic operand checks, and the resource-limit errors.
//! Callers pass the spans to blame: a tree walker has one per operand, an IR
//! op has only its own and passes it for all three.

use sky_diag::Span;
use sky_units::{Dimension, Rational};

use crate::{RuntimeError, Value};

/// The result type of every fallible evaluation step.
pub type R<T> = Result<T, RuntimeError>;

/// Integer exponents up to this size use `powi` (repeated multiplication,
/// exact whenever the true result is representable); larger or fractional
/// ones use `powf`.
pub const POWI_MAX: f64 = 64.0;

/// Most user-function calls active at once, like Python's recursion limit:
/// a friendly error for runaway recursion.
pub const MAX_CALL_DEPTH: usize = 1000;

/// How much of the stack a run may use before it stops with an error.
/// Executors recurse on the Rust stack, and how much each level costs
/// depends on the build (a debug build needs several KiB per nested
/// expression), so a fixed depth count can't be both safe and generous.
/// Instead a [`StackMeter`] measures how far the stack has grown since the
/// run started. The budget leaves 1 MiB of `sky_ast::DEEP_STACK_BYTES` (a
/// run must happen under `with_deep_stack`) for the frames between two
/// checks and for printing.
pub const STACK_BUDGET: usize = sky_ast::DEEP_STACK_BYTES - (1 << 20);

/// The address of a local: where the stack is now. Only ever compared with
/// another such address, never dereferenced.
#[inline(never)]
fn stack_position() -> usize {
    let marker = 0u8;
    std::hint::black_box(&marker) as *const u8 as usize
}

/// Measures stack growth since it was created; see [`STACK_BUDGET`].
#[derive(Clone, Copy, Debug)]
pub struct StackMeter {
    base: usize,
}

impl Default for StackMeter {
    fn default() -> Self {
        StackMeter::new()
    }
}

impl StackMeter {
    /// Starts measuring from the current stack position.
    pub fn new() -> Self {
        StackMeter {
            base: stack_position(),
        }
    }

    /// An error if the run has used up [`STACK_BUDGET`]: deep recursion
    /// combined with deeply nested code ends here, cleanly, rather than in a
    /// stack overflow.
    pub fn check(&self, span: Span) -> R<()> {
        // Direction-agnostic: the stack grows down on every target Sky
        // supports, but `abs_diff` doesn't need to know.
        if self.base.abs_diff(stack_position()) > STACK_BUDGET {
            return Err(RuntimeError::new(
                "the program nests too deeply for the interpreter",
                span,
            )
            .with_label("stopped here")
            .with_note("`sky run` stops before it runs out of stack")
            .with_help("reduce the recursion depth, or use a `while` loop"));
        }
        Ok(())
    }
}

/// The error for calling a user function [`MAX_CALL_DEPTH`] calls deep.
pub fn recursion_too_deep(callee: &str, span: Span) -> RuntimeError {
    RuntimeError::new(
        format!("recursion is too deep: more than {MAX_CALL_DEPTH} nested calls"),
        span,
    )
    .with_label(format!("calling `{callee}` here"))
    .with_note("`sky run` limits nested calls instead of overflowing the stack")
    .with_help("check the recursion's base case, or use a `while` loop")
}

/// The error for a call with the wrong number of arguments.
pub fn arity_mismatch(name: &str, expected: usize, found: usize, span: Span) -> RuntimeError {
    RuntimeError::new(
        format!(
            "`{name}` takes {expected} argument{}, found {found}",
            if expected == 1 { "" } else { "s" },
        ),
        span,
    )
}

pub fn arrays_unsupported(span: Span) -> RuntimeError {
    RuntimeError::new("arrays are not supported by the interpreter yet", span)
        .with_label("this needs an array")
        .with_note("the checker accepts array types in v0.1, but `sky run` cannot execute them yet")
}

/// `length`, or `L·M` for a dimension without a name (as the checker says it).
pub fn describe(d: &Dimension) -> String {
    match d.name() {
        Some(n) => n.to_string(),
        None => d.to_string(),
    }
}

pub fn overflow(span: Span) -> RuntimeError {
    RuntimeError::new("arithmetic overflow", span)
        .with_label("the result does not fit in a 64-bit float")
        .with_note("Sky numbers never become inf or NaN; the largest is about 1.8e308")
}

/// `x` if it is finite. Every operand is finite, so a non-finite result of
/// `+ - * / ^` is an overflow.
pub fn finite(x: f64, span: Span) -> R<f64> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(overflow(span))
    }
}

pub fn div_zero(whole: Span, zero: Span) -> RuntimeError {
    RuntimeError::new("division by zero", whole)
        .with_label("this divides by zero")
        .with_secondary(zero, "this is zero")
        .with_note("Sky reports division by zero instead of returning inf or NaN")
}

pub fn cannot_compare(l: &Value, r: &Value, whole: Span, lspan: Span, rspan: Span) -> RuntimeError {
    let (a, b) = match (l, r) {
        (Value::Number { dim: da, .. }, Value::Number { dim: db, .. }) => {
            (describe(da), describe(db))
        }
        _ => (l.kind().to_string(), r.kind().to_string()),
    };
    RuntimeError::new(format!("cannot compare {a} and {b}"), whole)
        .with_secondary(lspan, a)
        .with_secondary(rspan, b)
}

/// The Bool inside `v`, for the operand of `and`, `or`, `not`, `if` or
/// `while` (`kw` names which). v0.1 has no truthiness.
pub fn expect_bool(kw: &str, v: &Value, span: Span) -> R<bool> {
    match v {
        Value::Bool(b) => Ok(*b),
        other => Err(RuntimeError::new(
            format!("`{kw}` needs a Bool operand, found {}", other.kind()),
            span,
        )
        .with_label("not a Bool")
        .with_note("v0.1 has no truthiness")),
    }
}

/// `-v` or `+v`.
pub fn sign(negate: bool, v: Value, operand: Span) -> R<Value> {
    match v {
        Value::Array { dims, data, dim } => Ok(if negate {
            crate::array::negate(&dims, &data, dim)
        } else {
            Value::array(dims, data, dim)
        }),
        Value::Number { value, dim } => Ok(Value::number(if negate { -value } else { value }, dim)),
        other => Err(RuntimeError::new(
            format!(
                "`{}` needs a number, found {}",
                if negate { "-" } else { "+" },
                other.kind()
            ),
            operand,
        )),
    }
}

/// `+ - * /`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// `l op r` on two evaluated operands. `whole` is the span of the
/// expression, `lspan` and `rspan` those of its operands.
pub fn arith(op: ArithOp, l: &Value, r: &Value, whole: Span, lspan: Span, rspan: Span) -> R<Value> {
    if matches!(l, Value::Array { .. }) || matches!(r, Value::Array { .. }) {
        return crate::array::arith(op, l, r, whole, lspan, rspan);
    }
    match op {
        ArithOp::Add | ArithOp::Sub => {
            let (verb, prep) = if op == ArithOp::Add {
                ("add", "and")
            } else {
                ("subtract", "from")
            };
            let (Value::Number { value: a, dim: da }, Value::Number { value: b, dim: db }) = (l, r)
            else {
                // `subtract b from a`: name the right operand first.
                let (x, y) = if op == ArithOp::Add { (l, r) } else { (r, l) };
                return Err(RuntimeError::new(
                    format!("cannot {verb} {} {prep} {}", x.kind(), y.kind()),
                    whole,
                ));
            };
            if da != db {
                let message = if op == ArithOp::Add {
                    format!("cannot add {} and {}", describe(da), describe(db))
                } else {
                    format!("cannot subtract {} from {}", describe(db), describe(da))
                };
                return Err(RuntimeError::new(message, whole)
                    .with_label("mismatched dimensions")
                    .with_secondary(lspan, describe(da))
                    .with_secondary(rspan, describe(db)));
            }
            let x = if op == ArithOp::Add { a + b } else { a - b };
            Ok(Value::number(finite(x, whole)?, *da))
        }
        ArithOp::Mul | ArithOp::Div => {
            let (Value::Number { value: a, dim: da }, Value::Number { value: b, dim: db }) = (l, r)
            else {
                return Err(RuntimeError::new(
                    format!(
                        "cannot {} {} and {}",
                        if op == ArithOp::Mul {
                            "multiply"
                        } else {
                            "divide"
                        },
                        l.kind(),
                        r.kind()
                    ),
                    whole,
                ));
            };
            let dim = if op == ArithOp::Mul {
                da.checked_mul(*db)
            } else {
                da.checked_div(*db)
            }
            .map_err(|_| RuntimeError::new("dimension exponent too large", whole))?;
            if op == ArithOp::Div {
                if *b == 0.0 {
                    return Err(div_zero(whole, rspan));
                }
                return Ok(Value::number(finite(a / b, whole)?, dim));
            }
            Ok(Value::number(finite(a * b, whole)?, dim))
        }
    }
}

/// `l == r` (negated by the caller for `!=`): numbers of one dimension, two
/// Bools, or two strings.
pub fn equal(l: &Value, r: &Value, whole: Span, lspan: Span, rspan: Span) -> R<bool> {
    match (l, r) {
        (Value::Number { value: a, dim: da }, Value::Number { value: b, dim: db }) => {
            if da != db {
                return Err(cannot_compare(l, r, whole, lspan, rspan));
            }
            Ok(a == b)
        }
        (Value::Bool(a), Value::Bool(b)) => Ok(a == b),
        (Value::Str(a), Value::Str(b)) => Ok(a == b),
        _ => Err(cannot_compare(l, r, whole, lspan, rspan)),
    }
}

/// `l < r` and friends: `symbol` is the operator's source text and `holds`
/// decides the comparison. Only numbers of one dimension order.
pub fn order(
    symbol: &str,
    holds: fn(f64, f64) -> bool,
    l: &Value,
    r: &Value,
    whole: Span,
    lspan: Span,
    rspan: Span,
) -> R<bool> {
    let (Value::Number { value: a, dim: da }, Value::Number { value: b, dim: db }) = (l, r) else {
        return Err(RuntimeError::new(
            format!("cannot order {} and {} with `{symbol}`", l.kind(), r.kind()),
            whole,
        ));
    };
    if da != db {
        return Err(cannot_compare(l, r, whole, lspan, rspan));
    }
    Ok(holds(*a, *b))
}

/// `base ^ exp`. The result's dimension comes from the constant rational
/// exponent, exactly as the checker computes it: `const_exp` supplies it
/// (from the source, or from the IR's attribute) and is only asked for when
/// the base has a unit.
pub fn pow(
    base: &Value,
    exp: &Value,
    const_exp: impl FnOnce() -> Option<Rational>,
    whole: Span,
    lspan: Span,
    rspan: Span,
) -> R<Value> {
    let (Value::Number { value: b, dim: db }, Value::Number { value: x, dim: dx }) = (base, exp)
    else {
        return Err(RuntimeError::new(
            format!("cannot raise {} to {}", base.kind(), exp.kind()),
            whole,
        ));
    };
    if !dx.is_dimensionless() {
        return Err(
            RuntimeError::new("the exponent must be dimensionless", rspan)
                .with_label(format!("this is {}", describe(dx))),
        );
    }
    let dim = if db.is_dimensionless() {
        Dimension::DIMENSIONLESS
    } else {
        let Some(r) = const_exp() else {
            return Err(RuntimeError::new(
                "the exponent must be a constant number when the base has a unit",
                rspan,
            )
            .with_secondary(lspan, format!("this is {}", describe(db))));
        };
        db.pow_rat(r)
            .map_err(|_| RuntimeError::new("dimension exponent too large", whole))?
    };
    Ok(Value::number(pow_scalar(*b, *x, whole, lspan)?, dim))
}

/// `b ^ x` on finite `f64`s: division by zero for `0 ^ negative`, an error
/// for a non-real result, `powi` for small integer exponents and `powf`
/// otherwise, and an overflow error if the result is not finite.
pub fn pow_scalar(b: f64, x: f64, whole: Span, base_span: Span) -> R<f64> {
    if b == 0.0 && x < 0.0 {
        return Err(div_zero(whole, base_span));
    }
    if b < 0.0 && x.fract() != 0.0 {
        return Err(RuntimeError::new("the result is not a real number", whole)
            .with_label("a negative base with a fractional exponent")
            .with_note("Sky numbers are real; it never produces NaN or a complex number"));
    }
    let value = if x.fract() == 0.0 && x.abs() <= POWI_MAX {
        b.powi(x as i32)
    } else {
        b.powf(x)
    };
    finite(value, whole)
}

/// `1/x` with the inverse dimension, or the matrix inverse for an array;
/// `what` names the spelling (`⁻¹` or
/// `inv`) in the error for a non-number.
pub fn reciprocal(what: &str, v: &Value, whole: Span, operand: Span) -> R<Value> {
    // A matrix has an inverse, not a reciprocal.
    if matches!(v, Value::Array { .. }) {
        return crate::array::inverse(v, whole, operand);
    }
    let Value::Number { value, dim } = v else {
        return Err(RuntimeError::new(
            format!("`{what}` needs a number, found {}", v.kind()),
            operand,
        ));
    };
    if *value == 0.0 {
        return Err(div_zero(whole, operand));
    }
    let dim = dim
        .inverse()
        .map_err(|_| RuntimeError::new("dimension exponent too large", whole))?;
    Ok(Value::number(finite(1.0 / value, whole)?, dim))
}

/// A declaration without a value (`A: Matrix[3, 3]`) reached at run time:
/// an input nothing supplies in v0.1. Shared by both interpreters (the SIR
/// one sees such inputs as parameters of `<main>`).
pub fn declared_without_value(name: &str, span: Span) -> RuntimeError {
    RuntimeError::new(format!("`{name}` is declared but has no value"), span)
        .with_label("an input with nothing to read it from")
        .with_note("a declaration without a value only gives the checker a type")
        .with_help("give it a value: `name: type = value`")
}
