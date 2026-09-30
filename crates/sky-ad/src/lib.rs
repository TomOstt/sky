//! Automatic differentiation of SIR, as a program transformation (roadmap
//! idea 32; `docs/design/ad.md`).
//!
//! [`derivative`] turns a scalar function `f(x, ...)` into a new SIR
//! function `df/dx` with the same parameters that returns
//! `(f(x, ...), ∂f/∂x)`. It is forward mode: every value `v` of `f` gets a
//! *tangent* `dv/dx` computed next to it (a dual number, written out as
//! ordinary SIR operations), so the result is plain SIR that the verifier,
//! the optimizer and the interpreter treat like any other function.
//!
//! # Units
//!
//! The derivative has the dimension `dim(f) / dim(x)`, so the derivative of
//! a position with respect to time is a velocity. This is not a check
//! bolted on afterwards: the tangent of every value `v` is typed
//! `dim(v) / dim(x)`, and each rule produces operands whose dimensions
//! multiply out to that type (the derivative of `a * b` is
//! `ta * b + a * tb`, and both products have dimension
//! `dim(a) * dim(b) / dim(x)`).
//!
//! # What it accepts
//!
//! * Functions of any number of parameters; `wrt` picks one. The other
//!   parameters may be numbers (constants for this derivative) or anything
//!   else (a Bool flag, say), and are passed through.
//! * One or more results, each a number of any shape. With one result `r`
//!   the derivative function returns `(r, ∂r/∂x)`; with several it returns
//!   the values, then their tangents (`(r0, r1, ∂r0/∂x, ∂r1/∂x)`), which
//!   is what lets a derivative function be differentiated again (a second
//!   derivative is the derivative of a function that calls `df/dx`). The
//!   tangent of an array is an array of the same shape with dimension
//!   `dim(value)/dim(x)`: differentiating a position vector by time gives a
//!   velocity vector.
//! * Every operation of SIR v1, arrays included (`array.literal`, `index`,
//!   `slice`, `zeros`, `identity`, `len`, elementwise arithmetic and
//!   scaling, `matmul`, `transpose`, `inverse`, `solve`, `fused`), `if`,
//!   `while` and `call`. The rules are in `rules.rs`; the math functions
//!   are scalar in SIR and have no array form.
//!
//! # Array parameters: gradients
//!
//! `wrt` may also be an *array* parameter `p` (a `Vector` or `Matrix`),
//! for a function with one scalar result `F`: the derivative is then the
//! gradient `∂F/∂p`, an array of `p`'s shape and of dimension
//! `dim(F)/dim(p)` (a potential of a position gives a force). Two engines
//! compute it, behind the one function `d<f>/d<p>(params...) -> (F, ∂F/∂p)`:
//!
//! * **reverse mode** (`reverse.rs`): one forward and one backward sweep,
//!   however large `p` is. It handles straight-line code, `if` and calls.
//! * **forward mode, one pass per entry** (`gradient.rs`): the fallback for
//!   a loop that carries a value the gradient depends on, which reverse mode
//!   would need a tape for. It costs `n` evaluations for `n` entries and
//!   needs `p`'s size to be known and at most 256.
//!
//! [`derivative_functions`] tries reverse mode and falls back.
//!
//! Calls that pass or return arrays are inlined first (`inline.rs`), so a
//! helper `displacement(p)` is part of the function and its derivative
//! flows through it; calls of scalar functions stay calls.

//! # `Unknown` types
//!
//! A value of type `Unknown` has no dimension, so its tangent cannot be
//! typed. We reject rather than guess: AD returns [`AdError::UnknownType`]
//! if the parameter differentiated against, or any value the derivative
//! flows through, is `Unknown`. `Unknown` values the derivative does not
//! touch (a flag, a constant) are copied unchanged. (Treating them
//! dynamically would give a derivative with `Unknown` results that hides
//! unit errors, the thing this pass exists to prevent.)
//!
//! # Control flow
//!
//! * `if`: both arms are differentiated and each yields its tangents; the
//!   condition's derivative is zero and does not flow (a derivative is taken
//!   at a point, and at a discontinuity the taken arm's derivative is
//!   returned).
//! * `while`: a tangent is carried next to each loop-carried value that
//!   depends on `x` (found by a fixpoint activity analysis), through the
//!   `before` block, the `condition`, the `after` block and the results.
//! * `call`: the callee is differentiated too, into `d<callee>/d<param>`; a
//!   recursive function's derivative calls itself, which is what makes
//!   recursion work. Derivatives are memoized by name, so each is built
//!   once however often it is called. A call with several parameters that
//!   depend on `x` calls one derivative per such parameter and sums
//!   (`∂f/∂a·ta + ∂f/∂b·tb`).
//!
//! # Deliverables
//!
//! Callee derivatives are module functions, so the function [`derivative`]
//! returns calls functions it does not contain when the original has calls.
//! [`derivative_functions`] returns all of them, and [`differentiate_into`]
//! adds them to a [`Module`].

#![forbid(unsafe_code)]

mod analysis;
mod arrays;
mod emit;
mod gradient;
mod inline;
mod math_rules;
mod reverse;
mod rules;

#[cfg(test)]
mod tests;

use std::collections::{HashSet, VecDeque};
use std::fmt;

use sky_ir::{Function, Module, Type};

/// Why a function could not be differentiated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdError {
    /// The module has no function of that name.
    NoSuchFunction(String),
    /// `wrt` is not a parameter index of the function.
    ParamOutOfRange {
        func: String,
        wrt: usize,
        params: usize,
    },
    /// The parameter differentiated against is not a number.
    NonNumberParam { func: String, wrt: usize },
    /// A value the derivative needs has type `Unknown` (see the crate docs).
    UnknownType { func: String, what: String },
    /// A value the derivative needs is an array; v1 AD is for scalars.
    NonScalar { func: String, what: String },
    /// The function does not return exactly one scalar number.
    BadResult { func: String, why: String },
    /// An operation the derivative flows through has no rule (yet).
    Unsupported {
        func: String,
        op: String,
        why: String,
    },
    /// A dimension computation overflowed.
    Dimension { func: String, why: String },
    /// The IR breaks a rule the verifier would have caught.
    Malformed { func: String, why: String },
}

impl fmt::Display for AdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdError::NoSuchFunction(n) => write!(f, "no function `{n}` to differentiate"),
            AdError::ParamOutOfRange { func, wrt, params } => write!(
                f,
                "cannot differentiate `{func}` with respect to parameter {wrt}: it has {params}"
            ),
            AdError::NonNumberParam { func, wrt } => write!(
                f,
                "cannot differentiate `{func}` with respect to parameter {wrt}: it is not a number"
            ),
            AdError::UnknownType { func, what } => write!(
                f,
                "cannot differentiate `{func}`: {what} has an unknown type, so its derivative \
                 has no unit"
            ),
            AdError::NonScalar { func, what } => write!(
                f,
                "cannot differentiate `{func}`: {what} is not a scalar (array AD is not \
                 supported yet)"
            ),
            AdError::BadResult { func, why } | AdError::Dimension { func, why } => {
                write!(f, "cannot differentiate `{func}`: {why}")
            }
            AdError::Unsupported { func, op, why } => {
                write!(f, "cannot differentiate `{op}` in `{func}`: {why}")
            }
            AdError::Malformed { func, why } => {
                write!(f, "cannot differentiate `{func}`: malformed IR: {why}")
            }
        }
    }
}

impl std::error::Error for AdError {}

/// The name of the derivative of `f` with respect to parameter `wrt`:
/// `df/dx`, using the parameter's source name, or its index when it has
/// none (`df/d0`). The slash keeps it from clashing with a Sky identifier.
pub fn derivative_name(f: &Function, wrt: usize) -> String {
    let param = f
        .param_values()
        .get(wrt)
        .and_then(|v| f.value(*v).name.clone())
        .unwrap_or_else(|| wrt.to_string());
    derivative_name_of(&f.name, &param)
}

/// [`derivative_name`] for a function called `func` whose parameter is
/// called `param` (write its index if it has no name), for a caller that
/// has to name the derivative before it is built.
pub fn derivative_name_of(func: &str, param: &str) -> String {
    format!("d{func}/d{param}")
}

/// What is built for one differentiation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    /// A scalar parameter, by forward mode: `d<f>/d<x>`.
    Forward,
    /// An array parameter in one direction (a Jacobian-vector product):
    /// `d<f>/d<p>#jvp`, with the direction as an extra last parameter.
    Jvp,
    /// An array parameter, the gradient `d<f>/d<p>`, by reverse mode or by
    /// one [`Kind::Jvp`] per entry.
    Gradient,
}

/// A derivative function to build: `func` with respect to parameter `wrt`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Task {
    pub func: String,
    pub wrt: usize,
    pub kind: Kind,
}

/// The name of the function of `kind` for `f` and `wrt`.
pub(crate) fn derivative_name_for(f: &Function, wrt: usize, kind: Kind) -> String {
    match kind {
        Kind::Jvp => format!("{}#jvp", derivative_name(f, wrt)),
        Kind::Forward | Kind::Gradient => derivative_name(f, wrt),
    }
}

/// What differentiating `f` with respect to its parameter `wrt` builds: the
/// gradient for an array parameter, else the (forward) derivative.
fn kind_of(f: &Function, wrt: usize) -> Kind {
    let array = f
        .param_values()
        .get(wrt)
        .is_some_and(|p| matches!(f.ty(*p), Type::Number(n) if n.shape.rank() > 0));
    if array {
        Kind::Gradient
    } else {
        Kind::Forward
    }
}

/// The derivative of `func` with respect to parameter `wrt`, as a new
/// function `d<func>/d<param>` with `func`'s parameters that returns
/// `(value, derivative)`.
///
/// If `func` calls other functions, the returned function calls their
/// derivatives, which [`derivative_functions`] builds.
pub fn derivative(m: &Module, func: &str, wrt: usize) -> Result<Function, AdError> {
    let mut all = derivative_functions(m, func, wrt)?;
    Ok(all.swap_remove(0))
}

/// The derivative of `func` and every derivative function it needs
/// (callees, and itself if recursive), main one first. A derivative
/// function of a callee already in `m` (by name) is reused, not rebuilt.
pub fn derivative_functions(m: &Module, func: &str, wrt: usize) -> Result<Vec<Function>, AdError> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let first = m
        .function(func)
        .ok_or_else(|| AdError::NoSuchFunction(func.to_string()))?;
    let mut queue = VecDeque::from([Task {
        func: func.to_string(),
        wrt,
        kind: kind_of(first, wrt),
    }]);
    while let Some(task) = queue.pop_front() {
        let f = m
            .function(&task.func)
            .ok_or_else(|| AdError::NoSuchFunction(task.func.clone()))?;
        let dname = derivative_name_for(f, task.wrt, task.kind);
        // Memoization; for recursion it is also the cycle check.
        if !seen.insert(dname.clone()) {
            continue;
        }
        if !out.is_empty() && m.function(&dname).is_some() {
            continue;
        }
        let (d, needs) = build_task(m, f, &task)?;
        out.push(d);
        queue.extend(needs);
    }
    Ok(out)
}

/// Builds the function of one task.
fn build_task(m: &Module, f: &Function, task: &Task) -> Result<(Function, Vec<Task>), AdError> {
    match task.kind {
        Kind::Forward => emit::build(m, f, task.wrt, emit::Seed::Unit),
        Kind::Jvp => emit::build(m, f, task.wrt, emit::Seed::Direction),
        Kind::Gradient => match reverse::build(m, f, task.wrt) {
            Err(AdError::Unsupported {
                op: reverse_op,
                why: reverse_why,
                ..
            }) => gradient::build(m, f, task.wrt).map_err(|e| match e {
                AdError::Unsupported { func, op, why } => AdError::Unsupported {
                    func,
                    op,
                    why: format!(
                        "{why}; reverse mode does not apply either (`{reverse_op}`: {reverse_why})"
                    ),
                },
                other => other,
            }),
            other => other,
        },
    }
}

/// Adds the derivative of `func` (and what it needs) to `m` and returns
/// its name. Idempotent.
pub fn differentiate_into(m: &mut Module, func: &str, wrt: usize) -> Result<String, AdError> {
    let f = m
        .function(func)
        .ok_or_else(|| AdError::NoSuchFunction(func.to_string()))?;
    let name = derivative_name(f, wrt);
    if m.function(&name).is_none() {
        let all = derivative_functions(m, func, wrt)?;
        m.functions.extend(all);
    }
    // The derivatives are new functions: their effects (and those of any
    // caller) follow from their bodies, and one pass owns that rule.
    let _ = sky_ir::effects::infer_effects(m);
    Ok(name)
}

/// Whether a type can carry a tangent (a number of any shape).
pub(crate) fn can_carry_tangent(t: &Type) -> bool {
    matches!(t, Type::Number(_) | Type::Unknown)
}
