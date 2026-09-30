//! The `extern "C"` functions a compiled kernel calls.
//!
//! A kernel's arithmetic is plain IEEE machine code, but a math function
//! (`exp`, `sin`, `atan2`, ...) is a call back into Rust so that it is the
//! same function, with the same rounding and the same domain rules, as in
//! the interpreters: [`math`] is `sky_interp::math::apply`.

use sky_diag::Span;
use sky_interp::Value;
use sky_units::MathFn;

/// The index of `f` in [`MathFn::ALL`], which is how a kernel names the
/// function it calls.
pub fn math_index(f: MathFn) -> u32 {
    MathFn::ALL
        .iter()
        .position(|g| *g == f)
        .map_or(0, |i| i as u32)
}

/// `f(x)` or `f(x, y)` for `f = MathFn::ALL[index]`, computed by
/// `sky_interp::math::apply` on dimensionless numbers. `y` is ignored by a
/// one-argument function.
///
/// The value of a call does not depend on the arguments' dimensions (they
/// only decide whether the call is allowed and what dimension it has, which
/// the kernel's compiler settled), so the number is the interpreters'. Where
/// `apply` reports an error (a domain error, an overflow) this returns NaN:
/// a successful call never returns a non-finite number, so the kernel's
/// finiteness check sees the failure and the caller re-runs the elements
/// through the interpreter path to build the exact diagnostic.
pub extern "C" fn math(index: u32, x: f64, y: f64) -> f64 {
    let Some(f) = MathFn::ALL.get(index as usize).copied() else {
        return f64::NAN;
    };
    let args = [Value::scalar(x), Value::scalar(y)];
    let spans = [Span::new(0, 0); 2];
    match sky_interp::math::apply(f, &args[..f.arity()], Span::new(0, 0), &spans) {
        Ok(Value::Number { value, .. }) => value,
        _ => f64::NAN,
    }
}
