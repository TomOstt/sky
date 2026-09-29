//! Typing of the built-in math functions (`sqrt`, `exp`, `sin`, ...).
//!
//! The names, arities and dimension rules are `sky_units::MathFn`, shared
//! with the interpreter; this file only turns a rule's rejection into a
//! diagnostic that teaches (what the rule is, why, and what to write
//! instead). `check.rs` calls [`Checker::math_call`] before it types a call
//! as a user function.
//!
//! * A user binding or `def` named like a builtin wins, exactly as for
//!   `print` and `inv`: the hook declines and the normal call path runs.
//! * Arrays are rejected ("elementwise math on arrays comes with arrays at
//!   run time"), since the interpreter has no array values yet and a
//!   silent scalar typing would be wrong.
//! * An argument whose type is already unknown (an earlier error, a string)
//!   makes the result unknown, so one mistake reports once.

use sky_ast::{Expr, ExprKind};
use sky_diag::{display_capped as quote, Diagnostic, Span};
use sky_units::{Dimension, MathDimError, MathFn};

use crate::check::{describe, Checker, Ty, ValueTy};

impl Checker<'_> {
    /// The type of `callee(args)` if `callee` names a builtin math function
    /// the program has not rebound; `None` leaves the call to the ordinary
    /// path.
    pub(crate) fn math_call(&mut self, callee: &Expr, args: &[Expr], span: Span) -> Option<Ty> {
        let ExprKind::Name(id) = &callee.kind else {
            return None;
        };
        let name = self.ident(id);
        let f = MathFn::from_name(name)?;
        if self.env.get(name).is_some() {
            return None;
        }
        let tys: Vec<Ty> = args.iter().map(|a| self.expr(a)).collect();
        if args.len() != f.arity() {
            self.math_arity(f, args, span);
            return Some(Ty::Unknown);
        }
        // Every argument is inspected before any rule error, so a bad
        // argument list reports its non-number arguments first.
        let mut dims = Vec::with_capacity(args.len());
        let mut bad = false;
        for (a, t) in args.iter().zip(&tys) {
            match t {
                Ty::Value(v) if v.is_scalar() => dims.push(v.dim),
                Ty::Value(v) => {
                    self.math_array(f, a, v);
                    bad = true;
                }
                Ty::Bool | Ty::Func(_) => {
                    let what = match t {
                        Ty::Bool => "Bool".to_string(),
                        other => other.to_string(),
                    };
                    self.error(
                        Diagnostic::error(
                            format!("`{}` needs a number, found {what}", f.name()),
                            a.span,
                        )
                        .with_label(format!("this is {what}")),
                    );
                    bad = true;
                }
                Ty::Unknown => bad = true,
            }
        }
        if bad {
            return Some(Ty::Unknown);
        }
        match f.result_dim(&dims) {
            Ok(d) => Some(Ty::scalar(d)),
            Err(e) => {
                self.math_dim_error(f, e, callee, args, &dims);
                Some(Ty::Unknown)
            }
        }
    }

    #[cold]
    fn math_arity(&mut self, f: MathFn, args: &[Expr], span: Span) {
        let n = f.arity();
        let plural = if n == 1 { "" } else { "s" };
        let (primary, label) = if args.len() > n {
            let extra = args[n].span;
            (
                args.last().map_or(extra, |l| extra.join(l.span)),
                "extra argument".to_string(),
            )
        } else {
            (span, "missing argument".to_string())
        };
        self.error(
            Diagnostic::error(
                format!(
                    "`{}` takes {n} argument{plural}, found {}",
                    f.name(),
                    args.len()
                ),
                primary,
            )
            .with_label(label)
            .with_help(format!("call it as {}", signature(f))),
        );
    }

    #[cold]
    fn math_array(&mut self, f: MathFn, arg: &Expr, v: &ValueTy) {
        self.error(
            Diagnostic::error(
                format!("`{}` takes a scalar, found an array", f.name()),
                arg.span,
            )
            .with_label(self.this_is(v))
            .with_note("elementwise math on arrays comes with arrays at run time")
            .with_help("apply it to one element at a time"),
        );
    }

    #[cold]
    fn math_dim_error(
        &mut self,
        f: MathFn,
        e: MathDimError,
        callee: &Expr,
        args: &[Expr],
        dims: &[Dimension],
    ) {
        let name = f.name();
        let d = match e {
            MathDimError::NotDimensionless => {
                let (arg, found) = (&args[0], dims[0]);
                let mut d = Diagnostic::error(
                    format!(
                        "the argument of `{name}` must be dimensionless, found {}",
                        describe(&found)
                    ),
                    arg.span,
                )
                .with_label(format!("this has dimension {}", describe(&found)))
                .with_note(why_dimensionless(f))
                .with_help(format!(
                    "divide by a reference {}, e.g. `{name}({})`",
                    found.name().unwrap_or("quantity"),
                    self.divided(arg, &found)
                ));
                if matches!(f, MathFn::Sin | MathFn::Cos | MathFn::Tan) {
                    d = d.with_help("angles are dimensionless: write `0.5 rad` or `30 deg`, or a plain number of radians");
                }
                d
            }
            MathDimError::Mismatch => Diagnostic::error(
                format!(
                    "the arguments of `{name}` must have the same dimension, found {} and {}",
                    describe(&dims[0]),
                    describe(&dims[1])
                ),
                callee.span,
            )
            .with_label("mismatched dimensions")
            .with_secondary(args[0].span, describe(&dims[0]))
            .with_secondary(args[1].span, describe(&dims[1]))
            .with_note(format!(
                "`{name}` compares or combines its two arguments, which only makes sense in one dimension"
            ))
            .with_help("convert one side, or fix the unit"),
            MathDimError::Overflow => Diagnostic::error(
                format!("the dimension exponent of `{name}` is too large"),
                args[0].span,
            )
            .with_label("this dimension has an exponent that overflows")
            .with_note("exponents are exact fractions of 64-bit integers"),
        };
        self.error(d);
    }

    /// `x / (1 m)` for the help: the argument's own text when it is a bare
    /// name (short and safe to echo), else `x`.
    fn divided(&self, arg: &Expr, dim: &Dimension) -> String {
        let x = match &arg.kind {
            ExprKind::Name(id) => quote(self.ident(id)).into_owned(),
            _ => "x".to_string(),
        };
        match reference_unit(dim) {
            Some(u) => format!("{x} / (1 {u})"),
            None => format!("{x} / x_ref"),
        }
    }
}

/// `sqrt(x)`, `hypot(a, b)`: the call shape for arity help.
fn signature(f: MathFn) -> String {
    match f.arity() {
        1 => format!("`{}(x)`", f.name()),
        _ => format!("`{}(a, b)`", f.name()),
    }
}

/// Why `f` refuses a dimensioned argument, for the note.
fn why_dimensionless(f: MathFn) -> &'static str {
    match f {
        MathFn::Exp => {
            "exp(x) = 1 + x + x²/2 + …, which adds terms of different dimensions unless x has none"
        }
        MathFn::Log | MathFn::Log10 | MathFn::Log2 => {
            "log(x·u) = log(x) + log(u): the result would depend on a unit that has no logarithm"
        }
        MathFn::Sin | MathFn::Cos | MathFn::Tan => {
            "trigonometric functions take an angle, and an angle is a pure number (`rad` is a dimensionless unit)"
        }
        MathFn::Asin | MathFn::Acos | MathFn::Atan => {
            "inverse trigonometric functions take a ratio, which is a pure number"
        }
        MathFn::Expm1 => "expm1(x) = exp(x) - 1, and exp needs a pure number",
        MathFn::Log1p => {
            "log1p(x) = log(1 + x) adds x to 1, which only makes sense when x is a pure number"
        }
        MathFn::Sinh | MathFn::Cosh | MathFn::Tanh => {
            "hyperbolic functions are built from exp(x) and exp(-x), which need a pure number"
        }
        MathFn::Asinh | MathFn::Acosh | MathFn::Atanh => {
            "inverse hyperbolic functions are logarithms of expressions in a pure number x, like log(x + sqrt(x² + 1))"
        }
        MathFn::Erf => {
            "erf(x) integrates exp(-t²) up to x, and exp needs a pure number, so x must be one"
        }
        _ => {
            "values are stored in SI, so rounding a quantity depends on the unit you would show it in, which Sky does not keep"
        }
    }
}

/// A unit to divide by for the help, for dimensions with an obvious one.
fn reference_unit(d: &Dimension) -> Option<&'static str> {
    const TABLE: [(Dimension, &str); 12] = [
        (Dimension::LENGTH, "m"),
        (Dimension::MASS, "kg"),
        (Dimension::TIME, "s"),
        (Dimension::CURRENT, "A"),
        (Dimension::TEMPERATURE, "K"),
        (Dimension::ENERGY, "J"),
        (Dimension::FREQUENCY, "Hz"),
        (Dimension::VELOCITY, "m/s"),
        (Dimension::FORCE, "N"),
        (Dimension::PRESSURE, "Pa"),
        (Dimension::POWER, "W"),
        (Dimension::VOLTAGE, "V"),
    ];
    TABLE.iter().find(|(t, _)| t == d).map(|&(_, u)| u)
}
