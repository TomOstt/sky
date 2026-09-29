//! The built-in math functions and their dimension rules: the single source
//! of truth for the checker (`sky-types`), the interpreter (`sky-interp`) and
//! any other executor, so they cannot disagree on which names exist, how
//! many arguments each takes, or what dimension comes out.
//!
//! Only dimensions live here (this crate is pure and knows no diagnostics or
//! values). Evaluation is `sky_interp::math`; the wording of errors is the
//! caller's, driven by [`MathDimError`].
//!
//! # Why the rules are what they are
//!
//! * `sqrt`/`cbrt` halve/third every exponent (`m²` gives `m`), which is
//!   exact because exponents are rationals. `sqrt(1 m)` is `m^(1/2)`.
//! * `exp`, `log*`, trig and inverse trig need a dimensionless argument: the
//!   Taylor series of `exp` adds `x`, `x²`, ..., which have different
//!   dimensions unless `x` has none, and `log(3 m)` has no unit to be in
//!   (`log(3 m) = log 3 + log m`). `rad` is dimensionless in this table, so
//!   `sin(30 rad)` works and `omega * t` (Hz times s) is a valid argument.
//! * `floor`/`ceil`/`round` are dimensionless-only: values are stored in SI,
//!   so `round(1.5 km)` would round 1500 m to 1500 m while a person reading
//!   it in km expects 2. The result depends on the display unit, which Sky
//!   does not store, so it is ill-defined and rejected. Divide by the unit
//!   you mean: `round(x / (1 km))`.

use crate::dimension::{DimError, Dimension};
use crate::rational::Rational;

/// A built-in math function.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MathFn {
    Sqrt,
    Cbrt,
    Abs,
    Exp,
    Log,
    Log10,
    Log2,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Hypot,
    Min,
    Max,
    Floor,
    Ceil,
    Round,
}

/// How a function's result dimension follows from its arguments'.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DimRule {
    /// One argument; every exponent is divided by `n` (`sqrt` 2, `cbrt` 3).
    Root(i64),
    /// One argument of any dimension, kept (`abs`).
    Keep,
    /// One dimensionless argument, dimensionless result.
    Dimensionless,
    /// Two arguments of one shared dimension, which is the result's
    /// (`hypot`, `min`, `max`).
    SameKeep,
    /// Two arguments of one shared dimension, dimensionless result
    /// (`atan2`).
    SameRatio,
}

/// Why a rule rejected its arguments.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MathDimError {
    /// The (only) argument must be dimensionless. Carries no dimension: the
    /// caller has the arguments' dimensions, and a `Dimension` is large
    /// enough to make every `Result` of this error expensive to move.
    NotDimensionless,
    /// The two arguments must share a dimension.
    Mismatch,
    /// A root of the dimension needs an exponent that overflows.
    Overflow,
}

impl MathFn {
    /// Every function, in the order the spec lists them.
    pub const ALL: [MathFn; 20] = [
        MathFn::Sqrt,
        MathFn::Cbrt,
        MathFn::Abs,
        MathFn::Exp,
        MathFn::Log,
        MathFn::Log10,
        MathFn::Log2,
        MathFn::Sin,
        MathFn::Cos,
        MathFn::Tan,
        MathFn::Asin,
        MathFn::Acos,
        MathFn::Atan,
        MathFn::Atan2,
        MathFn::Hypot,
        MathFn::Min,
        MathFn::Max,
        MathFn::Floor,
        MathFn::Ceil,
        MathFn::Round,
    ];

    /// The name as written in Sky.
    pub const fn name(self) -> &'static str {
        match self {
            MathFn::Sqrt => "sqrt",
            MathFn::Cbrt => "cbrt",
            MathFn::Abs => "abs",
            MathFn::Exp => "exp",
            MathFn::Log => "log",
            MathFn::Log10 => "log10",
            MathFn::Log2 => "log2",
            MathFn::Sin => "sin",
            MathFn::Cos => "cos",
            MathFn::Tan => "tan",
            MathFn::Asin => "asin",
            MathFn::Acos => "acos",
            MathFn::Atan => "atan",
            MathFn::Atan2 => "atan2",
            MathFn::Hypot => "hypot",
            MathFn::Min => "min",
            MathFn::Max => "max",
            MathFn::Floor => "floor",
            MathFn::Ceil => "ceil",
            MathFn::Round => "round",
        }
    }

    /// The function called `name` (case-sensitive).
    pub fn from_name(name: &str) -> Option<MathFn> {
        MathFn::ALL.into_iter().find(|f| f.name() == name)
    }

    /// The dimension rule.
    pub const fn rule(self) -> DimRule {
        match self {
            MathFn::Sqrt => DimRule::Root(2),
            MathFn::Cbrt => DimRule::Root(3),
            MathFn::Abs => DimRule::Keep,
            MathFn::Hypot | MathFn::Min | MathFn::Max => DimRule::SameKeep,
            MathFn::Atan2 => DimRule::SameRatio,
            MathFn::Exp
            | MathFn::Log
            | MathFn::Log10
            | MathFn::Log2
            | MathFn::Sin
            | MathFn::Cos
            | MathFn::Tan
            | MathFn::Asin
            | MathFn::Acos
            | MathFn::Atan
            | MathFn::Floor
            | MathFn::Ceil
            | MathFn::Round => DimRule::Dimensionless,
        }
    }

    /// How many arguments the function takes.
    pub const fn arity(self) -> usize {
        match self.rule() {
            DimRule::SameKeep | DimRule::SameRatio => 2,
            DimRule::Root(_) | DimRule::Keep | DimRule::Dimensionless => 1,
        }
    }

    /// The result dimension for arguments of dimensions `args` (exactly
    /// [`MathFn::arity`] of them; a wrong count is the caller's arity error
    /// and yields the dimensionless result here rather than panicking).
    pub fn result_dim(self, args: &[Dimension]) -> Result<Dimension, MathDimError> {
        let dimensionless = Dimension::DIMENSIONLESS;
        match (self.rule(), args) {
            (DimRule::Root(n), [d]) => d
                .pow_rat(Rational::new(1, n))
                .map_err(|_: DimError| MathDimError::Overflow),
            (DimRule::Keep, [d]) => Ok(*d),
            (DimRule::Dimensionless, [d]) if d.is_dimensionless() => Ok(dimensionless),
            (DimRule::Dimensionless, [_]) => Err(MathDimError::NotDimensionless),
            (DimRule::SameKeep | DimRule::SameRatio, [a, b]) if a != b => {
                Err(MathDimError::Mismatch)
            }
            (DimRule::SameKeep, [a, _]) => Ok(*a),
            (DimRule::SameRatio, [_, _]) => Ok(dimensionless),
            _ => Ok(dimensionless),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip_and_are_unique() {
        for f in MathFn::ALL {
            assert_eq!(MathFn::from_name(f.name()), Some(f));
        }
        assert_eq!(MathFn::from_name("Sqrt"), None);
        assert_eq!(MathFn::from_name("print"), None);
    }

    #[test]
    fn rules() {
        let l2 = Dimension::from_ints([2, 0, 0, 0, 0, 0, 0]);
        assert_eq!(MathFn::Sqrt.result_dim(&[l2]), Ok(Dimension::LENGTH));
        assert_eq!(
            MathFn::Cbrt.result_dim(&[Dimension::VOLUME]),
            Ok(Dimension::LENGTH)
        );
        assert_eq!(MathFn::Abs.result_dim(&[l2]), Ok(l2));
        assert_eq!(
            MathFn::Exp.result_dim(&[Dimension::LENGTH]),
            Err(MathDimError::NotDimensionless)
        );
        assert_eq!(
            MathFn::Hypot.result_dim(&[Dimension::LENGTH, Dimension::TIME]),
            Err(MathDimError::Mismatch)
        );
        assert_eq!(
            MathFn::Atan2.result_dim(&[Dimension::LENGTH, Dimension::LENGTH]),
            Ok(Dimension::DIMENSIONLESS)
        );
        assert_eq!(
            MathFn::Max.result_dim(&[Dimension::TIME, Dimension::TIME]),
            Ok(Dimension::TIME)
        );
    }
}
