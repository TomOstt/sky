//! The tangent rule of every scalar operation, in one `match`.
//!
//! Notation: for the operation `r = op(a, b)`, `ta` and `tb` are the
//! tangents of `a` and `b` (`d/dx`), and the rule computes `tr`. A `None`
//! tangent is a structural zero, and the rules skip the work it implies
//! (`ta * b + a * tb` with no `tb` is just `ta * b`). Every emitted value
//! is typed `dim(·) / dim(x)` of what it stands for; the comment on each
//! rule shows why the operand dimensions multiply out.
//!
//! | operation | tangent `tr` |
//! |---|---|
//! | `a + b` | `ta + tb` |
//! | `a - b` | `ta - tb` |
//! | `-a` | `-ta` |
//! | `a * b` | `ta*b + a*tb` |
//! | `a / b` | `(ta - r*tb) / b` |
//! | `a ^ e` (constant `e`) | `(e * a^(e-1)) * ta` |
//! | `a ^ e` (computed `e`) | `(e * a^(e-1)) * ta + (r * ln a) * te` |
//! | `a⁻¹` | `-(r*r) * ta` |
//! | comparisons, `not`, constants | none (Bool or constant) |
//!
//! # Math functions
//!
//! `OpKind::Math(f)` has one arm here and its rules in `math_rules.rs`. For
//! `r = f(a)` with tangent `ta`:
//!
//! | `f` | `tr` |
//! |---|---|
//! | `sqrt` | `ta / (2r)` |
//! | `cbrt` | `ta / (3r²)` |
//! | `abs` | `sign(a) * ta` (`sign(0) = 1`) |
//! | `exp` | `r * ta` |
//! | `log`, `log10`, `log2` | `ta / a`, `ta / (a ln 10)`, `ta / (a ln 2)` |
//! | `sin`, `cos` | `cos(a) * ta`, `-sin(a) * ta` |
//! | `tan` | `(1 + r²) * ta` |
//! | `asin`, `acos` | `ta / sqrt(1 - a²)`, its negative |
//! | `atan` | `ta / (1 + a²)` |
//! | `atan2(y, x)` | `(x*ty - y*tx) / (x² + y²)` |
//! | `hypot(a, b)` | `(a*ta + b*tb) / r` |
//! | `min`, `max` | the tangent of the operand chosen (first on a tie) |
//! | `floor`, `ceil`, `round` | none (zero) |
//!
//! The general power `a ^ e` adds `r * ln(a) * te` when the exponent has a
//! tangent (`ln` needs `a > 0`, and `ln(a)` is dimensionless because a
//! computed exponent needs a dimensionless base).

use sky_ir::{OpData, OpKind, ValueId};
use sky_units::Rational;

use crate::emit::{number_type, Diff, Res};

impl Diff<'_> {
    /// The tangent of the (single) result `r` of `op`, given that at least
    /// one operand has a tangent; `None` if it is structurally zero.
    pub(crate) fn tangent_rule(&mut self, op: &OpData, r: ValueId) -> Res<Option<ValueId>> {
        let span = op.span;
        match &op.kind {
            OpKind::Add => {
                let [a, b] = self.operands(op)?;
                let ty = self.tangent_ty(r)?;
                Ok(match (self.tan_of(a), self.tan_of(b)) {
                    (Some(x), Some(y)) => Some(self.emit(OpKind::Add, vec![x, y], ty, span)),
                    (Some(t), None) | (None, Some(t)) => Some(t),
                    (None, None) => None,
                })
            }
            OpKind::Sub => {
                let [a, b] = self.operands(op)?;
                let ty = self.tangent_ty(r)?;
                Ok(match (self.tan_of(a), self.tan_of(b)) {
                    (Some(x), Some(y)) => Some(self.emit(OpKind::Sub, vec![x, y], ty, span)),
                    (Some(t), None) => Some(t),
                    (None, Some(y)) => Some(self.emit(OpKind::Neg, vec![y], ty, span)),
                    (None, None) => None,
                })
            }
            OpKind::Neg => {
                let [a] = self.operands(op)?;
                let ty = self.tangent_ty(r)?;
                Ok(self
                    .tan_of(a)
                    .map(|t| self.emit(OpKind::Neg, vec![t], ty, span)))
            }
            OpKind::Mul => {
                // ta*b and a*tb both have dim(a)*dim(b)/dim(x) = dim(r)/dim(x).
                let [a, b] = self.operands(op)?;
                let ty = self.tangent_ty(r)?;
                let (pa, pb) = (self.primal(a)?, self.primal(b)?);
                let left = self
                    .tan_of(a)
                    .map(|ta| self.emit(OpKind::Mul, vec![ta, pb], ty.clone(), span));
                let right = self
                    .tan_of(b)
                    .map(|tb| self.emit(OpKind::Mul, vec![pa, tb], ty.clone(), span));
                Ok(match (left, right) {
                    (Some(l), Some(r)) => Some(self.emit(OpKind::Add, vec![l, r], ty, span)),
                    (Some(t), None) | (None, Some(t)) => Some(t),
                    (None, None) => None,
                })
            }
            OpKind::Div => {
                // d(a/b) = ta/b - a*tb/b^2 = (ta - r*tb)/b. r*tb has
                // dim(r)*dim(b)/dim(x) = dim(a)/dim(x), the type of ta, and
                // dividing by b gives dim(r)/dim(x).
                let [a, b] = self.operands(op)?;
                let ty = self.tangent_ty(r)?;
                let num_ty = self.tangent_ty(a)?;
                let (pb, pr) = (self.primal(b)?, self.primal(r)?);
                let num = match (self.tan_of(a), self.tan_of(b)) {
                    (Some(ta), Some(tb)) => {
                        let u = self.emit(OpKind::Mul, vec![pr, tb], num_ty.clone(), span);
                        self.emit(OpKind::Sub, vec![ta, u], num_ty, span)
                    }
                    (Some(ta), None) => ta,
                    (None, Some(tb)) => {
                        let u = self.emit(OpKind::Mul, vec![pr, tb], num_ty.clone(), span);
                        self.emit(OpKind::Neg, vec![u], num_ty, span)
                    }
                    (None, None) => return Ok(None),
                };
                Ok(Some(self.emit(OpKind::Div, vec![num, pb], ty, span)))
            }
            OpKind::PowConst { exponent } => {
                // r = a^e: tr = e * a^(e-1) * ta. a^(e-1) has dim(a)^(e-1),
                // times ta's dim(a)/dim(x) gives dim(a)^e / dim(x).
                let [a] = self.operands(op)?;
                let Some(ta) = self.tan_of(a) else {
                    return Ok(None);
                };
                if exponent.is_zero() {
                    return Ok(None);
                }
                if *exponent == Rational::ONE {
                    return Ok(Some(ta));
                }
                let em1 = exponent
                    .checked_sub(Rational::ONE)
                    .map_err(|_| self.unsupported("pow.const", "the exponent overflows"))?;
                let pa = self.primal(a)?;
                let pw_ty = number_type(self.dim_pow(self.dim_of(a)?, em1)?, self.repr_of(r));
                let pw = self.emit(
                    OpKind::PowConst { exponent: em1 },
                    vec![pa],
                    pw_ty.clone(),
                    span,
                );
                let e = self.konst_rational(*exponent, r, span);
                let scale = self.emit(OpKind::Mul, vec![e, pw], pw_ty, span);
                let ty = self.tangent_ty(r)?;
                Ok(Some(self.emit(OpKind::Mul, vec![scale, ta], ty, span)))
            }
            OpKind::Pow => {
                let [a, e] = self.operands(op)?;
                if !self.dim_of(a)?.is_dimensionless() || !self.dim_of(e)?.is_dimensionless() {
                    return Err(self.unsupported(
                        "pow",
                        "a computed exponent needs a dimensionless base and exponent",
                    ));
                }
                let (ta, te) = (self.tan_of(a), self.tan_of(e));
                if ta.is_none() && te.is_none() {
                    return Ok(None);
                }
                // r = a^e: tr = e * a^(e-1) * ta + r * ln(a) * te, all
                // dimensionless but the tangents.
                let (pa, pe, pr) = (self.primal(a)?, self.primal(e)?, self.primal(r)?);
                let dimless = number_type(sky_units::Dimension::DIMENSIONLESS, self.repr_of(r));
                let ty = self.tangent_ty(r)?;
                let base_term = ta.map(|ta| {
                    let one = self.konst(1.0, dimless.clone(), span);
                    let em1 = self.emit(OpKind::Sub, vec![pe, one], dimless.clone(), span);
                    let pw = self.emit(OpKind::Pow, vec![pa, em1], dimless.clone(), span);
                    let scale = self.emit(OpKind::Mul, vec![pe, pw], dimless.clone(), span);
                    self.emit(OpKind::Mul, vec![scale, ta], ty.clone(), span)
                });
                let exp_term = te.map(|te| {
                    let ln = OpKind::Math(sky_units::MathFn::Log);
                    let ln_a = self.emit(ln, vec![pa], dimless.clone(), span);
                    let scale = self.emit(OpKind::Mul, vec![pr, ln_a], dimless.clone(), span);
                    self.emit(OpKind::Mul, vec![scale, te], ty.clone(), span)
                });
                Ok(match (base_term, exp_term) {
                    (Some(l), Some(rt)) => Some(self.emit(OpKind::Add, vec![l, rt], ty, span)),
                    (Some(t), None) | (None, Some(t)) => Some(t),
                    (None, None) => None,
                })
            }
            OpKind::Inverse => {
                // r = 1/a: tr = -(r*r) * ta. r*r has dim(r)^2, times
                // dim(a)/dim(x) = dim(r)^-1/dim(x) gives dim(r)/dim(x).
                let [a] = self.operands(op)?;
                let Some(ta) = self.tan_of(a) else {
                    return Ok(None);
                };
                let pr = self.primal(r)?;
                let r_dim = self.dim_of(r)?;
                let sq_ty = number_type(self.dim_mul(r_dim, r_dim)?, self.repr_of(r));
                let sq = self.emit(OpKind::Mul, vec![pr, pr], sq_ty, span);
                let ty = self.tangent_ty(r)?;
                let prod = self.emit(OpKind::Mul, vec![sq, ta], ty.clone(), span);
                Ok(Some(self.emit(OpKind::Neg, vec![prod], ty, span)))
            }
            OpKind::MatMul
            | OpKind::Transpose
            | OpKind::Solve { .. }
            | OpKind::ArrayLiteral { .. }
            | OpKind::Index
            | OpKind::Slice { .. }
            | OpKind::Zeros
            | OpKind::Identity
            | OpKind::Len => Err(self.unsupported(
                op.kind.mnemonic(),
                "array differentiation is not supported yet",
            )),
            OpKind::Math(f) => self.math_tangent(*f, op, r),
            // New operations get an arm above.
            other => {
                Err(self.unsupported(other.mnemonic(), "no derivative rule for this operation"))
            }
        }
    }

    pub(crate) fn operands<const N: usize>(&self, op: &OpData) -> Res<[ValueId; N]> {
        op.operands[..].try_into().map_err(|_| {
            self.malformed(&format!(
                "`{}` takes {N} operands, found {}",
                op.kind.mnemonic(),
                op.operands.len()
            ))
        })
    }

    /// A dimensionless rational constant with the representation of `like`.
    fn konst_rational(&mut self, q: Rational, like: ValueId, span: sky_diag::Span) -> ValueId {
        let ty = number_type(sky_units::Dimension::DIMENSIONLESS, self.repr_of(like));
        self.emit(OpKind::ConstRational(q), vec![], ty, span)
    }
}
