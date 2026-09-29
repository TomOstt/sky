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
//! | `a ^ e` (computed `e`) | `(e * a^(e-1)) * ta`; `tb` needs `ln` |
//! | `a⁻¹` | `-(r*r) * ta` |
//! | comparisons, `not`, constants | none (Bool or constant) |
//!
//! # Rules ready to paste for `OpKind::Math`
//!
//! The e-graph work adds `OpKind::Math(f)` (one operand `a`, one result
//! `r = f(a)`). Each function is one more arm of [`Diff::tangent_rule`]
//! that builds `d = f'(a)` and returns `d * ta` (or a cheaper form below).
//! `a` is dimensionless for everything except `sqrt` and `abs`, so `tr` has
//! the dimension `1/dim(x) * dim(a)`, which is `tangent_ty(r)` when
//! `dim(r) = 1`; for `sqrt` it is `dim(a)^(1/2) / dim(x)`, again `tangent_ty(r)`.
//!
//! | `f` | `tr` | notes |
//! |---|---|---|
//! | `sqrt` | `ta / (2 * r)` | `ta` has `dim(a)/dim(x)`, `r` has `dim(a)^½`, so the quotient is `dim(r)/dim(x)`; fails at `a = 0` (division by zero), as `a^½` does |
//! | `exp` | `r * ta` | reuses the result |
//! | `ln` | `ta / a` | `a` dimensionless |
//! | `sin` | `cos(a) * ta` | angle in radians, dimensionless |
//! | `cos` | `-(sin(a)) * ta` | |
//! | `tan` | `(1 + r*r) * ta` | |
//! | `asin` | `ta / sqrt(1 - a*a)` | |
//! | `acos` | `-(ta / sqrt(1 - a*a))` | |
//! | `atan` | `ta / (1 + a*a)` | |
//! | `sinh` | `cosh(a) * ta` | |
//! | `cosh` | `sinh(a) * ta` | |
//! | `tanh` | `(1 - r*r) * ta` | |
//! | `abs` | `sign(a) * ta` | `sign` is `if a < 0 { -1 } else { 1 }`; at 0 the right-hand slope is returned |
//!
//! Two-operand functions (`atan2(y, x)`, `hypot`, `min`, `max`) follow the
//! product rule shape: `atan2`: `(x*ty - y*tx) / (x*x + y*y)`; `hypot`:
//! `(x*tx + y*ty) / r`; `min`/`max`: the tangent of the operand selected by
//! the comparison, like `if`.
//!
//! The general power `a ^ e` with a tangent for the exponent adds
//! `r * ln(a) * te` to the rule below (`ln` needs `a > 0`, and `ln(a)` is
//! dimensionless because a computed exponent needs a dimensionless base).

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
                if self.tan_of(e).is_some() {
                    return Err(self.unsupported(
                        "pow",
                        "the exponent depends on the differentiated parameter, which needs `ln` \
                         (rule: add `r * ln(a) * te`, see rules.rs)",
                    ));
                }
                let Some(ta) = self.tan_of(a) else {
                    return Ok(None);
                };
                // tr = e * a^(e-1) * ta, all dimensionless but ta.
                let (pa, pe) = (self.primal(a)?, self.primal(e)?);
                let dimless = number_type(sky_units::Dimension::DIMENSIONLESS, self.repr_of(r));
                let one = self.konst(1.0, dimless.clone(), span);
                let em1 = self.emit(OpKind::Sub, vec![pe, one], dimless.clone(), span);
                let pw = self.emit(OpKind::Pow, vec![pa, em1], dimless.clone(), span);
                let scale = self.emit(OpKind::Mul, vec![pe, pw], dimless, span);
                let ty = self.tangent_ty(r)?;
                Ok(Some(self.emit(OpKind::Mul, vec![scale, ta], ty, span)))
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
            // New operations (`OpKind::Math`, ...) get an arm above.
            other => {
                Err(self.unsupported(other.mnemonic(), "no derivative rule for this operation"))
            }
        }
    }

    fn operands<const N: usize>(&self, op: &OpData) -> Res<[ValueId; N]> {
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
