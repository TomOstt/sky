//! The tangent rules of the math functions (`OpKind::Math`), the table in
//! the header of `rules.rs`. Each builds `f'(a)` from the primal operand
//! or result and multiplies by the operand's tangent.
//!
//! # Typing
//!
//! `ta` has `dim(a)/dim(x)`. Every function but the roots, `abs` and the
//! two-operand ones takes a dimensionless `a`, so the tangent of the result
//! is `ta` scaled by a pure number and has `1/dim(x)`, which is
//! [`Diff::tangent_ty`] of the result. The other rules say in a comment why
//! their operand dimensions multiply out to `dim(r)/dim(x)`.
//!
//! # Where the derivative does not exist
//!
//! At the edge of a function's domain (`sqrt` at 0, `asin` at ±1) the rule
//! divides by zero, so the run fails there as the primal does at its own
//! domain edge, and never returns a made-up number. `floor`, `ceil`,
//! `round` and `sign` are constant between their jumps, so their tangent
//! is zero (at a jump the slope is zero on both sides). `acosh` at 1 and
//! `atanh` at ±1 have infinite slope and fail as `asin` does. `abs`, `min`
//! and `max` have a kink and take the slope on one side: the arm that runs, as `if` does.

use sky_diag::Span;
use sky_ir::{CmpOp, OpData, OpKind, Type, ValueId};
use sky_units::{Dimension, MathFn};

use crate::emit::{number_type, Diff, Res};

impl Diff<'_> {
    /// The tangent of `r = f(operands)` given that some operand has one;
    /// `None` if it is structurally zero.
    pub(crate) fn math_tangent(
        &mut self,
        f: MathFn,
        op: &OpData,
        r: ValueId,
    ) -> Res<Option<ValueId>> {
        let span = op.span;
        let ty = self.tangent_ty(r)?;
        Ok(match f {
            // r = sqrt(a): tr = ta / (2r). ta is dim(a)/dim(x) and r is
            // dim(a)^(1/2), so the quotient is dim(r)/dim(x).
            MathFn::Sqrt => {
                let [a] = self.operands(op)?;
                let Some(ta) = self.tan_of(a) else {
                    return Ok(None);
                };
                let pr = self.primal(r)?;
                let two = self.dimensionless(2.0, r, span);
                let den = self.emit(OpKind::Mul, vec![two, pr], self.value_ty(r)?, span);
                Some(self.emit(OpKind::Div, vec![ta, den], ty, span))
            }
            // r = cbrt(a): tr = ta / (3r²); r² is dim(a)^(2/3).
            MathFn::Cbrt => {
                let [a] = self.operands(op)?;
                let Some(ta) = self.tan_of(a) else {
                    return Ok(None);
                };
                let pr = self.primal(r)?;
                let r_dim = self.dim_of(r)?;
                let sq_ty = number_type(self.dim_mul(r_dim, r_dim)?, self.repr_of(r));
                let sq = self.emit(OpKind::Mul, vec![pr, pr], sq_ty.clone(), span);
                let three = self.dimensionless(3.0, r, span);
                let den = self.emit(OpKind::Mul, vec![three, sq], sq_ty, span);
                Some(self.emit(OpKind::Div, vec![ta, den], ty, span))
            }
            // tr = sign(a) ta, with sign(0) = 1 (the right-hand slope).
            // sign is dimensionless, so the product is dim(a)/dim(x), and
            // dim(r) = dim(a).
            MathFn::Abs => {
                let [a] = self.operands(op)?;
                let Some(ta) = self.tan_of(a) else {
                    return Ok(None);
                };
                let pa = self.primal(a)?;
                let zero = self.konst(0.0, self.value_ty(a)?, span);
                let negative = self.compare(CmpOp::Lt, pa, zero, span);
                let sign_ty = self.dimensionless_ty(r);
                let sign = self.select(
                    negative,
                    sign_ty,
                    span,
                    |d| Ok(d.dimensionless(-1.0, r, span)),
                    |d| Ok(d.dimensionless(1.0, r, span)),
                )?;
                Some(self.emit(OpKind::Mul, vec![sign, ta], ty, span))
            }
            // exp(a)' = exp(a) = r.
            MathFn::Exp => {
                let [a] = self.operands(op)?;
                let pr = self.primal(r)?;
                self.tan_of(a)
                    .map(|ta| self.emit(OpKind::Mul, vec![pr, ta], ty, span))
            }
            // log(a)' = 1/a; log10 and log2 divide by ln 10 and ln 2 too.
            MathFn::Log | MathFn::Log10 | MathFn::Log2 => {
                let [a] = self.operands(op)?;
                let Some(ta) = self.tan_of(a) else {
                    return Ok(None);
                };
                let pa = self.primal(a)?;
                let den = match f {
                    MathFn::Log10 => self.times_const(pa, std::f64::consts::LN_10, r, span),
                    MathFn::Log2 => self.times_const(pa, std::f64::consts::LN_2, r, span),
                    _ => pa,
                };
                Some(self.emit(OpKind::Div, vec![ta, den], ty, span))
            }
            // sin' = cos, cos' = -sin.
            MathFn::Sin | MathFn::Cos => {
                let [a] = self.operands(op)?;
                let Some(ta) = self.tan_of(a) else {
                    return Ok(None);
                };
                let pa = self.primal(a)?;
                let dimless = self.dimensionless_ty(r);
                let d = if f == MathFn::Sin {
                    self.emit(OpKind::Math(MathFn::Cos), vec![pa], dimless, span)
                } else {
                    let s = self.emit(OpKind::Math(MathFn::Sin), vec![pa], dimless.clone(), span);
                    self.emit(OpKind::Neg, vec![s], dimless, span)
                };
                Some(self.emit(OpKind::Mul, vec![d, ta], ty, span))
            }
            // tan' = 1 + tan².
            MathFn::Tan => {
                let [a] = self.operands(op)?;
                let Some(ta) = self.tan_of(a) else {
                    return Ok(None);
                };
                let pr = self.primal(r)?;
                let dimless = self.dimensionless_ty(r);
                let sq = self.emit(OpKind::Mul, vec![pr, pr], dimless.clone(), span);
                let one = self.dimensionless(1.0, r, span);
                let d = self.emit(OpKind::Add, vec![one, sq], dimless, span);
                Some(self.emit(OpKind::Mul, vec![d, ta], ty, span))
            }
            // asin' = 1/sqrt(1 - a²), acos' = -asin', atan' = 1/(1 + a²).
            MathFn::Asin | MathFn::Acos | MathFn::Atan => {
                let [a] = self.operands(op)?;
                let Some(ta) = self.tan_of(a) else {
                    return Ok(None);
                };
                let pa = self.primal(a)?;
                let dimless = self.dimensionless_ty(r);
                let sq = self.emit(OpKind::Mul, vec![pa, pa], dimless.clone(), span);
                let one = self.dimensionless(1.0, r, span);
                if f == MathFn::Atan {
                    let den = self.emit(OpKind::Add, vec![one, sq], dimless, span);
                    Some(self.emit(OpKind::Div, vec![ta, den], ty, span))
                } else {
                    let rest = self.emit(OpKind::Sub, vec![one, sq], dimless.clone(), span);
                    let root = self.emit(OpKind::Math(MathFn::Sqrt), vec![rest], dimless, span);
                    let q = self.emit(OpKind::Div, vec![ta, root], ty.clone(), span);
                    Some(if f == MathFn::Acos {
                        self.emit(OpKind::Neg, vec![q], ty, span)
                    } else {
                        q
                    })
                }
            }
            // atan2(y, x)' = (x ty - y tx) / (x² + y²). With both operands
            // of dimension D, the numerator is D²/dim(w) (w the parameter
            // differentiated against) and the denominator D².
            MathFn::Atan2 => {
                let [y, x] = self.operands(op)?;
                let (py, px) = (self.primal(y)?, self.primal(x)?);
                let d = self.dim_of(y)?;
                let d2 = self.dim_mul(d, d)?;
                let sq_ty = number_type(d2, self.repr_of(r));
                let num_ty = number_type(self.dim_div(d2, self.x_dim)?, self.repr_of(r));
                let left = self
                    .tan_of(y)
                    .map(|t| self.emit(OpKind::Mul, vec![px, t], num_ty.clone(), span));
                let right = self
                    .tan_of(x)
                    .map(|t| self.emit(OpKind::Mul, vec![py, t], num_ty.clone(), span));
                let num = match (left, right) {
                    (Some(l), Some(rt)) => self.emit(OpKind::Sub, vec![l, rt], num_ty, span),
                    (Some(l), None) => l,
                    (None, Some(rt)) => self.emit(OpKind::Neg, vec![rt], num_ty, span),
                    (None, None) => return Ok(None),
                };
                let xx = self.emit(OpKind::Mul, vec![px, px], sq_ty.clone(), span);
                let yy = self.emit(OpKind::Mul, vec![py, py], sq_ty.clone(), span);
                let den = self.emit(OpKind::Add, vec![xx, yy], sq_ty, span);
                Some(self.emit(OpKind::Div, vec![num, den], ty, span))
            }
            // hypot(a, b)' = (a ta + b tb) / r: D²/dim(w) over D.
            MathFn::Hypot => {
                let [a, b] = self.operands(op)?;
                let (pa, pb, pr) = (self.primal(a)?, self.primal(b)?, self.primal(r)?);
                let d = self.dim_of(r)?;
                let d2 = self.dim_mul(d, d)?;
                let num_ty = number_type(self.dim_div(d2, self.x_dim)?, self.repr_of(r));
                let left = self
                    .tan_of(a)
                    .map(|t| self.emit(OpKind::Mul, vec![pa, t], num_ty.clone(), span));
                let right = self
                    .tan_of(b)
                    .map(|t| self.emit(OpKind::Mul, vec![pb, t], num_ty.clone(), span));
                let num = match (left, right) {
                    (Some(l), Some(rt)) => self.emit(OpKind::Add, vec![l, rt], num_ty, span),
                    (Some(t), None) | (None, Some(t)) => t,
                    (None, None) => return Ok(None),
                };
                Some(self.emit(OpKind::Div, vec![num, pr], ty, span))
            }
            // The tangent of the operand the result equals; on a tie, the
            // first operand's.
            MathFn::Min | MathFn::Max => {
                let [a, b] = self.operands(op)?;
                let (pa, pb) = (self.primal(a)?, self.primal(b)?);
                let first = if f == MathFn::Min {
                    CmpOp::Le
                } else {
                    CmpOp::Ge
                };
                let cond = self.compare(first, pa, pb, span);
                let chosen = self.select(
                    cond,
                    ty,
                    span,
                    |d| d.tangent_or_zero(a, span),
                    |d| d.tangent_or_zero(b, span),
                )?;
                Some(chosen)
            }
            // expm1(a)' = exp(a). Not r + 1: for very negative a that
            // subtraction-free form is the accurate one.
            MathFn::Expm1 => {
                let [a] = self.operands(op)?;
                let Some(ta) = self.tan_of(a) else {
                    return Ok(None);
                };
                let pa = self.primal(a)?;
                let dimless = self.dimensionless_ty(r);
                let d = self.emit(OpKind::Math(MathFn::Exp), vec![pa], dimless, span);
                Some(self.emit(OpKind::Mul, vec![d, ta], ty, span))
            }
            // log1p(a)' = 1/(1 + a).
            MathFn::Log1p => {
                let [a] = self.operands(op)?;
                let Some(ta) = self.tan_of(a) else {
                    return Ok(None);
                };
                let pa = self.primal(a)?;
                let dimless = self.dimensionless_ty(r);
                let one = self.dimensionless(1.0, r, span);
                let den = self.emit(OpKind::Add, vec![one, pa], dimless, span);
                Some(self.emit(OpKind::Div, vec![ta, den], ty, span))
            }
            // sinh' = cosh, cosh' = sinh.
            MathFn::Sinh | MathFn::Cosh => {
                let [a] = self.operands(op)?;
                let Some(ta) = self.tan_of(a) else {
                    return Ok(None);
                };
                let pa = self.primal(a)?;
                let dimless = self.dimensionless_ty(r);
                let other = if f == MathFn::Sinh {
                    MathFn::Cosh
                } else {
                    MathFn::Sinh
                };
                let d = self.emit(OpKind::Math(other), vec![pa], dimless, span);
                Some(self.emit(OpKind::Mul, vec![d, ta], ty, span))
            }
            // tanh' = 1 - tanh².
            MathFn::Tanh => {
                let [a] = self.operands(op)?;
                let Some(ta) = self.tan_of(a) else {
                    return Ok(None);
                };
                let pr = self.primal(r)?;
                let dimless = self.dimensionless_ty(r);
                let sq = self.emit(OpKind::Mul, vec![pr, pr], dimless.clone(), span);
                let one = self.dimensionless(1.0, r, span);
                let d = self.emit(OpKind::Sub, vec![one, sq], dimless, span);
                Some(self.emit(OpKind::Mul, vec![d, ta], ty, span))
            }
            // asinh(a)' = 1/sqrt(a² + 1), as 1/hypot(1, a) so that a huge a
            // does not overflow a².
            MathFn::Asinh => {
                let [a] = self.operands(op)?;
                let Some(ta) = self.tan_of(a) else {
                    return Ok(None);
                };
                let pa = self.primal(a)?;
                let dimless = self.dimensionless_ty(r);
                let one = self.dimensionless(1.0, r, span);
                let root = self.emit(OpKind::Math(MathFn::Hypot), vec![one, pa], dimless, span);
                Some(self.emit(OpKind::Div, vec![ta, root], ty, span))
            }
            // acosh(a)' = 1/sqrt(a² - 1); infinite at a = 1, where the
            // division fails.
            MathFn::Acosh => {
                let [a] = self.operands(op)?;
                let Some(ta) = self.tan_of(a) else {
                    return Ok(None);
                };
                let pa = self.primal(a)?;
                let dimless = self.dimensionless_ty(r);
                let sq = self.emit(OpKind::Mul, vec![pa, pa], dimless.clone(), span);
                let one = self.dimensionless(1.0, r, span);
                let rest = self.emit(OpKind::Sub, vec![sq, one], dimless.clone(), span);
                let root = self.emit(OpKind::Math(MathFn::Sqrt), vec![rest], dimless, span);
                Some(self.emit(OpKind::Div, vec![ta, root], ty, span))
            }
            // atanh(a)' = 1/(1 - a²).
            MathFn::Atanh => {
                let [a] = self.operands(op)?;
                let Some(ta) = self.tan_of(a) else {
                    return Ok(None);
                };
                let pa = self.primal(a)?;
                let dimless = self.dimensionless_ty(r);
                let sq = self.emit(OpKind::Mul, vec![pa, pa], dimless.clone(), span);
                let one = self.dimensionless(1.0, r, span);
                let den = self.emit(OpKind::Sub, vec![one, sq], dimless, span);
                Some(self.emit(OpKind::Div, vec![ta, den], ty, span))
            }
            // erf(a)' = 2/sqrt(π) exp(-a²).
            MathFn::Erf => {
                let [a] = self.operands(op)?;
                let Some(ta) = self.tan_of(a) else {
                    return Ok(None);
                };
                let pa = self.primal(a)?;
                let dimless = self.dimensionless_ty(r);
                let sq = self.emit(OpKind::Mul, vec![pa, pa], dimless.clone(), span);
                let neg = self.emit(OpKind::Neg, vec![sq], dimless.clone(), span);
                let e = self.emit(OpKind::Math(MathFn::Exp), vec![neg], dimless, span);
                let d = self.times_const(e, std::f64::consts::FRAC_2_SQRT_PI, r, span);
                Some(self.emit(OpKind::Mul, vec![d, ta], ty, span))
            }
            // Piecewise constant: zero slope.
            MathFn::Floor | MathFn::Ceil | MathFn::Round | MathFn::Sign => None,
        })
    }

    /// A dimensionless constant with the representation of `like`.
    fn dimensionless(&mut self, x: f64, like: ValueId, span: Span) -> ValueId {
        let ty = self.dimensionless_ty(like);
        self.konst(x, ty, span)
    }

    fn dimensionless_ty(&self, like: ValueId) -> Type {
        number_type(Dimension::DIMENSIONLESS, self.repr_of(like))
    }

    /// The type of source value `v` as a scalar number (its primal type).
    fn value_ty(&self, v: ValueId) -> Res<Type> {
        Ok(number_type(self.dim_of(v)?, self.repr_of(v)))
    }

    /// `a * k` for a dimensionless constant `k`, typed as `like` (which is
    /// dimensionless: the operand of `log`).
    fn times_const(&mut self, a: ValueId, k: f64, like: ValueId, span: Span) -> ValueId {
        let ty = self.dimensionless_ty(like);
        let k = self.konst(k, ty.clone(), span);
        self.emit(OpKind::Mul, vec![a, k], ty, span)
    }

    fn compare(&mut self, c: CmpOp, a: ValueId, b: ValueId, span: Span) -> ValueId {
        self.emit(OpKind::Cmp(c), vec![a, b], Type::Bool, span)
    }

    /// `if cond { then } else { els }` with one result of type `ty`; each
    /// arm is built by its closure, which returns the value it yields.
    fn select(
        &mut self,
        cond: ValueId,
        ty: Type,
        span: Span,
        then: impl FnOnce(&mut Self) -> Res<ValueId>,
        els: impl FnOnce(&mut Self) -> Res<ValueId>,
    ) -> Res<ValueId> {
        let outer = self.b.insertion_block();
        let (op, results) = self.b.op(OpKind::If, vec![cond], vec![ty], span);
        self.arm(op, span, then)?;
        self.arm(op, span, els)?;
        self.b.set_insertion_block(outer);
        Ok(results[0])
    }

    /// A new one-block region of `op` that yields what `build` returns.
    fn arm(
        &mut self,
        op: sky_ir::OpId,
        span: Span,
        build: impl FnOnce(&mut Self) -> Res<ValueId>,
    ) -> Res<()> {
        let region = self.b.add_region(op);
        let block = self.b.add_block(region, vec![]);
        self.b.set_insertion_block(block);
        let v = build(self)?;
        self.b.op(OpKind::Yield, vec![v], vec![], span);
        Ok(())
    }
}
