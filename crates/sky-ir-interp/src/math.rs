//! `OpKind::Math`: evaluated by `sky_interp::math::apply`, the one
//! implementation of Sky's math functions, so a SIR program and `sky run`
//! agree on every value, domain error and message.

use sky_diag::Span;
use sky_interp::{RuntimeError, Value};
use sky_ir::MathFn;

type R<T> = Result<T, RuntimeError>;

/// `f(args)`; `args` has `f.arity()` entries (the caller checks), `span` is
/// the call's and `spans` the arguments'.
pub(crate) fn eval(f: MathFn, args: &[&Value], span: Span, spans: &[Span]) -> R<Value> {
    let owned: Vec<Value> = args.iter().map(|v| (*v).clone()).collect();
    sky_interp::math::apply(f, &owned, span, spans)
}
