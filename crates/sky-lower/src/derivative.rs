//! `derivative(f, x)(args)`: a call of the derivative function that
//! `sky_ad` builds.
//!
//! The checker types the callee `derivative(f, x)` as a function with `f`'s
//! parameters whose signature says what it differentiates
//! ([`FuncSig::derivative_of`]). Lowering emits a `call` of `df/dx`, the
//! function `sky_ad::differentiate_into` adds to the module when lowering is
//! done (one per `(f, x)` pair, however often it is called), and keeps the
//! second of its two results: `df/dx` returns `(f(args), ∂f/∂x)`.
//!
//! The call passes the arguments and then the module-level names `f` reads,
//! like a call of `f` itself: the derivative has the parameters of `f`,
//! and `x` is one of the leading ones, so the trailing names are constants
//! of the derivative.
//!
//! A function `sky_ad` cannot differentiate (an operation it has no rule
//! for, an `Unknown` value the derivative flows through) is a
//! [`LowerError`] on the call, before anything runs.

use sky_ast::{Expr, ExprKind};
use sky_diag::Span;
use sky_ir::{OpKind, Type, ValueId};
use sky_types::{FuncSig, Ty};

use crate::ctx::{map_ty, Lower, R};
use crate::LowerError;

/// A derivative function to build once lowering has produced its function.
pub(crate) struct Wanted {
    /// The SIR name of the function that is differentiated.
    pub(crate) func: String,
    /// The index of the parameter differentiated against.
    pub(crate) wrt: usize,
    /// The call that needs it, for an error.
    pub(crate) span: Span,
}

impl Lower<'_> {
    /// The `def` a callee `derivative(f, x)` differentiates, if `callee`
    /// (parentheses aside) is one, for the analysis of what a function
    /// reads.
    pub(crate) fn derivative_def(&self, callee: &Expr) -> Option<Span> {
        let sig = self.derivative_sig(callee)?;
        Some(sig.derivative_of()?.func().def_span())
    }

    /// The signature of the callee `derivative(f, x)`, if it is one.
    pub(crate) fn derivative_sig(&self, callee: &Expr) -> Option<std::sync::Arc<FuncSig>> {
        let mut target = callee;
        while let ExprKind::Paren(inner) = &target.kind {
            target = inner;
        }
        if !matches!(target.kind, ExprKind::Call { .. }) {
            return None;
        }
        match self.checked.expr_types.get(&target.span) {
            Some(Ty::Func(sig)) if sig.derivative_of().is_some() => Some(sig.clone()),
            _ => None,
        }
    }

    /// `derivative(f, x)(args)`, the call of the derivative function `sig`
    /// describes, and the derivative it returns.
    pub(crate) fn derivative_call(&mut self, e: &Expr, sig: &FuncSig, args: &[Expr]) -> R<ValueId> {
        let Some(d) = sig.derivative_of() else {
            return self.err("internal error: not a derivative", e.span);
        };
        let f = d.func();
        let Some(info) = self.fns.get(&f.def_span()) else {
            return self.err(
                format!(
                    "`{}` is not available here",
                    sky_diag::display_capped(f.name())
                ),
                e.span,
            );
        };
        let (name, primal, arity, globals) = (
            info.name.clone(),
            info.result.clone(),
            info.params,
            info.globals.clone(),
        );
        if args.len() != arity {
            return self.err(
                format!(
                    "`{}` takes {arity} arguments, found {}",
                    f.name(),
                    args.len()
                ),
                e.span,
            );
        }
        // The call's own type is the derivative's: the checker's
        // `dim(f)/dim(x)`, which is what `sky_ad` types the tangent as.
        let tangent = match self.checked.expr_ty(e.span).as_ref().and_then(map_ty) {
            Some(t @ Type::Number(_)) => t,
            _ => return self.err("internal error: a derivative without a number type", e.span),
        };
        let mut vals = Vec::with_capacity(args.len() + globals.len());
        for a in args {
            vals.push(self.expr(a)?);
        }
        for g in &globals {
            vals.push(self.module_value(g, e.span)?);
        }
        let param = f
            .param_name(d.wrt())
            .map_or_else(|| d.wrt().to_string(), str::to_string);
        let callee = sky_ad::derivative_name_of(&name, &param);
        let (_, results) =
            self.st
                .b
                .op(OpKind::Call { callee }, vals, vec![primal, tangent], e.span);
        self.derivatives.push(Wanted {
            func: name,
            wrt: d.wrt(),
            span: e.span,
        });
        Ok(results[1])
    }
}

/// An error from `sky_ad` as a lowering error on the call that asked.
pub(crate) fn ad_error(e: &sky_ad::AdError, span: Span) -> LowerError {
    let note = match e {
        sky_ad::AdError::UnknownType { .. } => {
            "give the parameters and results the derivative flows through a unit annotation"
        }
        sky_ad::AdError::NonScalar { .. } => "derivatives of arrays come with array AD",
        _ => "`derivative` differentiates the function's SIR, which needs a rule for every operation the derivative flows through",
    };
    LowerError::new(e.to_string(), span).with_note(note)
}
