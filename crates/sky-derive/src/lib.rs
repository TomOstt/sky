//! `derivative(f, x)(args)` for `sky run`.
//!
//! The tree-walking interpreter (`sky-interp`) has no automatic
//! differentiation, and this crate does not add one. Differentiation is a
//! program transformation on SIR (`sky-ad`), so [`SirDerivatives`], the
//! interpreter's [`DerivativeHook`], evaluates a derivative by
//!
//! 1. lowering the program's `def`s (and only those: `sky_lower::lower_defs`)
//!    to SIR, once, on the first derivative;
//! 2. building the derivative function of the requested `def` with
//!    `sky_ad::differentiate_into` (once per function and parameter);
//! 3. running it with the reference interpreter (`sky-ir-interp`) on the
//!    arguments the tree walker evaluated, and the module-level names the
//!    function reads, and returning the derivative it computed.
//!
//! So the derivative is computed by the same code in both engines: the
//! lowered program calls `df/dx` and this crate calls it too, and there is
//! no second set of differentiation rules to keep in step. The dimension of
//! the value is the derivative function's, `dim(f)/dim(x)`.
//!
//! # What can go wrong
//!
//! Only the `def`s are lowered: the module's values come from the tree
//! walker, so top-level statements SIR cannot express yet (`equation`,
//! `solve`, `±`) do not stop a derivative. Nor does a `def` SIR cannot
//! express (one that `solve`s, say) that the differentiated function never
//! calls: such `def`s are left out (`sky_lower::lower_defs_partial`). A
//! function that does reach one cannot be differentiated, and the error
//! names the chain of calls and points at what stopped it (the `solve`). A
//! function `sky-ad` cannot differentiate (an unannotated parameter the
//! derivative flows through, or the gradient of a loop over an array whose
//! size is not known) is an error too. Both are runtime errors on the
//! derivative call; the checker cannot know, because `sky check` never
//! lowers.
//!
//! # Arrays
//!
//! The derivative with respect to an array parameter is its gradient: an
//! array of the parameter's shape, which is the value this hook returns
//! (`grad(U)(p)` for a potential `U` of a position vector `p`). `sky-ad`
//! computes it in reverse mode, or by forward mode one entry at a time when
//! the function loops; the function `d<f>/d<p>` has the same signature
//! either way, so nothing here depends on which.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::io::Write;

use sky_ast::Module as Ast;
use sky_diag::Span;
use sky_interp::{DerivativeHook, DerivativeRequest, RuntimeError, Value};

/// Evaluates derivatives for a program by SIR AD; see the crate docs.
pub struct SirDerivatives<'a> {
    module: &'a Ast,
    src: &'a str,
    /// The lowered program, built by the first derivative that needs it.
    sir: Option<Result<Lowered, RuntimeError>>,
}

/// The program as SIR, growing derivative functions as they are asked for.
struct Lowered {
    module: sky_ir::Module,
    /// The SIR function of each `def`, by the span of its name.
    defs: HashMap<Span, String>,
    /// Why each `def` that is not in `module` is not.
    failed: HashMap<Span, sky_lower::DefFailure>,
}

impl<'a> SirDerivatives<'a> {
    /// A hook for `module`, which was parsed from `src` and checked without
    /// errors (as `sky run` does before it runs anything).
    pub fn new(module: &'a Ast, src: &'a str) -> Self {
        SirDerivatives {
            module,
            src,
            sir: None,
        }
    }

    /// Lowers the program on first use. The error (when lowering fails) is
    /// kept, so it is reported the same way each time.
    fn lowered(&mut self, at: Span) -> Result<&mut Lowered, RuntimeError> {
        let (module, src) = (self.module, self.src);
        let sir = self.sir.get_or_insert_with(|| {
            let fail = |why: String, note: Option<String>| {
                let mut e = RuntimeError::new(format!("cannot evaluate a derivative: {why}"), at)
                    .with_label("this derivative");
                if let Some(n) = note {
                    e = e.with_note(n);
                }
                e.with_note(
                    "derivatives run on SIR, so every `def` in the program must lower to SIR",
                )
            };
            let checked = sky_types::check_module_with(module, src, sky_types::Env::with_prelude());
            if checked.has_errors() {
                return Err(fail("the program does not check".into(), None));
            }
            let partial = sky_lower::lower_defs_partial(module, src, &checked)
                .map_err(|e| fail(e.message, e.note))?;
            verified(&partial.module, at)?;
            Ok(Lowered {
                module: partial.module,
                defs: partial.defs,
                failed: partial.failed,
            })
        });
        sir.as_mut().map_err(|e| e.clone())
    }
}

/// The error for a derivative of `name`, a `def` that did not lower: what
/// stopped it, through which calls, pointing at the cause.
fn unlowered(name: &str, failure: &sky_lower::DefFailure, at: Span) -> RuntimeError {
    // `d<f>/d<x>` is a derivative SIR builds for a `derivative(f, x)` call.
    let show = |callee: &str| match callee
        .strip_prefix('d')
        .and_then(|rest| Some(&rest[..rest.rfind("/d")?]))
    {
        Some(f) => format!("the derivative of `{f}`"),
        None => format!("`{callee}`"),
    };
    let e = &failure.error;
    let message = if failure.calls.is_empty() {
        format!(
            "cannot evaluate a derivative: `{name}` cannot run on SIR: {}",
            e.message
        )
    } else {
        // "`f` calls `g`, which calls `h`, which cannot run on SIR: ..."
        let mut text = format!("`{name}`");
        for (i, c) in failure.calls.iter().enumerate() {
            text.push_str(if i == 0 { " calls " } else { ", which calls " });
            text.push_str(&show(c));
        }
        format!(
            "cannot evaluate a derivative: {text}, which cannot run on SIR: {}",
            e.message
        )
    };
    let mut err = RuntimeError::new(message, at)
        .with_label("this derivative")
        .with_secondary(e.span, "SIR stops here");
    if let Some(n) = &e.note {
        err = err.with_note(n.clone());
    }
    err.with_note(
        "derivatives run on SIR, so the function differentiated and every function it calls must lower to SIR",
    )
}

/// SIR that `sky-lower` or `sky-ad` built must verify; anything else is a
/// bug in them, reported as one.
fn verified(m: &sky_ir::Module, at: Span) -> Result<(), RuntimeError> {
    sky_ir::verify(m).map_err(|errors| {
        let why = errors
            .first()
            .map_or_else(String::new, |e| e.message.clone());
        RuntimeError::new(format!("internal error: invalid SIR: {why}"), at)
    })
}

impl DerivativeHook for SirDerivatives<'_> {
    fn derivative(
        &mut self,
        request: &DerivativeRequest<'_>,
        out: &mut dyn Write,
    ) -> Result<Value, RuntimeError> {
        let at = request.span;
        let lowered = self.lowered(at)?;
        if let Some(failure) = lowered.failed.get(&request.func.def_span) {
            return Err(unlowered(&request.func.name, failure, at));
        }
        let Some(func) = lowered.defs.get(&request.func.def_span).cloned() else {
            return Err(RuntimeError::new(
                format!(
                    "internal error: `{}` was not lowered to SIR",
                    request.func.name
                ),
                at,
            ));
        };
        let known = lowered.module.functions.len();
        let name = sky_ad::differentiate_into(&mut lowered.module, &func, request.wrt)
            .map_err(|e| RuntimeError::new(e.to_string(), at).with_label("this derivative"))?;
        if lowered.module.functions.len() != known {
            verified(&lowered.module, at)?;
        }
        let Some(target) = lowered.module.function(&name) else {
            return Err(RuntimeError::new(
                format!("internal error: `{name}` is missing"),
                at,
            ));
        };
        // The derivative has the parameters of `f`, then the module names
        // `f` reads, which the function reads from the module at run time.
        let mut args = request.args.clone();
        for p in target.param_values().iter().skip(args.len()) {
            let global = target.value(*p).name.clone().unwrap_or_default();
            match (request.globals)(&global) {
                Some(v) => args.push(v),
                None => {
                    return Err(RuntimeError::new(
                        format!("cannot find `{global}` in this scope"),
                        at,
                    )
                    .with_label("not defined"));
                }
            }
        }
        let results =
            sky_ir_interp::call_metered(&lowered.module, &name, args, out, request.stack, at)
                .map_err(|e| (*e).clone())?;
        match results.as_slice() {
            // `(f(args), ∂f/∂x)`
            [_, derivative] => Ok(derivative.clone()),
            _ => Err(RuntimeError::new(
                format!("internal error: `{name}` must return a value and a derivative"),
                at,
            )),
        }
    }
}
