//! Typing of `derivative(f, x)` (spec "Derivatives").
//!
//! `derivative(f, x)` is the derivative of the `def` function `f` with
//! respect to its parameter `x`: a function with `f`'s parameters whose
//! result has the dimension `dim(result of f) / dim(x)`, so the derivative
//! of a position with respect to time is a velocity and adding it to a
//! length is a unit error like any other. Sky has no function values in
//! SIR, so the form is only valid where it is called on the spot:
//! `derivative(position, t)(3 s)`. The checker types the callee as a
//! [`Ty::Func`] whose signature has `f`'s parameters and the derivative's
//! result, so the ordinary call rules check the arguments (arity, shape,
//! and a dimension for every annotated parameter) and give the result.
//!
//! `derivative` is recognised by name, unless the program bound that name
//! itself, exactly as for `print` and `inv` ([`Checker::derivative_call`]
//! is the one hook in `check.rs`). The second argument names a parameter
//! of `f`; it is not looked up as a variable, so it is not typed and its
//! entry in `Checked::expr_types` is `Unknown`.
//!
//! What is rejected, each with a message that says how to fix it:
//!
//! * a wrong number of arguments, or a first argument that is not a `def`
//!   (a built-in function, a number, another derivative);
//! * a second argument that is not the name of a parameter of `f`;
//! * a parameter without a unit annotation (its dimension is unknown, so
//!   the derivative's unit cannot be computed; this applies to every
//!   parameter, since the derivative flows through all of them);
//! * `x` or the result of `f` not being a scalar number, or the result's
//!   unit being unknown;
//! * a bare `derivative(f, x)` that is not called at once.
//!
//! An argument or a function whose type is already unknown (an earlier
//! error) makes the result unknown without a second error.

use std::sync::Arc;

use sky_ast::{Expr, ExprKind};
use sky_diag::{display_capped as quote, Diagnostic, Span};
use sky_units::MathFn;

use crate::check::{describe, strip_parens, Checker, Ty};
use crate::flow::FuncSig;

/// What the function `derivative(f, x)` differentiates: it is the
/// derivative of [`DerivativeOf::func`] with respect to its parameter
/// number [`DerivativeOf::wrt`] (counted from 0). Lowering reads this to
/// build the derivative function; see [`FuncSig::derivative_of`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DerivativeOf {
    pub(crate) func: Arc<FuncSig>,
    pub(crate) wrt: usize,
}

impl DerivativeOf {
    /// The function that is differentiated (the `def`'s own signature).
    pub fn func(&self) -> &FuncSig {
        &self.func
    }

    /// The index of the parameter differentiated against.
    pub fn wrt(&self) -> usize {
        self.wrt
    }
}

/// The names the language provides that are not `def`s, for the message
/// that says a built-in cannot be differentiated.
fn is_builtin_name(name: &str) -> bool {
    MathFn::from_name(name).is_some()
        || matches!(
            name,
            "print" | "inv" | "zeros" | "identity" | "len" | "derivative"
        )
}

impl Checker<'_> {
    /// The type of `callee(args)` when it involves `derivative`: either the
    /// call `derivative(f, x)(args...)`, or a `derivative(f, x)` that is not
    /// called at once (an error). `None` leaves any other call to the
    /// ordinary path.
    pub(crate) fn derivative_call(
        &mut self,
        callee: &Expr,
        args: &[Expr],
        span: Span,
    ) -> Option<Ty> {
        let inner = strip_parens(callee);
        if let ExprKind::Call {
            callee: name,
            args: dargs,
        } = &inner.kind
        {
            if self.is_derivative(name) {
                let t = self.derivative_function(dargs, inner.span);
                // The callee (and any parentheses around it) has the type
                // of the derivative function, as a name bound to it would.
                let mut node = callee;
                loop {
                    self.expr_types.insert(node.span, t.clone());
                    match &node.kind {
                        ExprKind::Paren(next) => node = next,
                        _ => break,
                    }
                }
                return Some(self.call(t, callee, args, span));
            }
        }
        if self.is_derivative(callee) {
            let t = self.derivative_function(args, span);
            if matches!(t, Ty::Func(_)) {
                self.derivative_not_called(args, span);
            }
            return Some(Ty::Unknown);
        }
        None
    }

    /// `derivative` while the program has not bound that name itself.
    fn is_derivative(&self, callee: &Expr) -> bool {
        matches!(&callee.kind, ExprKind::Name(id)
            if self.ident(id) == "derivative" && self.env.get("derivative").is_none())
    }

    /// The type of the function `derivative(args...)`: [`Ty::Func`] when
    /// the arguments are valid, otherwise [`Ty::Unknown`] after reporting.
    fn derivative_function(&mut self, args: &[Expr], span: Span) -> Ty {
        let [f_arg, x_arg] = args else {
            self.derivative_arity(args, span);
            for (i, a) in args.iter().enumerate() {
                self.derivative_argument(a, i == 0);
            }
            return Ty::Unknown;
        };
        self.derivative_argument(x_arg, false);
        let Some(sig) = self.derivative_target(f_arg) else {
            return Ty::Unknown;
        };
        let ExprKind::Name(xid) = &x_arg.kind else {
            self.derivative_param_not_a_name(&sig, x_arg);
            return Ty::Unknown;
        };
        let xname = self.ident(xid).to_string();
        let Some(wrt) = sig.params.iter().position(|p| p.name == xname) else {
            self.derivative_no_such_param(&sig, x_arg);
            return Ty::Unknown;
        };
        // Every parameter needs a unit: the derivative flows through all.
        let mut ok = true;
        for (i, p) in sig.params.iter().enumerate() {
            match &p.ty {
                None if p.annotation.is_some() => ok = false, // reported already
                None => {
                    self.derivative_unannotated(&sig, i, wrt, x_arg);
                    ok = false;
                }
                Some(v) if i == wrt && !v.is_scalar() => {
                    self.derivative_wrt_array(&sig, wrt, x_arg);
                    ok = false;
                }
                Some(_) => {}
            }
        }
        if !ok {
            return Ty::Unknown;
        }
        let Some(x_dim) = sig.params[wrt].ty.as_ref().map(|v| v.dim) else {
            return Ty::Unknown;
        };
        let dim = match &sig.ret {
            Ty::Value(v) if v.is_scalar() => match v.dim.checked_div(x_dim) {
                Ok(d) => d,
                Err(_) => {
                    self.error(Diagnostic::error(
                        "the dimension of the derivative is too large",
                        span,
                    ));
                    return Ty::Unknown;
                }
            },
            other => {
                self.derivative_bad_result(&sig, other, f_arg);
                return Ty::Unknown;
            }
        };
        Ty::Func(Arc::new(FuncSig {
            name: format!(
                "derivative({}, {})",
                quote(sig.name()),
                quote(&sig.params[wrt].name)
            ),
            def_span: sig.def_span,
            params: sig.params.clone(),
            ret: Ty::scalar(dim),
            generic: sig.generic.clone(),
            derivative: Some(Arc::new(DerivativeOf {
                func: sig.clone(),
                wrt,
            })),
        }))
    }

    /// Gives an argument of `derivative` its entry in the type table. The
    /// function is an ordinary expression; the parameter name is not a
    /// value, so a name there is recorded as unknown instead of being
    /// looked up.
    fn derivative_argument(&mut self, arg: &Expr, is_function: bool) {
        if is_function || !matches!(arg.kind, ExprKind::Name(_)) {
            self.expr(arg);
        } else {
            self.expr_types.insert(arg.span, Ty::Unknown);
        }
    }

    /// The signature of the function `f_arg` names, or `None` after
    /// reporting why it is not one `derivative` accepts.
    fn derivative_target(&mut self, f_arg: &Expr) -> Option<Arc<FuncSig>> {
        let inner = strip_parens(f_arg);
        if let ExprKind::Call { callee, .. } = &inner.kind {
            if self.is_derivative(callee) {
                if let ExprKind::Call { args, .. } = &inner.kind {
                    for (i, a) in args.iter().enumerate() {
                        self.derivative_argument(a, i == 0);
                    }
                }
                self.expr_types.insert(inner.span, Ty::Unknown);
                self.expr_types.insert(f_arg.span, Ty::Unknown);
                self.error(
                    Diagnostic::error(
                        "`derivative` needs a function defined with `def`, found another derivative",
                        f_arg.span,
                    )
                    .with_label("this is `derivative(...)`, which can only be called")
                    .with_help(
                        "name the first derivative with a `def` that calls it, and differentiate that",
                    ),
                );
                return None;
            }
        }
        if let ExprKind::Name(id) = &f_arg.kind {
            let name = self.ident(id);
            if is_builtin_name(name) && self.env.get(name).is_none() {
                self.expr_types.insert(f_arg.span, Ty::Unknown);
                self.error(
                    Diagnostic::error(
                        format!(
                            "`derivative` needs a function defined with `def`, found the built-in `{}`",
                            quote(name)
                        ),
                        f_arg.span,
                    )
                    .with_label("a built-in function")
                    .with_note(
                        "a `def` states the units of its parameters and result, which the unit of the derivative is computed from",
                    )
                    .with_help("wrap it in a `def` that states its units, and differentiate that"),
                );
                return None;
            }
        }
        match self.expr(f_arg) {
            Ty::Func(sig) => Some(sig),
            Ty::Unknown => None,
            other => {
                let what = match &other {
                    Ty::Value(v) if v.is_scalar() => format!("this is {}", describe(&v.dim)),
                    Ty::Value(v) => self.this_is(v),
                    _ => "this is Bool".to_string(),
                };
                self.error(
                    Diagnostic::error(
                        "`derivative` needs a function defined with `def`",
                        f_arg.span,
                    )
                    .with_label(what)
                    .with_help("define the function with `def f(x: unit) -> unit: ...` and differentiate that"),
                );
                None
            }
        }
    }

    #[cold]
    fn derivative_arity(&mut self, args: &[Expr], span: Span) {
        let n = args.len();
        self.error(
            Diagnostic::error(
                format!(
                    "`derivative` takes 2 arguments, found {n}: a function and the name of its parameter"
                ),
                span,
            )
            .with_label(if n < 2 {
                "missing argument"
            } else {
                "extra argument"
            })
            .with_help("write `derivative(f, x)(...)`, then the arguments of `f`"),
        );
    }

    #[cold]
    fn derivative_param_not_a_name(&mut self, sig: &FuncSig, x_arg: &Expr) {
        let mut d = Diagnostic::error(
            format!(
                "the second argument of `derivative` is the name of a parameter of `{}`",
                quote(sig.name())
            ),
            x_arg.span,
        )
        .with_label("not a parameter name")
        .with_secondary(
            sig.def_span,
            format!("`{}` is defined here", quote(sig.name())),
        );
        if let Some(first) = sig.params.first() {
            d = d.with_help(format!(
                "write the parameter's name, not its position: `derivative({}, {})`",
                quote(sig.name()),
                quote(&first.name)
            ));
        }
        self.error(d);
    }

    #[cold]
    fn derivative_no_such_param(&mut self, sig: &FuncSig, x_arg: &Expr) {
        let fname = quote(sig.name());
        let names: Vec<String> = sig
            .params
            .iter()
            .map(|p| format!("`{}`", quote(&p.name)))
            .collect();
        let mut d = Diagnostic::error(
            format!("`{fname}` has no parameter `{}`", self.quoted(x_arg.span)),
            x_arg.span,
        )
        .with_label(format!("not a parameter of `{fname}`"))
        .with_secondary(sig.def_span, format!("`{fname}` is defined here"));
        d = match names.as_slice() {
            [] => d.with_note(format!("`{fname}` takes no parameters")),
            [only] => d.with_note(format!("its parameter is {only}")),
            _ => d.with_note(format!("its parameters are {}", names.join(", "))),
        };
        if let Some(first) = sig.params.first() {
            d = d.with_help(format!(
                "differentiate with respect to a parameter: `derivative({fname}, {})`",
                quote(&first.name)
            ));
        }
        self.error(d);
    }

    /// Parameter `i` of `sig` has no unit annotation; `wrt` is the one
    /// differentiated against.
    #[cold]
    fn derivative_unannotated(&mut self, sig: &FuncSig, i: usize, wrt: usize, at: &Expr) {
        let (fname, pname) = (quote(sig.name()), quote(&sig.params[i].name));
        let d = if i == wrt {
            Diagnostic::error(
                format!(
                    "cannot differentiate `{fname}` with respect to `{pname}`: it has no unit"
                ),
                at.span,
            )
            .with_label(format!("`{pname}` is not annotated"))
            .with_note(format!(
                "the derivative has the unit of the result divided by the unit of `{pname}`, and an unannotated parameter accepts any unit"
            ))
        } else {
            Diagnostic::error(
                format!(
                    "cannot differentiate `{fname}`: its parameter `{pname}` has no unit"
                ),
                at.span,
            )
            .with_label(format!("while differentiating with respect to `{}`", quote(&sig.params[wrt].name)))
            .with_note("the derivative is computed through every parameter, so each needs a unit to type it")
        };
        self.error(
            d.with_secondary(sig.def_span, format!("`{fname}` is defined here"))
                .with_help(format!(
                    "annotate the parameter with its unit, for example `{pname}: s`"
                )),
        );
    }

    #[cold]
    fn derivative_wrt_array(&mut self, sig: &FuncSig, wrt: usize, at: &Expr) {
        let p = &sig.params[wrt];
        let mut d = Diagnostic::error(
            format!(
                "cannot differentiate `{}` with respect to `{}`: it is not a scalar",
                quote(sig.name()),
                quote(&p.name)
            ),
            at.span,
        );
        if let Some(v) = &p.ty {
            d = d.with_label(self.this_is(v));
        }
        if let Some(ann) = p.annotation {
            d = d.with_secondary(ann, "declared here");
        }
        self.error(
            d.with_note(
                "the derivative with respect to an array is a gradient, which comes with arrays at run time",
            )
            .with_help("differentiate with respect to a scalar parameter"),
        );
    }

    #[cold]
    fn derivative_bad_result(&mut self, sig: &FuncSig, ret: &Ty, f_arg: &Expr) {
        let fname = quote(sig.name());
        let (msg, note) = match ret {
            Ty::Value(v) => (
                format!("cannot differentiate `{fname}`: its result is {}, not a number", {
                    let mut shown = v.clone();
                    shown.ty = self.env.shapes.apply_type(&v.ty);
                    shown.to_string()
                }),
                "the derivative of an array-valued function is a Jacobian, which comes with arrays at run time",
            ),
            Ty::Bool => (
                format!("cannot differentiate `{fname}`: it returns a Bool, not a number"),
                "a Bool has no derivative",
            ),
            _ => (
                format!("cannot differentiate `{fname}`: the unit of its result is not known"),
                "the derivative has the unit of the result divided by the unit of the parameter",
            ),
        };
        let mut d = Diagnostic::error(msg, f_arg.span)
            .with_label("this function's result")
            .with_secondary(sig.def_span, format!("`{fname}` is defined here"))
            .with_note(note);
        if matches!(ret, Ty::Unknown | Ty::Func(_)) {
            d = d.with_help("declare the result's unit, for example `def f(t: s) -> m`");
        }
        self.error(d);
    }

    #[cold]
    fn derivative_not_called(&mut self, args: &[Expr], span: Span) {
        let text = match args {
            [f, x] => format!(
                "derivative({}, {})",
                self.quoted(f.span),
                self.quoted(x.span)
            ),
            _ => "derivative(f, x)".to_string(),
        };
        self.error(
            Diagnostic::error(
                format!("`{text}` is a function, and must be called where it is written"),
                span,
            )
            .with_label("not called")
            .with_note("Sky has no function values yet: a function can be called, but not stored or passed around")
            .with_help(format!(
                "pass the arguments of the function: `{text}(...)`, or name the derivative with a `def` that calls it"
            )),
        );
    }
}
