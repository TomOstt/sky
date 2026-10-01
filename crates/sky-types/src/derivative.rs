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
//! # What the derivative is, by the shapes of `x` and `f`
//!
//! | `x` | `f` | `derivative(f, x)` | unit |
//! |---|---|---|---|
//! | scalar | scalar | a scalar: the slope | `dim(f)/dim(x)` |
//! | scalar | `Vector`/`Matrix` | an array of `f`'s shape: the velocity of a position vector | `dim(f)/dim(x)` |
//! | `Vector`/`Matrix` | scalar | an array of `x`'s shape: the **gradient** ∂f/∂x, the force of a potential of a position | `dim(f)/dim(x)` |
//! | `Vector`/`Matrix` | array | not available: that is a Jacobian | |
//!
//! `grad(f)` is the same for a function of *one array parameter*, and for
//! scalar parameters it is the vector of the derivatives with respect to
//! every one of them.
//!
//! What is rejected, each with a message that says how to fix it:
//!
//! * a wrong number of arguments, or a first argument that is not a `def`
//!   (a built-in function, a number, another derivative);
//! * a second argument that is not the name of a parameter of `f`;
//! * a parameter without a unit annotation (its dimension is unknown, so
//!   the derivative's unit cannot be computed; this applies to every
//!   parameter, since the derivative flows through all of them);
//! * an array `x` and an array-valued `f` (a Jacobian), or a result that is
//!   not a number or whose unit is unknown;
//! * a bare `derivative(f, x)` that is not called at once.
//!
//! An argument or a function whose type is already unknown (an earlier
//! error) makes the result unknown without a second error.

use std::sync::Arc;

use sky_ast::{BinaryOp, Expr, ExprKind, UnaryOp};
use sky_diag::{display_capped as quote, Diagnostic, Span};
use sky_units::{Dimension, MathFn};

use crate::check::{describe, strip_parens, Checker, Ty, ValueTy};
use crate::flow::FuncSig;
use crate::rules::TensorType;
use crate::shape::Shape;

/// What the function `derivative(f, x)` differentiates: it is the
/// derivative of [`DerivativeOf::func`] with respect to its parameter
/// number [`DerivativeOf::wrt`] (counted from 0). Lowering reads this to
/// build the derivative function; see [`FuncSig::derivative_of`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DerivativeOf {
    pub(crate) func: Arc<FuncSig>,
    pub(crate) wrt: usize,
    /// `grad(f)`: the derivatives with respect to every parameter, as a
    /// vector, instead of the one at `wrt`.
    pub(crate) gradient: bool,
}

impl DerivativeOf {
    /// The function that is differentiated (the `def`'s own signature).
    pub fn func(&self) -> &FuncSig {
        &self.func
    }

    /// The index of the parameter differentiated against (for a
    /// gradient, see [`DerivativeOf::is_gradient`], the first: it is
    /// differentiated against all of them).
    pub fn wrt(&self) -> usize {
        self.wrt
    }

    /// Whether this is `grad(f)`, the vector of the derivatives with
    /// respect to every parameter of the function, in order, rather than
    /// the derivative with respect to [`DerivativeOf::wrt`].
    pub fn is_gradient(&self) -> bool {
        self.gradient
    }

    /// The parameters differentiated against, in the order the results
    /// come: all of them for a gradient, else just [`DerivativeOf::wrt`].
    pub fn wrts(&self) -> std::ops::Range<usize> {
        if self.gradient {
            0..self.func.arity()
        } else {
            self.wrt..self.wrt + 1
        }
    }
}

/// The names the language provides that are not `def`s, for the message
/// that says a built-in cannot be differentiated.
fn is_builtin_name(name: &str) -> bool {
    MathFn::from_name(name).is_some()
        || matches!(
            name,
            "print" | "inv" | "transpose" | "zeros" | "identity" | "len" | "derivative" | "grad"
        )
}

/// The forms that make a function from a `def`, as they are written where a
/// call's callee is.
enum Form<'e> {
    /// `derivative(f, x)`: the arguments.
    Derivative(&'e [Expr]),
    /// `grad(f)`: the arguments.
    Gradient(&'e [Expr]),
    /// `∂f/∂x`: the two names.
    Partial(&'e Expr, &'e Expr),
}

/// `x` in `∂x`.
fn partial_operand(e: &Expr) -> Option<&Expr> {
    match &e.kind {
        ExprKind::Unary {
            op: UnaryOp::Partial,
            operand,
            ..
        } => Some(operand),
        _ => None,
    }
}

/// What to do with the `def` a help message asks for.
fn then_do(who: &str) -> &'static str {
    if who == "grad" {
        "take the gradient of that"
    } else {
        "differentiate that"
    }
}

impl Checker<'_> {
    /// The type of `callee(args)` when it involves `derivative` or `grad`:
    /// either the call `derivative(f, x)(args...)`, `grad(f)(args...)` or
    /// `(∂f/∂x)(args...)`, or `derivative(f, x)` or `grad(f)` that is not
    /// called at once (an error). `None` leaves any other call to the
    /// ordinary path.
    pub(crate) fn derivative_call(
        &mut self,
        callee: &Expr,
        args: &[Expr],
        span: Span,
    ) -> Option<Ty> {
        let inner = strip_parens(callee);
        let called = match self.callee_form(inner) {
            Some(Form::Derivative(dargs)) => Some(self.derivative_function(dargs, inner.span)),
            Some(Form::Gradient(gargs)) => Some(self.gradient_function(gargs, inner.span)),
            Some(Form::Partial(f, x)) => Some(self.partial_function(inner, f, x)),
            None => None,
        };
        if let Some(t) = called {
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
            let result = self.call(t.clone(), callee, args, span);
            self.uncertain_derivative_args(&t, callee, args);
            return Some(result);
        }
        // `derivative(...)` or `grad(...)` written without a call after it.
        let (t, text) = if self.is_derivative(callee) {
            let t = self.derivative_function(args, span);
            let text = match args {
                [f, x] => format!(
                    "derivative({}, {})",
                    self.quoted(f.span),
                    self.quoted(x.span)
                ),
                _ => "derivative(f, x)".to_string(),
            };
            (t, text)
        } else if self.is_grad(callee) {
            let t = self.gradient_function(args, span);
            let text = match args {
                [f] => format!("grad({})", self.quoted(f.span)),
                _ => "grad(f)".to_string(),
            };
            (t, text)
        } else {
            return None;
        };
        if matches!(t, Ty::Func(_)) {
            self.derivative_not_called(&text, span);
        }
        Some(Ty::Unknown)
    }

    /// `∂f/∂t` as a value, not called: the checker reports it (`None` when
    /// `e` is not that shape). It is the same form as `derivative(f, t)`,
    /// so its names are checked the same way first.
    pub(crate) fn partial_not_called(&mut self, e: &Expr) -> Option<Ty> {
        let (f, x) = e.partial_quotient()?;
        let t = self.partial_function(e, f, x);
        if matches!(t, Ty::Func(_)) {
            let text = self.quoted(e.span);
            self.derivative_not_called(&text, e.span);
        }
        Some(Ty::Unknown)
    }

    /// `∂f/∂t(2 s)` written without parentheses: a call binds tighter than
    /// `∂`, so it reads `∂f / ∂(t(2 s))`, which is not the derivative. Reports
    /// it and says whether `e` was that shape.
    pub(crate) fn partial_misparsed(&mut self, e: &Expr) -> bool {
        let ExprKind::Binary {
            op: BinaryOp::Div,
            lhs,
            rhs,
            ..
        } = &e.kind
        else {
            return false;
        };
        let (Some(f), Some(call)) = (partial_operand(lhs), partial_operand(rhs)) else {
            return false;
        };
        let (ExprKind::Name(_), ExprKind::Call { callee, args }) = (&f.kind, &call.kind) else {
            return false;
        };
        if !matches!(callee.kind, ExprKind::Name(_)) {
            return false;
        }
        let (fname, tname) = (self.quoted(f.span), self.quoted(callee.span));
        let shown = self
            .text(call.span)
            .strip_prefix(self.text(callee.span))
            .unwrap_or("(...)")
            .to_string();
        for a in args {
            self.expr(a);
        }
        self.expr_types.insert(lhs.span, Ty::Unknown);
        self.expr_types.insert(rhs.span, Ty::Unknown);
        self.expr_types.insert(f.span, Ty::Unknown);
        self.expr_types.insert(call.span, Ty::Unknown);
        self.expr_types.insert(callee.span, Ty::Unknown);
        self.error(
            Diagnostic::error(
                format!("`∂{fname}/∂{tname}{shown}` is not a derivative: it divides `∂{fname}` by `∂` of a call"),
                e.span,
            )
            .with_label(format!("this is `∂{tname}{shown}`"))
            .with_note(format!(
                "a call binds tighter than `∂`, so the arguments belong to `{tname}`"
            ))
            .with_help(format!("call the derivative: `(∂{fname}/∂{tname}){shown}`")),
        );
        true
    }

    /// Which form `callee` is, if any. `derivative` and `grad` are
    /// recognised by name while the program has not bound the name itself.
    fn callee_form<'e>(&self, callee: &'e Expr) -> Option<Form<'e>> {
        if let ExprKind::Call { callee: name, args } = &callee.kind {
            if self.is_derivative(name) {
                return Some(Form::Derivative(args));
            }
            if self.is_grad(name) {
                return Some(Form::Gradient(args));
            }
        }
        let (f, x) = callee.partial_quotient()?;
        Some(Form::Partial(f, x))
    }

    /// `derivative` while the program has not bound that name itself.
    fn is_derivative(&self, callee: &Expr) -> bool {
        matches!(&callee.kind, ExprKind::Name(id)
            if self.ident(id) == "derivative" && self.env.get("derivative").is_none())
    }

    /// `grad` while the program has not bound that name itself.
    fn is_grad(&self, callee: &Expr) -> bool {
        matches!(&callee.kind, ExprKind::Name(id)
            if self.ident(id) == "grad" && self.env.get("grad").is_none())
    }

    /// `∂f/∂t`, the function `derivative(f, t)`; `e` is the quotient, `f`
    /// and `t` its names. The parameter name is not a value (as in
    /// `derivative`), and the two `∂` nodes are not values either.
    fn partial_function(&mut self, e: &Expr, f: &Expr, x: &Expr) -> Ty {
        if let ExprKind::Binary { lhs, rhs, .. } = &e.kind {
            for op in [lhs, rhs] {
                self.expr_types.insert(op.span, Ty::Unknown);
            }
        }
        self.derivative_function_as("∂", &[f.clone(), x.clone()], e.span)
    }

    /// The type of the function `derivative(args...)`: [`Ty::Func`] when
    /// the arguments are valid, otherwise [`Ty::Unknown`] after reporting.
    fn derivative_function(&mut self, args: &[Expr], span: Span) -> Ty {
        self.derivative_function_as("derivative", args, span)
    }

    /// [`Checker::derivative_function`] for a form that is written `who`
    /// (`derivative`, or `∂` for `∂f/∂t`), which its messages call it.
    fn derivative_function_as(&mut self, who: &str, args: &[Expr], span: Span) -> Ty {
        let [f_arg, x_arg] = args else {
            self.derivative_arity(args, span);
            for (i, a) in args.iter().enumerate() {
                self.derivative_argument(a, i == 0);
            }
            return Ty::Unknown;
        };
        self.derivative_argument(x_arg, false);
        let Some(sig) = self.derivative_target(f_arg, who) else {
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
                Some(v) if i == wrt && v.ty.shape().rank() > 2 => {
                    self.derivative_wrt_tensor(&sig, wrt, x_arg);
                    ok = false;
                }
                Some(_) => {}
            }
        }
        if !ok {
            return Ty::Unknown;
        }
        let Some(x_ty) = sig.params[wrt].ty.clone() else {
            return Ty::Unknown;
        };
        // The derivative has the shape of whichever of `x` and the result
        // is the array (the gradient of a scalar has the shape of `x`, the
        // rate of change of an array with a scalar has the array's), and
        // the unit of the result over the unit of `x`.
        let shape = match &sig.ret {
            Ty::Value(v) if x_ty.is_scalar() => v.ty.shape().clone(),
            Ty::Value(v) if v.is_scalar() => x_ty.ty.shape().clone(),
            Ty::Value(_) => {
                self.derivative_jacobian(&sig, &sig.params[wrt].name, f_arg);
                return Ty::Unknown;
            }
            other => {
                self.derivative_bad_result(&sig, other, f_arg);
                return Ty::Unknown;
            }
        };
        let Ty::Value(ret) = &sig.ret else {
            return Ty::Unknown;
        };
        let dim = match ret.dim.checked_div(x_ty.dim) {
            Ok(d) => d,
            Err(_) => {
                self.error(Diagnostic::error(
                    "the dimension of the derivative is too large",
                    span,
                ));
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
            ret: Ty::value(dim, TensorType::untagged(shape)),
            generic: sig.generic.clone(),
            derivative: Some(Arc::new(DerivativeOf {
                func: sig.clone(),
                wrt,
                gradient: false,
            })),
        }))
    }

    /// The type of the function `grad(args...)`: the gradient of a scalar
    /// function, a [`Ty::Func`] with the function's parameters. Either
    ///
    /// * every parameter is a scalar and they all have one dimension: the
    ///   result is a `Vector[n]` of `dim(result) / dim(parameter)`, one entry
    ///   per parameter in order; or
    /// * there is one parameter, and it is a `Vector` or `Matrix`: the
    ///   result has its shape and the unit `dim(result) / dim(parameter)`
    ///   (the force of a potential of a position).
    ///
    /// Otherwise [`Ty::Unknown`] after reporting.
    fn gradient_function(&mut self, args: &[Expr], span: Span) -> Ty {
        let [f_arg] = args else {
            self.gradient_arity(args, span);
            for (i, a) in args.iter().enumerate() {
                self.derivative_argument(a, i == 0);
            }
            return Ty::Unknown;
        };
        let Some(sig) = self.derivative_target(f_arg, "grad") else {
            return Ty::Unknown;
        };
        if sig.params.is_empty() {
            self.gradient_no_params(&sig, f_arg);
            return Ty::Unknown;
        }
        for (i, p) in sig.params.iter().enumerate() {
            match &p.ty {
                None if p.annotation.is_some() => return Ty::Unknown, // reported already
                None => {
                    self.gradient_unannotated(&sig, i, f_arg);
                    return Ty::Unknown;
                }
                Some(_) => {}
            }
        }
        let arrays: Vec<usize> = sig
            .params
            .iter()
            .enumerate()
            .filter(|(_, p)| p.ty.as_ref().is_some_and(|v| !v.is_scalar()))
            .map(|(i, _)| i)
            .collect();
        if let Some(&first) = arrays.first() {
            if sig.params.len() > 1 {
                self.gradient_array_among_others(&sig, first, f_arg);
                return Ty::Unknown;
            }
            return self.gradient_of_array(&sig, f_arg, span);
        }
        let Some(x_dim) = sig.params[0].ty.as_ref().map(|v| v.dim) else {
            return Ty::Unknown;
        };
        let (ret_dim, mixed) = match &sig.ret {
            Ty::Value(v) if v.is_scalar() => (
                v.dim,
                sig.params
                    .iter()
                    .any(|p| p.ty.as_ref().is_some_and(|t| t.dim != x_dim)),
            ),
            Ty::Value(_) => {
                self.gradient_jacobian(&sig, f_arg);
                return Ty::Unknown;
            }
            other => {
                self.derivative_bad_result(&sig, other, f_arg);
                return Ty::Unknown;
            }
        };
        if mixed {
            self.gradient_mixed_units(&sig, ret_dim, f_arg);
            return Ty::Unknown;
        }
        let Ok(dim) = ret_dim.checked_div(x_dim) else {
            self.error(Diagnostic::error(
                "the dimension of the gradient is too large",
                span,
            ));
            return Ty::Unknown;
        };
        let shape = Shape::vector(sig.params.len() as u64);
        Ty::Func(Arc::new(FuncSig {
            name: format!("grad({})", quote(sig.name())),
            def_span: sig.def_span,
            params: sig.params.clone(),
            ret: Ty::value(dim, TensorType::untagged(shape)),
            generic: sig.generic.clone(),
            derivative: Some(Arc::new(DerivativeOf {
                func: sig.clone(),
                wrt: 0,
                gradient: true,
            })),
        }))
    }

    /// `grad(f)` for a `def` with one array parameter `p`: the gradient
    /// with respect to it, an array of its shape. It is the derivative with
    /// respect to `p` (see [`Checker::derivative_function_as`]).
    fn gradient_of_array(&mut self, sig: &Arc<FuncSig>, f_arg: &Expr, span: Span) -> Ty {
        let Some(x_ty) = sig.params[0].ty.clone() else {
            return Ty::Unknown;
        };
        if x_ty.ty.shape().rank() > 2 {
            self.derivative_wrt_tensor(sig, 0, f_arg);
            return Ty::Unknown;
        }
        let ret = match &sig.ret {
            Ty::Value(v) if v.is_scalar() => v.dim,
            Ty::Value(_) => {
                self.derivative_jacobian(sig, &sig.params[0].name, f_arg);
                return Ty::Unknown;
            }
            other => {
                self.derivative_bad_result(sig, other, f_arg);
                return Ty::Unknown;
            }
        };
        let Ok(dim) = ret.checked_div(x_ty.dim) else {
            self.error(Diagnostic::error(
                "the dimension of the gradient is too large",
                span,
            ));
            return Ty::Unknown;
        };
        Ty::Func(Arc::new(FuncSig {
            name: format!("grad({})", quote(sig.name())),
            def_span: sig.def_span,
            params: sig.params.clone(),
            ret: Ty::value(dim, TensorType::untagged(x_ty.ty.shape().clone())),
            generic: sig.generic.clone(),
            derivative: Some(Arc::new(DerivativeOf {
                func: sig.clone(),
                wrt: 0,
                gradient: false,
            })),
        }))
    }

    #[cold]
    fn gradient_arity(&mut self, args: &[Expr], span: Span) {
        let n = args.len();
        let mut d = Diagnostic::error(
            format!("`grad` takes 1 argument, found {n}: a function"),
            span,
        )
        .with_label(if n < 1 {
            "missing argument"
        } else {
            "extra argument"
        });
        d = if n < 1 {
            d.with_help("write `grad(f)(...)`, then the arguments of `f`")
        } else {
            d.with_note("the gradient is with respect to every parameter, so no parameter is named")
                .with_help(
                    "write `grad(f)(...)` for all of them, or `derivative(f, x)(...)` for one parameter",
                )
        };
        self.error(d);
    }

    #[cold]
    fn gradient_no_params(&mut self, sig: &FuncSig, f_arg: &Expr) {
        let fname = quote(sig.name());
        self.error(
            Diagnostic::error(
                format!("cannot take the gradient of `{fname}`: it has no parameters"),
                f_arg.span,
            )
            .with_label("nothing to differentiate with respect to")
            .with_secondary(sig.def_span, format!("`{fname}` is defined here"))
            .with_help("give the function the parameters the gradient is taken with respect to"),
        );
    }

    /// Parameter `i` of `sig` has no unit annotation.
    #[cold]
    fn gradient_unannotated(&mut self, sig: &FuncSig, i: usize, f_arg: &Expr) {
        let (fname, pname) = (quote(sig.name()), quote(&sig.params[i].name));
        self.error(
            Diagnostic::error(
                format!("cannot take the gradient of `{fname}`: its parameter `{pname}` has no unit"),
                f_arg.span,
            )
            .with_label("the gradient is with respect to every parameter")
            .with_secondary(sig.def_span, format!("`{fname}` is defined here"))
            .with_note(
                "each derivative has the unit of the result divided by the unit of its parameter, and an unannotated parameter accepts any unit"
            )
            .with_help(format!(
                "annotate the parameter with its unit, for example `{pname}: m`"
            )),
        );
    }

    /// `grad` of a function with an array parameter and others.
    #[cold]
    fn gradient_array_among_others(&mut self, sig: &FuncSig, i: usize, f_arg: &Expr) {
        let (fname, pname) = (quote(sig.name()), quote(&sig.params[i].name));
        let v = sig.params[i].ty.clone();
        let mut d = Diagnostic::error(
            format!(
                "cannot take the gradient of `{fname}`: its array parameter `{pname}` is not its only parameter"
            ),
            f_arg.span,
        )
        .with_label("the gradient is with respect to every parameter");
        if let (Some(ann), Some(v)) = (sig.params[i].annotation, &v) {
            d = d.with_secondary(ann, self.this_is(v));
        }
        self.error(
            d.with_note(
                "`grad` is the gradient with respect to one array parameter, or the vector of the derivatives with respect to scalar parameters; an array among other parameters has no single unit-consistent vector to gather them in",
            )
            .with_help(format!(
                "name the array parameter: `derivative({fname}, {pname})(...)`"
            )),
        );
    }

    /// `grad` of an array-valued function.
    #[cold]
    fn gradient_jacobian(&mut self, sig: &FuncSig, f_arg: &Expr) {
        let fname = quote(sig.name());
        let mut d = Diagnostic::error(
            format!("cannot take the gradient of `{fname}`: its result is not a scalar"),
            f_arg.span,
        )
        .with_label("this function's result")
        .with_secondary(sig.def_span, format!("`{fname}` is defined here"));
        if let Ty::Value(v) = &sig.ret {
            d = d.with_note(format!(
                "the result is {}; the derivative of an array-valued function with respect to several variables is a Jacobian, which Sky does not have yet",
                self.shown(v)
            ));
        }
        self.error(d.with_help(
            "take the gradient of a scalar function, for example one component of the result, or use `derivative(f, t)` for the rate of change of an array with one scalar",
        ));
    }

    /// The parameters of `sig` do not all have one dimension, so the
    /// derivatives have different units and no vector holds them.
    #[cold]
    fn gradient_mixed_units(&mut self, sig: &FuncSig, ret: Dimension, f_arg: &Expr) {
        let fname = quote(sig.name());
        let mut d = Diagnostic::error(
            format!("cannot take the gradient of `{fname}`: its parameters have different units"),
            f_arg.span,
        )
        .with_label("the gradient is with respect to every parameter");
        let mut kinds: Vec<String> = Vec::new();
        for p in &sig.params {
            let Some(v) = &p.ty else { continue };
            if let Some(ann) = p.annotation {
                d = d.with_secondary(ann, format!("`{}` is {}", quote(&p.name), describe(&v.dim)));
            }
            if let Ok(q) = ret.checked_div(v.dim) {
                let q = describe(&q);
                if !kinds.contains(&q) {
                    kinds.push(q);
                }
            }
        }
        let list = match kinds.as_slice() {
            [] | [_] => String::new(),
            [a, b] => format!(": {a} and {b}"),
            [a, b, ..] => format!(": {a}, {b}, ..."),
        };
        d = d.with_note(format!(
            "the gradient is a Vector, which has one unit for all its entries, but the derivatives of `{fname}` are different quantities{list}"
        ));
        let one_by_one: Vec<String> = sig
            .params
            .iter()
            .map(|p| format!("`derivative({fname}, {})(...)`", quote(&p.name)))
            .collect();
        self.error(d.with_help(format!(
            "take one derivative per parameter instead: {}",
            one_by_one.join(", ")
        )));
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
    fn derivative_target(&mut self, f_arg: &Expr, who: &str) -> Option<Arc<FuncSig>> {
        let inner = strip_parens(f_arg);
        if let ExprKind::Call { callee, .. } = &inner.kind {
            let another = if self.is_derivative(callee) {
                Some(("derivative", "another derivative"))
            } else if self.is_grad(callee) {
                Some(("grad", "a gradient"))
            } else {
                None
            };
            if let Some((form, found)) = another {
                if let ExprKind::Call { args, .. } = &inner.kind {
                    for (i, a) in args.iter().enumerate() {
                        self.derivative_argument(a, i == 0);
                    }
                }
                self.expr_types.insert(inner.span, Ty::Unknown);
                self.expr_types.insert(f_arg.span, Ty::Unknown);
                self.error(
                    Diagnostic::error(
                        format!("`{who}` needs a function defined with `def`, found {found}"),
                        f_arg.span,
                    )
                    .with_label(format!("this is `{form}(...)`, which can only be called"))
                    .with_help(format!(
                        "name the first {} with a `def` that calls it, and {}",
                        if form == "grad" {
                            "gradient"
                        } else {
                            "derivative"
                        },
                        then_do(who)
                    )),
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
                            "`{who}` needs a function defined with `def`, found the built-in `{}`",
                            quote(name)
                        ),
                        f_arg.span,
                    )
                    .with_label("a built-in function")
                    .with_note(
                        "a `def` states the units of its parameters and result, which the unit of the derivative is computed from",
                    )
                    .with_help(format!(
                        "wrap it in a `def` that states its units, and {}",
                        then_do(who)
                    )),
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
                let mut d = Diagnostic::error(
                    format!("`{who}` needs a function defined with `def`"),
                    f_arg.span,
                )
                .with_label(what);
                if who == "∂" {
                    d = d.with_note(
                        "`∂f/∂t` is the derivative of a function, not of a quantity that depends on another: Sky has no dependent variables",
                    );
                }
                self.error(d.with_help(format!(
                    "define the function with `def f(x: unit) -> unit: ...` and {}",
                    then_do(who)
                )));
                None
            }
        }
    }

    /// `Vector[3] [L]`: the type with its shape variables resolved.
    fn shown(&self, v: &ValueTy) -> String {
        ValueTy {
            dim: v.dim,
            ty: self.env.shapes.apply_type(&v.ty),
        }
        .to_string()
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

    /// The parameter differentiated against has more than two axes.
    #[cold]
    fn derivative_wrt_tensor(&mut self, sig: &FuncSig, wrt: usize, at: &Expr) {
        let p = &sig.params[wrt];
        let mut d = Diagnostic::error(
            format!(
                "cannot differentiate `{}` with respect to `{}`: it has more than two axes",
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
            d.with_note("a derivative is with respect to a scalar, a vector or a matrix")
                .with_help("differentiate with respect to a vector or matrix parameter"),
        );
    }

    /// An array-valued function, differentiated with respect to an array.
    #[cold]
    fn derivative_jacobian(&mut self, sig: &FuncSig, pname: &str, f_arg: &Expr) {
        let fname = quote(sig.name());
        let mut d = Diagnostic::error(
            format!(
                "cannot differentiate `{fname}` with respect to `{}`: the derivative of an array with respect to an array is a Jacobian",
                quote(pname)
            ),
            f_arg.span,
        )
        .with_label("this function's result")
        .with_secondary(sig.def_span, format!("`{fname}` is defined here"));
        if let Ty::Value(v) = &sig.ret {
            d = d.with_note(format!(
                "the result is {}, so the derivative would have an entry for every result and every entry of `{}`, and Sky does not have Jacobians yet",
                self.shown(v),
                quote(pname)
            ));
        }
        self.error(d.with_help(format!(
            "differentiate a scalar result (a component, a norm, an energy) with respect to `{}`, or an array result with respect to a scalar",
            quote(pname)
        )));
    }

    #[cold]
    fn derivative_bad_result(&mut self, sig: &FuncSig, ret: &Ty, f_arg: &Expr) {
        let fname = quote(sig.name());
        let (msg, note) = match ret {
            Ty::Value(v) => (
                format!("cannot differentiate `{fname}`: its result is {}", self.shown(v)),
                "the derivative of an array-valued function with respect to an array is a Jacobian, which Sky does not have yet",
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
    fn derivative_not_called(&mut self, text: &str, span: Span) {
        let (what, call) = if text.starts_with('∂') {
            ("derivative", format!("({text})(...)"))
        } else if text.starts_with("grad") {
            ("gradient", format!("{text}(...)"))
        } else {
            ("derivative", format!("{text}(...)"))
        };
        self.error(
            Diagnostic::error(
                format!("`{text}` is a function, and must be called where it is written"),
                span,
            )
            .with_label("not called")
            .with_note("Sky has no function values yet: a function can be called, but not stored or passed around")
            .with_help(format!(
                "pass the arguments of the function: `{call}`, or name the {what} with a `def` that calls it"
            )),
        );
    }
}
