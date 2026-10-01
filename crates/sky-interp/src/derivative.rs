//! `derivative(f, x)(args)` in the tree-walking interpreter.
//!
//! The interpreter has no automatic differentiation, and does not grow a
//! second implementation of the rules: differentiation is a program
//! transformation on SIR (`sky-ad`), so the interpreter hands the call to a
//! [`DerivativeHook`], which lowers the function to SIR, differentiates it
//! and runs the result with the reference interpreter. `sky run` installs
//! one (the `sky-derive` crate); a run without one reports that it cannot
//! evaluate derivatives. The hook lives behind a trait because SIR lowering
//! itself depends on this crate.
//!
//! What the interpreter does itself is what `derivative(f, x)(args)` means
//! in the AST: find the function value `f`, the position of the parameter
//! `x` in it, evaluate the arguments in the caller's scope (left to right,
//! as for a call of `f`), and give the hook the arguments and a way to read
//! the module-level names `f` may read. The value it returns is the
//! derivative's, with its dimension.

use std::io::Write;
use std::sync::Arc;

use sky_ast::{Expr, ExprKind};
use sky_diag::Span;

use crate::arith::{arity_mismatch, StackMeter, R};
use crate::eval::Interp;
use crate::value::Function;
use crate::{RuntimeError, Value};

/// What a [`DerivativeHook`] is asked for: the value of the derivative of
/// `func` with respect to its parameter number `wrt`, at `args`.
pub struct DerivativeRequest<'a> {
    /// The function differentiated: a `def` as the interpreter holds it.
    /// [`Function::def_span`] says which `def` of the program it is.
    pub func: &'a Function,
    /// The index of the parameter differentiated against.
    pub wrt: usize,
    /// One value per parameter of `func`, evaluated in the caller.
    pub args: Vec<Value>,
    /// The whole call expression, for an error.
    pub span: Span,
    /// The value a module-level name has now, for the names `func` reads
    /// when it runs (a callee sees the module, never its caller's locals).
    pub globals: &'a dyn Fn(&str) -> Option<Value>,
    /// The stack the interpreter has used so far, so the hook can keep
    /// the run inside the same budget.
    pub stack: StackMeter,
}

/// The engine behind `derivative(f, x)(args)`; see the module docs.
pub trait DerivativeHook {
    /// The derivative asked for, as a number with its dimension
    /// (`dim(f's result) / dim(x)`). `print` output the function makes
    /// while it runs goes to `out`, once, as in a call of `f`.
    fn derivative(
        &mut self,
        request: &DerivativeRequest<'_>,
        out: &mut dyn Write,
    ) -> Result<Value, RuntimeError>;
}

impl Interp<'_> {
    /// Evaluates the call `callee(args)` if it involves `derivative`:
    /// `derivative(f, x)(args...)`, or a `derivative(f, x)` that is not
    /// called at once (the checker rejects that; here it is an error too).
    /// `None` leaves any other call to the ordinary path.
    pub(crate) fn derivative_call(
        &mut self,
        call: &Expr,
        callee: &Expr,
        args: &[Expr],
    ) -> Option<R<Value>> {
        let mut inner = callee;
        while let ExprKind::Paren(next) = &inner.kind {
            inner = next;
        }
        if let ExprKind::Call {
            callee: name,
            args: dargs,
        } = &inner.kind
        {
            if self.is_derivative(name) {
                return Some(self.derivative_of(call, inner, dargs, args));
            }
            if self.is_grad(name) {
                return Some(self.gradient_of(call, inner, dargs, args));
            }
        }
        // `∂f/∂x`, the same as `derivative(f, x)`.
        if let Some((f, x)) = inner.partial_quotient() {
            return Some(self.derivative_of(call, inner, &[f.clone(), x.clone()], args));
        }
        for (is_form, form) in [
            (self.is_derivative(callee), "derivative(f, x)"),
            (self.is_grad(callee), "grad(f)"),
        ] {
            if is_form {
                return Some(Err(RuntimeError::new(
                    format!("`{form}` is a function, and must be called where it is written"),
                    call.span,
                )));
            }
        }
        None
    }

    /// `grad` while the program has not bound that name itself.
    fn is_grad(&self, callee: &Expr) -> bool {
        matches!(&callee.kind, ExprKind::Name(id)
            if self.text(id.span) == "grad" && self.lookup("grad").is_none())
    }

    /// `derivative` while the program has not bound that name itself.
    fn is_derivative(&self, callee: &Expr) -> bool {
        matches!(&callee.kind, ExprKind::Name(id)
            if self.text(id.span) == "derivative" && !self.is_user_bound("derivative"))
    }

    /// The function a `derivative` or `grad` argument names.
    fn derivative_function(&mut self, who: &str, f_arg: &Expr) -> R<Arc<Function>> {
        // A name in a function body is the function in scope where that
        // function was defined, as for a call (early binding).
        let named = match &f_arg.kind {
            ExprKind::Name(id) => self.lexical_function(self.text(id.span)).map(Value::Func),
            _ => None,
        };
        match named.map_or_else(|| self.eval(f_arg), Ok)? {
            Value::Func(f) => Ok(f),
            other => Err(RuntimeError::new(
                format!(
                    "`{who}` needs a function defined with `def`, found {}",
                    other.kind()
                ),
                f_arg.span,
            )),
        }
    }

    /// `derivative(dargs...)(args...)`, where `form` is the expression
    /// `derivative(dargs...)`.
    fn derivative_of(
        &mut self,
        call: &Expr,
        form: &Expr,
        dargs: &[Expr],
        args: &[Expr],
    ) -> R<Value> {
        let [f_arg, x_arg] = dargs else {
            return Err(RuntimeError::new(
                format!("`derivative` takes 2 arguments, found {}", dargs.len()),
                form.span,
            ));
        };
        let func = self.derivative_function("derivative", f_arg)?;
        let wrt = match &x_arg.kind {
            ExprKind::Name(id) => {
                let name = self.text(id.span);
                func.params.iter().position(|p| p == name).ok_or_else(|| {
                    RuntimeError::new(
                        format!("`{}` has no parameter `{name}`", func.name),
                        x_arg.span,
                    )
                })?
            }
            _ => {
                return Err(RuntimeError::new(
                    "the second argument of `derivative` is the name of a parameter",
                    x_arg.span,
                ));
            }
        };
        let label = format!("derivative({}, {})", func.name, func.params[wrt]);
        let values = self.derivative_arguments(&label, &func, call, args)?;
        let mut derivatives = self.derivatives_at(call, &func, &[wrt], values)?;
        Ok(derivatives.remove(0))
    }

    /// `grad(dargs...)(args...)`: the derivatives of the function with
    /// respect to each of its parameters, in order, as a vector. The
    /// arguments are evaluated once, and the function runs once per
    /// parameter (forward mode: one derivative each).
    ///
    /// A function of one array parameter has one derivative, its gradient
    /// (an array of the parameter's shape), which is the value: the checker
    /// types `grad(f)` for such a function as `derivative(f, p)`.
    fn gradient_of(&mut self, call: &Expr, form: &Expr, dargs: &[Expr], args: &[Expr]) -> R<Value> {
        let [f_arg] = dargs else {
            return Err(RuntimeError::new(
                format!("`grad` takes 1 argument, found {}", dargs.len()),
                form.span,
            ));
        };
        let func = self.derivative_function("grad", f_arg)?;
        let label = format!("grad({})", func.name);
        let values = self.derivative_arguments(&label, &func, call, args)?;
        let wrts: Vec<usize> = (0..func.params.len()).collect();
        let mut derivatives = self.derivatives_at(call, &func, &wrts, values)?;
        if let [Value::Array { .. }] = derivatives[..] {
            return Ok(derivatives.remove(0));
        }
        // The checker made sure the parameters share one dimension, so the
        // derivatives do; the vector has that one.
        let mut data = Vec::with_capacity(derivatives.len());
        let mut dim = None;
        for d in &derivatives {
            match d {
                Value::Number { value, dim: d } if dim.is_none_or(|x| x == *d) => {
                    data.push(*value);
                    dim = Some(*d);
                }
                _ => {
                    return Err(RuntimeError::new(
                        "the derivatives of a gradient must be numbers of one dimension",
                        call.span,
                    ));
                }
            }
        }
        let Some(dim) = dim else {
            return Err(RuntimeError::new(
                "cannot take the gradient of a function without parameters",
                call.span,
            ));
        };
        Ok(Value::array(vec![data.len()], data, dim))
    }

    /// The arguments of a call of the derivative function `label` of
    /// `func`, evaluated left to right in the caller.
    fn derivative_arguments(
        &mut self,
        label: &str,
        func: &Function,
        call: &Expr,
        args: &[Expr],
    ) -> R<Vec<Value>> {
        if args.len() != func.params.len() {
            return Err(arity_mismatch(
                label,
                func.params.len(),
                args.len(),
                call.span,
            ));
        }
        let mut values = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            let v = self.eval(a)?;
            self.check_parameter_tags(func, i, &v, a.span)?;
            values.push(v);
        }
        Ok(values)
    }

    /// The derivative of `func` at `values` with respect to each parameter
    /// in `wrts`, in that order, from the hook.
    fn derivatives_at(
        &mut self,
        call: &Expr,
        func: &Function,
        wrts: &[usize],
        values: Vec<Value>,
    ) -> R<Vec<Value>> {
        // Disjoint borrows: the hook, the output, and the module names.
        let Interp {
            env,
            out,
            derivatives,
            stack,
            ..
        } = self;
        let Some(hook) = derivatives.as_deref_mut() else {
            return Err(
                RuntimeError::new("this run cannot evaluate derivatives", call.span).with_note(
                    "`derivative(f, x)` needs the SIR pipeline, which `sky run` provides",
                ),
            );
        };
        let env = &**env;
        let globals = |name: &str| env.get(name).cloned();
        wrts.iter()
            .map(|&wrt| {
                hook.derivative(
                    &DerivativeRequest {
                        func,
                        wrt,
                        args: values.clone(),
                        span: call.span,
                        globals: &globals,
                        stack: *stack,
                    },
                    &mut **out,
                )
            })
            .collect()
    }
}
