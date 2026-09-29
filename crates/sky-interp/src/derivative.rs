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
        }
        if self.is_derivative(callee) {
            return Some(Err(RuntimeError::new(
                "`derivative(f, x)` is a function, and must be called where it is written",
                call.span,
            )));
        }
        None
    }

    /// `derivative` while the program has not bound that name itself.
    fn is_derivative(&self, callee: &Expr) -> bool {
        matches!(&callee.kind, ExprKind::Name(id)
            if self.text(id.span) == "derivative" && self.lookup("derivative").is_none())
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
        let func = match self.eval(f_arg)? {
            Value::Func(f) => f,
            other => {
                return Err(RuntimeError::new(
                    format!(
                        "`derivative` needs a function defined with `def`, found {}",
                        other.kind()
                    ),
                    f_arg.span,
                ));
            }
        };
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
        if args.len() != func.params.len() {
            return Err(arity_mismatch(
                &format!("derivative({}, {})", func.name, func.params[wrt]),
                func.params.len(),
                args.len(),
                call.span,
            ));
        }
        let values = args
            .iter()
            .map(|a| self.eval(a))
            .collect::<R<Vec<Value>>>()?;
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
        hook.derivative(
            &DerivativeRequest {
                func: &func,
                wrt,
                args: values,
                span: call.span,
                globals: &globals,
                stack: *stack,
            },
            &mut **out,
        )
    }
}
