//! Typing of the vector builtins `norm(x)` and `dot(a, b)`.
//!
//! * `norm(x)` is the length of a vector (the Euclidean norm; of a matrix,
//!   the Frobenius norm; of a scalar, its absolute value). Its dimension is
//!   the dimension of `x`: the length of a velocity is a speed.
//! * `dot(a, b)` is the inner product of two vectors of one length. Its
//!   dimension is the product of theirs (`dot(F, d)` is a force times a
//!   length, an energy).
//!
//! Both are ordinary builtins: a program that binds the name itself wins,
//! as with `len` and `print`. The interpreter's twin is `sky_interp::vecfn`.

use std::sync::Arc;

use sky_ast::{Expr, ExprKind};
use sky_diag::{Diagnostic, Span};

use crate::check::{describe, Checker, Ty, ValueTy};
use crate::rules::elementwise;
use crate::shape::ShapeContext;

impl Checker<'_> {
    /// The type of `callee(args)` if `callee` is the builtin `norm` or
    /// `dot`; `None` leaves the call to the ordinary path.
    pub(crate) fn vector_call(&mut self, callee: &Expr, args: &[Expr], call: Span) -> Option<Ty> {
        let ExprKind::Name(id) = &callee.kind else {
            return None;
        };
        let name = self.ident(id);
        let arity = match name {
            "norm" => 1,
            "dot" => 2,
            _ => return None,
        };
        if self.env.get(name).is_some() {
            return None;
        }
        let tys: Vec<Ty> = args.iter().map(|a| self.expr(a)).collect();
        if args.len() != arity {
            let plural = if arity == 1 { "" } else { "s" };
            let help = if arity == 1 {
                "call it as `norm(v)`"
            } else {
                "call it as `dot(a, b)`"
            };
            self.error(
                Diagnostic::error(
                    format!(
                        "`{name}` takes {arity} argument{plural}, found {}",
                        args.len()
                    ),
                    call,
                )
                .with_label("wrong number of arguments")
                .with_help(help),
            );
            return Some(Ty::Unknown);
        }
        // Every argument is inspected first, so one bad call reports once
        // per bad argument and never cascades.
        let mut values = Vec::with_capacity(arity);
        let mut bad = false;
        for (a, t) in args.iter().zip(&tys) {
            match t {
                Ty::Value(v) => values.push(v.clone()),
                Ty::Unknown => bad = true,
                other => {
                    self.error(
                        Diagnostic::error(
                            format!("`{name}` needs a vector, found {}", self.kind_of(other)),
                            a.span,
                        )
                        .with_label(format!("this is {}", self.kind_of(other))),
                    );
                    bad = true;
                }
            }
        }
        if bad {
            return Some(Ty::Unknown);
        }
        Some(if name == "norm" {
            Ty::scalar(values[0].dim)
        } else {
            self.dot_type(args, &values, call)
        })
    }

    /// Wording for a non-number in an error.
    fn kind_of(&self, t: &Ty) -> String {
        match t {
            Ty::Bool => "Bool".to_string(),
            other => other.to_string(),
        }
    }

    fn dot_type(&mut self, args: &[Expr], values: &[ValueTy], call: Span) -> Ty {
        let (a, b) = (&values[0], &values[1]);
        for (arg, v) in args.iter().zip(values) {
            if v.ty.shape().rank() != 1 {
                self.error(
                    Diagnostic::error(
                        format!(
                            "`dot` needs two vectors, found {}",
                            if v.is_scalar() {
                                "a scalar".to_string()
                            } else {
                                format!("{}", self.env.shapes.apply_type(&v.ty))
                            }
                        ),
                        arg.span,
                    )
                    .with_label(self.this_is(v))
                    .with_help(if v.is_scalar() {
                        "multiply a scalar with `*`"
                    } else {
                        "for matrices, use `@`"
                    }),
                );
                return Ty::Unknown;
            }
        }
        self.set_origin(call, "`dot`".to_string());
        let unified = elementwise(&mut self.env.shapes, &a.ty, &b.ty);
        self.env.shapes.set_origin(None);
        if let Err(e) = unified {
            let e = e.in_context(ShapeContext::Operator(Arc::from("dot")));
            let (l, r) = (self.this_is(a), self.this_is(b));
            self.error(
                self.shape_diagnostic(&e, call)
                    .with_secondary(args[0].span, l)
                    .with_secondary(args[1].span, r),
            );
            return Ty::Unknown;
        }
        match a.dim.checked_mul(b.dim) {
            Ok(dim) => Ty::scalar(dim),
            Err(_) => {
                self.error(Diagnostic::error(
                    format!(
                        "the dimension of `dot` ({} times {}) has an exponent that overflows",
                        describe(&a.dim),
                        describe(&b.dim)
                    ),
                    call,
                ));
                Ty::Unknown
            }
        }
    }
}
