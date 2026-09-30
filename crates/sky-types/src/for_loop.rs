//! `for` loops: `for i in range(...)` and `for x in v`. The rules are in the
//! spec's "Blocks and control flow"; this file types the loop variable and
//! picks how the loop's exits are merged. Nothing here evaluates user code.
//!
//! A `for` is a `while` over an index, so scoping is `while`'s
//! ([`Checker::loop_body`]): the body shares the enclosing scope and is
//! checked once, and the loop variable stays bound after the loop, as in
//! Python. What changes is the zero-iteration path. A `while` may never run,
//! so a name first bound in its body is only maybe-defined after it. A `for`
//! over an array always runs (arrays are never empty in v0.1), and so does
//! a `range` whose bounds are constants that make at least one step, so for
//! those the loop variable and everything the body binds is defined after
//! the loop. A `range` with a variable bound may run zero times: the loop
//! variable is maybe-defined after it, and using it is the usual "may be
//! undefined" error.
//!
//! # Types
//!
//! * `range(...)`: the loop variable is a dimensionless scalar. An index
//!   is already any dimensionless scalar (`v[i]`, see [`crate::array`]), so
//!   it indexes without conversion. The bounds must be dimensionless
//!   integers: a constant is checked here, a variable one when the program
//!   runs.
//! * A vector: the loop variable is a scalar with the vector's element
//!   dimension.
//! * A matrix: the loop variable is a row, a vector with the matrix's
//!   element dimension (`for r in A` is `for i in range(len(A))` with
//!   `r = A[i]`). No tags: a row of a symmetric matrix is just a vector.
//! * Anything else (a scalar, a Bool, a function, a string literal) is an
//!   error with a help. A string held in a variable has no type here, so
//!   iterating it is a run-time error.
//!
//! `range` is not a value: it is recognised only as the whole iterable of a
//! `for`, and only while the program has not bound the name itself (a
//! user's `range` shadows it, as with `print`). Anywhere else it is an
//! error that says so.

use sky_ast::{Block, Expr, ExprKind, Ident, Literal};
use sky_diag::{display_capped as quote, Diagnostic, Span};
use sky_units::Dimension;

use crate::check::{const_exponent, describe, Checker, Ty, ValueTy};
use crate::constant::MAX_RANGE_BOUND;
use crate::flow::Runs;
use crate::rules::TensorType;
use crate::shape::Shape;

/// `(start, stop, step)` of `range(args)` when every argument is an integer
/// constant, in the same reading the checker uses for exponents and sizes
/// ([`const_exponent`]); `None` if any is not, or `args` is not 1 to 3
/// long. Shared with lowering, which must agree with the checker on
/// whether a loop is known to run (see [`range_runs_at_least_once`]).
pub fn range_constants(src: &str, args: &[Expr]) -> Option<(i64, i64, i64)> {
    let c = |a: &Expr| {
        const_exponent(src, a)
            .filter(|r| r.is_integer())
            .map(|r| r.numer())
    };
    match args {
        [stop] => Some((0, c(stop)?, 1)),
        [start, stop] => Some((c(start)?, c(stop)?, 1)),
        [start, stop, step] => Some((c(start)?, c(stop)?, c(step)?)),
        _ => None,
    }
}

/// Whether `range(start, stop, step)` has at least one item: the checker
/// then counts the loop as running, so what its body binds is defined
/// after it.
pub fn range_runs_at_least_once((start, stop, step): (i64, i64, i64)) -> bool {
    (step > 0 && start < stop) || (step < 0 && start > stop)
}

impl Checker<'_> {
    /// `for var in iter: body`.
    #[inline(never)]
    pub(crate) fn for_stmt(&mut self, span: Span, var: &Ident, iter: &Expr, body: &Block) {
        let before = self.env.snapshot();
        let (elem, runs) = match self.range_args(iter) {
            Some(args) => self.range_header(iter.span, args),
            None => self.iterable(iter),
        };
        self.bind_name(var, elem);
        self.loop_body(span, before, body, runs);
    }

    /// The arguments if `iter` is a call of the builtin `range`: callee
    /// `range` while the program has not bound that name itself.
    fn range_args<'e>(&self, iter: &'e Expr) -> Option<&'e [Expr]> {
        let ExprKind::Call { callee, args } = &iter.kind else {
            return None;
        };
        match &callee.kind {
            ExprKind::Name(id) if self.ident(id) == "range" && self.env.get("range").is_none() => {
                Some(args)
            }
            _ => None,
        }
    }

    /// Types `range(...)` as a loop header: the index type, and whether
    /// the loop is known to run. A bad header counts as running, so its one
    /// error does not turn into "may be undefined" errors after the loop.
    fn range_header(&mut self, call: Span, args: &[Expr]) -> (Ty, Runs) {
        let index = Ty::scalar(Dimension::DIMENSIONLESS);
        if !(1..=3).contains(&args.len()) {
            for a in args {
                self.expr(a);
            }
            self.error(
                Diagnostic::error(
                    format!("`range` takes 1 to 3 arguments, found {}", args.len()),
                    call,
                )
                .with_label("wrong number of arguments")
                .with_help(
                    "write `range(stop)`, `range(start, stop)` or `range(start, stop, step)`",
                ),
            );
            return (index, Runs::AtLeastOnce);
        }
        let mut consts = Vec::with_capacity(args.len());
        let mut ok = true;
        for a in args {
            match self.range_arg(a) {
                Ok(c) => consts.push(c),
                Err(()) => {
                    ok = false;
                    consts.push(None);
                }
            }
        }
        if !ok {
            return (index, Runs::AtLeastOnce);
        }
        if let (Some(&Some(0)), Some(step)) = (consts.get(2), args.get(2)) {
            self.error(
                Diagnostic::error("the step of `range` cannot be zero", step.span)
                    .with_label("zero step")
                    .with_note("the index would never move, so the loop would never end"),
            );
            return (index, Runs::AtLeastOnce);
        }
        let runs = match range_constants(self.src, args) {
            Some(known) if range_runs_at_least_once(known) => Runs::AtLeastOnce,
            _ => Runs::Maybe,
        };
        (index, runs)
    }

    /// Types one argument of `range`: a dimensionless integer. `Ok` holds
    /// its value if it is a constant; `Err` means an error was reported.
    fn range_arg(&mut self, a: &Expr) -> Result<Option<i64>, ()> {
        let t = self.expr(a);
        match &t {
            Ty::Value(v) if !v.is_scalar() => {
                let label = self.this_is(v);
                self.error(
                    Diagnostic::error("`range` needs a number, not an array", a.span)
                        .with_label(label)
                        .with_help(
                            "to visit the elements of an array, loop over it: `for x in v:`",
                        ),
                );
                return Err(());
            }
            Ty::Value(v) if !v.dim.is_dimensionless() => {
                self.error(
                    Diagnostic::error(
                        format!(
                            "`range` needs a dimensionless integer, found {}",
                            describe(&v.dim)
                        ),
                        a.span,
                    )
                    .with_label(format!("this is {}", describe(&v.dim)))
                    .with_note("a count of steps has no unit")
                    .with_help(
                        "divide by a reference quantity and round: e.g. `range(round(t / dt))`",
                    ),
                );
                return Err(());
            }
            Ty::Bool | Ty::Func(_) | Ty::Ode(_) => {
                self.error(
                    Diagnostic::error(format!("`range` needs an integer, found {t}"), a.span)
                        .with_label(format!("this is {t}")),
                );
                return Err(());
            }
            _ => {}
        }
        let Some(c) = const_exponent(self.src, a) else {
            return Ok(None);
        };
        if !c.is_integer() {
            self.error(
                Diagnostic::error(
                    format!("`range` needs an integer, found `{}`", self.quoted(a.span)),
                    a.span,
                )
                .with_label("not an integer")
                .with_help("round it first: `round(x)`, `floor(x)` or `ceil(x)`"),
            );
            return Err(());
        }
        let n = c.numer();
        if n.unsigned_abs() > MAX_RANGE_BOUND.unsigned_abs() {
            self.error(
                Diagnostic::error(
                    format!(
                        "the bound `{}` of `range` is too large",
                        self.quoted(a.span)
                    ),
                    a.span,
                )
                .with_label("too large")
                .with_note(
                    "a bound may be at most 2^53 in size, the largest integer an f64 holds exactly",
                ),
            );
            return Err(());
        }
        Ok(Some(n))
    }

    /// Types the iterable of `for x in iter` when it is not `range`: the
    /// loop variable's type, and that the loop runs (arrays are never
    /// empty). An error gives an unknown variable and the same, so it
    /// reports once.
    fn iterable(&mut self, iter: &Expr) -> (Ty, Runs) {
        let t = self.expr(iter);
        if let ExprKind::Literal(Literal::Str(_)) = iter.kind {
            self.error(
                Diagnostic::error("cannot iterate over a string", iter.span)
                    .with_label("this is a string")
                    .with_note("in v0.1 a string is only printed; it has no elements to loop over")
                    .with_help("loop over a vector, a matrix or `range(n)`"),
            );
            return (Ty::Unknown, Runs::AtLeastOnce);
        }
        let elem = match &t {
            Ty::Value(v) if !v.is_scalar() => {
                let shape = self.env.shapes.apply(v.ty.shape());
                // Iteration walks the first axis, so each item is what
                // indexing with one scalar gives (`A[i]`).
                let rest = shape.dims()[1..].to_vec();
                if rest.is_empty() {
                    Ty::scalar(v.dim)
                } else {
                    Ty::value(v.dim, TensorType::untagged(Shape::new(rest)))
                }
            }
            Ty::Value(v) => {
                self.not_iterable(iter, &t, Some(v));
                Ty::Unknown
            }
            Ty::Bool | Ty::Func(_) | Ty::Ode(_) => {
                self.not_iterable(iter, &t, None);
                Ty::Unknown
            }
            Ty::Unknown => Ty::Unknown,
        };
        (elem, Runs::AtLeastOnce)
    }

    /// `for x in 3 m` and friends: a scalar, a Bool, or a function.
    #[cold]
    fn not_iterable(&mut self, iter: &Expr, t: &Ty, scalar: Option<&ValueTy>) {
        let (noun, label) = match (t, scalar) {
            (_, Some(v)) => (
                match v.dim.name() {
                    Some("dimensionless") => "a dimensionless number".to_string(),
                    Some(n) => format!("a {n}"),
                    None => format!("a number of dimension `{}`", v.dim),
                },
                format!("this is {}", describe(&v.dim)),
            ),
            (Ty::Func(f), _) => (
                format!("function `{}`", quote(f.name())),
                "this is a function".to_string(),
            ),
            (Ty::Ode(o), _) => (o.describe(), format!("this is {o}")),
            _ => ("a Bool".to_string(), "this is Bool".to_string()),
        };
        let mut d = Diagnostic::error(format!("cannot iterate over {noun}"), iter.span)
            .with_label(label)
            .with_note("only vectors and matrices have elements to loop over");
        d = match (t, scalar) {
            (_, Some(v)) if v.dim.is_dimensionless() => d.with_help(format!(
                "to repeat something that many times, write `for i in range({}):`",
                self.quoted(iter.span)
            )),
            (Ty::Func(_), _) => d.with_help("call it to get a value, or loop over `range(n)`"),
            _ => d.with_help("loop over a vector, a matrix or `range(n)`"),
        };
        self.error(d);
    }

    /// `range(...)` anywhere but as the iterable of a `for`. `None` if
    /// `callee` is not the builtin `range`.
    pub(crate) fn range_outside_for(
        &mut self,
        callee: &Expr,
        args: &[Expr],
        call: Span,
    ) -> Option<Ty> {
        let ExprKind::Name(id) = &callee.kind else {
            return None;
        };
        if self.ident(id) != "range" || self.env.get("range").is_some() {
            return None;
        }
        for a in args {
            self.expr(a);
        }
        self.error(
            Diagnostic::error("`range` can only be the iterable of a `for` loop", call)
                .with_label("not directly after `for i in`")
                .with_note("`range` is not a value: it only tells a `for` loop what to count")
                .with_help("write `for i in range(n):`; to make a vector, write its entries"),
        );
        Some(Ty::Unknown)
    }
}
