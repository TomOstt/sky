//! Uncertainty in the checker (spec "Uncertainty").
//!
//! Uncertainty is a run-time property of a number, not part of its type: a
//! value of type `mass` may or may not carry an uncertainty, and every rule
//! about dimensions and shapes is exactly what it was. So this file is small:
//!
//! * `x ± dx` types as `x + dx` would: both are scalars of one dimension,
//!   and the result has that dimension. A constant negative `dx` is an error
//!   here (at run time it is one too);
//! * `nominal(x)` and `uncertainty(x)` take a scalar and give a scalar of
//!   the same dimension;
//! * a **warning** for comparing uncertain values. The checker cannot see
//!   the property in the type, so it tracks it the cheap, honest way: an
//!   expression *may be uncertain* if it contains a `±`, or a name that was
//!   bound (anywhere earlier in the walk, in any branch) to a value that may
//!   be. `nominal(...)` and `uncertainty(...)` make it exact again. A
//!   comparison with such an operand compares only the nominal values, which
//!   is rarely what a comparison of measurements means, so it warns once,
//!   with the way to ask the real question.
//!
//! The tracking is approximate on purpose: it is per name (a function
//! parameter, a name reassigned in one branch), never blocks a program, and
//! its worst case is a warning that is not needed or one that is missing.

use std::collections::HashSet;

use sky_ast::{BinaryOp, Expr, ExprKind};
use sky_diag::{Diagnostic, Span};

use crate::check::{Checker, Ty};
use crate::constant::entry_value;
use crate::display::a_dimension;
use crate::provenance::label_dim;

/// Most expression nodes one "may be uncertain" walk visits. A walk is
/// iterative and bounded, so a huge expression costs a constant amount.
const MAX_VISIT: usize = 256;

/// What the checker remembers about uncertainty.
#[derive(Clone, Debug, Default)]
pub(crate) struct UncertainState {
    /// Names bound to a value that may be uncertain.
    names: HashSet<String>,
    /// Comparisons already warned about (a loop body is walked more than
    /// once).
    warned: HashSet<Span>,
}

impl Checker<'_> {
    /// The type of `value ± sigma`.
    pub(crate) fn plus_minus(&mut self, value: &Expr, sigma: &Expr, op_span: Span) -> Ty {
        let lt = self.expr(value);
        let rt = self.expr(sigma);
        let (lv, rv) = match (&lt, &rt) {
            (Ty::Value(l), Ty::Value(r)) => (l, r),
            _ => {
                // Bool and function operands are reported once each; an
                // unknown one was already reported.
                self.reject_uncertain_operand("±", value, &lt);
                self.reject_uncertain_operand("±", sigma, &rt);
                return Ty::Unknown;
            }
        };
        if !lv.is_scalar() || !rv.is_scalar() {
            let (ll, rl) = (self.this_is(lv), self.this_is(rv));
            self.error(
                Diagnostic::error("`±` needs scalar operands", op_span)
                    .with_secondary(value.span, ll)
                    .with_secondary(sigma.span, rl)
                    .with_note("arrays cannot carry an uncertainty yet")
                    .with_help("write `±` on each element you need"),
            );
            return Ty::Unknown;
        }
        if lv.dim != rv.dim {
            let d = Diagnostic::error(
                format!(
                    "cannot use {} as the uncertainty of {}",
                    a_dimension(&rv.dim),
                    a_dimension(&lv.dim)
                ),
                op_span,
            )
            .with_label("mismatched dimensions")
            .with_secondary(value.span, label_dim(&lv.dim))
            .with_secondary(sigma.span, label_dim(&rv.dim))
            .with_note("the uncertainty of a value has the dimension of the value");
            let d = self.with_derivations(d, &[(value, lv.dim), (sigma, rv.dim)]);
            self.error(
                d.with_help("give both sides the same unit: `5.31 ± 0.02 kg`, or `5.31 kg ± 20 g`"),
            );
            return Ty::Unknown;
        }
        if entry_value(self.src, sigma).is_some_and(|s| s < 0.0) {
            self.error(
                Diagnostic::error("an uncertainty cannot be negative", sigma.span)
                    .with_label("this is negative")
                    .with_note("`±` takes a standard uncertainty, which is never negative")
                    .with_help("write its magnitude: `x ± 0.1`"),
            );
        }
        Ty::scalar(lv.dim)
    }

    /// `±` needs a number: a Bool or a function operand is an error.
    fn reject_uncertain_operand(&mut self, what: &str, e: &Expr, t: &Ty) {
        match t {
            Ty::Bool => {
                self.error(
                    Diagnostic::error(format!("`{what}` needs a number, found Bool"), e.span)
                        .with_label("this is Bool")
                        .with_note("v0.1 has no truthiness: a Bool is never a number"),
                );
            }
            Ty::Func(sig) => {
                self.error(
                    Diagnostic::error(
                        format!(
                            "`{what}` needs a number, found function `{}`",
                            sky_diag::display_capped(sig.name())
                        ),
                        e.span,
                    )
                    .with_label("this is a function")
                    .with_help("call it to get a value"),
                );
            }
            Ty::Ode(o) => {
                let found = o.describe();
                self.error(
                    Diagnostic::error(format!("`{what}` needs a number, found {found}"), e.span)
                        .with_label(format!("this is {found}")),
                );
            }
            Ty::Value(_) | Ty::Unknown => {}
        }
    }

    /// The type of `nominal(x)` or `uncertainty(x)` if `callee` names one of
    /// the builtins the program has not rebound; `None` leaves the call to
    /// the ordinary path.
    pub(crate) fn uncertain_call(
        &mut self,
        callee: &Expr,
        args: &[Expr],
        span: Span,
    ) -> Option<Ty> {
        let ExprKind::Name(id) = &callee.kind else {
            return None;
        };
        let name = self.ident(id);
        if !matches!(name, "nominal" | "uncertainty") || self.env.get(name).is_some() {
            return None;
        }
        let tys: Vec<Ty> = args.iter().map(|a| self.expr(a)).collect();
        let [arg] = args else {
            self.error(
                Diagnostic::error(
                    format!("`{name}` takes 1 argument, found {}", args.len()),
                    span,
                )
                .with_label("wrong number of arguments")
                .with_help(format!("call it as `{name}(x)`")),
            );
            return Some(Ty::Unknown);
        };
        match &tys[0] {
            Ty::Value(v) if v.is_scalar() => Some(Ty::scalar(v.dim)),
            Ty::Value(v) => {
                self.error(
                    Diagnostic::error(format!("`{name}` takes a scalar, found an array"), arg.span)
                        .with_label(self.this_is(v))
                        .with_note("arrays cannot carry an uncertainty yet"),
                );
                Some(Ty::Unknown)
            }
            other => {
                self.reject_uncertain_operand(name, arg, other);
                Some(Ty::Unknown)
            }
        }
    }

    /// Remembers whether `name` was just bound to a value that may be
    /// uncertain.
    pub(crate) fn note_uncertain_binding(&mut self, name: &str, value: &Expr) {
        if self.may_be_uncertain(value) {
            self.unc.names.insert(name.to_string());
        } else {
            self.unc.names.remove(name);
        }
    }

    /// `name op= value`: uncertain if it was, or if `value` is.
    pub(crate) fn note_uncertain_update(&mut self, name: &str, value: &Expr) {
        if self.may_be_uncertain(value) {
            self.unc.names.insert(name.to_string());
        }
    }

    /// Whether `e` may evaluate to an uncertain number (see the module
    /// docs). Bounded: at most [`MAX_VISIT`] nodes.
    fn may_be_uncertain(&self, e: &Expr) -> bool {
        let mut stack = vec![e];
        let mut visited = 0;
        while let Some(e) = stack.pop() {
            visited += 1;
            if visited > MAX_VISIT {
                return false;
            }
            match &e.kind {
                ExprKind::PlusMinus { .. } => return true,
                ExprKind::Name(id) => {
                    if self.unc.names.contains(self.ident(id)) {
                        return true;
                    }
                }
                ExprKind::Literal(_) | ExprKind::UnitLiteral { .. } => {}
                ExprKind::Unary { operand, .. }
                | ExprKind::Inverse { operand, .. }
                | ExprKind::Transpose { operand, .. } => stack.push(operand),
                ExprKind::Binary { lhs, rhs, .. } => stack.extend([&**lhs, &**rhs]),
                ExprKind::BoolOp { operands, .. } => stack.extend(operands.iter()),
                ExprKind::Paren(inner) => stack.push(inner),
                ExprKind::IfElse { then, else_, .. } => stack.extend([&**then, &**else_]),
                ExprKind::As { value, .. } => stack.push(value),
                ExprKind::Array(elems) => stack.extend(elems.iter()),
                ExprKind::Index { base, .. } => stack.push(base),
                ExprKind::Attribute { base, .. } => stack.push(base),
                ExprKind::Call { callee, args } => {
                    // `nominal(x)` and `uncertainty(x)` are exact.
                    if self.is_exact_part_call(callee) {
                        continue;
                    }
                    stack.extend(args.iter());
                }
            }
        }
        false
    }

    /// Whether `callee` is the builtin `nominal` or `uncertainty`.
    fn is_exact_part_call(&self, callee: &Expr) -> bool {
        match &callee.kind {
            ExprKind::Name(id) => {
                let name = self.ident(id);
                matches!(name, "nominal" | "uncertainty") && self.env.get(name).is_none()
            }
            _ => false,
        }
    }

    /// Warns that a comparison of uncertain values looks only at their
    /// nominal values, once per comparison.
    pub(crate) fn warn_uncertain_comparison(
        &mut self,
        op: BinaryOp,
        op_span: Span,
        lhs: &Expr,
        rhs: &Expr,
    ) {
        let uncertain = [lhs, rhs].into_iter().find(|e| self.may_be_uncertain(e));
        let Some(uncertain) = uncertain else {
            return;
        };
        if !self.unc.warned.insert(op_span) {
            return;
        }
        let sym = op.symbol();
        let d = Diagnostic::warning(
            "comparing an uncertain value compares only its nominal value",
            op_span,
        )
        .with_label(format!("`{sym}` ignores the uncertainty"))
        .with_secondary(uncertain.span, "this value may carry an uncertainty")
        .with_note(
            "a comparison looks at the central values alone, so `a < b` can hold even when \
             `a` and `b` agree within their uncertainties",
        )
        .with_help(
            "to ask whether two values agree, compare their difference with its uncertainty: \
             `abs(nominal(a - b)) < 2 * uncertainty(a - b)`",
        );
        self.diagnostics.push(d);
    }

    /// After a dimension mismatch in `a +- b`: adds the help that `+-` is
    /// `+` then `-` and the uncertainty operator is `±` (or `+/-`), since
    /// `5.31 +- 0.02 kg` is what a reader of other languages writes.
    pub(crate) fn plus_dash_hint(&mut self, op: BinaryOp, op_span: Span) {
        let dash_follows = self
            .src
            .get(op_span.end as usize..)
            .is_some_and(|rest| rest.starts_with('-'));
        if op != BinaryOp::Add || !dash_follows {
            return;
        }
        if let Some(d) = self.diagnostics.last_mut() {
            d.help.push(
                "`+-` is `+` followed by `-`, not an uncertainty; write `±` or `+/-`".to_string(),
            );
        }
    }
}
