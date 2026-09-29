//! Typing of `v as unit`, the display value (spec "Showing values in a
//! unit").
//!
//! A display value is a number or array written in a chosen unit. It is not
//! a number: no value remembers a unit (spec "Unit values"), so `v as km`
//! has nothing to compute with. It is only *shown*, so it is allowed in two
//! places: as an argument of `print` and as an expression statement (which
//! the REPL echoes). Anywhere else, including as an operand of arithmetic,
//! is an error with a help. This is a syntactic rule, not a new [`Ty`]: the
//! parent that allows a display value calls [`Checker::allow_show`] for the
//! expression it is about to type, and every other path reaches
//! [`Checker::show`] without permission.
//!
//! The target unit must have the value's dimension ("cannot show a
//! velocity in kg"). The unit is checked by the same code as a unit
//! literal's, so prefixes, `^` and unknown-unit suggestions are identical.
//! A value whose type is unknown (an earlier error, an unannotated
//! parameter) is not checked here: the interpreter checks it dynamically.

use sky_ast::Expr;
use sky_ast::UnitExpr;
use sky_diag::Diagnostic;

use crate::check::{describe, strip_parens, Checker, Ty};
use crate::provenance::label_dim;

/// "a velocity", "an energy", "a plain number": a dimension in a sentence.
/// The checker and the interpreter word "cannot show a velocity in kg" alike.
pub fn a_dimension(d: &sky_units::Dimension) -> String {
    if d.is_dimensionless() {
        return "a plain number".to_string();
    }
    match d.name() {
        Some(n) if n.starts_with(['a', 'e', 'i', 'o', 'u']) => format!("an {n}"),
        Some(n) => format!("a {n}"),
        None => format!("a quantity of dimension {d}"),
    }
}

impl Checker<'_> {
    /// Permits a display value at the top of `e` (through parentheses)
    /// while it is typed next: the caller is `print` or an expression
    /// statement.
    pub(crate) fn allow_show(&mut self, e: &Expr) {
        self.shown_ok = Some(strip_parens(e).span);
    }

    /// The type of `value as unit`, which is only ever [`Ty::Unknown`]: a
    /// display value has no type to compute with. Reports a misplaced use,
    /// then checks the value against the unit's dimension.
    pub(crate) fn show(&mut self, e: &Expr, value: &Expr, unit: &UnitExpr) -> Ty {
        let allowed = self.shown_ok == Some(e.span);
        let value_ty = self.expr(value);
        let unit_dim = self.unit(unit);
        if !allowed {
            self.misplaced_show(e, unit);
            return Ty::Unknown;
        }
        let (Ty::Value(v), Some(want)) = (&value_ty, unit_dim) else {
            if matches!(value_ty, Ty::Bool | Ty::Func(_)) && unit_dim.is_some() {
                self.show_non_number(value, unit, &value_ty);
            }
            return Ty::Unknown;
        };
        if v.dim != want {
            let unit_text = self.quoted(unit.span());
            let d = Diagnostic::error(
                format!(
                    "cannot show {} in {unit_text}",
                    a_dimension(&v.dim),
                ),
                value.span,
            )
            .with_label(format!("this is {}", label_dim(&v.dim)))
            .with_secondary(
                unit.span(),
                format!("`{unit_text}` measures {}", describe(&want)),
            )
            .with_help(if v.dim.is_dimensionless() {
                format!("a plain number has no unit to convert from; write it with one, e.g. `3 {unit_text}`")
            } else {
                format!(
                    "`as` changes the unit a value is shown in, not what it measures; choose a unit of {}",
                    describe(&v.dim)
                )
            });
            let d = self.with_derivations(d, &[(value, v.dim)]);
            self.error(d);
        }
        Ty::Unknown
    }

    /// `true as km`, `f as km`: only numbers and arrays have units.
    fn show_non_number(&mut self, value: &Expr, unit: &UnitExpr, t: &Ty) {
        let what = match t {
            Ty::Bool => "a Bool".to_string(),
            other => other.to_string(),
        };
        let unit_text = self.quoted(unit.span());
        self.error(
            Diagnostic::error(format!("cannot show {what} in {unit_text}"), value.span)
                .with_label(format!("this is {what}"))
                .with_help("only numbers and arrays of numbers can be shown in a unit"),
        );
    }

    /// A display value where a number is needed (`(v as km) + 1`, `x = v as
    /// km`, `f(v as km)`), or anywhere but `print` and a statement.
    fn misplaced_show(&mut self, e: &Expr, unit: &UnitExpr) {
        let unit_text = self.quoted(unit.span());
        self.error(
            Diagnostic::error(
                format!("a value shown in {unit_text} can only be printed"),
                e.span,
            )
            .with_label("this is for display, not a number to compute with")
            .with_help(format!(
                "values are computed in SI and converted only when shown: put `as` last, \
                 `print(… as {unit_text})`; to get a plain number of {unit_text}, divide by \
                 one: `x / (1 {unit_text})`"
            )),
        );
    }
}
