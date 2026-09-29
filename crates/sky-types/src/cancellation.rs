//! The catastrophic-cancellation lint: recognizes a handful of textbook
//! expressions that subtract nearly equal numbers and suggests the stable
//! form. Warnings only, and only for patterns the lint is confident about.
//!
//! # Patterns
//!
//! | written | suggested |
//! | --- | --- |
//! | `sqrt(x^2 + c) - x` | `c / (sqrt(x^2 + c) + x)` |
//! | `(1 + x) - 1` | `x` |
//! | `x^2 - y^2` | `(x - y) * (x + y)` |
//! | `exp(x) - 1` | `expm1(x)` |
//! | `log(1 + x)` | `log1p(x)` |
//! | `(-b + sqrt(b^2 - 4*a*c)) / (2*a)` | `2*c / (-b - sqrt(b^2 - 4*a*c))` |
//!
//! # Confidence
//!
//! The lint reads the syntax tree, and never evaluates user code. A pattern
//! matches only when the operands that must be the same are the same source
//! text (ignoring spaces and outer parentheses), and every builtin it names
//! (`sqrt`, `exp`, `log`) is the builtin: a program that defines its own
//! `exp` is not warned about. Whether the subtraction really cancels depends
//! on the values, which the checker does not know, so:
//!
//! * with a variable operand, the lint fires on the pattern (each note says
//!   when the cancellation happens);
//! * when the relevant operands are numeric constants (`exp(1e-9) - 1`) it
//!   evaluates the closeness itself and fires only when at least about three
//!   digits are lost: `exp(1) - 1` and `(1 + 5) - 1` are silent.
//!
//! The suggested expression is the user's own text spliced into the
//! rewrite, and is left out when any piece is not quotable verbatim (too
//! long, or spanning lines); the note still explains the identity.
//!
//! `expm1` and `log1p` are builtins (`sky_units::MathFn`), so the help names
//! them.

use sky_ast::{BinaryOp, Expr, ExprKind, UnaryOp};
use sky_units::{MathFn, Rational};

use crate::check::{const_exponent, strip_parens, Checker, Ty};
use crate::constant::entry_value;
use sky_diag::{Diagnostic, Severity};

/// A constant operand this close to the critical value (a relative gap of
/// 1e-3) loses at least three digits.
const CLOSE: f64 = 1e-3;

/// The explanation every pattern shares.
const WHY: &str = "subtracting two nearly equal numbers cancels their leading digits, so the \
     rounding errors they already carry become the leading digits of the result: two numbers \
     correct to 16 digits that agree in their first 12 leave a difference correct to about 4";

impl Checker<'_> {
    /// Called for every typed expression: warns if `e` is one of the
    /// cancellation patterns. `t` is the type just computed for `e`, and
    /// `reported` the number of diagnostics before `e` was typed: an
    /// expression with an error inside it (a unit mismatch in the
    /// `sqrt(...)`, say) is not linted, so one mistake gives one message.
    /// Unknown types are linted: an unannotated parameter is unknown, and
    /// the patterns are about the arithmetic, not the units.
    pub(crate) fn lint_cancellation(&mut self, e: &Expr, t: &Ty, reported: usize) {
        let candidate = matches!(
            e.kind,
            ExprKind::Binary {
                op: BinaryOp::Sub | BinaryOp::Div,
                ..
            } | ExprKind::Call { .. }
        );
        if !candidate
            || matches!(t, Ty::Bool | Ty::Func(_))
            || self.diagnostics[reported..]
                .iter()
                .any(|d| d.severity == Severity::Error)
        {
            return;
        }
        match &e.kind {
            ExprKind::Binary {
                op: BinaryOp::Sub,
                lhs,
                rhs,
                ..
            } => {
                let _ = self.sqrt_minus_x(e, lhs, rhs)
                    || self.one_plus_x_minus_one(e, lhs, rhs)
                    || self.difference_of_squares(e, lhs, rhs)
                    || self.exp_minus_one(e, lhs, rhs);
            }
            ExprKind::Binary {
                op: BinaryOp::Div,
                lhs,
                rhs,
                ..
            } => {
                self.quadratic_formula(e, lhs, rhs);
            }
            ExprKind::Call { callee, args } => {
                self.log_one_plus_x(e, callee, args);
            }
            _ => {}
        }
    }

    // ----- helpers -----

    /// `e`'s source text if it can be quoted in a help verbatim.
    fn verbatim(&self, e: &Expr) -> Option<String> {
        let raw = self.text(e.span);
        (self.quoted(e.span) == raw).then(|| raw.to_string())
    }

    /// [`Checker::operand_text`] if `e` can be quoted verbatim: a piece to
    /// splice into a larger suggested expression.
    fn piece(&self, e: &Expr) -> Option<String> {
        let raw = self.verbatim(e)?;
        match strip_parens(e).kind {
            ExprKind::Binary { .. }
            | ExprKind::Unary { .. }
            | ExprKind::Inverse { .. }
            | ExprKind::Transpose { .. }
                if !matches!(e.kind, ExprKind::Paren(_)) =>
            {
                Some(format!("({raw})"))
            }
            _ => Some(raw),
        }
    }

    /// Whether `a` and `b` are the same source text, ignoring whitespace
    /// and outer parentheses.
    fn same(&self, a: &Expr, b: &Expr) -> bool {
        let text = |e: &Expr| -> String {
            self.text(strip_parens(e).span)
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect()
        };
        text(a) == text(b)
    }

    fn const_val(&self, e: &Expr) -> Option<f64> {
        entry_value(self.src, e)
    }

    /// `e` is `name(arg)`, the builtin called `name` (not rebound by the
    /// program) with one argument: the argument.
    fn math_arg<'e>(&self, e: &'e Expr, name: &str) -> Option<&'e Expr> {
        let ExprKind::Call { callee, args } = &strip_parens(e).kind else {
            return None;
        };
        self.call_of(callee, args, name)
    }

    fn call_of<'e>(&self, callee: &Expr, args: &'e [Expr], name: &str) -> Option<&'e Expr> {
        let ExprKind::Name(id) = &callee.kind else {
            return None;
        };
        let shown = self.ident(id);
        if shown != name || self.env.get(shown).is_some() || MathFn::from_name(name).is_none() {
            return None;
        }
        match args {
            [arg] => Some(arg),
            _ => None,
        }
    }

    /// `x` if `e` is `x^2` or `x*x`.
    fn square_of<'e>(&self, e: &'e Expr) -> Option<&'e Expr> {
        match &strip_parens(e).kind {
            ExprKind::Binary {
                op: BinaryOp::Pow,
                lhs,
                rhs,
                ..
            } if const_exponent(self.src, rhs) == Some(Rational::integer(2)) => Some(lhs),
            ExprKind::Binary {
                op: BinaryOp::Mul,
                lhs,
                rhs,
                ..
            } if self.same(lhs, rhs) && !is_number(self, lhs) => Some(lhs),
            _ => None,
        }
    }

    /// Whether `e` is a numeric constant equal to `v`.
    fn is_const(&self, e: &Expr, v: f64) -> bool {
        self.const_val(e) == Some(v)
    }

    /// `x` if `e` is `1 + x` or `x + 1` (`1` a numeric literal).
    fn one_plus<'e>(&self, e: &'e Expr) -> Option<&'e Expr> {
        let ExprKind::Binary {
            op: BinaryOp::Add,
            lhs,
            rhs,
            ..
        } = &strip_parens(e).kind
        else {
            return None;
        };
        if self.is_const(lhs, 1.0) {
            Some(rhs)
        } else if self.is_const(rhs, 1.0) {
            Some(lhs)
        } else {
            None
        }
    }

    /// Emits `d` if the site `e` has not been warned about and the budget
    /// allows.
    fn cancellation_warning(&mut self, e: &Expr, d: Diagnostic) -> bool {
        if !self.numeric_warning_budget(e.span) {
            return true; // matched, so no other pattern is tried
        }
        self.error(d);
        true
    }

    // ----- patterns -----

    /// `sqrt(x^2 + c) - x`, `sqrt(c + x^2) - x` and `sqrt(x^2 - c) - x`.
    fn sqrt_minus_x(&mut self, e: &Expr, lhs: &Expr, rhs: &Expr) -> bool {
        let Some(arg) = self.math_arg(lhs, "sqrt") else {
            return false;
        };
        let ExprKind::Binary {
            op: inner @ (BinaryOp::Add | BinaryOp::Sub),
            lhs: p,
            rhs: q,
            ..
        } = &strip_parens(arg).kind
        else {
            return false;
        };
        // `x^2 + c`, `c + x^2` (a sum only) or `x^2 - c`.
        let c = match self.square_of(p) {
            Some(b) if self.same(b, rhs) => &**q,
            _ => match self.square_of(q) {
                Some(b) if *inner == BinaryOp::Add && self.same(b, rhs) => &**p,
                _ => return false,
            },
        };
        let subtract = *inner == BinaryOp::Sub;
        // Constants: cancellation needs x > 0 and c small next to x².
        let x = self.const_val(rhs);
        if x.is_some_and(|x| x <= 0.0) {
            return false;
        }
        if let (Some(x), Some(c)) = (x, self.const_val(c)) {
            if c.abs() >= CLOSE * x * x {
                return false;
            }
        }
        let shown = self.quoted(e.span);
        let help = match (self.verbatim(lhs), self.piece(c), self.piece(rhs)) {
            (Some(root), Some(c), Some(x)) => {
                let numerator = if subtract { format!("-{c}") } else { c };
                format!(
                    "write `{numerator} / ({root} + {x})`, the same value without the subtraction"
                )
            }
            _ => "multiply by the conjugate: the same value is c / (sqrt(x^2 + c) + x)".to_string(),
        };
        let d = Diagnostic::warning(format!("`{shown}` can lose digits to cancellation"), e.span)
            .with_label(format!(
                "subtracts nearly equal numbers when `{}` is large and positive",
                self.quoted(rhs.span)
            ))
            .with_note(format!(
                "{WHY}. Here, when x > 0 and c is small next to x^2, sqrt(x^2 + c) and x agree \
                 in their leading digits. Multiplying by the conjugate, sqrt(x^2 + c) - x = \
                 c / (sqrt(x^2 + c) + x), replaces the subtraction with an addition of positive \
                 numbers. (For x < 0 the original is already stable, and the rewrite is not.)"
            ))
            .with_help(help);
        self.cancellation_warning(e, d)
    }

    /// `(1 + x) - 1`, `(x + 1) - 1`.
    fn one_plus_x_minus_one(&mut self, e: &Expr, lhs: &Expr, rhs: &Expr) -> bool {
        if !self.is_const(rhs, 1.0) {
            return false;
        }
        let Some(x) = self.one_plus(lhs) else {
            return false;
        };
        if let Some(v) = self.const_val(x) {
            if v == 0.0 || v.abs() >= CLOSE {
                return false;
            }
        }
        let shown = self.quoted(e.span);
        let help = match self.verbatim(x) {
            Some(x) => {
                format!("write `{x}`: it is the same value, without rounding it through 1 + …")
            }
            None => "the value is just the term added to 1: write that".to_string(),
        };
        let d = Diagnostic::warning(
            format!("`{shown}` loses the low digits of the small term"),
            e.span,
        )
        .with_label("`1 + …` rounds away digits before the subtraction")
        .with_note(format!(
            "{WHY}. Here 1 + x is rounded to a multiple of about 2.2e-16 (an `f64` has 52 \
                 fraction bits), so when x is small the low digits of x are gone before 1 is \
                 subtracted: the result is x plus an error of up to 1.1e-16, which is a large \
                 relative error when x is 1e-8 and everything when x is 1e-17"
        ))
        .with_help(help);
        self.cancellation_warning(e, d)
    }

    /// `x^2 - y^2`.
    fn difference_of_squares(&mut self, e: &Expr, lhs: &Expr, rhs: &Expr) -> bool {
        let (Some(x), Some(y)) = (self.square_of(lhs), self.square_of(rhs)) else {
            return false;
        };
        if self.same(x, y) {
            return false;
        }
        if let (Some(a), Some(b)) = (self.const_val(x), self.const_val(y)) {
            if a == b || (a - b).abs() >= CLOSE * a.abs().max(b.abs()) {
                return false;
            }
        }
        let shown = self.quoted(e.span);
        let help = match (self.piece(x), self.piece(y)) {
            (Some(x), Some(y)) => format!("write `({x} - {y}) * ({x} + {y})`"),
            _ => "factor it: x^2 - y^2 = (x - y) * (x + y)".to_string(),
        };
        let d = Diagnostic::warning(format!("`{shown}` can lose digits to cancellation"), e.span)
            .with_label("subtracts two nearly equal squares when the two bases are close")
            .with_note(format!(
                "{WHY}. Here, when x and y agree in their leading digits, so do x^2 and y^2. \
                 The difference x - y is computed exactly when x and y are within a factor of two \
                 of each other (Sterbenz's lemma), so x^2 - y^2 = (x - y) * (x + y) keeps every \
                 digit that the inputs have"
            ))
            .with_help(help);
        self.cancellation_warning(e, d)
    }

    /// `exp(x) - 1`.
    fn exp_minus_one(&mut self, e: &Expr, lhs: &Expr, rhs: &Expr) -> bool {
        if !self.is_const(rhs, 1.0) {
            return false;
        }
        let Some(x) = self.math_arg(lhs, "exp") else {
            return false;
        };
        if let Some(v) = self.const_val(x) {
            if v == 0.0 || v.abs() >= CLOSE {
                return false;
            }
        }
        let shown = self.quoted(e.span);
        let help = match self.verbatim(x) {
            Some(x) => format!("write `expm1({x})`"),
            None => "write `expm1(x)`, with your argument for x".to_string(),
        };
        let d = Diagnostic::warning(
            format!("`{shown}` loses digits to cancellation when the argument is small"),
            e.span,
        )
        .with_label("exp(x) is close to 1, so the subtraction cancels")
        .with_note(format!(
            "{WHY}. Here, when |x| is small, exp(x) is 1 plus a small number, and the \
                 subtraction leaves only the digits that survived rounding to a multiple of \
                 2.2e-16: for x = 1e-10 about 6 of 16 are correct. The function `expm1(x)`, \
                 exp(x) - 1 computed without the subtraction, is the standard cure"
        ))
        .with_help(help);
        self.cancellation_warning(e, d)
    }

    /// `log(1 + x)`, `log(x + 1)`.
    fn log_one_plus_x(&mut self, e: &Expr, callee: &Expr, args: &[Expr]) -> bool {
        let Some(arg) = self.call_of(callee, args, "log") else {
            return false;
        };
        let Some(x) = self.one_plus(arg) else {
            return false;
        };
        if let Some(v) = self.const_val(x) {
            if v == 0.0 || v.abs() >= CLOSE {
                return false;
            }
        }
        let shown = self.quoted(e.span);
        let help = match self.verbatim(x) {
            Some(x) => format!("write `log1p({x})`"),
            None => "write `log1p(x)`, with the term added to 1 for x".to_string(),
        };
        let d = Diagnostic::warning(
            format!("`{shown}` loses digits when the term added to 1 is small"),
            e.span,
        )
        .with_label("`1 + …` rounds away digits before the logarithm")
        .with_note(format!(
            "{WHY}. Here 1 + x is rounded to a multiple of about 2.2e-16 before `log` sees \
                 it, so when x is small the low digits of x are gone (the result log(1 + x) ≈ x \
                 has a large relative error, and is 0 for x below 1.1e-16). The function \
                 `log1p(x)`, log(1 + x) computed without forming 1 + x, is the standard cure"
        ))
        .with_help(help);
        self.cancellation_warning(e, d)
    }

    /// `(-b + sqrt(b^2 - 4*a*c)) / (2*a)` and its variants: `sqrt(D) - b`
    /// and `-b - sqrt(D)`.
    fn quadratic_formula(&mut self, e: &Expr, num: &Expr, den: &Expr) -> bool {
        let ExprKind::Binary {
            op: top @ (BinaryOp::Add | BinaryOp::Sub),
            lhs,
            rhs,
            ..
        } = &strip_parens(num).kind
        else {
            return false;
        };
        // (b, the sqrt call, whether the sqrt is added)
        fn neg(x: &Expr) -> Option<&Expr> {
            match &strip_parens(x).kind {
                ExprKind::Unary {
                    op: UnaryOp::Neg,
                    operand,
                    ..
                } => Some(&**operand),
                _ => None,
            }
        }
        let (b, root, plus) = match (top, neg(lhs), neg(rhs)) {
            (BinaryOp::Add, Some(b), _) => (b, &**rhs, true), // -b + sqrt(D)
            (BinaryOp::Sub, Some(b), _) => (b, &**rhs, false), // -b - sqrt(D)
            (BinaryOp::Sub, None, _) => (&**rhs, &**lhs, true), // sqrt(D) - b
            _ => return false,
        };
        let Some(disc) = self.math_arg(root, "sqrt") else {
            return false;
        };
        let ExprKind::Binary {
            op: BinaryOp::Sub,
            lhs: bsq,
            rhs: prod,
            ..
        } = &strip_parens(disc).kind
        else {
            return false;
        };
        let Some(b2) = self.square_of(bsq) else {
            return false;
        };
        if !self.same(b2, b) {
            return false;
        }
        // 4*a*c: three factors, one of them 4; 2*a: two factors, one 2.
        let mut factors = Vec::new();
        flatten_product(prod, &mut factors);
        let mut den_factors = Vec::new();
        flatten_product(den, &mut den_factors);
        let (Some(four), Some(two)) = (
            factors.iter().position(|f| self.is_const(f, 4.0)),
            den_factors.iter().position(|f| self.is_const(f, 2.0)),
        ) else {
            return false;
        };
        if factors.len() != 3 || den_factors.len() != 2 {
            return false;
        }
        let rest: Vec<&Expr> = factors
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != four)
            .map(|(_, f)| *f)
            .collect();
        let a = den_factors[1 - two];
        let c = if self.same(rest[0], a) {
            rest[1]
        } else if self.same(rest[1], a) {
            rest[0]
        } else {
            return false;
        };
        // All-constant coefficients: cancellation needs b² >> |4ac| and a
        // real root.
        if let (Some(av), Some(bv), Some(cv)) =
            (self.const_val(a), self.const_val(b), self.const_val(c))
        {
            let d = bv * bv - 4.0 * av * cv;
            if d < 0.0 || (4.0 * av * cv).abs() >= CLOSE * bv * bv {
                return false;
            }
        }
        let shown = self.quoted(e.span);
        let (cancels, sign) = if plus {
            ("-b + sqrt(…)", "b > 0")
        } else {
            ("-b - sqrt(…)", "b < 0")
        };
        let help = match (self.piece(b), self.piece(c), self.verbatim(root)) {
            (Some(b), Some(c), Some(root)) => format!(
                "the other root has no cancellation: write `2 * {c} / (-{b} {} {root})`",
                if plus { "-" } else { "+" }
            ),
            _ => "compute the other root as 2c / (-b ∓ sqrt(b^2 - 4ac)), with the opposite sign \
                  before the square root"
                .to_string(),
        };
        let d = Diagnostic::warning(
            format!("the quadratic formula `{shown}` can lose digits to cancellation"),
            e.span,
        )
        .with_label("one of the two roots subtracts nearly equal numbers")
        .with_note(format!(
            "{WHY}. Here, when b^2 is much larger than |4ac|, sqrt(b^2 - 4ac) is nearly |b|, so \
             the root `{cancels}` cancels when {sign} (for the other sign of b, the other root \
             does). Since the \
             roots multiply to c/a, that root equals 2c / (-b ∓ sqrt(b^2 - 4ac)) (the \
             citardauq form), which adds numbers of the same sign exactly when the textbook \
             form subtracts"
        ))
        .with_help(help);
        self.cancellation_warning(e, d)
    }
}

/// Whether `e` is a numeric constant.
fn is_number(c: &Checker<'_>, e: &Expr) -> bool {
    c.const_val(e).is_some()
}

/// The factors of a product `a * b * c`, looking through parentheses and
/// nested products; anything else is one factor. At most 8 are collected
/// (a longer product is not a quadratic coefficient).
fn flatten_product<'e>(e: &'e Expr, out: &mut Vec<&'e Expr>) {
    let e = strip_parens(e);
    match &e.kind {
        ExprKind::Binary {
            op: BinaryOp::Mul,
            lhs,
            rhs,
            ..
        } if out.len() < 8 => {
            flatten_product(lhs, out);
            flatten_product(rhs, out);
        }
        _ => out.push(e),
    }
}
