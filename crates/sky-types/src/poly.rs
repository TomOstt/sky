//! Polynomials in symbolic sizes, with exact rational coefficients.
//!
//! The cost model (`sky_ir::cost`) states what a function costs as a
//! polynomial in the sizes of its arrays (`2·N·K·M` flops for a matrix
//! product), and a performance contract (`cost flops <= 2 * N^3`, spec
//! "Performance contracts") is a polynomial too. Both are a [`Poly`]: a sum
//! of [`Monomial`]s (products of named sizes to whole powers) with
//! [`Rational`] coefficients, kept in a canonical form, so equal polynomials
//! are `==`.
//!
//! # Domain and soundness
//!
//! A size is a positive integer: every array has at least one entry per
//! axis. Everything here that compares polynomials is about **all sizes at
//! least 1** (real numbers `>= 1`, which contain the integers), and is
//! sound: [`Poly::le_for_all_sizes`] answers `true` only if `p(x) <= q(x)`
//! for every such `x`. It is not complete (a `false` means "not proven"),
//! because it substitutes `x = 1 + y` and asks that every coefficient of the
//! difference in `y` be non-negative, which is enough and quick but is not a
//! decision procedure for polynomial inequalities. The two ways to say
//! "cannot prove it" are kept apart from "false": to *refute* a claim the
//! caller needs a witness point ([`Poly::eval`]), which is exact.
//!
//! # Overflow
//!
//! Coefficients are `i64` fractions and the number of terms and the degree
//! are capped ([`MAX_TERMS`], [`MAX_DEGREE`]). Every operation that could
//! exceed them returns `None`; nothing panics and nothing wraps.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use sky_ast::ast::span_text;
use sky_ast::{BinaryOp, Expr, ExprKind, Literal, UnaryOp};
use sky_diag::Span;
use sky_units::Rational;

/// The most terms a polynomial may have.
pub const MAX_TERMS: usize = 256;
/// The largest total degree of a monomial.
pub const MAX_DEGREE: u32 = 64;

/// A product of sizes to whole powers: `N²·M`. The empty product is `1`.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct Monomial {
    powers: BTreeMap<String, u32>,
}

impl Monomial {
    /// The monomial `1`.
    pub fn one() -> Monomial {
        Monomial::default()
    }

    /// The size `name` to the first power.
    pub fn var(name: &str) -> Monomial {
        let mut powers = BTreeMap::new();
        powers.insert(name.to_string(), 1);
        Monomial { powers }
    }

    /// The total degree.
    pub fn degree(&self) -> u32 {
        self.powers.values().sum()
    }

    /// The sizes and their powers, sorted by name.
    pub fn powers(&self) -> impl Iterator<Item = (&str, u32)> {
        self.powers.iter().map(|(k, v)| (k.as_str(), *v))
    }

    fn mul(&self, other: &Monomial) -> Option<Monomial> {
        let mut powers = self.powers.clone();
        for (k, e) in &other.powers {
            let slot = powers.entry(k.clone()).or_insert(0);
            *slot = slot.checked_add(*e)?;
        }
        let m = Monomial { powers };
        (m.degree() <= MAX_DEGREE).then_some(m)
    }
}

/// A polynomial in named sizes. Zero coefficients are never stored.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct Poly {
    terms: BTreeMap<Monomial, Rational>,
}

impl Poly {
    /// The polynomial `0`.
    pub fn zero() -> Poly {
        Poly::default()
    }

    /// The constant `c`.
    pub fn constant(c: Rational) -> Poly {
        let mut terms = BTreeMap::new();
        if !c.is_zero() {
            terms.insert(Monomial::one(), c);
        }
        Poly { terms }
    }

    /// The constant `n`.
    pub fn integer(n: i64) -> Poly {
        Poly::constant(Rational::integer(n))
    }

    /// The size `name`.
    pub fn var(name: &str) -> Poly {
        let mut terms = BTreeMap::new();
        terms.insert(Monomial::var(name), Rational::ONE);
        Poly { terms }
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// The value if there are no sizes in it.
    pub fn as_constant(&self) -> Option<Rational> {
        match self.terms.len() {
            0 => Some(Rational::ZERO),
            1 => self.terms.get(&Monomial::one()).copied(),
            _ => None,
        }
    }

    /// The terms, from the highest degree down (ties by monomial).
    pub fn terms(&self) -> Vec<(&Monomial, Rational)> {
        let mut v: Vec<_> = self.terms.iter().map(|(m, c)| (m, *c)).collect();
        v.sort_by(|a, b| b.0.degree().cmp(&a.0.degree()).then_with(|| a.0.cmp(b.0)));
        v
    }

    /// The names of the sizes that occur.
    pub fn vars(&self) -> BTreeSet<String> {
        self.terms
            .keys()
            .flat_map(|m| m.powers.keys().cloned())
            .collect()
    }

    /// The highest total degree (0 for a constant or zero).
    pub fn degree(&self) -> u32 {
        self.terms.keys().map(Monomial::degree).max().unwrap_or(0)
    }

    fn add_term(&mut self, m: Monomial, c: Rational) -> Option<()> {
        if c.is_zero() {
            return Some(());
        }
        let sum = match self.terms.get(&m) {
            Some(old) => old.checked_add(c).ok()?,
            None => c,
        };
        if sum.is_zero() {
            self.terms.remove(&m);
        } else {
            if !self.terms.contains_key(&m) && self.terms.len() >= MAX_TERMS {
                return None;
            }
            self.terms.insert(m, sum);
        }
        Some(())
    }

    pub fn checked_add(&self, other: &Poly) -> Option<Poly> {
        let mut out = self.clone();
        for (m, c) in &other.terms {
            out.add_term(m.clone(), *c)?;
        }
        Some(out)
    }

    pub fn checked_neg(&self) -> Option<Poly> {
        let mut terms = BTreeMap::new();
        for (m, c) in &self.terms {
            terms.insert(m.clone(), c.checked_neg().ok()?);
        }
        Some(Poly { terms })
    }

    pub fn checked_sub(&self, other: &Poly) -> Option<Poly> {
        self.checked_add(&other.checked_neg()?)
    }

    pub fn checked_mul(&self, other: &Poly) -> Option<Poly> {
        let mut out = Poly::zero();
        for (m1, c1) in &self.terms {
            for (m2, c2) in &other.terms {
                out.add_term(m1.mul(m2)?, c1.checked_mul(*c2).ok()?)?;
            }
        }
        Some(out)
    }

    pub fn checked_scale(&self, k: Rational) -> Option<Poly> {
        if k.is_zero() {
            return Some(Poly::zero());
        }
        let mut terms = BTreeMap::new();
        for (m, c) in &self.terms {
            terms.insert(m.clone(), c.checked_mul(k).ok()?);
        }
        Some(Poly { terms })
    }

    /// `self` to the power `n`.
    pub fn checked_pow(&self, n: u32) -> Option<Poly> {
        if n > MAX_DEGREE {
            return None;
        }
        let mut out = Poly::integer(1);
        for _ in 0..n {
            out = out.checked_mul(self)?;
        }
        Some(out)
    }

    /// The value at a point: each size in `point`. `None` if a size is
    /// missing from it or the arithmetic overflows.
    pub fn eval(&self, point: &BTreeMap<String, Rational>) -> Option<Rational> {
        let mut total = Rational::ZERO;
        for (m, c) in &self.terms {
            let mut t = *c;
            for (name, e) in m.powers() {
                let x = *point.get(name)?;
                for _ in 0..e {
                    t = t.checked_mul(x).ok()?;
                }
            }
            total = total.checked_add(t).ok()?;
        }
        Some(total)
    }

    /// Replaces sizes by polynomials, all at once. A size not in `map` stays.
    pub fn subst(&self, map: &BTreeMap<String, Poly>) -> Option<Poly> {
        let mut out = Poly::zero();
        for (m, c) in &self.terms {
            let mut t = Poly::constant(*c);
            for (name, e) in m.powers() {
                let factor = match map.get(name) {
                    Some(p) => p.checked_pow(e)?,
                    None => Poly::var(name).checked_pow(e)?,
                };
                t = t.checked_mul(&factor)?;
            }
            out = out.checked_add(&t)?;
        }
        Some(out)
    }

    /// The polynomial as `cost` lines write it: `2 * N^3 + 5 * N`,
    /// `N^3 / 3`.
    pub fn to_source(&self) -> String {
        let terms = self.terms();
        if terms.is_empty() {
            return "0".to_string();
        }
        let mut out = String::new();
        for (i, (m, c)) in terms.iter().enumerate() {
            match (i, c.numer() < 0) {
                (0, false) => {}
                (0, true) => out.push('-'),
                (_, false) => out.push_str(" + "),
                (_, true) => out.push_str(" - "),
            }
            let (n, d) = (c.numer().unsigned_abs(), c.denom());
            let vars: Vec<String> = m
                .powers()
                .map(|(name, e)| {
                    if e == 1 {
                        name.to_string()
                    } else {
                        format!("{name}^{e}")
                    }
                })
                .collect();
            let mut factors: Vec<String> = Vec::new();
            if vars.is_empty() || n != 1 {
                factors.push(n.to_string());
            }
            factors.extend(vars);
            out.push_str(&factors.join(" * "));
            if d != 1 {
                out.push_str(&format!(" / {d}"));
            }
        }
        out
    }

    /// The value with every size equal to `x`, as an `f64`, for ranking
    /// polynomials by how they grow. Not exact.
    pub fn weight_at(&self, x: f64) -> f64 {
        self.terms
            .iter()
            .map(|(m, c)| (c.numer() as f64 / c.denom() as f64) * x.powi(m.degree() as i32))
            .sum()
    }

    /// Whether every coefficient is at least zero.
    pub fn all_coefficients_nonnegative(&self) -> bool {
        self.terms.values().all(|c| c.numer() >= 0)
    }

    /// Whether the polynomial is `>= 0` for all sizes `>= 1`. Sound, not
    /// complete (see the module documentation): with `x = 1 + y`, all the
    /// coefficients in `y` are non-negative.
    pub fn nonnegative_for_all_sizes(&self) -> bool {
        let shift: BTreeMap<String, Poly> = self
            .vars()
            .into_iter()
            .filter_map(|v| Poly::integer(1).checked_add(&Poly::var(&v)).map(|p| (v, p)))
            .collect();
        match self.subst(&shift) {
            Some(shifted) => shifted.all_coefficients_nonnegative(),
            // Too large to decide: not proven.
            None => false,
        }
    }

    /// `self <= other` for all sizes `>= 1`, proven (`false` means not
    /// proven, not refuted).
    pub fn le_for_all_sizes(&self, other: &Poly) -> bool {
        other
            .checked_sub(self)
            .is_some_and(|d| d.nonnegative_for_all_sizes())
    }

    /// A polynomial at least both `self` and `other` for all sizes `>= 1`:
    /// the larger when one is proven larger, and otherwise the larger
    /// coefficient of each monomial, which is at least both wherever the
    /// sizes are non-negative.
    pub fn upper_join(&self, other: &Poly) -> Poly {
        if self.le_for_all_sizes(other) {
            return other.clone();
        }
        if other.le_for_all_sizes(self) {
            return self.clone();
        }
        self.coefficientwise(other, |a, b| a.max(b))
    }

    /// A polynomial at most both `self` and `other`, as [`Poly::upper_join`].
    pub fn lower_join(&self, other: &Poly) -> Poly {
        if self.le_for_all_sizes(other) {
            return self.clone();
        }
        if other.le_for_all_sizes(self) {
            return other.clone();
        }
        self.coefficientwise(other, |a, b| a.min(b))
    }

    fn coefficientwise(&self, other: &Poly, pick: fn(Rational, Rational) -> Rational) -> Poly {
        let mut terms = BTreeMap::new();
        let keys: BTreeSet<&Monomial> = self.terms.keys().chain(other.terms.keys()).collect();
        for m in keys {
            let a = self.terms.get(m).copied().unwrap_or(Rational::ZERO);
            let b = other.terms.get(m).copied().unwrap_or(Rational::ZERO);
            let c = pick(a, b);
            if !c.is_zero() {
                terms.insert(m.clone(), c);
            }
        }
        Poly { terms }
    }
}

fn superscript(n: u32) -> String {
    const DIGITS: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    n.to_string()
        .bytes()
        .map(|b| DIGITS[usize::from(b - b'0')])
        .collect()
}

fn monomial_text(m: &Monomial) -> String {
    let parts: Vec<String> = m
        .powers()
        .map(|(name, e)| {
            if e == 1 {
                name.to_string()
            } else {
                format!("{name}{}", superscript(e))
            }
        })
        .collect();
    parts.join("·")
}

/// A term without its sign: `N³/3`, `2·N·M`, `7`, `1/2`.
fn term_text(m: &Monomial, c: Rational) -> String {
    let (n, d) = (c.numer().unsigned_abs(), c.denom());
    let vars = monomial_text(m);
    if vars.is_empty() {
        return if d == 1 {
            n.to_string()
        } else {
            format!("{n}/{d}")
        };
    }
    match (n, d) {
        (1, 1) => vars,
        (_, 1) => format!("{n}·{vars}"),
        (1, _) => format!("{vars}/{d}"),
        _ => format!("{n}·{vars}/{d}"),
    }
}

impl fmt::Display for Poly {
    /// `2·N·K·M`, `N³/3 + 2·N²`, `0`: the highest degree first, the
    /// constant last.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let terms = self.terms();
        if terms.is_empty() {
            return f.write_str("0");
        }
        for (i, (m, c)) in terms.iter().enumerate() {
            let negative = c.numer() < 0;
            match (i, negative) {
                (0, false) => {}
                (0, true) => f.write_str("-")?,
                (_, false) => f.write_str(" + ")?,
                (_, true) => f.write_str(" - ")?,
            }
            f.write_str(&term_text(m, *c))?;
        }
        Ok(())
    }
}

/// Why an expression is not a polynomial in sizes.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PolyError {
    pub span: Span,
    pub message: String,
    pub help: Option<String>,
}

impl PolyError {
    fn new(span: Span, message: impl Into<String>) -> PolyError {
        PolyError {
            span,
            message: message.into(),
            help: None,
        }
    }

    fn help(mut self, help: impl Into<String>) -> PolyError {
        self.help = Some(help.into());
        self
    }
}

const MAX_EXPR_DEPTH: usize = 128;

/// The polynomial an expression writes: numbers, names of sizes, `+ - *`,
/// `/` by a number, and `^` with a whole-number exponent, in parentheses as
/// needed. Every name it uses is appended to `names` with its span, for the
/// caller to check against the sizes in scope. A number with a unit is an
/// error (`2 N` is two newtons; write `2 * N`).
pub fn poly_from_expr(
    src: &str,
    e: &Expr,
    names: &mut Vec<(String, Span)>,
) -> Result<Poly, PolyError> {
    from_expr(src, e, names, 0)
}

fn too_large(span: Span) -> PolyError {
    PolyError::new(span, "this expression is too large").with_limit()
}

impl PolyError {
    fn with_limit(self) -> PolyError {
        self.help("keep the number of terms and the degree small")
    }
}

fn from_expr(
    src: &str,
    e: &Expr,
    names: &mut Vec<(String, Span)>,
    depth: usize,
) -> Result<Poly, PolyError> {
    if depth > MAX_EXPR_DEPTH {
        return Err(too_large(e.span));
    }
    let go = |x: &Expr, names: &mut Vec<(String, Span)>| from_expr(src, x, names, depth + 1);
    match &e.kind {
        ExprKind::Paren(inner) => go(inner, names),
        ExprKind::Literal(Literal::Int(s) | Literal::Float(s)) => {
            let text = span_text(src, *s).unwrap_or("");
            number(text)
                .map(Poly::constant)
                .ok_or_else(|| PolyError::new(e.span, "this number is too large or too precise"))
        }
        ExprKind::UnitLiteral { value, unit } => {
            let n = span_text(src, value.span()).unwrap_or("");
            let u = span_text(src, unit.span()).unwrap_or("");
            Err(
                PolyError::new(e.span, "a size is not a unit: this reads as a quantity")
                    .help(format!("write `{n} * {u}` for {n} times the size `{u}`")),
            )
        }
        ExprKind::Name(id) => {
            let name = span_text(src, id.span).unwrap_or("?").to_string();
            let p = Poly::var(&name);
            names.push((name, id.span));
            Ok(p)
        }
        ExprKind::Unary {
            op: UnaryOp::Neg,
            operand,
            ..
        } => go(operand, names)?
            .checked_neg()
            .ok_or_else(|| too_large(e.span)),
        ExprKind::Unary {
            op: UnaryOp::Pos,
            operand,
            ..
        } => go(operand, names),
        ExprKind::Binary { op, lhs, rhs, .. } => {
            let l = go(lhs, names)?;
            let r = go(rhs, names)?;
            let big = || too_large(e.span);
            match op {
                BinaryOp::Add => l.checked_add(&r).ok_or_else(big),
                BinaryOp::Sub => l.checked_sub(&r).ok_or_else(big),
                BinaryOp::Mul => l.checked_mul(&r).ok_or_else(big),
                BinaryOp::Div => {
                    let d = r.as_constant().filter(|d| !d.is_zero()).ok_or_else(|| {
                        PolyError::new(rhs.span, "a size can only be divided by a number")
                            .help("write the cost as a sum of terms with fractional coefficients, e.g. `N^3 / 3`")
                    })?;
                    let inv = d.checked_recip().map_err(|_| big())?;
                    l.checked_scale(inv).ok_or_else(big)
                }
                BinaryOp::Pow => {
                    let n = r
                        .as_constant()
                        .filter(|n| {
                            n.is_integer() && (0..=i64::from(MAX_DEGREE)).contains(&n.numer())
                        })
                        .ok_or_else(|| {
                            PolyError::new(
                                rhs.span,
                                "the exponent must be a whole number, at most 64",
                            )
                        })?;
                    let n = u32::try_from(n.numer()).map_err(|_| big())?;
                    l.checked_pow(n).ok_or_else(big)
                }
                _ => Err(not_poly(e.span)),
            }
        }
        _ => Err(not_poly(e.span)),
    }
}

fn not_poly(span: Span) -> PolyError {
    PolyError::new(span, "this is not a polynomial in the sizes")
        .help("use numbers, sizes, `+`, `-`, `*`, `^` with a whole exponent, and `/` by a number")
}

/// The exact value of a number literal's text: digits with an optional
/// fraction and an optional exponent (`2`, `0.5`, `1e9`, `1_000`).
fn number(text: &str) -> Option<Rational> {
    let text: String = text.chars().filter(|c| *c != '_').collect();
    let (mantissa, exp) = match text.split_once(['e', 'E']) {
        Some((m, e)) => (m, e.parse::<i32>().ok()?),
        None => (text.as_str(), 0),
    };
    if exp.unsigned_abs() > 18 {
        return None;
    }
    let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let frac = frac.trim_end_matches('0');
    let digits = |s: &str| s.bytes().all(|b| b.is_ascii_digit());
    if (int.is_empty() && frac.is_empty()) || !digits(int) || !digits(frac) {
        return None;
    }
    let num: i64 = format!("{int}{frac}").parse().ok()?;
    let mut r =
        Rational::checked_new(num, 10i64.checked_pow(u32::try_from(frac.len()).ok()?)?).ok()?;
    let scale = Rational::integer(10i64.checked_pow(exp.unsigned_abs())?);
    r = if exp >= 0 {
        r.checked_mul(scale).ok()?
    } else {
        r.checked_div(scale).ok()?
    };
    Some(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Poly {
        let src = s.to_string();
        let lexed = sky_syntax::lex(&src).expect("lexes");
        let (module, errors) = sky_ast::parse_module(&lexed.tokens, &src);
        assert!(errors.is_empty(), "{errors:?}");
        let sky_ast::StmtKind::Expr(e) = &module.stmts[0].kind else {
            panic!("not an expression")
        };
        poly_from_expr(&src, e, &mut Vec::new()).expect("polynomial")
    }

    #[test]
    fn arithmetic_is_exact() {
        assert_eq!(p("(N + 1) * (N - 1)"), p("N^2 - 1"));
        assert_eq!(p("N^3 / 3 + N^3 / 3 + N^3 / 3"), p("N^3"));
        assert_eq!(p("0.5 * N"), p("N / 2"));
        assert_eq!(p("1e3 + 1"), p("1001"));
        assert_eq!(p("N - N"), Poly::zero());
    }

    #[test]
    fn printing_puts_the_dominant_term_first() {
        assert_eq!(p("2 * N * K * M").to_string(), "2·K·M·N");
        assert_eq!(p("N^3 / 3 + 2 * N^2 + 7").to_string(), "N³/3 + 2·N² + 7");
        assert_eq!(p("N - 1").to_string(), "N - 1");
        assert_eq!(p("2 * N^2 / 3").to_string(), "2·N²/3");
        assert_eq!(Poly::zero().to_string(), "0");
    }

    #[test]
    fn comparison_is_for_all_sizes_at_least_one() {
        // N^2 <= N^3 holds for N >= 1 only because of the domain.
        assert!(p("N^2").le_for_all_sizes(&p("N^3")));
        assert!(!p("N^3").le_for_all_sizes(&p("N^2")));
        assert!(p("N").le_for_all_sizes(&p("N")));
        // 2 N^3 + 5 N is not within 2 N^3, but 2 N^3 is within 3 N^3.
        assert!(!p("2 * N^3 + 5 * N").le_for_all_sizes(&p("2 * N^3")));
        assert!(p("2 * N^3").le_for_all_sizes(&p("3 * N^3")));
        // 3 N^2 <= N^3 + 2 N^2 needs N >= 1: proven; N^3 <= 4 N^2 is false for large N.
        assert!(p("3 * N^2").le_for_all_sizes(&p("N^3 + 2 * N^2")));
        assert!(!p("N^3").le_for_all_sizes(&p("4 * N^2")));
        // Two sizes.
        assert!(p("N * M").le_for_all_sizes(&p("N * M * K")));
        assert!(!p("N * M * K").le_for_all_sizes(&p("N * M")));
        assert!(p("1").le_for_all_sizes(&p("N")));
    }

    #[test]
    fn soundness_against_sampled_points() {
        // Whenever `le_for_all_sizes` says yes, no sampled point disagrees.
        let polys = [
            p("N"),
            p("N^2"),
            p("2 * N^2 + N"),
            p("N^3 / 3 + 2 * N^2"),
            p("5"),
            p("3 * N"),
            p("N^2 + N + 1"),
            p("N - 1"),
        ];
        for a in &polys {
            for b in &polys {
                if !a.le_for_all_sizes(b) {
                    continue;
                }
                for n in [1i64, 2, 3, 7, 100, 12345] {
                    let mut pt = BTreeMap::new();
                    pt.insert("N".to_string(), Rational::integer(n));
                    let (x, y) = (a.eval(&pt).unwrap(), b.eval(&pt).unwrap());
                    assert!(x <= y, "{a} <= {b} claimed, but at N={n}: {x} > {y}");
                }
            }
        }
    }

    #[test]
    fn source_text_reads_back() {
        for text in [
            "2 * N^3 + 5 * N",
            "N^3 / 3 + 2 * N^2",
            "M^2 * N",
            "7",
            "2 * N^2 / 3",
        ] {
            assert_eq!(p(text).to_source(), text, "{text}");
            assert_eq!(p(&p(text).to_source()), p(text));
        }
    }

    #[test]
    fn joins_bound_both_sides() {
        let (a, b) = (p("N^2 + 5"), p("N^3"));
        let hi = a.upper_join(&b);
        assert!(a.le_for_all_sizes(&hi) && b.le_for_all_sizes(&hi));
        let lo = a.lower_join(&b);
        assert!(lo.le_for_all_sizes(&a) && lo.le_for_all_sizes(&b));
    }

    #[test]
    fn substitution_is_simultaneous() {
        let mut map = BTreeMap::new();
        map.insert("N".to_string(), p("M"));
        map.insert("M".to_string(), p("N + 1"));
        assert_eq!(p("N * M").subst(&map).unwrap(), p("M * (N + 1)"));
    }

    #[test]
    fn what_is_not_a_polynomial_is_an_error() {
        for bad in ["N / M", "N ^ M", "N ^ 0.5", "2 N", "sqrt(N)", "N < 2"] {
            let src = bad.to_string();
            let lexed = sky_syntax::lex(&src).expect("lexes");
            let (module, errors) = sky_ast::parse_module(&lexed.tokens, &src);
            assert!(errors.is_empty(), "{bad}: {errors:?}");
            let sky_ast::StmtKind::Expr(e) = &module.stmts[0].kind else {
                panic!()
            };
            assert!(poly_from_expr(&src, e, &mut Vec::new()).is_err(), "{bad}");
        }
    }

    #[test]
    fn size_and_degree_are_capped() {
        let big = p("N + 1").checked_pow(60).unwrap();
        assert!(big.checked_pow(2).is_none());
    }
}
