//! `value ± sigma` (and `+/-`): precedence, the shared unit, spans, depth.

mod common;
use common::{at, expr_errors, module_sexpr, sexpr};
use sky_ast::{parse_expr, ExprKind, MAX_DEPTH};

fn check(cases: &[(&str, &str)]) {
    for (src, want) in cases {
        assert_eq!(sexpr(src), *want, "source: {src:?}");
    }
}

#[test]
fn glyph_and_ascii_spelling_parse_alike() {
    for (glyph, ascii) in [
        ("5.31 ± 0.02 kg", "5.31 +/- 0.02 kg"),
        ("x ± dx", "x +/- dx"),
        ("a+b±c", "a+b+/-c"),
    ] {
        assert_eq!(sexpr(glyph), sexpr(ascii), "{glyph:?}");
    }
}

#[test]
fn the_unit_is_shared_when_a_bare_number_precedes() {
    check(&[
        (
            "5.31 ± 0.02 kg",
            "(plusminus (unit 5.31 kg) (unit 0.02 kg))",
        ),
        (
            "5 +/- 1 m/s^2",
            "(plusminus (unit 5 (/ m (^ s 2))) (unit 1 (/ m (^ s 2))))",
        ),
        // A sign on the number is kept; the unit goes under it.
        (
            "-5.31 ± 0.02 kg",
            "(plusminus (neg (unit 5.31 kg)) (unit 0.02 kg))",
        ),
        (
            "+5.31 ± 0.02 kg",
            "(plusminus (pos (unit 5.31 kg)) (unit 0.02 kg))",
        ),
        // Each side may carry its own unit, so `kg ± g` works.
        ("5.31 kg ± 20 g", "(plusminus (unit 5.31 kg) (unit 20 g))"),
    ]);
}

#[test]
fn the_unit_is_not_shared_in_any_other_shape() {
    check(&[
        // A plain number on both sides: a dimensionless value.
        ("5.31 ± 0.02", "(plusminus 5.31 0.02)"),
        // A name or a group is not a bare number: the checker decides.
        ("x ± 0.02 kg", "(plusminus x (unit 0.02 kg))"),
        (
            "(5.31) ± 0.02 kg",
            "(plusminus (paren 5.31) (unit 0.02 kg))",
        ),
        ("-x ± 0.02 kg", "(plusminus (neg x) (unit 0.02 kg))"),
        // The sigma is not a unit literal.
        ("5.31 ± dx", "(plusminus 5.31 dx)"),
        ("5.31 kg ± dx", "(plusminus (unit 5.31 kg) dx)"),
    ]);
}

#[test]
fn precedence_is_that_of_plus_and_minus() {
    check(&[
        // Looser than `*` and `^`, tighter than comparison.
        ("a * b ± c * d", "(plusminus (* a b) (* c d))"),
        ("a ± b ^ 2", "(plusminus a (^ b 2))"),
        ("a ± b == c", "(== (plusminus a b) c)"),
        ("a < b ± c", "(< a (plusminus b c))"),
        // Left-associative, and level with `+` and `-`.
        ("a + b ± c", "(plusminus (+ a b) c)"),
        ("a ± b + c", "(+ (plusminus a b) c)"),
        ("a ± b - c", "(- (plusminus a b) c)"),
        ("a - b ± c", "(plusminus (- a b) c)"),
        ("a ± b ± c", "(plusminus (plusminus a b) c)"),
        // Prefix and postfix bind tighter.
        ("-a ± -b", "(plusminus (neg a) (neg b))"),
        (
            "f(x) ± g(y)[0]",
            "(plusminus (call f x) (index (call g y) 0))",
        ),
        // `as` is looser still: it shows the whole uncertain value.
        ("x ± dx as km", "(as (plusminus x dx) km)"),
        ("(x ± dx) as km", "(as (paren (plusminus x dx)) km)"),
        // `if`, `and` and `or` are looser.
        ("a ± b if c else d", "(ifelse c (plusminus a b) d)"),
    ]);
}

#[test]
fn a_unit_sum_after_the_sigma_is_not_absorbed() {
    // The right operand stops before `+`, so this is `(a ± 1 m) + 2 m`.
    check(&[("a ± 1 m + 2 m", "(+ (plusminus a (unit 1 m)) (unit 2 m))")]);
}

#[test]
fn spans_cover_both_operands() {
    let src = "m = 5.31 ± 0.02 kg\n";
    let tokens = common::lex(src);
    let (module, errors) = sky_ast::parse_module(&tokens, src);
    assert!(errors.is_empty(), "{errors:?}");
    let sky_ast::StmtKind::Assign { value, .. } = &module.stmts[0].kind else {
        panic!("not an assignment");
    };
    assert_eq!(at(src, value.span), "5.31 ± 0.02 kg");
    let ExprKind::PlusMinus {
        value: v,
        sigma,
        op_span,
    } = &value.kind
    else {
        panic!("not a ±: {:?}", value.kind);
    };
    assert_eq!(at(src, *op_span), "±");
    assert_eq!(at(src, sigma.span), "0.02 kg");
    // The shared-unit value is a synthetic unit literal that runs from the
    // number to the end of the sigma, where its unit is.
    assert_eq!(at(src, v.span), "5.31 ± 0.02 kg");
    let ExprKind::UnitLiteral { value: n, unit } = &v.kind else {
        panic!("not a unit literal: {:?}", v.kind);
    };
    assert_eq!(at(src, n.span()), "5.31");
    assert_eq!(at(src, unit.span()), "kg");

    let src = "x +/- dx";
    let (e, errors) = parse_expr(&common::lex(src), src);
    assert!(errors.is_empty());
    let e = e.expect("expression");
    let ExprKind::PlusMinus { op_span, .. } = &e.kind else {
        panic!("not a ±");
    };
    assert_eq!(at(src, *op_span), "+/-");
}

#[test]
fn statement_forms() {
    assert_eq!(
        module_sexpr("m = 5.31 ± 0.02 kg\nprint(m as g)\n"),
        "(= m (plusminus (unit 5.31 kg) (unit 0.02 kg)))\n(call print (as m g))"
    );
}

#[test]
fn errors() {
    // Missing right operand.
    assert!(!expr_errors("5 ±").is_empty());
    // Missing left operand: `±` is not a prefix operator.
    assert!(!expr_errors("± 0.1").is_empty());
    // Not an assignment target.
    let src = "x ± 1 = 2\n";
    let (_, errs) = sky_ast::parse_module(&common::lex(src), src);
    assert!(!errs.is_empty());
}

#[test]
fn each_plus_minus_is_one_nesting_level() {
    // `a ± a ± ... ± a` is left-nested, like `a + a + ... + a`.
    let chain = |n: usize| vec!["a"; n + 1].join(" ± ");
    assert!(expr_errors(&chain(MAX_DEPTH as usize - 1)).is_empty());
    assert!(!expr_errors(&chain(MAX_DEPTH as usize + 1)).is_empty());
}
