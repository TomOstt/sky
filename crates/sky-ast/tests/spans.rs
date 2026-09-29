//! Every node's span covers its full source range.

mod common;
use common::{at, lex};
use sky_ast::{parse_expr, parse_module, Expr, ExprKind, StmtKind, UnitExpr};

fn parse(src: &str) -> Expr {
    let (expr, errors) = parse_expr(&lex(src), src);
    assert!(errors.is_empty(), "{errors:#?}");
    expr.unwrap()
}

#[test]
fn binary_and_unit_literal_spans() {
    let src = "9.8 m/s^2 + v";
    let e = parse(src);
    assert_eq!(at(src, e.span), "9.8 m/s^2 + v");
    let ExprKind::Binary {
        lhs, rhs, op_span, ..
    } = &e.kind
    else {
        panic!()
    };
    assert_eq!(at(src, *op_span), "+");
    assert_eq!(at(src, lhs.span), "9.8 m/s^2");
    assert_eq!(at(src, rhs.span), "v");
    let ExprKind::UnitLiteral { value, unit } = &lhs.kind else {
        panic!()
    };
    assert_eq!(at(src, value.span()), "9.8");
    assert_eq!(at(src, unit.span()), "m/s^2");
    let UnitExpr::Div {
        lhs: m, rhs: s2, ..
    } = unit
    else {
        panic!()
    };
    assert_eq!(at(src, m.span()), "m");
    assert_eq!(at(src, s2.span()), "s^2");
    let UnitExpr::Pow {
        base,
        exponent_span,
        ..
    } = &**s2
    else {
        panic!()
    };
    assert_eq!(at(src, base.span()), "s");
    assert_eq!(at(src, *exponent_span), "2");
}

#[test]
fn negative_and_rational_exponent_spans() {
    let src = "3 s^-2";
    let ExprKind::UnitLiteral { unit, .. } = parse(src).kind else {
        panic!()
    };
    let UnitExpr::Pow {
        exponent_span,
        span,
        ..
    } = unit
    else {
        panic!()
    };
    assert_eq!(at(src, exponent_span), "-2");
    assert_eq!(at(src, span), "s^-2");

    let src = "1 kg*m^(1/2)";
    let ExprKind::UnitLiteral { unit, .. } = parse(src).kind else {
        panic!()
    };
    assert_eq!(at(src, unit.span()), "kg*m^(1/2)");
}

#[test]
fn postfix_prefix_and_delimited_spans() {
    for (src, whole) in [
        ("f(a, b)", "f(a, b)"),
        ("f()", "f()"),
        ("Matrix[3, 4]", "Matrix[3, 4]"),
        ("( a )", "( a )"),
        ("-x", "-x"),
        ("∂f", "∂f"),
        ("A⁻¹", "A⁻¹"),
        ("a.b.c", "a.b.c"),
        ("10 km", "10 km"),
        ("-x^2", "-x^2"),
    ] {
        assert_eq!(at(src, parse(src).span), whole, "{src:?}");
    }
    let src = "A⁻¹ @ b";
    let ExprKind::Binary { lhs, .. } = parse(src).kind else {
        panic!()
    };
    let ExprKind::Inverse { op_span, operand } = &lhs.kind else {
        panic!()
    };
    assert_eq!(at(src, *op_span), "⁻¹");
    assert_eq!(at(src, operand.span), "A");

    let src = "obj.attr";
    let ExprKind::Attribute { name, .. } = parse(src).kind else {
        panic!()
    };
    assert_eq!(name.text(src), Some("attr"));
}

#[test]
fn statement_spans_exclude_the_newline() {
    let src = "A: Matrix[3, 4] = zeros(3, 4)\nx = 1\n";
    let (module, errors) = parse_module(&lex(src), src);
    assert!(errors.is_empty());
    assert_eq!(
        at(src, module.stmts[0].span),
        "A: Matrix[3, 4] = zeros(3, 4)"
    );
    assert_eq!(at(src, module.stmts[1].span), "x = 1");
    assert_eq!(module.span.range(), 0..src.len());
    let StmtKind::Binding { name, ty, value } = &module.stmts[0].kind else {
        panic!()
    };
    assert_eq!(at(src, name.span), "A");
    assert_eq!(at(src, ty.span), "Matrix[3, 4]");
    assert_eq!(at(src, value.as_ref().unwrap().span), "zeros(3, 4)");
}
