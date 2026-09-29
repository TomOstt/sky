//! Every node's span covers its full source range.

mod common;
use common::{at, lex};
use sky_ast::{
    parse_expr, parse_module, AssignTarget, CompoundOp, Expr, ExprKind, IndexItem, Stmt, StmtKind,
    UnitExpr,
};

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

fn stmt(src: &str) -> Stmt {
    let (module, errors) = parse_module(&lex(src), src);
    assert!(errors.is_empty(), "{src:?}: {errors:#?}");
    assert_eq!(module.stmts.len(), 1, "{src:?}");
    module.stmts.into_iter().next().unwrap()
}

#[test]
fn index_and_slice_item_spans() {
    let src = "A[0, :, 1:3, :n, 2:]";
    let ExprKind::Index { base, items } = parse(src).kind else {
        panic!()
    };
    assert_eq!(at(src, base.span), "A");
    let spans: Vec<&str> = items.iter().map(|i| at(src, i.span())).collect();
    assert_eq!(spans, ["0", ":", "1:3", ":n", "2:"]);
    assert!(items[0].as_expr().is_some());
    assert!(items[1..].iter().all(|i| i.as_expr().is_none()));
    let IndexItem::Slice {
        start,
        stop,
        colon_span,
        ..
    } = &items[2]
    else {
        panic!()
    };
    assert_eq!(at(src, start.as_ref().unwrap().span), "1");
    assert_eq!(at(src, stop.as_ref().unwrap().span), "3");
    assert_eq!(colon_span.range(), 9..10);
    let IndexItem::Slice {
        start: None,
        stop: None,
        colon_span,
        span,
    } = &items[1]
    else {
        panic!()
    };
    assert_eq!((colon_span.range(), span.range()), (5..6, 5..6));
    // Spaces inside a slice are not part of the item's span.
    let src = "A[ 1 : 3 ]";
    let ExprKind::Index { items, .. } = parse(src).kind else {
        panic!()
    };
    assert_eq!(at(src, items[0].span()), "1 : 3");
}

#[test]
fn assignment_target_spans() {
    let src = "A[i, j] = x";
    let s = stmt(src);
    assert_eq!(at(src, s.span), src);
    let StmtKind::Assign { target, value } = &s.kind else {
        panic!("{s:#?}")
    };
    assert_eq!(at(src, target.span()), "A[i, j]");
    assert_eq!(at(src, value.span), "x");
    let AssignTarget::Index {
        base,
        items,
        bracket_span,
        ..
    } = target
    else {
        panic!()
    };
    assert_eq!(at(src, base.span), "A");
    assert_eq!(at(src, *bracket_span), "[i, j]");
    assert_eq!(items.len(), 2);

    // The bracket span starts at `[`, not at the end of the base.
    let src = "a.b [0, :] = v";
    let StmtKind::Assign {
        target: AssignTarget::Index {
            base, bracket_span, ..
        },
        ..
    } = &stmt(src).kind
    else {
        panic!()
    };
    assert_eq!(at(src, base.span), "a.b");
    assert_eq!(at(src, *bracket_span), "[0, :]");

    let src = "a.b.c = x";
    let StmtKind::Assign {
        target: AssignTarget::Attribute { base, name, span },
        ..
    } = &stmt(src).kind
    else {
        panic!()
    };
    assert_eq!(at(src, base.span), "a.b");
    assert_eq!(at(src, name.span), "c");
    assert_eq!(at(src, *span), "a.b.c");

    let src = "x = 1";
    let StmtKind::Assign {
        target: AssignTarget::Name(id),
        ..
    } = &stmt(src).kind
    else {
        panic!()
    };
    assert_eq!(at(src, id.span), "x");
}

#[test]
fn compound_assignment_spans() {
    for (src, op, symbol, target, value) in [
        ("A += B", CompoundOp::Add, "+=", "A", "B"),
        ("x -= 1 m", CompoundOp::Sub, "-=", "x", "1 m"),
        ("A[i, j] *= 2", CompoundOp::Mul, "*=", "A[i, j]", "2"),
        ("v.x /= n", CompoundOp::Div, "/=", "v.x", "n"),
        ("x ^= 2", CompoundOp::Pow, "^=", "x", "2"),
        ("P @= F", CompoundOp::MatMul, "@=", "P", "F"),
        ("x^=2", CompoundOp::Pow, "^=", "x", "2"),
    ] {
        let s = stmt(src);
        assert_eq!(at(src, s.span), src);
        let StmtKind::CompoundAssign {
            target: t,
            op: o,
            op_span,
            value: v,
        } = &s.kind
        else {
            panic!("{src:?}: {s:#?}")
        };
        assert_eq!(*o, op, "{src:?}");
        assert_eq!(o.symbol(), symbol);
        // op_span covers the whole two-character token.
        assert_eq!(at(src, *op_span), symbol, "{src:?}");
        assert_eq!(at(src, t.span()), target, "{src:?}");
        assert_eq!(at(src, v.span), value, "{src:?}");
    }
}
