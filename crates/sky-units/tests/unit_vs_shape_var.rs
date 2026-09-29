//! `N` is both a unit (the newton) and a common shape-variable name
//! (`Matrix[N, N]`). The parser tells them apart by position only: a unit is
//! an identifier directly after a numeric literal (`10 N`), and everything
//! else is a name. These tests pin that, so adding `N` to the unit table
//! cannot change how `Matrix[N, N]` parses.

use sky_ast::{ast::span_text, Expr, ExprKind, Module, StmtKind, UnitExpr};
use sky_units::{lookup, Dimension};

fn parse(src: &str) -> Module {
    let lexed = sky_syntax::lex(src).expect("lexes");
    let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
    assert!(errors.is_empty(), "{src:?}: {errors:#?}");
    module
}

fn value(module: &Module, i: usize) -> &Expr {
    match &module.stmts[i].kind {
        StmtKind::Assign { value, .. } => value,
        StmtKind::Binding {
            value: Some(value), ..
        } => value,
        other => panic!("statement {i} has no value: {other:?}"),
    }
}

fn unit_name<'s>(e: &Expr, src: &'s str) -> &'s str {
    match &e.kind {
        ExprKind::UnitLiteral {
            unit: UnitExpr::Name(id),
            ..
        } => span_text(src, id.span).unwrap(),
        other => panic!("not a single-unit literal: {other:?}"),
    }
}

#[test]
fn newton_after_a_number_is_a_unit() {
    let src = "F = 10 N\n";
    let m = parse(src);
    let name = unit_name(value(&m, 0), src);
    assert_eq!(name, "N");
    assert_eq!(lookup(name).unwrap().dim, Dimension::FORCE);
}

#[test]
fn n_in_a_shape_annotation_is_a_name_not_a_unit() {
    let src = "A: Matrix[N, N]\nB: Matrix[N, N] = A\n";
    let m = parse(src);
    for stmt in &m.stmts {
        let StmtKind::Binding { ty, .. } = &stmt.kind else {
            panic!("{stmt:?}");
        };
        let ExprKind::Index { indices, .. } = &ty.kind else {
            panic!("{ty:?}");
        };
        for ix in indices {
            let ExprKind::Name(id) = &ix.kind else {
                panic!("shape dim is not a plain name: {ix:?}");
            };
            assert_eq!(span_text(src, id.span), Some("N"));
        }
    }
}

#[test]
fn a_value_named_n_and_the_unit_n_coexist() {
    // `N = 3` binds a value; `2 N` is still 2 newtons, and `2 * N` is the
    // value. `2 N * N` is 2 newtons times the value (spaces around `*`),
    // while `2 N*N` is the unit N² (no spaces).
    let src = "N = 3\na = 2 N\nb = 2 * N\nc = 2 N * N\nd = 2 N*N\n";
    let m = parse(src);
    assert_eq!(unit_name(value(&m, 1), src), "N");
    assert!(matches!(value(&m, 2).kind, ExprKind::Binary { .. }));
    let ExprKind::Binary { lhs, rhs, .. } = &value(&m, 3).kind else {
        panic!();
    };
    assert_eq!(unit_name(lhs, src), "N");
    assert!(matches!(rhs.kind, ExprKind::Name(_)));
    assert!(matches!(
        value(&m, 4).kind,
        ExprKind::UnitLiteral {
            unit: UnitExpr::Mul { .. },
            ..
        }
    ));
}

#[test]
fn a_unit_literal_inside_a_shape_is_not_a_shape_variable() {
    // `Matrix[2 N, N]` parses its first dim as the quantity "2 newtons".
    // The checker must reject it as a shape dimension, never read it as the
    // variable N.
    let src = "A: Matrix[2 N, N]\n";
    let m = parse(src);
    let StmtKind::Binding { ty, .. } = &m.stmts[0].kind else {
        panic!();
    };
    let ExprKind::Index { indices, .. } = &ty.kind else {
        panic!();
    };
    assert_eq!(unit_name(&indices[0], src), "N");
    assert!(matches!(indices[1].kind, ExprKind::Name(_)));
}
