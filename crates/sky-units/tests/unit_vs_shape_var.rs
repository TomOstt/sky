//! `N` is both a unit (the newton) and a common shape-variable name
//! (`Matrix[N, N]`). The parser tells them apart by position only: a unit is
//! an identifier directly after a numeric literal (`10 N`) or in a type
//! annotation outside `[...]` (`f: N`); inside a shape's brackets, and
//! everywhere else, it is a name. These tests pin that, so adding `N` to the
//! unit table cannot change how `Matrix[N, N]` parses.

use sky_ast::{ast::span_text, DimExpr, Expr, ExprKind, Module, StmtKind, UnitExpr};
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
        let shape = ty.shape.as_ref().expect("a shape");
        assert_eq!(shape.dims.len(), 2, "{ty:?}");
        assert!(ty.unit.is_none(), "{ty:?}");
        for dim in &shape.dims {
            let DimExpr::Name(id) = dim else {
                panic!("shape dim is not a plain name: {dim:?}");
            };
            assert_eq!(span_text(src, id.span), Some("N"));
        }
    }
}

#[test]
fn a_lone_n_or_k_after_the_colon_is_a_unit() {
    // A bare name is always a unit (newtons, kelvin), never a size
    // variable: those only exist inside `[...]`. And after a shape, `N` is
    // the unit of the elements while the `N`s inside are size variables.
    let src = "f: N\nt: K\nA: Matrix[N, N] N\n";
    let m = parse(src);
    let units: Vec<&UnitExpr> = m
        .stmts
        .iter()
        .map(|s| match &s.kind {
            StmtKind::Binding { ty, .. } => ty.unit.as_ref().expect("a unit"),
            other => panic!("{other:?}"),
        })
        .collect();
    for (u, want) in units.iter().zip(["N", "K", "N"]) {
        let UnitExpr::Name(id) = u else {
            panic!("{u:?}")
        };
        assert_eq!(span_text(src, id.span), Some(want));
        assert!(lookup(want).is_some());
    }
    let StmtKind::Binding { ty, .. } = &m.stmts[0].kind else {
        panic!()
    };
    assert!(ty.shape.is_none());
    let StmtKind::Binding { ty, .. } = &m.stmts[2].kind else {
        panic!()
    };
    assert_eq!(ty.shape.as_ref().map(|s| s.dims.len()), Some(2));
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
fn a_unit_literal_inside_a_shape_is_a_parse_error() {
    // `Matrix[2 N, N]`: a dimension is an integer or a name, so "2 newtons"
    // is never read as the variable N.
    let src = "A: Matrix[2 N, N]\n";
    let lexed = sky_syntax::lex(src).expect("lexes");
    let (_, errors) = sky_ast::parse_module(&lexed.tokens, src);
    assert_eq!(errors.len(), 1, "{errors:#?}");
    assert!(
        errors[0].message.contains("expected `,` or `]`"),
        "{errors:#?}"
    );
}
