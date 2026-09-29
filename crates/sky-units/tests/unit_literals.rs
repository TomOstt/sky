//! `dt = 10 ms` parses as a unit literal whose unit is the table's `ms`
//! (time, scale 0.001), not `m` times `s`.

use sky_ast::{ast::span_text, ExprKind, StmtKind, UnitExpr};
use sky_units::{lookup, Dimension};

#[test]
fn dt_ten_ms_is_time() {
    let src = "dt = 10 ms\n";
    let lexed = sky_syntax::lex(src).expect("lexes");
    let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
    assert!(errors.is_empty(), "{errors:#?}");
    let StmtKind::Assign { value, .. } = &module.stmts[0].kind else {
        panic!("{:?}", module.stmts[0]);
    };
    let ExprKind::UnitLiteral {
        unit: UnitExpr::Name(id),
        ..
    } = &value.kind
    else {
        panic!("not a single-unit literal: {value:?}");
    };
    let name = span_text(src, id.span).unwrap();
    assert_eq!(name, "ms");
    let unit = lookup(name).unwrap();
    assert_eq!(unit.dim, Dimension::TIME);
    assert_eq!(unit.scale, 0.001);
}
