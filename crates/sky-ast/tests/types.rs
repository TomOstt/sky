//! Type annotations (`name: tag* shape [unit]`): spans of every part, the
//! disambiguation rules, and bounded error output.

mod common;
use common::{at, lex, module_sexpr};
use sky_ast::{parse_module, DimExpr, ShapeKind, StmtKind, TypeExpr, UnitExpr, MAX_QUOTED_BYTES};

fn ty(src: &str) -> TypeExpr {
    let (module, errors) = parse_module(&lex(src), src);
    assert!(errors.is_empty(), "{src:?}: {errors:#?}");
    let StmtKind::Binding { ty, .. } = module.stmts.into_iter().next().unwrap().kind else {
        panic!("{src:?} is not a binding");
    };
    ty
}

fn errors(src: &str) -> Vec<String> {
    let (_, errors) = parse_module(&lex(src), src);
    errors.into_iter().map(|e| e.message).collect()
}

#[test]
fn every_part_has_an_exact_span() {
    let src = "K: Symmetric Diagonal Matrix[3, N,] N/m = x";
    let t = ty(src);
    assert_eq!(at(src, t.span), "Symmetric Diagonal Matrix[3, N,] N/m");
    let tags: Vec<&str> = t.tags.iter().map(|i| at(src, i.span)).collect();
    assert_eq!(tags, ["Symmetric", "Diagonal"]);
    let shape = t.shape.expect("a shape");
    assert_eq!(shape.kind, ShapeKind::Matrix);
    assert_eq!(at(src, shape.kind_span), "Matrix");
    assert_eq!(at(src, shape.span), "Matrix[3, N,]");
    assert!(matches!(shape.dims[0], DimExpr::Int(_)));
    assert!(matches!(shape.dims[1], DimExpr::Name(_)));
    assert_eq!(at(src, t.unit.expect("a unit").span()), "N/m");
}

#[test]
fn a_lone_name_is_a_unit_never_a_size_variable() {
    for (src, want) in [("x: N", "N"), ("x: K", "K"), ("x: m", "m")] {
        let t = ty(src);
        assert!(t.tags.is_empty() && t.shape.is_none(), "{src}");
        let Some(UnitExpr::Name(id)) = t.unit else {
            panic!("{src}: {:?}", t.unit)
        };
        assert_eq!(at(src, id.span), want);
    }
}

#[test]
fn a_capitalized_name_before_another_name_is_a_tag() {
    // The parser has no tag table: `Foo` is a tag here and the checker
    // rejects it. A lowercase name is never a tag.
    assert_eq!(
        module_sexpr("x: Foo Matrix[2, 2]"),
        "(: x (type (tags Foo) (Matrix 2 2)))"
    );
    assert_eq!(module_sexpr("x: Scalar"), "(: x (type (Scalar)))");
    assert!(!errors("x: m s").is_empty());
}

#[test]
fn errors_recover_at_the_end_of_the_line() {
    let src = "a: 5 m\nb: Matrix[3, 3]\nc: m s\nd: Vector[2]\n";
    let (module, errs) = parse_module(&lex(src), src);
    assert_eq!(errs.len(), 2, "{errs:#?}");
    let names: Vec<&str> = module
        .stmts
        .iter()
        .map(|s| match &s.kind {
            StmtKind::Binding { name, .. } => at(src, name.span),
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(names, ["b", "d"]);
}

#[test]
fn huge_names_in_type_errors_are_capped() {
    let big = "y".repeat(100_000);
    let cap = MAX_QUOTED_BYTES + 200;
    for src in [
        format!("x: {big}[3]\n"),
        format!("x: Matrix[{big}+1]\n"),
        format!("x: 5 {big}\n"),
        format!("x: m {big}\n"),
        format!("x: m / {big}\n"),
        format!("x: Scalar {big}\n"),
    ] {
        let (_, errors) = parse_module(&lex(&src), &src);
        assert!(!errors.is_empty(), "{}", &src[..20]);
        for e in errors {
            let text = e.message.len() + e.help.as_ref().map_or(0, String::len);
            assert!(text < 4 * cap, "{} bytes: {}", text, &e.message[..40]);
        }
    }
}

#[test]
fn a_long_tag_list_is_flat() {
    // Tags loop, they don't recurse: a huge list is fine.
    let tags = "Symmetric ".repeat(50_000);
    let src = format!("x: {tags}Matrix[2, 2]\n");
    let t = ty(&src);
    assert_eq!(t.tags.len(), 50_000);
}
