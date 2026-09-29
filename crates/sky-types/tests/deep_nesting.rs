//! The checker walk recurses once per nesting level. The parser caps
//! nesting at `sky_ast::MAX_DEPTH` (256), so checking the deepest module the
//! parser accepts must fit on the stack callers run it on:
//! [`sky_ast::with_deep_stack`] ([`sky_ast::DEEP_STACK_BYTES`], 8 MiB).

use sky_ast::{with_deep_stack, DEEP_STACK_BYTES, MAX_DEPTH};
use sky_types::{check_module_with, Env, Shape, TagSet, Ty};
use sky_units::Dimension;

fn nested(kind: &str, n: usize) -> String {
    match kind {
        "paren" => format!("{}x{}", "(".repeat(n), ")".repeat(n)),
        "call" => format!("{}x{}", "inv(".repeat(n), ")".repeat(n)),
        "index" => format!("{}0{}", "a[".repeat(n), "]".repeat(n)),
        "power" => format!("x{}", "^x".repeat(n)),
        "prefix" => format!("{}x", "-".repeat(n)),
        "sum" => format!("x{}", " + x".repeat(n)),
        "product" => format!("x{}", " * x".repeat(n)),
        "inverse" => format!("x{}", "⁻¹".repeat(n)),
        "matmul" => format!("A{}", " @ A".repeat(n)),
        "matrix_prefix" => format!("{}A", "-".repeat(n)),
        "not" => format!("{}p", "not ".repeat(n)),
        "and" => format!("p{}", " and p".repeat(n)),
        // Each level is a paren and an `or`, two parser depth levels.
        "mixed_bool" => format!("{}p{}", "(p or ".repeat(n / 2), ")".repeat(n / 2)),
        _ => unreachable!(),
    }
}

fn env() -> Env {
    let mut env = Env::new();
    env.bind("x", Ty::scalar(Dimension::DIMENSIONLESS));
    env.bind("a", Ty::scalar(Dimension::DIMENSIONLESS));
    env.bind("p", Ty::Bool);
    env.bind_array(
        "A",
        Shape::matrix(3u64, 3u64),
        TagSet::EMPTY,
        Dimension::DIMENSIONLESS,
    )
    .expect("square");
    env
}

#[test]
fn max_depth_checks_on_an_8_mib_stack() {
    assert_eq!(MAX_DEPTH, 256);
    let kinds = [
        "paren",
        "call",
        "index",
        "power",
        "prefix",
        "sum",
        "product",
        "inverse",
        "matmul",
        "matrix_prefix",
        "not",
        "and",
        "mixed_bool",
    ];
    assert_eq!(DEEP_STACK_BYTES, 8 * 1024 * 1024);
    with_deep_stack(move || {
        for kind in kinds {
            let src = format!("y = {}\n", nested(kind, MAX_DEPTH as usize));
            let lexed = sky_syntax::lex(&src).expect("lexes");
            let (module, errors) = sky_ast::parse_module(&lexed.tokens, &src);
            assert!(errors.is_empty(), "{kind}: {errors:#?}");
            let checked = check_module_with(&module, &src, env());
            assert!(
                checked.diagnostics.is_empty(),
                "{kind}: {:#?}",
                checked.diagnostics
            );
            assert!(checked.env.get("y").is_some(), "{kind}");
        }
    });
}
