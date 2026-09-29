//! The checker walk recurses once per nesting level. The parser caps
//! nesting at `sky_ast::MAX_DEPTH` (256), so checking the deepest module the
//! parser accepts must fit on the stack callers run it on.

use sky_ast::MAX_DEPTH;
use sky_types::{check_module_with, Env, Shape, TagSet, Ty};
use sky_units::Dimension;

/// The stack the checker is run on: 8 MiB, like the parser's callers.
// TODO: switch to `sky_ast::with_deep_stack(|| ...)` and `DEEP_STACK_BYTES`
// once Gilfoyle's shared helper (branch `sky-ast/bool-ops`) lands on main.
const STACK_BYTES: usize = 8 * 1024 * 1024;

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
        _ => unreachable!(),
    }
}

fn env() -> Env {
    let mut env = Env::new();
    env.bind("x", Ty::scalar(Dimension::DIMENSIONLESS));
    env.bind("a", Ty::scalar(Dimension::DIMENSIONLESS));
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
    ];
    let handle = std::thread::Builder::new()
        .stack_size(STACK_BYTES)
        .spawn(move || {
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
        })
        .expect("spawn");
    handle.join().expect("no stack overflow");
}
