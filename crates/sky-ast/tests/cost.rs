//! `cost flops <= 2 * N^3`: the performance contract in a function's header
//! (spec "Performance contracts"), and that `cost` is not reserved.

mod common;
use common::{module_errors, module_sexpr};

#[test]
fn the_line_parses_in_the_header_next_to_requires_and_ensures() {
    let src = "def gram(A: Matrix[N, M]):\n    requires len(A) > 0\n    cost flops <= 2 * N * M^2\n    ensures true\n    cost allocations == 0\n    return A\n";
    assert_eq!(
        module_sexpr(src),
        "(def gram (params (: A (type (Matrix N M)))) (block (requires (> (call len A) 0)) \
         (cost flops <= (* (* 2 N) (^ M 2))) (ensures true) (cost allocations == 0) (return A)))"
    );
}

#[test]
fn every_measure_is_a_word() {
    for m in [
        "flops",
        "bytes_read",
        "bytes_written",
        "bytes",
        "allocations",
        "allocated_bytes",
    ] {
        let src = format!("def f(A: Vector[N]):\n    cost {m} <= N\n    pass\n");
        assert!(module_errors(&src).is_empty(), "{m}");
    }
}

#[test]
fn cost_is_an_ordinary_name_everywhere_else() {
    assert_eq!(module_sexpr("cost = 3\n"), "(= cost 3)");
    assert_eq!(module_sexpr("cost(2)\n"), "(call cost 2)");
    assert_eq!(module_sexpr("x = cost + 1\n"), "(= x (+ cost 1))");
    assert_eq!(module_sexpr("cost <= 1\n"), "(<= cost 1)");
    assert_eq!(module_sexpr("cost\n"), "cost");
    // Also as the name of a parameter or a function.
    assert!(module_errors("def cost(cost):\n    return cost\n").is_empty());
    assert!(module_errors("def f(flops, cost):\n    return flops * cost\n").is_empty());
}

#[test]
fn it_is_only_allowed_at_the_start_of_a_function_body() {
    let errors = module_errors("cost flops <= 1\n");
    assert_eq!(errors.len(), 1, "{errors:#?}");
    assert_eq!(
        errors[0].message,
        "`cost` is only allowed at the start of a function body"
    );
    let errors = module_errors("def f(x):\n    if x:\n        cost flops <= 1\n");
    assert_eq!(errors.len(), 1, "{errors:#?}");
    let errors = module_errors("def f(x):\n    y = x\n    cost flops <= 1\n");
    assert_eq!(errors.len(), 1, "{errors:#?}");
    assert_eq!(
        errors[0].message,
        "`cost` must come before the other statements of the function body"
    );
}

#[test]
fn a_nested_function_has_its_own_head() {
    let src = "def f(x: Vector[N]):\n    cost flops <= N\n    def g(y: Vector[N]):\n        cost flops <= N\n        return y\n    return g(x)\n";
    assert!(module_errors(src).is_empty());
}

#[test]
fn an_unknown_measure_lists_the_known_ones() {
    let errors = module_errors("def f(x):\n    cost speed <= 3\n    pass\n");
    assert_eq!(errors.len(), 1, "{errors:#?}");
    assert!(errors[0].message.contains("`speed` is not something"));
    let help = errors[0].help.clone().unwrap_or_default();
    assert!(help.contains("`flops`") && help.contains("`allocated_bytes`"));
}

#[test]
fn only_at_most_and_exactly() {
    for op in ["<", ">", ">=", "=", "!="] {
        let errors = module_errors(&format!("def f(x):\n    cost flops {op} 3\n    pass\n"));
        assert_eq!(errors.len(), 1, "{op}: {errors:#?}");
        assert!(
            errors[0]
                .message
                .contains("expected `<=` or `==` after `cost flops`"),
            "{op}: {errors:#?}"
        );
    }
    let errors = module_errors("def f(x):\n    cost flops\n    pass\n");
    assert_eq!(errors.len(), 1, "{errors:#?}");
}

#[test]
fn a_missing_bound_is_one_error_and_recovers() {
    let errors = module_errors("def f(x):\n    cost flops <=\n    pass\ny = = 2\n");
    assert_eq!(errors.len(), 2, "{errors:#?}");
}
