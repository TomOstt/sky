//! Checker behaviour for `cost` lines: the bound must be a polynomial in the
//! sizes of the parameters (spec "Performance contracts"). Whether a
//! function meets its contract is `sky check`'s question, answered from SIR
//! (see `crates/sky-cli/tests/cost.rs`).

use sky_ast::with_deep_stack;
use sky_types::{check_module, Checked};

fn check(src: &str) -> Checked {
    with_deep_stack(|| {
        let lexed = sky_syntax::lex(src).expect("lexes");
        let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
        assert!(errors.is_empty(), "{errors:#?}");
        check_module(&module, src)
    })
}

fn messages(c: &Checked) -> Vec<&str> {
    c.diagnostics.iter().map(|d| d.message.as_str()).collect()
}

#[test]
fn a_polynomial_in_the_sizes_checks() {
    let src = "def f(A: Matrix[N, K], B: Matrix[K, M]) -> Matrix[N, M]:\n    cost flops <= 2 * N * K * M + N^2 / 3 - 1\n    cost allocations == 1\n    cost bytes <= 16 * (N * K + K * M)\n    return A @ B\n";
    let c = check(src);
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
}

#[test]
fn a_bound_without_sizes_checks() {
    let c = check("def f(x: m) -> m:\n    cost flops <= 4\n    return x * 2\n");
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
}

#[test]
fn a_name_that_is_not_a_size_is_an_error_naming_the_sizes() {
    let c = check("def f(A: Matrix[N, M]):\n    cost flops <= 2 * N * K\n    return A\n");
    assert_eq!(messages(&c), ["`K` is not a size of `f`"]);
    assert_eq!(
        c.diagnostics[0].help.first().map(String::as_str),
        Some("the sizes of `f` are `N`, `M`")
    );
    // Reported once per name.
    let c = check("def f(A: Matrix[N, M]):\n    cost flops <= K * K\n    return A\n");
    assert_eq!(messages(&c), ["`K` is not a size of `f`"]);
}

#[test]
fn a_function_without_array_parameters_has_no_sizes() {
    let c = check("def f(x):\n    cost flops <= N\n    return x\n");
    assert_eq!(messages(&c), ["`N` is not a size of `f`"]);
    assert!(c.diagnostics[0]
        .help
        .iter()
        .any(|h| h.contains("name them in the parameters")));
}

#[test]
fn the_value_of_a_number_is_not_a_size() {
    let c = check("def f(A: Matrix[N, N], n):\n    cost flops <= n\n    return A\n");
    assert_eq!(messages(&c), ["`n` is not a size of `f`"]);
    assert!(c.diagnostics[0]
        .notes
        .iter()
        .any(|n| n.contains("not on the values of numbers")));
}

#[test]
fn what_is_not_a_polynomial_is_an_error_with_a_help_line() {
    for (bound, message) in [
        ("N / M", "a size can only be divided by a number"),
        ("N ^ M", "the exponent must be a whole number, at most 64"),
        ("sqrt(N)", "this is not a polynomial in the sizes"),
        ("N < 3", "this is not a polynomial in the sizes"),
        ("N ^ 0.5", "the exponent must be a whole number, at most 64"),
        (
            "N ^ 99999",
            "the exponent must be a whole number, at most 64",
        ),
    ] {
        let c = check(&format!(
            "def f(A: Matrix[N, M]):\n    cost flops <= {bound}\n    return A\n"
        ));
        assert_eq!(messages(&c), [message], "{bound}");
    }
}

#[test]
fn a_size_next_to_a_number_reads_as_a_unit_and_says_so() {
    let c = check("def f(A: Matrix[N, N]):\n    cost flops <= 2 N\n    return A\n");
    assert_eq!(
        messages(&c),
        ["a size is not a unit: this reads as a quantity"]
    );
    assert_eq!(
        c.diagnostics[0].help.first().map(String::as_str),
        Some("write `2 * N` for 2 times the size `N`")
    );
}

#[test]
fn cost_lines_do_not_disturb_requires_and_ensures() {
    let src = "def f(x: m) -> m:\n    cost flops <= 4\n    requires x >= 0 m\n    cost allocations == 0\n    ensures result >= 0 m\n    return x\n";
    let c = check(src);
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    // The `requires` after a `cost` line still guards calls with constants.
    let c = check(&format!("{src}y = f(-(2 m))\n"));
    assert!(
        c.diagnostics
            .iter()
            .any(|d| d.message.contains("breaks the precondition")),
        "{:#?}",
        c.diagnostics
    );
}

#[test]
fn cost_is_an_ordinary_name_in_a_program() {
    let c = check("cost = 3\nprice = cost * 2\n");
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
}
