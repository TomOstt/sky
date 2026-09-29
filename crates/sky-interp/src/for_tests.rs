//! `for` loops in the tree walker: `range`, arrays, `break` and
//! `continue`, scoping, and the run-time checks (no checker runs here).

use sky_ast::parse_module;
use sky_syntax::lex;

use crate::{run_module, RuntimeEnv, RuntimeError};

/// Runs `src` and returns what it printed, or the runtime error.
fn run(src: &str) -> Result<String, RuntimeError> {
    let src = format!("{src}\n");
    let lexed = lex(&src).expect("lexes");
    let (module, errors) = parse_module(&lexed.tokens, &src);
    assert!(errors.is_empty(), "{errors:?} in {src:?}");
    let mut out = Vec::new();
    run_module(&module, &src, &mut RuntimeEnv::new(), &mut out)?;
    Ok(String::from_utf8(out).expect("utf-8 output"))
}

fn out(src: &str) -> String {
    run(src).expect("runs")
}

fn err(src: &str) -> String {
    run(src).expect_err("fails").message.clone()
}

/// The items a loop over `header` visits, one line each.
fn visits(header: &str) -> String {
    out(&format!("for i in {header}:\n    print(i)"))
}

#[test]
fn range_forms() {
    assert_eq!(visits("range(3)"), "0\n1\n2\n");
    assert_eq!(visits("range(2, 5)"), "2\n3\n4\n");
    assert_eq!(visits("range(0, 10, 4)"), "0\n4\n8\n");
    assert_eq!(visits("range(5, 0, -2)"), "5\n3\n1\n");
    assert_eq!(visits("range(-2, 1)"), "-2\n-1\n0\n");
}

#[test]
fn an_empty_range_runs_nothing() {
    assert_eq!(visits("range(0)"), "");
    assert_eq!(visits("range(3, 3)"), "");
    assert_eq!(visits("range(5, 2)"), "");
    assert_eq!(visits("range(2, 5, -1)"), "");
    // ... and leaves the variable as it was, or unbound.
    assert_eq!(out("i = 7\nfor i in range(0):\n    pass\nprint(i)"), "7\n");
    assert!(err("for i in range(0):\n    pass\nprint(i)").contains("cannot find `i`"));
}

#[test]
fn bounds_are_evaluated_once_and_left_to_right() {
    let src = "\
def show(x):
    print(\"bound\", x)
    return x
for i in range(show(1), show(4), show(2)):
    print(i)
";
    assert_eq!(out(src), "bound 1\nbound 4\nbound 2\n1\n3\n");
    // Changing `n` inside the loop does not change how often it runs.
    let src = "n = 3\nfor i in range(n):\n    n = 10\n    print(i)\nprint(n)";
    assert_eq!(out(src), "0\n1\n2\n10\n");
}

#[test]
fn the_variable_keeps_its_last_value_and_rebinding_it_is_harmless() {
    assert_eq!(out("for i in range(3):\n    pass\nprint(i)"), "2\n");
    let src = "for i in range(3):\n    print(i)\n    i = 100\nprint(i)";
    assert_eq!(out(src), "0\n1\n2\n100\n");
}

#[test]
fn break_and_continue_work_as_in_while() {
    let src = "\
for i in range(10):
    if i == 2:
        continue
    if i == 5:
        break
    print(i)
print(i)
";
    assert_eq!(out(src), "0\n1\n3\n4\n5\n");
    // `break` leaves only the inner loop.
    let src = "\
for i in range(2):
    for j in range(5):
        if j == 1:
            break
        print(i, j)
";
    assert_eq!(out(src), "0 0\n1 0\n");
}

#[test]
fn return_leaves_the_function_from_inside_a_loop() {
    let src = "\
def first_over(v, limit):
    for x in v:
        if x > limit:
            return x
    return 0 m
print(first_over([1 m, 5 m, 9 m], 3 m))
print(first_over([1 m, 2 m], 3 m))
";
    assert_eq!(out(src), "5 m\n0 m\n");
}

#[test]
fn a_vector_gives_its_entries_with_their_unit() {
    let src = "total = 0 N\nfor f in [1 N, 2 N, 3.5 N]:\n    total += f\nprint(total, f)";
    assert_eq!(out(src), "6.5 N 3.5 N\n");
    assert_eq!(
        out("for x in [1 km, 250 cm]:\n    print(x)"),
        "1000 m\n2.5 m\n"
    );
}

#[test]
fn a_matrix_gives_its_rows_as_vectors() {
    let src = "for r in [[1, 2], [3, 4], [5, 6]]:\n    print(r)";
    assert_eq!(out(src), "[1, 2]\n[3, 4]\n[5, 6]\n");
    let src = "for r in [[1 m, 2 m]]:\n    print(r, len(r))";
    assert_eq!(out(src), "[1 m, 2 m] 2\n");
    // A row is indexable and iterable in turn.
    let src = "s = 0\nfor r in [[1, 2], [3, 4]]:\n    for x in r:\n        s += x\nprint(s)";
    assert_eq!(out(src), "10\n");
}

#[test]
fn the_index_indexes() {
    let src = "v = [10 m, 20 m, 30 m]\nfor i in range(len(v)):\n    print(i, v[i])";
    assert_eq!(out(src), "0 10 m\n1 20 m\n2 30 m\n");
    assert!(err("v = [1, 2]\nfor i in range(3):\n    print(v[i])").contains("out of bounds"));
}

#[test]
fn the_iterable_is_evaluated_once() {
    let src = "v = [1, 2, 3]\nfor x in v:\n    v = [9]\n    print(x)";
    assert_eq!(out(src), "1\n2\n3\n");
}

#[test]
fn a_riemann_sum() {
    // The integral of x^2 on [0, 1] by the midpoint rule.
    let src = "\
n = 1000
dx = 1 / n
s = 0
for i in range(n):
    x = (i + 0.5) * dx
    s += x^2 * dx
print(s)
";
    let s: f64 = out(src).trim().parse().expect("a number");
    assert!((s - 1.0 / 3.0).abs() < 1e-6, "{s}");
}

#[test]
fn loops_in_functions_use_the_functions_locals() {
    let src = "\
def total(v):
    t = 0 m
    for x in v:
        t += x
    return t
i = 5
print(total([1 m, 2 m]), i)
";
    assert_eq!(out(src), "3 m 5\n");
}

#[test]
fn a_user_range_shadows_the_builtin() {
    let src = "def range(n):\n    return [1, 2]\nfor i in range(9):\n    print(i)";
    assert_eq!(out(src), "1\n2\n");
}

#[test]
fn range_arguments_are_checked_at_run_time() {
    assert_eq!(
        err("n = 2.5\nfor i in range(n):\n    pass"),
        "`range` needs an integer, found 2.5"
    );
    assert_eq!(
        err("for i in range(3 m):\n    pass"),
        "`range` needs a dimensionless integer, found length"
    );
    assert_eq!(
        err("for i in range(true):\n    pass"),
        "`range` needs an integer, found a Bool"
    );
    assert_eq!(
        err("for i in range(0, 5, 0):\n    pass"),
        "the step of `range` cannot be zero"
    );
    assert_eq!(
        err("for i in range(1e16):\n    pass"),
        "the bound 1e16 of `range` is too large"
    );
    assert_eq!(
        err("for i in range(1, 2, 3, 4):\n    pass"),
        "`range` takes 1 to 3 arguments, found 4"
    );
}

#[test]
fn the_largest_bounds_do_not_overflow() {
    // 2^53 is allowed; the loop steps past it without wrapping.
    let big = "9007199254740992";
    let src = format!("for i in range({big}, {big} - 2, -1):\n    print(i)");
    assert_eq!(out(&src), "9007199254740992\n9007199254740991\n");
    let src = format!("for i in range(-{big}, {big}, {big}):\n    print(i)");
    assert_eq!(out(&src), "-9007199254740992\n0\n");
}

#[test]
fn only_arrays_can_be_iterated() {
    for (src, kind) in [
        ("for x in 3:\n    pass", "a number"),
        ("for x in true:\n    pass", "a Bool"),
        ("for x in \"abc\":\n    pass", "a string"),
        ("s = \"abc\"\nfor x in s:\n    pass", "a string"),
    ] {
        assert_eq!(err(src), format!("cannot iterate over {kind}"), "{src}");
    }
    assert_eq!(
        err("def f():\n    return 1\nfor x in f:\n    pass"),
        "cannot iterate over a function"
    );
}

#[test]
fn an_error_in_the_body_stops_the_loop() {
    let src = "for i in range(3):\n    print(i)\n    print(1 / (1 - i))";
    let e = run(src).expect_err("division by zero at i = 1");
    assert_eq!(e.message, "division by zero");
}
