//! Checker behaviour for `for` loops that the goldens in `tests/ui/` don't
//! pin: the type of the loop variable, what is defined after a loop, and
//! how `range` is recognised.

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

fn ty(c: &Checked, name: &str) -> String {
    c.env
        .get(name)
        .map_or_else(|| "<unbound>".to_string(), |t| t.to_string())
}

#[test]
fn a_range_index_is_a_dimensionless_scalar_that_indexes() {
    let c = check("v = [1 m, 2 m, 3 m]\nfor i in range(3):\n    x = v[i]\n    y = v[i + 0]\n");
    assert_eq!(messages(&c), Vec::<&str>::new());
    assert_eq!(ty(&c, "i"), "Scalar [1]");
    assert_eq!(ty(&c, "x"), "Scalar [L]");
    // Also with a variable bound, and for a matrix row.
    let c =
        check("n = 2\nA = [[1, 2], [3, 4]]\nfor i in range(n):\n    r = A[i]\n    e = A[i, i]\n");
    assert_eq!(messages(&c), Vec::<&str>::new());
    assert_eq!(ty(&c, "r"), "<unbound>");
}

#[test]
fn a_vector_gives_scalars_of_its_element_dimension() {
    let c = check("for f in [1 N, 2 N]:\n    g = f\n");
    assert_eq!(messages(&c), Vec::<&str>::new());
    assert_eq!(ty(&c, "f"), "Scalar [L·M·T⁻²]");
    assert_eq!(ty(&c, "g"), "Scalar [L·M·T⁻²]");
}

#[test]
fn a_matrix_gives_rows_without_tags() {
    // Symmetric positive definite, but its rows are plain vectors.
    let src = "A: PositiveDefinite Matrix[3, 3] m\nfor r in A:\n    s = r\n";
    let c = check(src);
    assert_eq!(messages(&c), Vec::<&str>::new());
    assert_eq!(ty(&c, "s"), "Vector[3] [L]");
    let c = check("for r in [[1 s, 2 s]]:\n    pass\n");
    assert_eq!(ty(&c, "r"), "Vector[2] [T]");
}

#[test]
fn a_loop_variable_is_bound_after_a_loop_that_always_runs() {
    for src in [
        "for i in range(3):\n    pass\n",
        "for i in range(2, 9, 3):\n    pass\n",
        "for i in range(5, 0, -1):\n    pass\n",
        "for i in [1, 2]:\n    pass\n",
        "for i in [[1], [2]]:\n    pass\n",
    ] {
        let c = check(&format!("{src}z = i\n"));
        assert_eq!(messages(&c), Vec::<&str>::new(), "{src}");
    }
}

#[test]
fn a_loop_that_may_not_run_leaves_its_variable_maybe_undefined() {
    for src in [
        "n = 3\nfor i in range(n):\n    pass\n",
        "for i in range(0):\n    pass\n",
        "for i in range(3, 3):\n    pass\n",
        "for i in range(3, 0):\n    pass\n",
        "for i in range(0, 3, -1):\n    pass\n",
        "for i in range(1, 2 + 1):\n    pass\n",
    ] {
        let c = check(&format!("{src}z = i\n"));
        assert_eq!(messages(&c), ["`i` may be undefined here"], "{src}");
    }
    // Bound before the loop, it keeps its type and its value.
    let c = check("n = 3\ni = 7\nfor i in range(n):\n    pass\nz = i\n");
    assert_eq!(messages(&c), Vec::<&str>::new());
}

#[test]
fn what_the_body_binds_follows_the_same_rule() {
    let c = check("for i in range(2):\n    y = 1 m\nz = y\n");
    assert_eq!(messages(&c), Vec::<&str>::new());
    let c = check("n = 2\nfor i in range(n):\n    y = 1 m\nz = y\n");
    assert_eq!(messages(&c), ["`y` may be undefined here"]);
    // A `break` before the binding leaves it maybe-defined.
    let c = check("for i in range(2):\n    if i == 1:\n        break\n    y = 1\nz = y\n");
    assert_eq!(messages(&c), ["`y` may be undefined here"]);
}

#[test]
fn a_loop_that_always_returns_makes_the_rest_unreachable() {
    // Every path through the body returns, and the body runs at least once,
    // so nothing falls out of the loop: the `y` bound in the other branch
    // is defined after the `if`.
    let src = "\
def f(v, c):
    if c > 0:
        for x in v:
            return x
    else:
        y = 1
    return y
";
    let c = check(src);
    assert_eq!(messages(&c), Vec::<&str>::new());
}

#[test]
fn break_and_continue_are_valid_only_inside_a_loop_of_the_same_function() {
    let c = check("for i in range(3):\n    def f():\n        break\n");
    assert_eq!(messages(&c), ["`break` outside a loop"]);
    let c = check("for i in range(3):\n    if i == 1:\n        continue\n    break\n");
    assert_eq!(messages(&c), Vec::<&str>::new());
}

#[test]
fn range_is_only_a_for_header() {
    let c = check("x = range(3)\n");
    assert_eq!(
        messages(&c),
        ["`range` can only be the iterable of a `for` loop"]
    );
    // Its arguments are still checked, once.
    let c = check("x = range(3 m + 2 s)\n");
    assert_eq!(messages(&c).len(), 2, "{:?}", messages(&c));
    // Parentheses make it a value.
    let c = check("for i in (range(3)):\n    pass\n");
    assert_eq!(
        messages(&c),
        ["`range` can only be the iterable of a `for` loop"]
    );
}

#[test]
fn a_user_binding_named_range_wins() {
    let c = check("def range(n):\n    return [1 m, 2 m]\nfor i in range(3):\n    d = i\n");
    assert_eq!(messages(&c), Vec::<&str>::new());
    assert_eq!(ty(&c, "d"), "Scalar [L]");
    let c = check("range = 3\nfor i in range:\n    pass\n");
    assert_eq!(messages(&c), ["cannot iterate over a dimensionless number"]);
}

#[test]
fn an_unknown_iterable_is_silent_and_gives_an_unknown_variable() {
    // A value the checker cannot type (a parameter without an annotation)
    // gives no error; the interpreter checks it at run time.
    let c = check("def f(s):\n    for ch in s:\n        d = ch\n    return d\n");
    assert_eq!(messages(&c), Vec::<&str>::new());
    // A string in a variable is typed, and rejected like a literal.
    let c = check("s = \"abc\"\nfor ch in s:\n    d = ch\nz = d\n");
    assert_eq!(messages(&c), ["cannot iterate over a string"]);
}

#[test]
fn an_error_in_the_header_reports_once_and_does_not_cascade() {
    let c = check("for i in range(2.5):\n    x = i\nz = x + i\n");
    assert_eq!(messages(&c), ["`range` needs an integer, found `2.5`"]);
    let c = check("for x in 3 m:\n    y = x\nz = y\n");
    assert_eq!(messages(&c), ["cannot iterate over a length"]);
}

#[test]
fn nested_loops_and_shadowed_variables() {
    let src = "for i in range(2):\n    for i in range(3):\n        pass\nz = i\n";
    let c = check(src);
    assert_eq!(messages(&c), Vec::<&str>::new());
    assert_eq!(ty(&c, "z"), "Scalar [1]");
}

#[test]
fn deeply_nested_loops_do_not_overflow_the_checker() {
    let depth = 200;
    let mut src = String::new();
    for d in 0..depth {
        src.push_str(&"    ".repeat(d));
        src.push_str(&format!("for i{d} in range(2):\n"));
    }
    src.push_str(&"    ".repeat(depth));
    src.push_str("pass\n");
    let c = check(&src);
    assert_eq!(messages(&c), Vec::<&str>::new());
}
