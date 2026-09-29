//! Checker behaviour for blocks, control flow, and functions that the
//! goldens in `tests/ui/` don't pin: bindings after merges, the no-cascade
//! rule, generic size variables, and the depth limit on the checker's own
//! recursion.

use sky_ast::{with_deep_stack, MAX_DEPTH};
use sky_types::{check_module, Checked, Ty};
use sky_units::Dimension;

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
fn merged_names_keep_their_type_and_tags_are_joined() {
    let src = "\
x = 3 m
if x > 1 m:
    y = 1 s
else:
    y = 2 s
";
    let c = check(src);
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    assert_eq!(c.env.get("y"), Some(Ty::scalar(Dimension::TIME)));
}

#[test]
fn a_name_bound_in_only_some_branches_is_unbound_afterwards() {
    let c = check("x = 1 m\nif x > 0 m:\n    y = 1\n");
    assert!(c.diagnostics.is_empty());
    // Not a value any more: `Env::get` says so, and a use is an error.
    assert_eq!(c.env.get("y"), None);
    let c = check("x = 1 m\nif x > 0 m:\n    y = 1\nz = y\nw = y\n");
    assert_eq!(messages(&c), ["`y` may be undefined here"]);
}

#[test]
fn a_diverging_branch_takes_no_part_in_the_merge() {
    let c = check(
        "def f(x):\n    if x > 0 m:\n        y = 1 m\n    else:\n        return 0 m\n    return y\n",
    );
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    // All branches diverging: the code after is unreachable, and names
    // bound before stay bound.
    let c = check(
        "def g(x):\n    a = 1\n    if x > 0 m:\n        return 1\n    else:\n        return 2\n    return a\n",
    );
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
}

#[test]
fn a_conflict_is_reported_once_and_does_not_cascade() {
    let src = "\
x = 3 m
if x > 1 m:
    t = 1 s
else:
    t = 2 m
u = t + 1 kg
v = u * t
";
    let c = check(src);
    assert_eq!(
        messages(&c),
        ["`t` is a length here but a time in the other branch"]
    );
    assert_eq!(c.env.get("t"), Some(Ty::Unknown));
}

#[test]
fn breaks_and_continues_feed_the_state_after_a_loop() {
    // Bound before the `break` on every path out: defined afterwards.
    let c = check(
        "n = 0\nwhile true:\n    n = n + 1\n    v = 1 m\n    if n > 3:\n        break\nw = v + 1 m\n",
    );
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    // A loop that can also end because its condition failed leaves `v`
    // maybe-defined.
    let c = check("n = 0\nwhile n < 3:\n    n = n + 1\n    v = 1 m\nw = v\n");
    assert_eq!(messages(&c), ["`v` may be undefined here"]);
}

#[test]
fn a_function_binds_its_name_and_its_locals_do_not_escape() {
    let c = check("def f(x):\n    inner = 1\n    return x\n");
    assert!(c.diagnostics.is_empty());
    assert_eq!(c.env.get("inner"), None);
    assert!(matches!(c.env.get("f"), Some(Ty::Func(_))));
}

#[test]
fn unannotated_parameters_are_unknown_and_never_an_error() {
    let c = check("def f(a, b):\n    return a + b * 2 m\nx = f(1 s, 2 kg) + 3 K\n");
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    assert_eq!(c.env.get("x"), Some(Ty::Unknown));
}

#[test]
fn inferred_return_types_reach_the_call_site() {
    let c = check("def f():\n    return 2 m\nx = f()\ny = x + 1 s\n");
    assert_eq!(messages(&c), ["cannot add length and time"]);
}

#[test]
fn recursion_is_unknown_when_the_return_type_is_inferred() {
    let c = check("def f(n):\n    if n < 1:\n        return 1 m\n    return f(n - 1)\nx = f(2)\n");
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
}

#[test]
fn size_variables_of_a_signature_are_generic() {
    let src = "\
def dot(a: Vector[N], b: Vector[N]) -> Vector[N]:
    return a

u: Vector[3]
v: Vector[4]
x = dot(u, u)
y = dot(v, v)
z = dot(u, v)
";
    let c = check(src);
    assert_eq!(messages(&c).len(), 1, "{:#?}", c.diagnostics);
    assert!(messages(&c)[0].starts_with("mismatched shapes"));
    assert_eq!(ty(&c, "x"), "Vector[3] [1]");
    assert_eq!(ty(&c, "y"), "Vector[4] [1]");
}

#[test]
fn a_body_that_fixes_a_size_makes_the_function_specific() {
    let src = "\
def f(a: Vector[N]) -> Vector[3]:
    return a

u: Vector[3]
v: Vector[4]
x = f(u)
y = f(v)
";
    let c = check(src);
    assert_eq!(messages(&c).len(), 1, "{:#?}", c.diagnostics);
}

#[test]
fn module_size_variables_and_signature_variables_are_separate() {
    let src = "\
A: Matrix[N, N]
def f(b: Vector[N]) -> Vector[N]:
    return b
v: Vector[3]
d = f(v)
";
    let c = check(src);
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    assert_eq!(ty(&c, "d"), "Vector[3] [1]");
    // The module's N is untouched by the call.
    assert_eq!(ty(&c, "A"), "Matrix[N,N] [1]");
}

#[test]
fn a_function_can_be_redefined_and_rebound() {
    let c = check("def f():\n    return 1 m\ndef f():\n    return 1 s\nx = f()\n");
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    assert_eq!(c.env.get("x"), Some(Ty::scalar(Dimension::TIME)));
    // A function in one branch only is maybe-defined.
    let c = check("if true:\n    def g():\n        return 1\nx = g()\n");
    assert_eq!(messages(&c), ["`g` may be undefined here"]);
}

#[test]
fn a_name_can_be_a_function_and_then_a_value() {
    let c = check("def f():\n    return 1\nf = 3 m\n");
    assert!(c.diagnostics.is_empty());
    let c = check("def f():\n    return 1\nif true:\n    f = 3 m\n");
    assert_eq!(
        messages(&c),
        ["`f` is a length here but function `f` in the other branch"]
    );
}

#[test]
fn nested_functions_read_the_enclosing_names_bound_so_far() {
    let c = check("a = 1 m\ndef f():\n    b = 2 m\n    def g():\n        return a + b\n    return g()\nx = f()\n");
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    assert_eq!(c.env.get("x"), Some(Ty::scalar(Dimension::LENGTH)));
}

// ----- depth -----

/// `n` nested blocks of `kind`, one tab per level, with `body` innermost.
fn nested_blocks(kinds: &[&str], n: usize, body: &str) -> String {
    let mut src = String::new();
    for i in 0..n {
        src.push_str(&"\t".repeat(i));
        src.push_str(kinds[i % kinds.len()]);
        src.push_str(":\n");
    }
    src.push_str(&"\t".repeat(n));
    src.push_str(body);
    src.push('\n');
    src
}

#[test]
fn the_deepest_block_nesting_the_parser_accepts_checks_cleanly() {
    let n = MAX_DEPTH as usize;
    for kinds in [
        &["if x > 0 m"][..],
        &["while x > 0 m"],
        &["def f()"],
        &["if x > 0 m", "while x > 0 m", "def f()"],
    ] {
        let mut src = String::from("x = 1 m\n");
        src.push_str(&nested_blocks(kinds, n, "pass"));
        // MAX_DEPTH blocks use the whole depth budget, so the innermost
        // statement can't be an expression with a level of its own.
        let c = with_deep_stack(|| {
            let lexed = sky_syntax::lex(&src).expect("lexes");
            let (module, errors) = sky_ast::parse_module(&lexed.tokens, &src);
            assert!(errors.is_empty(), "{kinds:?}: {errors:#?}");
            check_module(&module, &src)
        });
        assert!(c.diagnostics.is_empty(), "{kinds:?}: {:#?}", c.diagnostics);
    }
}

#[test]
fn a_return_at_the_bottom_of_nested_defs_checks() {
    // 255 nested defs (each a scope with a cloned environment), then a
    // return that reads the outermost name.
    let n = MAX_DEPTH as usize - 1;
    let mut src = String::from("a = 1 m\n");
    src.push_str(&nested_blocks(&["def f()"], n, "return a"));
    let c = check(&src);
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
}
