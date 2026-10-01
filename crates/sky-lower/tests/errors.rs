//! What lowering rejects: everything `sky run` cannot execute either
//! (attributes, derivatives, writes through an index) and what SIR v1 has no
//! representation for (function values).

mod common;

use common::*;

/// The error, and the source text it points at.
fn rejected(src: &str) -> (String, &str) {
    let e = try_lower(src).expect_err("must be rejected");
    (e.message.clone(), &src[e.span.range()])
}

#[test]
fn attributes_are_rejected() {
    // The checker rejects an attribute of a number or a string; one of an
    // unannotated parameter (which it does not type) reaches lowering.
    let (msg, at) = rejected("def f(x):\n    return x.value\nprint(f(1))\n");
    assert!(msg.contains("attribute `value`"), "{msg}");
    assert_eq!(at, "x.value");
}

#[test]
fn derivative_operators_are_rejected() {
    let (msg, at) = rejected("x = 2 m\ny = ∂x\n");
    assert!(msg.contains("derivative"), "{msg}");
    assert_eq!(at, "∂x");
    let (msg, _) = rejected("x = 2 m\ny = ∇x\n");
    assert!(msg.contains("derivative"), "{msg}");
}

#[test]
fn an_error_has_a_note_saying_why() {
    let e = try_lower("def f(w):\n    for x in w:\n        print(x)\nf([1 m, 2 m])\n")
        .expect_err("rejected");
    let note = e.note.as_deref().unwrap_or("");
    assert!(note.contains("`w: Vector[N]`"), "{note}");
    assert!(e.to_string().contains("type is not known"));
    let d = e.to_diagnostic();
    assert_eq!(d.primary.span, e.span);
}

#[test]
fn a_function_used_as_a_value_is_rejected() {
    let (msg, at) = rejected("def f(x):\n    return x\ng = f\n");
    assert!(msg.contains("function `f` is used as a value"), "{msg}");
    assert_eq!(at, "f");
    let (msg, at) =
        rejected("def f(x):\n    return x\ndef apply(h, x):\n    return x\ny = apply(f, 1)\n");
    assert!(msg.contains("used as a value"), "{msg}");
    assert_eq!(at, "f");
}

#[test]
fn calling_a_parameter_is_rejected() {
    let (msg, at) = rejected("def apply(h, x):\n    return h(x)\n");
    assert!(
        msg.contains("`h` is not a function known at compile time"),
        "{msg}"
    );
    assert_eq!(at, "h");
}

#[test]
fn an_input_declaration_must_be_at_the_top_level_of_the_file() {
    let (msg, at) = rejected("def f():\n    v: Vector[3]\n    return v\n");
    assert!(msg.contains("top level of the file"), "{msg}");
    assert_eq!(at, "v: Vector[3]");
    let (msg, _) = rejected("x = 1\nif x > 0:\n    v: Vector[3]\n");
    assert!(msg.contains("top level of the file"), "{msg}");
}

#[test]
fn a_nested_def_does_not_see_its_enclosing_functions_locals() {
    // The checker lets the name through; at run time the interpreter finds no
    // `y` (a def captures nothing but module names), and lowering cannot
    // give the inner function a value for it.
    let (msg, at) = rejected(
        "def outer(a):\n    y = a + 1\n    def inner():\n        return y\n    return inner()\n",
    );
    assert!(msg.contains("cannot find `y`"), "{msg}");
    assert_eq!(at, "y");
}

/// A literal `sky run` cannot read is an error only where it is evaluated,
/// so lowering does not refuse the program: the literal becomes an operation
/// that fails when it runs (`literal.error`), at the span `sky run` reports.
#[test]
fn literals_the_interpreter_cannot_read_lower_to_an_operation_that_fails() {
    for (src, kind, at) in [
        ("x = 1e999\n", "literal.error[too_large]", "1e999"),
        ("x = 1e999 m\n", "literal.error[too_large]", "1e999"),
        ("x = 1e306 km\n", "literal.error[overflow]", "1e306 km"),
    ] {
        let m = lower_ok(src);
        let f = main_fn(&m);
        assert_eq!(count(f, "literal.error"), 1, "{src}");
        let op = f
            .ops
            .iter()
            .find(|o| o.kind.mnemonic() == "literal.error")
            .expect("the literal");
        assert_eq!(&src[op.span.range()], at, "{src}");
        let text = sky_ir::print(&m);
        assert!(text.contains(kind), "{text}");
        assert_eq!(sky_ir::print(&sky_ir::parse(&text).expect("parses")), text);
    }
}

#[test]
fn source_that_does_not_check_is_reported_by_the_front_end() {
    let e = lower_err("x = 1 m + 2 s\n");
    assert!(e.contains("cannot add"), "{e}");
    let e = lower_err("x = (\n");
    assert!(e.contains("error"), "{e}");
}

#[test]
fn what_lowers_is_not_rejected() {
    // `@` and matrix inverses lower (SIR represents them), even though the
    // interpreter cannot run them yet.
    lower_ok("A: PositiveDefinite Matrix[3, 3]\nb: Vector[3]\nx = A⁻¹ @ b\ny = A @ b\n");
    // A function may take an array-typed parameter.
    lower_ok("def apply(A: Matrix[N, N], b: Vector[N]) -> Vector[N]:\n    return A @ b\n");
}

/// Lowers `src` the way `sky run` does, for execution.
fn try_lower_for_run(src: &str) -> Result<sky_ir::Module, sky_lower::LowerError> {
    sky_ast::with_deep_stack(|| {
        let lexed = sky_syntax::lex(src).expect("lexes");
        let (ast, errors) = sky_ast::parse_module(&lexed.tokens, src);
        assert!(errors.is_empty(), "{errors:#?}");
        let checked = sky_types::check_module(&ast, src);
        assert!(!checked.has_errors(), "{:#?}", checked.diagnostics);
        sky_lower::lower_for_run(&ast, src, &checked)
    })
}

#[test]
fn a_computed_range_bound_or_step_lowers_with_run_time_checks() {
    // The interpreter stops `range(5 / 2)` with an error when the loop
    // starts; SIR makes the same checks with `expect.range`, after each
    // argument, on the argument's span.
    let src = "n = 5 / 2\nfor i in range(n):\n    print(i)\n";
    for lowered in [try_lower(src), try_lower_for_run(src)] {
        let m = lowered.expect("lowers");
        let text = sky_ir::print(&m);
        assert!(text.contains("expect.range[bound]"), "{text}");
        assert!(!text.contains("expect.range[step]"), "{text}");
    }
    let src = "a = 2\nb = 8\ns = 3\nfor i in range(a, b, s):\n    print(i)\n";
    let text = sky_ir::print(&try_lower_for_run(src).expect("lowers"));
    assert_eq!(text.matches("expect.range[bound]").count(), 3, "{text}");
    assert_eq!(text.matches("expect.range[step]").count(), 1, "{text}");
    // Literals are checked by the checker: no run-time check.
    let literals = try_lower_for_run("for i in range(2, 10, 3):\n    print(i)\n");
    let text = sky_ir::print(&literals.expect("lowers"));
    assert!(!text.contains("expect.range"), "{text}");
}
