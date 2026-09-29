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
    let (msg, at) = rejected("x = 2 m\ny = x.value\n");
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
    let e = try_lower("x = 2 m\ny = x.value\n").expect_err("rejected");
    let note = e.note.as_deref().unwrap_or("");
    assert!(note.contains("sky run"), "{note}");
    assert!(e.to_string().contains("attribute"));
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

#[test]
fn literals_the_interpreter_cannot_read_are_rejected() {
    let (msg, at) = rejected("x = 1e999\n");
    assert!(msg.contains("number too large"), "{msg}");
    assert_eq!(at, "1e999");
    let (msg, at) = rejected("x = 1e306 km\n");
    assert!(msg.contains("overflow"), "{msg}");
    assert_eq!(at, "1e306 km");
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
