//! `derivative(f, x)(args)` lowers to a call of the function `sky_ad`
//! builds, and the reference interpreter runs it.

mod common;

use common::eval::run_main;
use common::{func, lower_err, lower_ok, main_fn, names};
use sky_ir::OpKind;

/// The physics example: x(t) = 5 m/s² t², so x'(2 s) = 20 m/s.
const POSITION: &str = "def x(t: s) -> m:\n    return 5 m/s^2 * t^2\n";

fn run(src: &str) -> String {
    let m = lower_ok(src);
    let (out, result) = run_main(&m);
    assert_eq!(result, Ok(()), "{src}");
    out
}

#[test]
fn the_derivative_of_a_position_is_a_velocity() {
    let src = format!("{POSITION}print(x(2 s))\nprint(derivative(x, t)(2 s))\n");
    assert_eq!(run(&src), "20 m\n20 m/s\n");
}

#[test]
fn the_derivative_function_is_added_once_and_called() {
    let src = format!(
        "{POSITION}print(derivative(x, t)(1 s))\nprint(derivative(x, t)(2 s))\nprint(derivative(x, t)(3 s))\n"
    );
    let m = lower_ok(&src);
    assert_eq!(names(&m), ["<main>", "x", "dx/dt"]);
    let calls = main_fn(&m)
        .ops
        .iter()
        .filter(|o| matches!(&o.kind, OpKind::Call { callee } if callee == "dx/dt"))
        .count();
    assert_eq!(calls, 3);
    // The derivative function returns (value, derivative).
    assert_eq!(func(&m, "dx/dt").results.len(), 2);
    assert_eq!(run(&src), "10 m/s\n20 m/s\n30 m/s\n");
}

#[test]
fn several_parameters_pick_the_one_named() {
    let src = "def spring(x: m, k: N/m) -> J:\n    return 0.5 * k * x^2\n\
               print(derivative(spring, x)(0.1 m, 200 N/m))\n\
               print(derivative(spring, k)(0.1 m, 200 N/m))\n";
    let m = lower_ok(src);
    assert_eq!(names(&m), ["<main>", "spring", "dspring/dx", "dspring/dk"]);
    assert_eq!(run(src), "20 N\n0.005000000000000001 m^2\n");
}

#[test]
fn a_derivative_reads_module_names_like_the_function_does() {
    let src = "g = 9.5 m/s^2\ndef fall(t: s) -> m:\n    return 0.5 * g * t^2\n\
               def speed(t: s) -> m/s:\n    return derivative(fall, t)(t)\n\
               g = 10 m/s^2\nprint(speed(2 s))\n";
    assert_eq!(run(src), "20 m/s\n");
}

#[test]
fn a_derivative_inside_a_loop_and_a_function() {
    let src = format!(
        "{POSITION}def v(t: s) -> m/s:\n    return derivative(x, t)(t)\n\
         t = 0 s\ntotal = 0 m/s\nwhile t < 3 s:\n    total += v(t)\n    t += 1 s\nprint(total)\n"
    );
    // v(t) = 10 t m/s^2: 0 + 10 + 20
    assert_eq!(run(&src), "30 m/s\n");
}

#[test]
fn control_flow_and_math_in_the_function() {
    let src = "def f(t: s) -> m:\n    if t > 1 s:\n        return sqrt(t / 1 s) * 1 m\n    return 2 m/s * t\n\
               print(derivative(f, t)(0.5 s))\nprint(derivative(f, t)(4 s))\n";
    assert_eq!(run(src), "2 m/s\n0.25 m/s\n");
}

#[test]
fn recursion_and_calls_between_functions() {
    let src = "def sq(a: m) -> m^2:\n    return a * a\n\
               def f(a: m) -> m^2:\n    return sq(a) + sq(2 * a)\n\
               print(derivative(f, a)(3 m))\n";
    // f = 5 a²: f' = 10 a
    assert_eq!(run(src), "30 m\n");
}

#[test]
fn a_second_derivative_differentiates_the_function_that_calls_the_first() {
    // a(t) = d/dt (d/dt 5 m/s^2 t^2) = 10 m/s^2
    let src = format!(
        "{POSITION}def v(t: s) -> m/s:\n    return derivative(x, t)(t)\n\
         print(derivative(v, t)(2 s))\n"
    );
    let m = lower_ok(&src);
    assert_eq!(
        names(&m),
        ["<main>", "x", "v", "dx/dt", "dv/dt", "ddx/dt/dt"]
    );
    assert_eq!(run(&src), "10 m/s^2\n");
}

#[test]
fn a_function_the_ad_pass_cannot_differentiate_is_a_lowering_error() {
    // `g` has an unannotated parameter, so the derivative flowing through
    // it has no unit: rejected before anything runs.
    let src = "def g(a):\n    return a * 2\n\
               def f(t: s) -> s:\n    return g(t)\n\
               print(1)\nprint(derivative(f, t)(1 s))\n";
    let e = lower_err(src);
    assert!(
        e.contains("cannot differentiate `f`: the result of the call to `g` has an unknown type"),
        "{e}"
    );
    assert!(e.contains("unit annotation"), "{e}");
    assert!(e.contains("print(derivative(f, t)(1 s))"), "{e}");
}

#[test]
fn a_gradient_calls_one_derivative_per_parameter_and_gathers_them_in_a_vector() {
    let src = "k = 3 N/m\ndef U(x: m, y: m) -> J:\n    return k * (x^2 + y^2)\n\
               print(grad(U)(1 m, 2 m))\n";
    let m = lower_ok(src);
    assert_eq!(names(&m), ["<main>", "U", "dU/dx", "dU/dy"]);
    let main = main_fn(&m);
    for callee in ["dU/dx", "dU/dy"] {
        let calls = main
            .ops
            .iter()
            .filter(|o| matches!(&o.kind, OpKind::Call { callee: c } if c == callee))
            .count();
        assert_eq!(calls, 1, "{callee}");
    }
    let literal = main
        .ops
        .iter()
        .find(|o| matches!(o.kind, OpKind::ArrayLiteral { cols: None }))
        .expect("the derivatives are gathered in an array literal");
    assert_eq!(literal.operands.len(), 2);
    assert_eq!(run(src), "[6 N, 12 N]\n");
}

#[test]
fn partial_notation_lowers_to_the_derivative_function() {
    let src = format!("{POSITION}print((∂x/∂t)(2 s))\n");
    let m = lower_ok(&src);
    assert_eq!(names(&m), ["<main>", "x", "dx/dt"]);
    assert_eq!(run(&src), "20 m/s\n");
}
