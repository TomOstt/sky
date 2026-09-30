//! `norm(v)` and `dot(a, b)` differentiate: the derivative of `norm` is
//! `(v·dv)/‖v‖` (forward) and `(r̄/‖v‖) v` (reverse), and that of `dot` is the
//! product rule. Each case lowers a Sky program, asks `sky_ad` for the
//! derivative, verifies the SIR (a rule with a wrong dimension is
//! ill-typed), runs it, and compares with an exact answer worked out by
//! hand and with a central finite difference.

mod common;

use common::lower_ok;
use sky_ir::Module;
use sky_ir_interp::{call, Value};
use sky_units::Dimension;

const L: Dimension = Dimension::LENGTH;
const ONE: Dimension = Dimension::DIMENSIONLESS;

fn vector(xs: &[f64], dim: Dimension) -> Value {
    Value::array(vec![xs.len()], xs.to_vec(), dim)
}

fn matrix(rows: usize, cols: usize, xs: &[f64], dim: Dimension) -> Value {
    Value::array(vec![rows, cols], xs.to_vec(), dim)
}

fn scalar(x: f64, dim: Dimension) -> Value {
    Value::number(x, dim)
}

fn try_run(m: &Module, name: &str, args: &[Value]) -> Result<Vec<Value>, String> {
    sky_ast::with_deep_stack(|| call(m, name, args.to_vec(), &mut Vec::new()))
        .map_err(|e| e.message.clone())
}

fn run(m: &Module, name: &str, args: &[Value]) -> Vec<Value> {
    try_run(m, name, args).unwrap_or_else(|e| panic!("`{name}` failed: {e}"))
}

fn differentiate(src: &str, func: &str, wrt: usize) -> (Module, String) {
    let mut m = lower_ok(src);
    let name = sky_ad::differentiate_into(&mut m, func, wrt)
        .unwrap_or_else(|e| panic!("cannot differentiate `{func}`: {e}"));
    if let Err(errors) = sky_ir::verify(&m) {
        panic!(
            "the derivative is ill-typed SIR: {errors:?}\n{}",
            sky_ir::print(&m)
        );
    }
    (m, name)
}

fn entries(v: &Value) -> (Vec<f64>, Dimension) {
    match v {
        Value::Array { data, dim, .. } => (data.clone(), *dim),
        Value::Number { value, dim } => (vec![*value], *dim),
        other => panic!("not a number: {other:?}"),
    }
}

fn assert_close(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: {got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
        let scale = w.abs().max(1.0);
        assert!((g - w).abs() <= 1e-6 * scale, "{what}: {got:?} vs {want:?}");
    }
}

/// The gradient of the scalar function `func` with respect to the array
/// `args[wrt]`: checked against `exact` (if given) and central finite
/// differences, and for its dimension. Returns the gradient.
fn gradient(
    src: &str,
    func: &str,
    wrt: usize,
    args: &[Value],
    exact: Option<&[f64]>,
    want_dim: Dimension,
) -> Vec<f64> {
    let (m, name) = differentiate(src, func, wrt);
    let out = run(&m, &name, args);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0], run(&m, func, args)[0], "the primal is unchanged");
    let (got, dim) = entries(&out[1]);
    assert_eq!(dim, want_dim, "the dimension of the gradient");
    assert!(matches!(out[1], Value::Array { .. }), "an array");
    if let Some(exact) = exact {
        assert_close(&got, exact, "against the exact gradient");
    }
    let Value::Array { dims, data, dim } = &args[wrt] else {
        panic!("not an array");
    };
    let fd: Vec<f64> = (0..data.len())
        .map(|i| {
            let h = 1e-6 * data[i].abs().max(1.0);
            let at = |dx: f64| {
                let mut d = data.clone();
                d[i] += dx;
                let mut a = args.to_vec();
                a[wrt] = Value::array(dims.clone(), d, *dim);
                entries(&run(&m, func, &a)[0]).0[0]
            };
            (at(h) - at(-h)) / (2.0 * h)
        })
        .collect();
    assert_close(&got, &fd, "against finite differences");
    got
}

#[test]
fn the_gradient_of_the_natural_gravitational_potential() {
    // U = -k / |p|, ∇U = k p / |p|³. For p = [1, 2, 2] m, |p| = 3 m exactly.
    let src = "def U(p: Vector[3] m) -> J:\n    return -2 J*m / norm(p)\n";
    let g = gradient(
        src,
        "U",
        0,
        &[vector(&[1.0, 2.0, 2.0], L)],
        Some(&[2.0 / 27.0, 4.0 / 27.0, 4.0 / 27.0]),
        Dimension::FORCE,
    );
    assert_eq!(g.len(), 3);
}

#[test]
fn norm_and_the_square_root_form_have_one_gradient() {
    let with_norm =
        "def U(p: Vector[3] m) -> J:\n    return -7 J*m / norm(p) + 3 N/m * norm(p)^2\n";
    let with_sqrt =
        "def U(p: Vector[3] m) -> J:\n    return -7 J*m / sqrt(p @ p) + 3 N/m * (p @ p)\n";
    let p = vector(&[0.5, -1.25, 2.0], L);
    let a = gradient(with_norm, "U", 0, &[p.clone()], None, Dimension::FORCE);
    let b = gradient(with_sqrt, "U", 0, &[p], None, Dimension::FORCE);
    assert_close(&a, &b, "norm against sqrt(p @ p)");
}

#[test]
fn the_gradient_of_the_norm_is_the_unit_vector() {
    // ∇|p| = p / |p|, dimensionless; [3, 4] has length 5.
    let src = "def f(p: Vector[2] m) -> m:\n    return norm(p)\n";
    gradient(
        src,
        "f",
        0,
        &[vector(&[3.0, 4.0], L)],
        Some(&[0.6, 0.8]),
        ONE,
    );
}

#[test]
fn the_norm_of_a_matrix_and_of_a_scalar() {
    // The Frobenius norm: ∇‖A‖ = A / ‖A‖. ‖[[1, 2], [2, 4]]‖ = 5.
    let src = "def f(A: Matrix[2, 2] m) -> m:\n    return norm(A)\n";
    let a = matrix(2, 2, &[1.0, 2.0, 2.0, 4.0], L);
    let g = gradient(src, "f", 0, &[a], Some(&[0.2, 0.4, 0.4, 0.8]), ONE);
    assert_eq!(g.len(), 4);
    // The norm of a scalar is its absolute value.
    let src = "def f(p: Vector[2] m) -> m:\n    return norm(p[0] - p[1])\n";
    for (p, want) in [([1.0, 3.0], [-1.0, 1.0]), ([3.0, 1.0], [1.0, -1.0])] {
        gradient(src, "f", 0, &[vector(&p, L)], Some(&want), ONE);
    }
}

#[test]
fn the_gradient_of_dot() {
    // f = p·a with a constant: ∇f = a.
    let src = "def f(p: Vector[3] m) -> J:\n    return dot(p, [1 N, 2 N, 3 N])\n";
    gradient(
        src,
        "f",
        0,
        &[vector(&[0.5, 1.0, -2.0], L)],
        Some(&[1.0, 2.0, 3.0]),
        Dimension::FORCE,
    );
    // Both factors depend on p: f = p·(2p + c), ∇f = 4p + c.
    let src =
        "def f(p: Vector[3] m) -> J:\n    q = 2 N/m * p + [1 N, 0 N, -1 N]\n    return dot(p, q)\n";
    gradient(
        src,
        "f",
        0,
        &[vector(&[0.5, 1.0, -2.0], L)],
        Some(&[4.0 * 0.5 + 1.0, 4.0 * 1.0, 4.0 * -2.0 - 1.0]),
        Dimension::FORCE,
    );
    // With respect to either of two arguments, `norm` and `dot` together.
    let src =
        "def f(p: Vector[2] m, q: Vector[2] m) -> m:\n    return norm(p - q) + dot(q, p) / 1 m\n";
    let args = [vector(&[0.5, 1.0], L), vector(&[1.5, -0.5], L)];
    gradient(src, "f", 0, &args, None, ONE);
    gradient(src, "f", 1, &args, None, ONE);
}

#[test]
fn the_derivative_of_norm_and_dot_in_forward_mode() {
    // |[3 t, 4 t]| = 5 t: the derivative is 5 m/s.
    let src = "def f(t: s) -> m:\n    return norm([3 m/s * t, 4 m/s * t])\n";
    let (m, name) = differentiate(src, "f", 0);
    let out = run(&m, &name, &[scalar(2.0, Dimension::TIME)]);
    assert_eq!(entries(&out[0]).0, [10.0]);
    assert_eq!(entries(&out[1]), (vec![5.0], Dimension::VELOCITY));
    // r = [t², 1] m, v = [1, t] m/s: r·v = t² + t, so d/dt = 2t + 1.
    let src = "def f(t: s) -> m^2/s:\n    r = [1 m/s^2 * t^2, 1 m]\n    v = [1 m/s, 1 m/s^2 * t]\n    return dot(r, v)\n";
    let (m, name) = differentiate(src, "f", 0);
    let out = run(&m, &name, &[scalar(3.0, Dimension::TIME)]);
    assert_eq!(entries(&out[0]).0, [12.0]);
    let (d, dim) = entries(&out[1]);
    assert_eq!(d, [7.0]);
    assert_eq!(dim, Dimension::from_ints([2, 0, -2, 0, 0, 0, 0]));
}

#[test]
fn a_loop_falls_back_to_forward_mode_through_norm_and_dot() {
    // Reverse mode has no tape for a loop, so the gradient is one forward
    // pass per entry: the tangent rules of `norm` and `dot` again.
    let src = "def f(p: Vector[3] m) -> m^2:\n    acc = 0 m^2\n    i = 0\n    while i < 2:\n        acc = acc + norm(p) * p[i] + dot(p, p)\n        i += 1\n    return acc\n";
    let (m, name) = differentiate(src, "f", 0);
    assert!(
        m.functions.iter().any(|f| f.name.contains("#jvp")),
        "the forward fallback ran"
    );
    let p = vector(&[0.5, -1.0, 2.0], L);
    let out = run(&m, &name, &[p.clone()]);
    let (got, dim) = entries(&out[1]);
    assert_eq!(dim, L);
    let h = 1e-6;
    let want: Vec<f64> = (0..3)
        .map(|i| {
            let at = |dx: f64| {
                let mut d = entries(&p).0;
                d[i] += dx;
                entries(&run(&m, "f", &[vector(&d, L)])[0]).0[0]
            };
            (at(h) - at(-h)) / (2.0 * h)
        })
        .collect();
    assert_close(&got, &want, "against finite differences");
}

#[test]
fn the_norm_has_no_derivative_at_zero_and_fails_like_sqrt() {
    // The norm of the zero vector has no derivative: the run fails with the
    // error the derivative of `sqrt` has at 0, and never returns a made-up
    // number.
    let norm_src = "def f(p: Vector[2] m) -> m:\n    return norm(p)\n";
    let sqrt_src = "def f(p: Vector[2] m) -> m:\n    return sqrt(p @ p)\n";
    let zero = vector(&[0.0, 0.0], L);
    let failure = |src: &str| {
        let (m, name) = differentiate(src, "f", 0);
        try_run(&m, &name, &[zero.clone()]).expect_err("no derivative at the origin")
    };
    let (a, b) = (failure(norm_src), failure(sqrt_src));
    assert!(a.contains("division by zero"), "{a}");
    assert_eq!(a, b);
    // The value itself is fine at 0.
    let m = lower_ok(norm_src);
    assert_eq!(entries(&run(&m, "f", &[zero])[0]).0, [0.0]);
    // Forward mode says the same.
    let src = "def g(t: s) -> m:\n    return norm([0 m/s * t, 0 m/s * t])\n";
    let (m, name) = differentiate(src, "g", 0);
    let err = try_run(&m, &name, &[scalar(1.0, Dimension::TIME)]).expect_err("no derivative");
    assert!(err.contains("division by zero"), "{err}");
}
