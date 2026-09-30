//! Automatic differentiation through arrays: the gradient of a scalar
//! function of an array parameter (reverse mode, and the forward-mode
//! fallback), and the derivative of an array-valued function of a scalar
//! (forward mode).
//!
//! Each case lowers a Sky program to SIR, asks `sky_ad` for the derivative,
//! verifies the SIR it built (a rule with a wrong dimension or shape makes
//! ill-typed IR), runs it in the reference interpreter, and compares the
//! result with an exact answer worked out by hand and with a central finite
//! difference of the function itself.

mod common;

use common::lower_ok;
use sky_ir::Module;
use sky_ir_interp::{call, Value};
use sky_units::Dimension;

fn vector(xs: &[f64], dim: Dimension) -> Value {
    Value::array(vec![xs.len()], xs.to_vec(), dim)
}

fn matrix(rows: usize, cols: usize, xs: &[f64], dim: Dimension) -> Value {
    Value::array(vec![rows, cols], xs.to_vec(), dim)
}

fn scalar(x: f64, dim: Dimension) -> Value {
    Value::number(x, dim)
}

fn run(m: &Module, name: &str, args: &[Value]) -> Vec<Value> {
    sky_ast::with_deep_stack(|| call(m, name, args.to_vec(), &mut Vec::new()))
        .unwrap_or_else(|e| panic!("`{name}` failed: {}", e.message))
}

/// Differentiates `func` with respect to parameter `wrt`, verifies the
/// module, and runs the derivative: `(value, derivative)`.
fn differentiate(src: &str, func: &str, wrt: usize, args: &[Value]) -> (Module, Value, Value) {
    let mut m = lower_ok(src);
    let name = sky_ad::differentiate_into(&mut m, func, wrt)
        .unwrap_or_else(|e| panic!("cannot differentiate `{func}`: {e}"));
    if let Err(errors) = sky_ir::verify(&m) {
        panic!(
            "the derivative is ill-typed SIR: {errors:?}\n{}",
            sky_ir::print(&m)
        );
    }
    let mut out = run(&m, &name, args);
    assert_eq!(out.len(), 2, "a derivative returns (value, derivative)");
    let d = out.pop().unwrap();
    let v = out.pop().unwrap();
    (m, v, d)
}

fn entries(v: &Value) -> (Vec<f64>, Dimension) {
    match v {
        Value::Array { data, dim, .. } => (data.clone(), *dim),
        Value::Number { value, dim } => (vec![*value], *dim),
        other => panic!("not a number: {other:?}"),
    }
}

fn number(v: &Value) -> f64 {
    entries(v).0[0]
}

/// The central finite difference of `func` (a scalar function) with respect
/// to every entry of the array `args[wrt]`.
fn finite_difference(m: &Module, func: &str, wrt: usize, args: &[Value]) -> Vec<f64> {
    let Value::Array { dims, data, dim } = &args[wrt] else {
        panic!("not an array");
    };
    (0..data.len())
        .map(|i| {
            let h = 1e-6 * data[i].abs().max(1.0);
            let at = |dx: f64| {
                let mut d = data.clone();
                d[i] += dx;
                let mut a = args.to_vec();
                a[wrt] = Value::array(dims.clone(), d, *dim);
                number(&run(m, func, &a)[0])
            };
            (at(h) - at(-h)) / (2.0 * h)
        })
        .collect()
}

fn assert_close(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: {got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
        let scale = w.abs().max(1.0);
        assert!((g - w).abs() <= 1e-6 * scale, "{what}: {got:?} vs {want:?}");
    }
}

/// The gradient of `func` at `args`, against `exact` (if given) and against
/// finite differences; also its dimension. Returns the gradient.
fn check_gradient(
    src: &str,
    func: &str,
    wrt: usize,
    args: &[Value],
    exact: Option<&[f64]>,
    want_dim: Dimension,
) -> Vec<f64> {
    let (m, value, grad) = differentiate(src, func, wrt, args);
    // The value is what the function returns.
    assert_eq!(
        value,
        run(&m, func, args)[0],
        "the primal value is unchanged"
    );
    let (got, dim) = entries(&grad);
    assert_eq!(dim, want_dim, "the dimension of the gradient");
    assert!(
        matches!(grad, Value::Array { .. }),
        "the gradient is an array"
    );
    if let Some(exact) = exact {
        assert_close(&got, exact, "against the exact gradient");
    }
    assert_close(
        &got,
        &finite_difference(&m, func, wrt, args),
        "against finite differences",
    );
    got
}

const L: Dimension = Dimension::LENGTH;
const ONE: Dimension = Dimension::DIMENSIONLESS;

// ----- the physics -----

#[test]
fn the_gradient_of_a_spring_potential_of_a_position_is_a_force() {
    // U = ½ k p·p, ∇U = k p.
    let src = "def U(p: Vector[3] m) -> J:\n    return 0.5 * 200 N/m * (p @ p)\n";
    let p = vector(&[0.1, -0.2, 0.3], L);
    let g = check_gradient(
        src,
        "U",
        0,
        &[p],
        Some(&[20.0, -40.0, 60.0]),
        Dimension::FORCE,
    );
    assert_eq!(g.len(), 3);
}

#[test]
fn the_gradient_of_a_gravitational_potential_points_to_the_origin() {
    // U = -G M m / |p|, F = -∇U = -G M m p / |p|³.
    let src = "def U(p: Vector[3] m) -> J:\n    return -6.674e-11 m^3/kg/s^2 * 5.972e24 kg * 1000 kg / sqrt(p @ p)\n";
    let p = vector(&[7.0e6, 0.0, 0.0], L);
    let (_, _, grad) = differentiate(src, "U", 0, &[p]);
    let (g, dim) = entries(&grad);
    assert_eq!(dim, Dimension::FORCE);
    let gmm = 6.674e-11 * 5.972e24 * 1000.0;
    let r = 7.0e6;
    // ∇U = +G M m p / |p|³ = G M m / r² along +x: the force -∇U points inward.
    assert!((g[0] - gmm / (r * r)).abs() < 1e-9 * gmm / (r * r), "{g:?}");
    assert_eq!(g[1], 0.0);
    assert_eq!(g[2], 0.0);
}

#[test]
fn a_spring_network_with_a_stiffness_matrix() {
    // U = ½ (p - p0)ᵀ K (p - p0) with K symmetric: ∇U = K (p - p0).
    let src = "def U(p: Vector[3] m) -> J:\n    K = [[300, -100, 0], [-100, 200, -50], [0, -50, 100]] * 1 N/m\n    p0 = [0.5 m, 0 m, 0.25 m]\n    d = p - p0\n    return 0.5 * (d @ (K @ d))\n";
    let p = vector(&[0.6, 0.1, 0.2], L);
    let exact = [
        300.0 * 0.1 - 100.0 * 0.1 + 0.0,
        -100.0 * 0.1 + 200.0 * 0.1 - 50.0 * -0.05,
        -50.0 * 0.1 + 100.0 * -0.05,
    ];
    check_gradient(src, "U", 0, &[p], Some(&exact), Dimension::FORCE);
}

// ----- the array operations, one rule at a time -----

#[test]
fn elementwise_arithmetic_and_scaling() {
    let src = "def f(p: Vector[3]) -> m:\n    q = 3 * p - p / 2 + [1, 2, 3]\n    r = -q + p\n    return 1 m * (r @ r)\n";
    check_gradient(src, "f", 0, &[vector(&[0.3, -1.0, 2.0], ONE)], None, L);
}

#[test]
fn a_scalar_that_depends_on_the_array_scales_an_array() {
    // The scalar factor s = p·p is used in every entry of s * p: its
    // adjoint is a sum over the entries.
    let src = "def f(p: Vector[3]) -> m:\n    s = p @ p\n    q = s * p\n    return 1 m * (q @ [1, 2, 3])\n";
    check_gradient(src, "f", 0, &[vector(&[0.3, -1.0, 2.0], ONE)], None, L);
    let src = "def f(p: Vector[3]) -> m:\n    s = p @ p\n    q = [1, 2, 3] / s\n    return 1 m * (q @ [1, 0, 0]) + 1 m * (q @ [0, 1, 1])\n";
    check_gradient(src, "f", 0, &[vector(&[0.3, -1.0, 2.0], ONE)], None, L);
}

#[test]
fn a_scalar_that_depends_on_the_array_scales_a_matrix() {
    let src = "def f(p: Vector[2]) -> m:\n    s = p @ p\n    A = s * [[1, 2], [3, 4]]\n    return 1 m * (p @ (A @ p))\n";
    check_gradient(src, "f", 0, &[vector(&[0.3, -1.0], ONE)], None, L);
    let src = "def f(p: Vector[2]) -> m:\n    s = p @ p\n    A = [[1, 2], [3, 4]] / s\n    return 1 m * (p @ (A @ p))\n";
    check_gradient(src, "f", 0, &[vector(&[0.3, -1.0], ONE)], None, L);
}

#[test]
fn index_slice_and_array_literals() {
    let src = "def f(p: Vector[4]) -> m:\n    a = p[1] * p[2]\n    v = [p[0], p[3], a]\n    w = p[1:3]\n    return 1 m * (v @ v + w @ w + p[3] * p[3])\n";
    check_gradient(src, "f", 0, &[vector(&[0.3, -1.0, 2.0, 0.5], ONE)], None, L);
}

#[test]
fn indexing_with_a_computed_position() {
    let src = "def f(p: Vector[4]) -> m:\n    i = 2\n    return 1 m * (p[i] * p[i + 1] + p[len(p) - 1])\n";
    check_gradient(src, "f", 0, &[vector(&[0.3, -1.0, 2.0, 0.5], ONE)], None, L);
}

#[test]
fn matrix_parameters_index_slice_and_transpose() {
    let src = "def f(A: Matrix[2, 3]) -> m:\n    r = A[1]\n    c = A[0, 1:3]\n    B = Aᵀ @ A\n    return 1 m * (r @ r + c @ c + B[2, 0] + A[0, 0] * A[1, 2])\n";
    let a = matrix(2, 3, &[0.5, -1.0, 2.0, 1.5, 0.25, -0.75], ONE);
    check_gradient(src, "f", 0, &[a], None, L);
}

#[test]
fn slices_of_matrices_in_every_form() {
    let src = "def f(A: Matrix[3, 4]) -> m:\n    a = A[1:3, 1:3]\n    b = A[0:2, 2]\n    c = A[2, 0:3]\n    d = A[1:, :2]\n    return 1 m * (a[0, 1] * a[1, 0] + b @ b + c @ c + d[1, 1] * d[0, 0])\n";
    let a = matrix(
        3,
        4,
        &[
            0.5, -1.0, 2.0, 1.5, 0.25, -0.75, 1.25, 0.5, -2.0, 0.3, 0.9, -0.4,
        ],
        ONE,
    );
    check_gradient(src, "f", 0, &[a], None, L);
}

#[test]
fn matrix_products_in_every_shape() {
    let src = "def f(A: Matrix[2, 2], x: Vector[2]) -> m:\n    y = A @ x\n    z = x @ A\n    B = A @ A\n    return 1 m * (y @ z + B[0, 1] + x @ x)\n";
    let a = matrix(2, 2, &[0.5, -1.0, 2.0, 1.5], ONE);
    let x = vector(&[0.3, -0.8], ONE);
    check_gradient(src, "f", 0, &[a.clone(), x.clone()], None, L);
    check_gradient(src, "f", 1, &[a, x], None, L);
}

#[test]
fn the_inverse_and_solve_of_a_matrix() {
    let src = "def f(A: Matrix[2, 2]) -> m:\n    B = inv(A)\n    return 1 m * (B[0, 0] + 2 * B[1, 0] - B[0, 1] * B[1, 1])\n";
    let a = matrix(2, 2, &[3.0, 1.0, 0.5, 2.0], ONE);
    check_gradient(src, "f", 0, &[a], None, L);
    // A⁻¹ b with A positive definite is a Cholesky solve.
    let src = "def g(b: Vector[2]) -> m:\n    A = [[4, 1], [1, 3]]\n    x = A⁻¹ @ b\n    return 1 m * (x @ x)\n";
    check_gradient(src, "g", 0, &[vector(&[1.0, -2.0], ONE)], None, L);
}

#[test]
fn units_in_the_inverse() {
    // A in N/m: A⁻¹ is m/N, and U = ½ fᵀ A⁻¹ f is in J for f in N; its
    // gradient with respect to f is A⁻¹ f, a length.
    let src = "def U(f: Vector[2] N) -> J:\n    K = [[4, 1], [1, 3]] * 1 N/m\n    x = K⁻¹ @ f\n    return 0.5 * (f @ x)\n";
    let f = vector(&[1.0, -2.0], Dimension::FORCE);
    check_gradient(src, "U", 0, &[f], None, L);
}

#[test]
fn zeros_identity_and_len_have_no_tangent() {
    let src = "def f(p: Vector[3]) -> m:\n    Z = zeros(3)\n    I = identity(3)\n    return 1 m * (p @ (I @ p) + (Z @ p))\n";
    check_gradient(
        src,
        "f",
        0,
        &[vector(&[0.3, -1.0, 2.0], ONE)],
        Some(&[0.6, -2.0, 4.0]),
        L,
    );
}

#[test]
fn the_gradient_of_a_constant_is_zero() {
    let src = "def f(p: Vector[3]) -> m:\n    return 2 m\n";
    check_gradient(
        src,
        "f",
        0,
        &[vector(&[0.3, -1.0, 2.0], ONE)],
        Some(&[0.0, 0.0, 0.0]),
        L,
    );
}

// ----- control flow and calls -----

#[test]
fn a_gradient_through_if() {
    let src = "def f(p: Vector[3] m) -> J:\n    e = 1 N/m * (p @ p)\n    if p[0] > 0 m:\n        e = e + 2 N * p[1]\n    else:\n        e = 3 * e\n    return e\n";
    for p in [[0.5, -1.0, 2.0], [-0.5, -1.0, 2.0]] {
        check_gradient(src, "f", 0, &[vector(&p, L)], None, Dimension::FORCE);
    }
}

#[test]
fn a_gradient_through_nested_ifs() {
    let src = "def f(p: Vector[2]) -> m:\n    x = p[0]\n    y = p[1]\n    r = 0\n    if x > 0:\n        if y > 0:\n            r = x * y\n        else:\n            r = x * x - y\n    else:\n        r = y * y * y\n    return 1 m * r\n";
    for p in [[0.5, 1.0], [0.5, -1.0], [-0.5, 2.0]] {
        check_gradient(src, "f", 0, &[vector(&p, ONE)], None, L);
    }
}

#[test]
fn helpers_that_take_and_return_arrays_are_part_of_the_function() {
    let src = "def displacement(p: Vector[3] m) -> Vector[3] m:\n    return p - [1 m, 0 m, 0 m]\n\
               def norm(v: Vector[3] m) -> m:\n    return sqrt(v @ v)\n\
               def U(p: Vector[3] m) -> J:\n    return 5 N/m * norm(displacement(p))^2 + 1 N * norm(p)\n";
    check_gradient(
        src,
        "U",
        0,
        &[vector(&[0.5, 1.0, -2.0], L)],
        None,
        Dimension::FORCE,
    );
}

#[test]
fn a_scalar_helper_is_called_through_its_derivative() {
    // `bump` takes and returns scalars, so it stays a call, and its
    // derivative is the callee's forward derivative.
    let src = "def bump(x: m) -> J:\n    return 2 N/m^2 * x * x * x\n\
               def U(p: Vector[3] m) -> J:\n    return bump(p[0]) + bump(p[1] + p[2])\n";
    check_gradient(
        src,
        "U",
        0,
        &[vector(&[0.5, 1.0, -2.0], L)],
        None,
        Dimension::FORCE,
    );
}

#[test]
fn a_scalar_helper_with_a_loop_is_reached_from_reverse_mode() {
    let src = "def power3(x: m) -> m:\n    r = 1 m\n    i = 0\n    while i < 3:\n        r = r * x / 1 m\n        i += 1\n    return r\n\
               def U(p: Vector[2] m) -> m:\n    return power3(p[0]) + power3(p[1]) * 2\n";
    check_gradient(
        src,
        "U",
        0,
        &[vector(&[0.5, 1.5], L)],
        Some(&[0.75, 13.5]),
        ONE,
    );
}

#[test]
fn a_loop_falls_back_to_one_pass_per_entry() {
    // Reverse mode has no tape for a loop, so the gradient is computed by
    // forward mode, one pass per entry, and is the same.
    let src = "def f(p: Vector[3]) -> m:\n    acc = 0\n    i = 0\n    while i < 3:\n        acc = acc + p[i] * p[i] * (i + 1)\n        i += 1\n    return 1 m * acc\n";
    let p = vector(&[0.3, -1.0, 2.0], ONE);
    let g = check_gradient(src, "f", 0, &[p], Some(&[0.6, -4.0, 12.0]), L);
    assert_eq!(g.len(), 3);
    // The same function with the loop unrolled, which reverse mode takes.
    let unrolled = "def f(p: Vector[3]) -> m:\n    return 1 m * (p[0] * p[0] + p[1] * p[1] * 2 + p[2] * p[2] * 3)\n";
    let (m, ..) = differentiate(unrolled, "f", 0, &[vector(&[0.3, -1.0, 2.0], ONE)]);
    let looped = differentiate(src, "f", 0, &[vector(&[0.3, -1.0, 2.0], ONE)]).0;
    // Reverse mode calls no `#jvp` function; the fallback does.
    assert!(m.functions.iter().all(|f| !f.name.contains("#jvp")));
    assert!(looped.functions.iter().any(|f| f.name.contains("#jvp")));
}

#[test]
fn the_fallback_gradient_of_a_matrix() {
    let src = "def f(A: Matrix[2, 2]) -> m:\n    acc = 0\n    i = 0\n    while i < 2:\n        acc = acc + A[i, 0] * A[i, 1] + A[i, i]\n        i += 1\n    return 1 m * acc\n";
    let a = matrix(2, 2, &[0.5, -1.0, 2.0, 1.5], ONE);
    check_gradient(
        src,
        "f",
        0,
        &[a],
        Some(&[1.0 + (-1.0), 0.5, 1.5, 2.0 + 1.0]),
        L,
    );
}

#[test]
fn a_recursive_function_of_an_array() {
    let src = "def f(p: Vector[2], n: m) -> m:\n    if n < 0.5 m:\n        return 1 m * (p @ p)\n    return f(p, n - 1 m) * (p[0] + 1)\n";
    check_gradient(
        src,
        "f",
        0,
        &[vector(&[0.3, 0.7], ONE), scalar(2.0, L)],
        None,
        L,
    );
}

#[test]
fn several_arguments_the_others_are_constants() {
    let src = "def f(p: Vector[3] m, w: Vector[3], t: s) -> J:\n    return 1 N/m * (p @ p) * (w @ w) * (t / 1 s)\n";
    let args = [
        vector(&[0.5, 1.0, -2.0], L),
        vector(&[1.0, 2.0, 3.0], ONE),
        scalar(2.0, Dimension::TIME),
    ];
    check_gradient(src, "f", 0, &args, None, Dimension::FORCE);
}

// ----- forward mode through arrays -----

#[test]
fn the_derivative_of_a_position_vector_is_a_velocity_vector() {
    let src = "def r(t: s) -> Vector[2] m:\n    return [5 m/s^2 * t^2, 2 m/s * t + 1 m]\n";
    let (_, v, d) = differentiate(src, "r", 0, &[scalar(2.0, Dimension::TIME)]);
    assert_eq!(entries(&v).0, [20.0, 5.0]);
    let (dd, dim) = entries(&d);
    assert_eq!(dd, [20.0, 2.0]);
    assert_eq!(dim, Dimension::VELOCITY);
}

#[test]
fn forward_mode_through_matrix_operations() {
    // x(t) = R(t) x0 with a rotation-like matrix of t: dx/dt = R'(t) x0.
    let src = "def x(t: s) -> Vector[2] m:\n    R = [[cos(t / 1 s), -sin(t / 1 s)], [sin(t / 1 s), cos(t / 1 s)]]\n    return R @ [1 m, 0 m]\n";
    let t = 0.7;
    let (_, _, d) = differentiate(src, "x", 0, &[scalar(t, Dimension::TIME)]);
    let (dd, dim) = entries(&d);
    assert_close(&dd, &[-t.sin(), t.cos()], "R'(t) x0");
    assert_eq!(dim, Dimension::VELOCITY);
    // Transpose, inverse and solve of a matrix that depends on t.
    let src = "def s(t: s) -> m:\n    A = [[4 + t / 1 s, 1], [1, 3]]\n    B = inv(A)\n    C = Aᵀ @ B\n    return 1 m * (B[0, 1] + C[1, 1] + (A⁻¹ @ [1, 2])[0])\n";
    let (m, _, d) = differentiate(src, "s", 0, &[scalar(0.5, Dimension::TIME)]);
    let h = 1e-6;
    let at = |t: f64| number(&run(&m, "s", &[scalar(t, Dimension::TIME)])[0]);
    let fd = (at(0.5 + h) - at(0.5 - h)) / (2.0 * h);
    assert_close(&[number(&d)], &[fd], "against finite differences");
}

#[test]
fn a_derivative_of_a_function_that_returns_a_vector_through_a_helper() {
    let src =
        "def shift(v: Vector[2] m, dt: s) -> Vector[2] m:\n    return v + [1 m/s, 2 m/s] * dt\n\
               def r(t: s) -> Vector[2] m:\n    return shift([t * 1 m/s^2 * t, 1 m], t)\n";
    let (_, _, d) = differentiate(src, "r", 0, &[scalar(2.0, Dimension::TIME)]);
    assert_eq!(entries(&d).0, [4.0 + 1.0, 2.0]);
}
