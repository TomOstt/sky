//! Automatic differentiation through the solves of a general matrix:
//! `solve[lu]` and `solve.right[lu]`.
//!
//! `A⁻¹ @ b` and `b @ A⁻¹` lower to solves for any square `A`
//! (`sky_types::chain`), and `sky_ad` differentiates the lowered SIR. The
//! forward tangent reuses the solve (`tr = A⁻¹ (tb - tA r)`, and for the
//! transposed solve `tr = (tx - r tA) A⁻¹`). The adjoint of `solve[lu]` is
//! **not** a solve with the same `A`, since `A` is not symmetric: it is the
//! transposed solve `r̄ A⁻¹` (and the adjoint of `solve.right` is a solve), so
//! a wrong rule shows up here as a gradient that disagrees with the central
//! finite difference of the function itself.

mod common;

use common::lower_ok;
use sky_ir::{Module, OpKind};
use sky_ir_interp::{call, Value};
use sky_units::Dimension;

const L: Dimension = Dimension::LENGTH;
const ONE: Dimension = Dimension::DIMENSIONLESS;

fn vector(xs: &[f64]) -> Value {
    Value::array(vec![xs.len()], xs.to_vec(), ONE)
}

fn matrix(rows: usize, cols: usize, xs: &[f64]) -> Value {
    Value::array(vec![rows, cols], xs.to_vec(), ONE)
}

fn run(m: &Module, name: &str, args: &[Value]) -> Vec<Value> {
    sky_ast::with_deep_stack(|| call(m, name, args.to_vec(), &mut Vec::new()))
        .unwrap_or_else(|e| panic!("`{name}` failed: {}", e.message))
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

/// Differentiates `func` with respect to parameter `wrt`, verifies the
/// module, and runs the derivative: `(module, value, derivative)`.
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
    let d = out.pop().unwrap();
    let v = out.pop().unwrap();
    (m, v, d)
}

/// The central finite difference of the scalar function `func` with respect
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

fn check_gradient(src: &str, func: &str, wrt: usize, args: &[Value]) {
    let (m, value, grad) = differentiate(src, func, wrt, args);
    assert_eq!(
        value,
        run(&m, func, args)[0],
        "the primal value is unchanged"
    );
    let (got, dim) = entries(&grad);
    assert_eq!(dim, L, "the dimension of the gradient");
    let want = finite_difference(&m, func, wrt, args);
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(&want) {
        assert!(
            (g - w).abs() <= 1e-6 * w.abs().max(1.0),
            "{func} wrt {wrt}: {got:?} vs {want:?}"
        );
    }
}

fn has_op(m: &Module, func: &str, pred: impl Fn(&OpKind) -> bool) -> bool {
    m.function(func)
        .expect("function")
        .ops
        .iter()
        .any(|o| pred(&o.kind))
}

#[test]
fn the_gradient_through_a_solve_with_a_matrix_that_is_not_symmetric() {
    let src = "def f(A: Matrix[2, 2], b: Vector[2]) -> m:\n    x = A⁻¹ @ b\n    return 1 m * (x @ x + x[0])\n";
    let a = matrix(2, 2, &[3.0, 1.0, -0.5, 2.0]);
    let b = vector(&[1.0, -2.0]);
    let (m, _, _) = differentiate(src, "f", 0, &[a.clone(), b.clone()]);
    assert!(
        has_op(&m, "f", |k| matches!(k, OpKind::Solve { .. })),
        "the source is a solve, not an inverse"
    );
    check_gradient(src, "f", 0, &[a.clone(), b.clone()]);
    check_gradient(src, "f", 1, &[a, b]);
}

#[test]
fn the_gradient_through_a_transposed_solve() {
    let src = "def g(A: Matrix[2, 2], b: Vector[2]) -> m:\n    y = b @ A⁻¹\n    return 1 m * (y @ y + y[1])\n";
    let a = matrix(2, 2, &[3.0, 1.0, -0.5, 2.0]);
    let b = vector(&[1.0, -2.0]);
    let (m, _, _) = differentiate(src, "g", 0, &[a.clone(), b.clone()]);
    assert!(
        has_op(&m, "g", |k| matches!(k, OpKind::SolveRight { .. })),
        "the source is a transposed solve"
    );
    check_gradient(src, "g", 0, &[a.clone(), b.clone()]);
    check_gradient(src, "g", 1, &[a, b]);
}

#[test]
fn the_gradient_through_solves_with_a_matrix_of_right_hand_sides() {
    let a = matrix(3, 3, &[4.0, 1.0, 0.5, -1.0, 3.0, 2.0, 0.25, -0.5, 2.0]);
    // A⁻¹ B: B has two right-hand sides.
    let src = "def h(A: Matrix[3, 3], B: Matrix[3, 2]) -> m:\n    X = A⁻¹ @ B\n    return 1 m * (X[0, 0] + 2 * X[2, 1] + X[0, 1] * X[1, 0])\n";
    let b = matrix(3, 2, &[1.0, -1.0, 0.5, 2.0, -0.25, 1.5]);
    check_gradient(src, "h", 0, &[a.clone(), b.clone()]);
    check_gradient(src, "h", 1, &[a.clone(), b]);
    // B A⁻¹: B has two rows.
    let src = "def k(A: Matrix[3, 3], B: Matrix[2, 3]) -> m:\n    X = B @ A⁻¹\n    return 1 m * (X[0, 0] + 2 * X[1, 2] + X[0, 1] * X[1, 0])\n";
    let b = matrix(2, 3, &[1.0, -1.0, 0.5, 2.0, -0.25, 1.5]);
    check_gradient(src, "k", 0, &[a.clone(), b.clone()]);
    check_gradient(src, "k", 1, &[a, b]);
}

#[test]
fn the_derivative_through_solves_of_a_matrix_that_depends_on_the_parameter() {
    // Forward mode, with the matrix and both right-hand sides moving.
    let src = "def s(t: s) -> m:\n    A = [[4 + t / 1 s, 1], [-0.5, 3]]\n    b = [1, 2]\n    return 1 m * ((A⁻¹ @ b)[0] + (b @ A⁻¹)[1])\n";
    let at = |t: f64| Value::number(t, Dimension::TIME);
    let (m, v, d) = differentiate(src, "s", 0, &[at(0.7)]);
    let f = |t: f64| number(&run(&m, "s", &[at(t)])[0]);
    let h = 1e-6;
    let fd = (f(0.7 + h) - f(0.7 - h)) / (2.0 * h);
    let got = number(&d);
    assert!(
        (got - fd).abs() <= 1e-6 * fd.abs().max(1.0),
        "{got} vs {fd}"
    );
    assert_eq!(number(&v), f(0.7));
}
