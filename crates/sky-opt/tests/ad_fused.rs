//! Differentiating fused kernels: the derivative of a function after the
//! `fuse` pass must equal the derivative of the same function before it.
//!
//! `sky-ad` differentiates a `fused` op elementwise (the tangent is a second
//! kernel whose body is the tangent of the body; the adjoint of each operand
//! is a kernel too), so nothing needs to unfuse first. The unfused function
//! is the oracle.

use sky_ir::{Module, OpKind};
use sky_ir_interp::{call, Value};
use sky_units::Dimension;

fn vector(xs: &[f64], dim: Dimension) -> Value {
    Value::array(vec![xs.len()], xs.to_vec(), dim)
}

fn run(m: &Module, name: &str, args: &[Value]) -> Vec<Value> {
    sky_ast::with_deep_stack(|| call(m, name, args.to_vec(), &mut Vec::new()))
        .unwrap_or_else(|e| panic!("`{name}` failed: {}", e.message))
}

fn count_fused(m: &Module, func: &str) -> usize {
    m.function(func)
        .expect("function")
        .ops
        .iter()
        .filter(|o| matches!(o.kind, OpKind::Fused))
        .count()
}

/// The derivative of `func` with respect to `wrt` on the unfused and on the
/// fused module, run at `args`: they must agree to rounding.
fn same_derivative(src: &str, func: &str, wrt: usize, args: &[Value]) -> Value {
    let plain = sky_lower::lower_source(src).unwrap_or_else(|e| panic!("{e}"));
    let mut fused = plain.clone();
    let rewrites = sky_opt::fuse(&mut fused);
    assert!(
        !rewrites.is_empty(),
        "the program has something to fuse:\n{src}"
    );
    assert!(count_fused(&fused, func) > 0, "`{func}` has a fused kernel");
    let derivative = |m: &Module| {
        let mut m = m.clone();
        let name = sky_ad::differentiate_into(&mut m, func, wrt)
            .unwrap_or_else(|e| panic!("cannot differentiate `{func}`: {e}"));
        if let Err(errors) = sky_ir::verify(&m) {
            panic!("ill-typed derivative: {errors:?}\n{}", sky_ir::print(&m));
        }
        let mut out = run(&m, &name, args);
        (out.remove(0), out.remove(0))
    };
    let (v0, d0) = derivative(&plain);
    let (v1, d1) = derivative(&fused);
    assert_eq!(v0, v1, "the value");
    let (
        Value::Array {
            data: a, dim: da, ..
        },
        Value::Array {
            data: b, dim: db, ..
        },
    ) = (&d0, &d1)
    else {
        // A scalar derivative (an array function of a scalar has an array one).
        let (Value::Number { value: a, dim: da }, Value::Number { value: b, dim: db }) = (&d0, &d1)
        else {
            panic!("derivatives of different kinds: {d0:?} {d1:?}");
        };
        assert_eq!(da, db);
        assert!((a - b).abs() <= 1e-12 * a.abs().max(1.0), "{a} vs {b}");
        return d1;
    };
    assert_eq!(da, db, "the dimension");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= 1e-12 * x.abs().max(1.0), "{a:?} vs {b:?}");
    }
    d1
}

#[test]
fn the_gradient_through_a_fused_chain() {
    let src = "def f(p: Vector[3] m) -> m^2:\n    q = (p + [1 m, 2 m, 3 m]) * 2 - p / 3\n    return q @ q\n";
    let p = vector(&[0.3, -1.0, 2.0], Dimension::LENGTH);
    let g = same_derivative(src, "f", 0, &[p]);
    // q = (5/3) p + 2 [1, 2, 3], f = q·q: ∇f = (10/3) q.
    let want: Vec<f64> = [0.3, -1.0, 2.0]
        .iter()
        .zip([1.0, 2.0, 3.0])
        .map(|(p, c)| 10.0 / 3.0 * (5.0 / 3.0 * p + 2.0 * c))
        .collect();
    let Value::Array { data, .. } = g else {
        panic!("not an array")
    };
    for (x, y) in data.iter().zip(&want) {
        assert!((x - y).abs() < 1e-9, "{data:?} vs {want:?}");
    }
}

#[test]
fn the_gradient_through_a_fused_kernel_with_a_broadcast_scalar() {
    // `s` is a scalar that depends on the parameter, used in every entry of
    // the fused kernel: its adjoint sums the kernel's partials.
    let src = "def f(p: Vector[3]) -> m:\n    s = p @ p\n    q = (p * s + [1, 2, 3]) * s - p\n    return 1 m * (q @ q)\n";
    let p = vector(&[0.3, -1.0, 2.0], Dimension::DIMENSIONLESS);
    same_derivative(src, "f", 0, &[p]);
}

#[test]
fn the_gradient_through_a_fused_kernel_and_a_conditional() {
    let src = "def f(p: Vector[3]) -> m:\n    q = (p + [1, 2, 3]) * 2 - p / 3\n    if p[0] > 0:\n        q = q * 3 + p\n    return 1 m * (q @ q)\n";
    for p in [[0.3, -1.0, 2.0], [-0.3, -1.0, 2.0]] {
        same_derivative(src, "f", 0, &[vector(&p, Dimension::DIMENSIONLESS)]);
    }
}

#[test]
fn the_fallback_gradient_through_a_fused_kernel() {
    let src = "def f(p: Vector[3]) -> m:\n    q = (p + [1, 2, 3]) * 2 - p / 3\n    acc = 0\n    i = 0\n    while i < 3:\n        acc = acc + q[i] * q[i]\n        i += 1\n    return 1 m * acc\n";
    let p = vector(&[0.3, -1.0, 2.0], Dimension::DIMENSIONLESS);
    same_derivative(src, "f", 0, &[p]);
}

#[test]
fn the_derivative_of_a_fused_vector_of_a_scalar() {
    // Forward mode: the tangent kernel takes the kernel's operands and
    // their tangents, a scalar operand (the broadcast `t`) included.
    let src = "def r(t: s) -> Vector[3] m:\n    v = [1, 2, 3] * 1 m/s\n    return (v * t + [1 m, 1 m, 1 m]) * 2 - v * t / 2\n";
    let d = same_derivative(src, "r", 0, &[Value::number(2.0, Dimension::TIME)]);
    let Value::Array { data, dim, .. } = d else {
        panic!("not an array")
    };
    assert_eq!(dim, Dimension::VELOCITY);
    assert_eq!(data, [1.5, 3.0, 4.5]);
}
