//! Differentiating the entry-by-entry array operations (`a * b`, `a / b`
//! and the math functions of arrays): the gradient and the derivative of an
//! array function of a number, on the reference interpreter, must match a
//! central finite difference; the same program before and after `fuse`
//! (a kernel whose body has the new operations) must differentiate to the
//! same numbers; and all engines must print the same gradient.
//!
//! Functions that would need a choice per entry (`min`, `max` of arrays)
//! must be refused with the crate's error, never give invalid SIR.

use sky_ir::{Module, OpKind};
use sky_ir_interp::{call, Value};
use sky_units::Dimension;

fn vector(xs: &[f64], dim: Dimension) -> Value {
    Value::array(vec![xs.len()], xs.to_vec(), dim)
}

fn dimless(xs: &[f64]) -> Value {
    vector(xs, Dimension::DIMENSIONLESS)
}

fn run(m: &Module, name: &str, args: &[Value]) -> Vec<Value> {
    sky_ast::with_deep_stack(|| call(m, name, args.to_vec(), &mut Vec::new()))
        .unwrap_or_else(|e| panic!("`{name}` failed: {}", e.message))
}

fn lowered(src: &str) -> Module {
    sky_lower::lower_source(src).unwrap_or_else(|e| panic!("{e}\n{src}"))
}

fn scalar(v: &Value) -> f64 {
    match v {
        Value::Number { value, .. } => *value,
        other => panic!("not a number: {other:?}"),
    }
}

fn entries(v: &Value) -> (Vec<f64>, Dimension) {
    match v {
        Value::Array { data, dim, .. } => (data.clone(), *dim),
        other => panic!("not an array: {other:?}"),
    }
}

/// The gradient of `f` at `p` by reverse mode, checked for well-typed SIR.
fn gradient(m: &Module, p: &Value) -> (f64, Vec<f64>, Dimension) {
    let mut m = m.clone();
    let name = sky_ad::differentiate_into(&mut m, "f", 0)
        .unwrap_or_else(|e| panic!("cannot differentiate: {e}\n{}", sky_ir::print(&m)));
    if let Err(errors) = sky_ir::verify(&m) {
        panic!("ill-typed derivative: {errors:?}\n{}", sky_ir::print(&m));
    }
    let out = run(&m, &name, std::slice::from_ref(p));
    let (g, dim) = entries(&out[1]);
    (scalar(&out[0]), g, dim)
}

/// The central difference of `f` in each entry of `p`.
fn numeric_gradient(m: &Module, p: &Value) -> Vec<f64> {
    let (xs, dim) = entries(p);
    (0..xs.len())
        .map(|i| {
            let h = 1e-6 * xs[i].abs().max(1.0);
            let at = |d: f64| {
                let mut q = xs.clone();
                q[i] += d;
                scalar(&run(m, "f", &[vector(&q, dim)])[0])
            };
            (at(h) - at(-h)) / (2.0 * h)
        })
        .collect()
}

fn close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g - w).abs() <= tol * w.abs().max(g.abs()).max(1.0),
            "{what}: {got:?} vs {want:?}"
        );
    }
}

/// `f`'s gradient at `p` against finite differences, on the plain and the
/// fused module, which must also agree with each other. Returns the
/// gradient and whether the fused module has a kernel.
fn check(src: &str, p: &Value, want_dim: Dimension) -> Vec<f64> {
    let plain = lowered(src);
    let mut fused = plain.clone();
    sky_opt::fuse(&mut fused);
    let (v0, g0, d0) = gradient(&plain, p);
    assert_eq!(d0, want_dim, "the gradient's dimension\n{src}");
    close(&g0, &numeric_gradient(&plain, p), 1e-6, src);
    let (v1, g1, d1) = gradient(&fused, p);
    assert_eq!(v0, v1, "the value\n{src}");
    assert_eq!(d0, d1);
    close(&g1, &g0, 1e-12, &format!("fused vs plain\n{src}"));
    g0
}

fn program(body: &str) -> String {
    format!("def f(p: Vector[3]) -> Scalar:\n    s = p[0] * p[2]\n{body}\n")
}

#[test]
fn products_and_quotients_of_arrays() {
    let p = dimless(&[0.3, -1.0, 2.0]);
    let g = check(
        &program("    q = p * p\n    return q @ [1, 2, 3]"),
        &p,
        Dimension::DIMENSIONLESS,
    );
    close(&g, &[0.6, -4.0, 12.0], 1e-12, "d(sum c p^2) = 2 c p");
    check(
        &program("    q = p * [2, 3, 4] / (p * p + [1, 1, 1])\n    return q @ q"),
        &p,
        Dimension::DIMENSIONLESS,
    );
    // The divisor is the differentiated array, the numerator too.
    check(
        &program("    q = (p + [3, 3, 3]) / (p * p + [1, 2, 3])\n    return q @ [1, 2, 3]"),
        &p,
        Dimension::DIMENSIONLESS,
    );
    // A scalar that depends on p, spread over an array on both sides.
    check(
        &program("    q = p * s / (p * p + [1, 1, 1]) + p * (p * s)\n    return q @ q"),
        &p,
        Dimension::DIMENSIONLESS,
    );
}

#[test]
fn math_functions_of_arrays() {
    let p = dimless(&[0.3, -1.0, 2.0]);
    for body in [
        "    return sqrt(p * p + [1, 1, 1]) @ [1, 2, 3]",
        "    return exp(p) @ sin(p)",
        "    return (cos(p) * tanh(p)) @ [1, 1, 1]",
        "    return (atan(p) + sinh(p) + cosh(p)) @ [1, 2, 3]",
        "    return log(p * p + [1, 1, 1]) @ [1, 2, 3]",
        "    return erf(p) @ [1, 2, 3]",
        "    return cbrt(p * p + [1, 1, 1]) @ [1, 2, 3]",
        "    return asinh(p) @ [1, 2, 3]",
        "    return (expm1(p) + log1p(p * p)) @ [1, 2, 3]",
        // Piecewise constant between the jumps: a zero slope.
        "    return floor(p * [0.5, 0.5, 0.5] + [0.25, 0.25, 0.25]) @ [1, 2, 3] + p @ p",
    ] {
        check(&program(body), &p, Dimension::DIMENSIONLESS);
    }
}

#[test]
fn two_argument_functions_of_arrays_and_numbers() {
    let p = dimless(&[0.3, -1.0, 2.0]);
    for body in [
        "    return atan2(p, [1, 2, 3]) @ [1, 2, 3]",
        "    return atan2([1, 2, 3], p) @ [1, 2, 3]",
        "    return atan2(p, p * p + [1, 1, 1]) @ [1, 2, 3]",
        "    return hypot(p, [3, 4, 5]) @ [1, 2, 3]",
        "    return hypot(p, p * p) @ [1, 2, 3]",
        // A number that depends on p, spread over the array.
        "    return hypot(p, s) @ [1, 2, 3]",
        "    return hypot(s, p) @ [1, 2, 3]",
        "    return atan2(p, s + 3) @ [1, 2, 3]",
        "    return atan2(s + 3, p) @ [1, 2, 3]",
        "    return hypot(p * s, p) @ hypot(p, s)",
    ] {
        check(&program(body), &p, Dimension::DIMENSIONLESS);
    }
}

#[test]
fn abs_of_an_array_takes_the_right_hand_slope_at_zero() {
    let p = dimless(&[0.3, -1.0, 2.0]);
    check(
        &program("    return abs(p - [0.5, 0.5, 0.5]) @ [1, 2, 3]"),
        &p,
        Dimension::DIMENSIONLESS,
    );
    // At a zero entry the slope is that of `sign(0) = 1`, as for a number.
    let z = dimless(&[0.0, -1.0, 2.0]);
    let plain = lowered(&program("    return abs(p) @ [1, 2, 3]"));
    let (_, g, _) = gradient(&plain, &z);
    assert_eq!(g, [1.0, -2.0, 3.0]);
    let mut fused = plain.clone();
    sky_opt::fuse(&mut fused);
    assert_eq!(gradient(&fused, &z).1, g);
}

#[test]
fn a_choice_per_entry_is_refused_not_miscompiled() {
    for body in [
        "    return min(p, [0, 0, 0]) @ [1, 2, 3]",
        "    return max(p, 0.5) @ [1, 2, 3]",
        "    return max(p, p * p) @ [1, 2, 3]",
    ] {
        let src = program(body);
        for fuse in [false, true] {
            let mut m = lowered(&src);
            if fuse {
                sky_opt::fuse(&mut m);
            }
            match sky_ad::differentiate_into(&mut m, "f", 0) {
                Err(sky_ad::AdError::Unsupported { op, .. }) => {
                    assert!(op.contains("min") || op.contains("max"), "{op}");
                }
                other => panic!("expected a clean refusal, got {other:?}\n{src}"),
            }
        }
    }
    // The primal alone is fine, and so is a derivative that does not go
    // through them.
    let m = lowered(&program(
        "    q = max(p, 0.5)\n    return q @ [1, 2, 3] + (p @ p)",
    ));
    let mut m2 = m.clone();
    let _ = sky_ad::differentiate_into(&mut m2, "f", 0);
    assert!(run(&m, "f", &[dimless(&[0.3, -1.0, 2.0])]).len() == 1);
}

#[test]
fn dimensions_flow_through_array_products_and_functions() {
    // The gradient of a force-like quantity with respect to a position.
    let p = vector(&[0.3, -1.0, 2.0], Dimension::LENGTH);
    let g = check(
        "def f(p: Vector[3] m) -> N:\n    r = sqrt(p * p + [1 m^2, 1 m^2, 1 m^2])\n    return [1 N/m, 2 N/m, 3 N/m] @ r\n",
        &p,
        Dimension::FORCE.checked_div(Dimension::LENGTH).unwrap(),
    );
    close(
        &g,
        &[
            1.0 * 0.3 / (0.3f64 * 0.3 + 1.0).sqrt(),
            2.0 * -1.0 / 2.0f64.sqrt(),
            3.0 * 2.0 / 5.0f64.sqrt(),
        ],
        1e-12,
        "c p / sqrt(p^2 + 1)",
    );
    // A quotient of lengths by times, and hypot and atan2 of lengths.
    check(
        "def f(p: Vector[3] m) -> m/s:\n    v = p / [1 s, 2 s, 4 s]\n    return [1, 2, 3] @ (v * v / [1 m/s, 2 m/s, 3 m/s] + hypot(v, [1 m/s, 1 m/s, 1 m/s]))\n",
        &p,
        Dimension::VELOCITY.checked_div(Dimension::LENGTH).unwrap(),
    );
    check(
        "def f(p: Vector[3] m) -> Scalar:\n    return atan2(p, [1 m, 2 m, 3 m]) @ [1, 2, 3]\n",
        &p,
        Dimension::LENGTH.inverse().unwrap(),
    );
    // A number with a unit spread over an array.
    check(
        "def f(p: Vector[3] m) -> m^2:\n    d = 2 m\n    q = p * d + hypot(p, d) * d\n    return q @ [1, 1, 1]\n",
        &p,
        Dimension::LENGTH.pow_int(2).unwrap()
            .checked_div(Dimension::LENGTH)
            .unwrap(),
    );
}

/// The derivative with respect to a number of an array function of it:
/// forward mode, whose tangent is an array.
fn forward(m: &Module, t: f64, dim: Dimension) -> (Vec<f64>, Vec<f64>, Dimension) {
    let mut m = m.clone();
    let name = sky_ad::differentiate_into(&mut m, "r", 0)
        .unwrap_or_else(|e| panic!("cannot differentiate: {e}"));
    if let Err(errors) = sky_ir::verify(&m) {
        panic!("ill-typed derivative: {errors:?}\n{}", sky_ir::print(&m));
    }
    let out = run(&m, &name, &[Value::number(t, dim)]);
    let (v, _) = entries(&out[0]);
    let (d, ddim) = entries(&out[1]);
    (v, d, ddim)
}

#[test]
fn the_derivative_of_an_array_function_of_a_number() {
    let srcs = [
        "def r(t: s) -> Vector[3] m:\n    v = [1 m/s, 2 m/s, 3 m/s]\n    x = v * t\n    return sqrt(x * x + [1 m^2, 1 m^2, 1 m^2])\n",
        "def r(t: s) -> Vector[3] m:\n    v = [1 m/s, 2 m/s, 3 m/s]\n    x = v * t\n    return x * x / (x + [1 m, 2 m, 3 m])\n",
        "def r(t: s) -> Vector[3] m:\n    v = [1 m/s, 2 m/s, 3 m/s]\n    x = v * t\n    return hypot(x, 2 m) + abs(x - [1 m, 5 m, 1 m])\n",
        "def r(t: s) -> Vector[3]:\n    x = [1 m/s, 2 m/s, 3 m/s] * t / 1 m\n    return exp(x) * sin(x) + atan2(x, t / 1 s)\n",
    ];
    for src in srcs {
        let plain = lowered(src);
        let mut fused = plain.clone();
        sky_opt::fuse(&mut fused);
        assert!(
            fused
                .function("r")
                .unwrap()
                .ops
                .iter()
                .any(|o| matches!(o.kind, OpKind::Fused)),
            "a kernel: {src}"
        );
        let t = 0.7;
        let (v0, d0, dim0) = forward(&plain, t, Dimension::TIME);
        let (v1, d1, dim1) = forward(&fused, t, Dimension::TIME);
        assert_eq!(v0, v1);
        assert_eq!(dim0, dim1);
        close(&d1, &d0, 1e-12, src);
        // Central difference of the primal.
        let h = 1e-6;
        let at = |t: f64| entries(&run(&plain, "r", &[Value::number(t, Dimension::TIME)])[0]).0;
        let (up, down) = (at(t + h), at(t - h));
        let fd: Vec<f64> = up
            .iter()
            .zip(&down)
            .map(|(a, b)| (a - b) / (2.0 * h))
            .collect();
        close(&d0, &fd, 1e-6, src);
    }
}

#[test]
fn the_gradient_after_fuse_uses_kernels_with_the_new_operations() {
    // `sqrt(x*x + y*y) * c + d` is one kernel of five operations, and its
    // gradient (an adjoint kernel per operand) equals the unfused one.
    let src = program(
        "    q = sqrt(p * p + [1, 2, 3] * [1, 2, 3]) * 0.5 + p / [2, 4, 8]\n    return q @ [1, 2, 3]",
    );
    let plain = lowered(&src);
    let mut fused = plain.clone();
    sky_opt::fuse(&mut fused);
    let kernels = fused
        .function("f")
        .unwrap()
        .ops
        .iter()
        .filter(|o| matches!(o.kind, OpKind::Fused))
        .count();
    assert!(kernels >= 1);
    let p = dimless(&[0.3, -1.0, 2.0]);
    check(&src, &p, Dimension::DIMENSIONLESS);
    // And the default pipeline (which fuses the gradient function itself).
    let mut m = plain.clone();
    let name = sky_ad::differentiate_into(&mut m, "f", 0).unwrap();
    let before = run(&m, &name, std::slice::from_ref(&p));
    sky_opt::default_pipeline()
        .verify_each(true)
        .run(&mut m)
        .unwrap_or_else(|e| panic!("the pipeline failed: {e:?}"));
    assert_eq!(run(&m, &name, std::slice::from_ref(&p)), before);
}
