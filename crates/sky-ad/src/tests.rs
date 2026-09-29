//! Hand-built SIR run through the reference interpreter, compared with the
//! analytic derivative and a central finite difference; typing of the
//! results; the error cases; and a proptest over random polynomials.

use std::sync::Mutex;

use proptest::prelude::*;
use sky_diag::Span;
use sky_ir::{
    CmpOp, Function, FunctionBuilder, Module, NumberType, OpKind, RegionId, Repr, Type, ValueId,
};
use sky_ir_interp::{call, Value};
use sky_units::{Dimension, MathFn, Rational};

use crate::{derivative, derivative_functions, differentiate_into, AdError};

fn sp() -> Span {
    Span::point(0)
}

fn num() -> Type {
    Type::scalar(Dimension::DIMENSIONLESS)
}

fn dimty(d: Dimension) -> Type {
    Type::scalar(d)
}

fn module(functions: Vec<Function>) -> Module {
    Module {
        functions,
        ..Module::default()
    }
}

/// A small wrapper so the test functions read like the math.
struct B {
    b: FunctionBuilder,
}

impl B {
    fn new(name: &str, params: Vec<Type>) -> B {
        let b = FunctionBuilder::new(name, params, vec![num()], sp());
        B { b }
    }

    fn param(&self, i: usize) -> ValueId {
        self.b.params()[i]
    }

    fn c(&mut self, x: f64) -> ValueId {
        self.b.op1(
            OpKind::ConstFloat { bits: x.to_bits() },
            vec![],
            num(),
            sp(),
        )
    }

    fn op(&mut self, k: OpKind, args: &[ValueId]) -> ValueId {
        self.b.op1(k, args.to_vec(), num(), sp())
    }

    fn add(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.op(OpKind::Add, &[a, b])
    }

    fn sub(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.op(OpKind::Sub, &[a, b])
    }

    fn mul(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.op(OpKind::Mul, &[a, b])
    }

    fn div(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.op(OpKind::Div, &[a, b])
    }

    fn powc(&mut self, a: ValueId, n: i64, d: i64) -> ValueId {
        self.op(
            OpKind::PowConst {
                exponent: Rational::new(n, d),
            },
            &[a],
        )
    }

    fn cmp(&mut self, c: CmpOp, a: ValueId, b: ValueId) -> ValueId {
        self.b.op1(OpKind::Cmp(c), vec![a, b], Type::Bool, sp())
    }

    fn ret(mut self, v: ValueId) -> Function {
        self.b.op(OpKind::Return, vec![v], vec![], sp());
        self.b.finish()
    }

    fn arm(&mut self, op: sky_ir::OpId, f: impl FnOnce(&mut B) -> ValueId) {
        let r = self.b.add_region(op);
        let blk = self.b.add_block(r, vec![]);
        self.b.set_insertion_block(blk);
        let v = f(self);
        self.b.op(OpKind::Yield, vec![v], vec![], sp());
    }

    /// `if cond { then } else { els }` where each arm is built by a closure
    /// that returns the yielded value.
    fn if_(
        &mut self,
        cond: ValueId,
        then: impl FnOnce(&mut B) -> ValueId,
        els: impl FnOnce(&mut B) -> ValueId,
    ) -> ValueId {
        let (op, res) = self.b.op(OpKind::If, vec![cond], vec![num()], sp());
        let outer = self.b.insertion_block();
        self.arm(op, then);
        self.arm(op, els);
        self.b.set_insertion_block(outer);
        res[0]
    }
}

/// Runs `name` on `args` and returns the numbers in its results.
fn run(m: &Module, name: &str, args: &[f64]) -> Vec<Value> {
    let args = args.iter().map(|x| Value::scalar(*x)).collect();
    sky_ast::with_deep_stack(|| call(m, name, args, &mut Vec::new())).expect("runs")
}

fn number(v: &Value) -> f64 {
    match v {
        Value::Number { value, .. } => *value,
        other => panic!("not a number: {other:?}"),
    }
}

fn dim_of_value(v: &Value) -> Dimension {
    match v {
        Value::Number { dim, .. } => *dim,
        other => panic!("not a number: {other:?}"),
    }
}

/// The worst finite-difference error seen (relative), for the report.
static WORST_FD: Mutex<f64> = Mutex::new(0.0);

/// Differentiates `func` wrt `wrt`, runs it at `args`, and checks that the
/// value equals the original and the derivative equals `exact` and a
/// central finite difference.
fn check(m: &Module, func: &str, wrt: usize, args: &[f64], exact: f64) {
    check_with(m, func, wrt, args, exact, true);
}

/// [`check`], optionally without the finite difference (for a parameter
/// the function is only piecewise constant in, like a trip count).
fn check_with(m: &Module, func: &str, wrt: usize, args: &[f64], exact: f64, fd: bool) {
    let mut m = m.clone();
    let name = differentiate_into(&mut m, func, wrt).expect("differentiates");
    if let Err(errors) = sky_ir::verify(&m) {
        panic!("the derivative is ill-typed SIR: {errors:?}");
    }
    let out = run(&m, &name, args);
    assert_eq!(out.len(), 2);
    let original = run(&m, func, args);
    assert_eq!(out[0], original[0], "the primal value is unchanged");
    let d = number(&out[1]);
    let scale = exact.abs().max(1.0);
    assert!(
        (d - exact).abs() <= 1e-12 * scale,
        "{func}{args:?}: AD {d} vs exact {exact}"
    );
    if !fd {
        return;
    }
    // Central finite difference.
    let h = 1e-5 * args[wrt].abs().max(1.0);
    let at = |dx: f64| {
        let mut a = args.to_vec();
        a[wrt] += dx;
        number(&run(&m, func, &a)[0])
    };
    let fd = (at(h) - at(-h)) / (2.0 * h);
    let err = (d - fd).abs() / scale;
    {
        let mut w = WORST_FD.lock().unwrap();
        if err > *w {
            *w = err;
            eprintln!("worst finite-difference error so far: {err:e} ({func} at {args:?})");
        }
    }
    assert!(
        err < 1e-6,
        "{func}{args:?}: AD {d} vs finite difference {fd}"
    );
}

// ----- polynomial and arithmetic -----

fn cube() -> Function {
    let mut b = B::new("cube", vec![num()]);
    let x = b.param(0);
    let r = b.powc(x, 3, 1);
    b.ret(r)
}

#[test]
fn cube_derivative() {
    let m = module(vec![cube()]);
    for x in [-2.0, 0.0, 0.5, 3.0] {
        check(&m, "cube", 0, &[x], 3.0 * x * x);
    }
}

#[test]
fn product_of_two_parameters() {
    let mut b = B::new("xy", vec![num(), num()]);
    let (x, y) = (b.param(0), b.param(1));
    let r = b.mul(x, y);
    let m = module(vec![b.ret(r)]);
    check(&m, "xy", 0, &[3.0, 5.0], 5.0);
    check(&m, "xy", 1, &[3.0, 5.0], 3.0);
}

#[test]
fn x_times_x_and_sub_neg() {
    // f = x*x - (-x) + 7  =>  2x + 1
    let mut b = B::new("f", vec![num()]);
    let x = b.param(0);
    let sq = b.mul(x, x);
    let nx = b.op(OpKind::Neg, &[x]);
    let d = b.sub(sq, nx);
    let seven = b.c(7.0);
    let r = b.add(d, seven);
    let m = module(vec![b.ret(r)]);
    check(&m, "f", 0, &[1.5], 4.0);
}

#[test]
fn quotient_and_inverse() {
    // f = (x + 1) / (x - 2)  =>  -3 / (x-2)^2
    let mut b = B::new("q", vec![num()]);
    let x = b.param(0);
    let one = b.c(1.0);
    let two = b.c(2.0);
    let n = b.add(x, one);
    let d = b.sub(x, two);
    let r = b.div(n, d);
    let m = module(vec![b.ret(r)]);
    check(&m, "q", 0, &[5.0], -3.0 / 9.0);
    check(&m, "q", 0, &[-1.0], -3.0 / 9.0);

    // g = 1/x + 1/(x*x)  =>  -1/x^2 - 2/x^3
    let mut b = B::new("g", vec![num()]);
    let x = b.param(0);
    let i = b.op(OpKind::Inverse, &[x]);
    let sq = b.mul(x, x);
    let j = b.op(OpKind::Inverse, &[sq]);
    let r = b.add(i, j);
    let m = module(vec![b.ret(r)]);
    check(&m, "g", 0, &[2.0], -0.25 - 0.25);

    // h = c / x with constant numerator (only tb)
    let mut b = B::new("h", vec![num()]);
    let x = b.param(0);
    let c = b.c(6.0);
    let r = b.div(c, x);
    let m = module(vec![b.ret(r)]);
    check(&m, "h", 0, &[2.0], -1.5);
}

#[test]
fn rational_powers() {
    for (n, d) in [(2, 3), (3, 2), (1, 2), (-1, 2), (-3, 1), (1, 1), (0, 1)] {
        let mut b = B::new("p", vec![num()]);
        let x = b.param(0);
        let r = b.powc(x, n, d);
        let m = module(vec![b.ret(r)]);
        let e = n as f64 / d as f64;
        for x in [0.5, 2.0, 7.25] {
            check(&m, "p", 0, &[x], e * x.powf(e - 1.0));
        }
    }
}

#[test]
fn general_pow_with_constant_exponent_value() {
    // f = x ^ y with y a parameter (not differentiated): y * x^(y-1).
    let mut b = B::new("f", vec![num(), num()]);
    let (x, y) = (b.param(0), b.param(1));
    let r = b.op(OpKind::Pow, &[x, y]);
    let m = module(vec![b.ret(r)]);
    check(&m, "f", 0, &[2.0, 3.5], 3.5 * 2f64.powf(2.5));
    // wrt the exponent: r * ln(a).
    check(&m, "f", 1, &[2.0, 3.5], 2f64.powf(3.5) * 2f64.ln());
}

#[test]
fn general_pow_with_both_depending_on_x() {
    // f = x ^ x: x^x * (ln x + 1).
    let mut b = B::new("f", vec![num()]);
    let x = b.param(0);
    let r = b.op(OpKind::Pow, &[x, x]);
    let m = module(vec![b.ret(r)]);
    let at = 1.7f64;
    check(&m, "f", 0, &[at], at.powf(at) * (at.ln() + 1.0));
}

// ----- math functions -----

/// `f(x) = op(x)` on a dimensionless parameter.
fn math1(f: MathFn) -> Module {
    let mut b = B::new("f", vec![num()]);
    let x = b.param(0);
    let r = b.op(OpKind::Math(f), &[x]);
    module(vec![b.ret(r)])
}

/// `f(a, b) = op(a, b)` on two dimensionless parameters.
fn math2(f: MathFn) -> Module {
    let mut b = B::new("f", vec![num(), num()]);
    let (x, y) = (b.param(0), b.param(1));
    let r = b.op(OpKind::Math(f), &[x, y]);
    module(vec![b.ret(r)])
}

#[test]
fn unary_math_matches_the_exact_derivative_and_a_finite_difference() {
    type Case = (MathFn, f64, fn(f64) -> f64);
    let cases: [Case; 31] = [
        (MathFn::Sqrt, 2.5, |x| 0.5 / x.sqrt()),
        (MathFn::Cbrt, 3.0, |x| 1.0 / (3.0 * x.cbrt().powi(2))),
        (MathFn::Abs, -1.5, |_| -1.0),
        (MathFn::Abs, 2.0, |_| 1.0),
        (MathFn::Exp, 0.7, f64::exp),
        (MathFn::Log, 2.2, |x| 1.0 / x),
        (MathFn::Log10, 2.2, |x| 1.0 / (x * std::f64::consts::LN_10)),
        (MathFn::Log2, 2.2, |x| 1.0 / (x * std::f64::consts::LN_2)),
        (MathFn::Sin, 0.9, f64::cos),
        (MathFn::Cos, 0.9, |x| -x.sin()),
        (MathFn::Tan, 0.4, |x| 1.0 / (x.cos() * x.cos())),
        (MathFn::Asin, 0.3, |x| 1.0 / (1.0 - x * x).sqrt()),
        (MathFn::Acos, 0.3, |x| -1.0 / (1.0 - x * x).sqrt()),
        (MathFn::Atan, 1.3, |x| 1.0 / (1.0 + x * x)),
        (MathFn::Floor, 2.5, |_| 0.0),
        (MathFn::Round, 2.4, |_| 0.0),
        (MathFn::Expm1, 0.7, f64::exp),
        (MathFn::Expm1, -30.0, f64::exp),
        (MathFn::Log1p, 2.2, |x| 1.0 / (1.0 + x)),
        (MathFn::Log1p, -0.6, |x| 1.0 / (1.0 + x)),
        (MathFn::Sinh, 0.9, f64::cosh),
        (MathFn::Cosh, 0.9, f64::sinh),
        (MathFn::Cosh, -1.4, f64::sinh),
        (MathFn::Tanh, 0.4, |x| 1.0 / (x.cosh() * x.cosh())),
        (MathFn::Tanh, 3.0, |x| 1.0 / (x.cosh() * x.cosh())),
        (MathFn::Asinh, 0.9, |x| 1.0 / (x * x + 1.0).sqrt()),
        (MathFn::Asinh, -4.0, |x| 1.0 / (x * x + 1.0).sqrt()),
        (MathFn::Acosh, 2.5, |x| 1.0 / (x * x - 1.0).sqrt()),
        (MathFn::Atanh, 0.3, |x| 1.0 / (1.0 - x * x)),
        (MathFn::Atanh, -0.8, |x| 1.0 / (1.0 - x * x)),
        (MathFn::Sign, -2.0, |_| 0.0),
    ];
    for (f, x, exact) in cases {
        check(&math1(f), "f", 0, &[x], exact(x));
    }
    check(&math1(MathFn::Ceil), "f", 0, &[2.5], 0.0);
}

#[test]
fn erf_derivative_is_the_gaussian() {
    // erf'(x) = 2/sqrt(pi) exp(-x^2), which the AD builds from exp, not
    // from a second erf, so the finite difference also checks erf itself.
    for x in [0.0_f64, 0.3, -0.8, 1.7, -2.4] {
        let exact = std::f64::consts::FRAC_2_SQRT_PI * (-x * x).exp();
        check(&math1(MathFn::Erf), "f", 0, &[x], exact);
    }
}

#[test]
fn asinh_derivative_survives_a_huge_argument() {
    // 1/sqrt(x² + 1) would overflow x²; hypot(1, x) does not.
    let x = 1e200;
    check_with(&math1(MathFn::Asinh), "f", 0, &[x], 1.0 / x, false);
}

#[test]
fn chain_rule_through_math() {
    // f(x) = sin(x^2) * exp(-x): (2x cos(x^2) - sin(x^2)) e^-x
    let mut b = B::new("f", vec![num()]);
    let x = b.param(0);
    let x2 = b.powc(x, 2, 1);
    let s = b.op(OpKind::Math(MathFn::Sin), &[x2]);
    let nx = b.op(OpKind::Neg, &[x]);
    let e = b.op(OpKind::Math(MathFn::Exp), &[nx]);
    let r = b.mul(s, e);
    let m = module(vec![b.ret(r)]);
    let at = 0.8f64;
    let exact = (2.0 * at * (at * at).cos() - (at * at).sin()) * (-at).exp();
    check(&m, "f", 0, &[at], exact);
}

#[test]
fn binary_math_matches_the_exact_derivative_and_a_finite_difference() {
    let (y, x) = (1.2f64, 0.7f64);
    let d = x * x + y * y;
    check(&math2(MathFn::Atan2), "f", 0, &[y, x], x / d);
    check(&math2(MathFn::Atan2), "f", 1, &[y, x], -y / d);
    check(&math2(MathFn::Hypot), "f", 0, &[3.0, 4.0], 3.0 / 5.0);
    check(&math2(MathFn::Hypot), "f", 1, &[3.0, 4.0], 4.0 / 5.0);
    // min and max: the operand that is chosen.
    check(&math2(MathFn::Min), "f", 0, &[1.0, 2.0], 1.0);
    check(&math2(MathFn::Min), "f", 1, &[1.0, 2.0], 0.0);
    check(&math2(MathFn::Min), "f", 0, &[3.0, 2.0], 0.0);
    check(&math2(MathFn::Min), "f", 1, &[3.0, 2.0], 1.0);
    check(&math2(MathFn::Max), "f", 0, &[1.0, 2.0], 0.0);
    check(&math2(MathFn::Max), "f", 1, &[1.0, 2.0], 1.0);
    check(&math2(MathFn::Max), "f", 0, &[3.0, 2.0], 1.0);
    check(&math2(MathFn::Max), "f", 1, &[3.0, 2.0], 0.0);
}

#[test]
fn binary_math_with_both_operands_depending_on_x() {
    // atan2(x, 2x + 1) and hypot(x, 2x) = sqrt(5) |x|.
    let mut b = B::new("f", vec![num()]);
    let x = b.param(0);
    let two = b.c(2.0);
    let one = b.c(1.0);
    let t = b.mul(two, x);
    let u = b.add(t, one);
    let r = b.op(OpKind::Math(MathFn::Atan2), &[x, u]);
    let m = module(vec![b.ret(r)]);
    let at = 0.9f64;
    let (dy, dx) = (1.0, 2.0);
    let (yy, xx) = (at, 2.0 * at + 1.0);
    check(&m, "f", 0, &[at], (xx * dy - yy * dx) / (xx * xx + yy * yy));

    let mut b = B::new("g", vec![num()]);
    let x = b.param(0);
    let two = b.c(2.0);
    let t = b.mul(two, x);
    let r = b.op(OpKind::Math(MathFn::Hypot), &[x, t]);
    let m = module(vec![b.ret(r)]);
    check(&m, "g", 0, &[-1.5], -(5f64).sqrt());
}

#[test]
fn math_tangents_have_the_right_dimension() {
    let len = dimty(Dimension::LENGTH);
    let area = dimty(Dimension::LENGTH.pow_int(2).unwrap());
    let per_len = dimty(Dimension::LENGTH.inverse().unwrap());
    let dimless = num();
    // (function, parameter types, result type, tangent wrt parameter 0)
    let cases: Vec<(MathFn, Vec<Type>, Type, Type)> = vec![
        // sqrt(area) is a length; d/d(area) is 1/length.
        (
            MathFn::Sqrt,
            vec![area.clone()],
            len.clone(),
            per_len.clone(),
        ),
        // abs(length) is a length; its slope is dimensionless.
        (MathFn::Abs, vec![len.clone()], len.clone(), dimless.clone()),
        // hypot(a, b): both lengths, a length; slope dimensionless.
        (
            MathFn::Hypot,
            vec![len.clone(), len.clone()],
            len.clone(),
            dimless.clone(),
        ),
        // min(a, b): slope dimensionless.
        (
            MathFn::Min,
            vec![len.clone(), len.clone()],
            len.clone(),
            dimless.clone(),
        ),
        // sign(length) is a pure number, and constant.
        (
            MathFn::Sign,
            vec![len.clone()],
            dimless.clone(),
            per_len.clone(),
        ),
        // atan2(y, x): dimensionless result, slope wrt a length is 1/length.
        (
            MathFn::Atan2,
            vec![len.clone(), len.clone()],
            dimless.clone(),
            per_len.clone(),
        ),
    ];
    for (f, params, result, tangent) in cases {
        let mut b = FunctionBuilder::new("f", params.clone(), vec![result.clone()], sp());
        let args: Vec<ValueId> = b.params().to_vec();
        let r = b.op1(OpKind::Math(f), args, result, sp());
        b.op(OpKind::Return, vec![r], vec![], sp());
        let m = module(vec![b.finish()]);
        let d = derivative(&m, "f", 0).expect("differentiates");
        let got = number_type_of(&d.results[1]).dim;
        let want = number_type_of(&tangent).dim;
        assert_eq!(got, want, "{f:?}");
        let mut full = m.clone();
        full.functions.push(d);
        // Built by hand: effects are derived from the bodies, as lowering and
        // `differentiate_into` do.
        let _ = sky_ir::effects::infer_effects(&mut full);
        if let Err(errors) = sky_ir::verify(&full) {
            panic!("{f:?}: ill-typed derivative: {errors:?}");
        }
    }
}

#[test]
fn math_domain_edges_fail_like_the_primal() {
    // d sqrt(x) at 0 divides by zero; asin at 1 too.
    for (f, x) in [
        (MathFn::Sqrt, 0.0),
        (MathFn::Asin, 1.0),
        (MathFn::Log, 0.0),
        (MathFn::Acosh, 1.0),
        (MathFn::Atanh, 1.0),
        (MathFn::Log1p, -1.0),
    ] {
        let mut m = math1(f);
        let name = differentiate_into(&mut m, "f", 0).unwrap();
        let args = vec![Value::scalar(x)];
        let e = sky_ast::with_deep_stack(|| call(&m, &name, args, &mut Vec::new()));
        assert!(e.is_err(), "{f:?} at {x}");
    }
}

// ----- control flow -----

#[test]
fn if_with_different_branches() {
    // if x < 1 { x*x } else { 3*x + 1 }
    let mut b = B::new("f", vec![num()]);
    let x = b.param(0);
    let one = b.c(1.0);
    let c = b.cmp(CmpOp::Lt, x, one);
    let r = b.if_(
        c,
        |b| {
            let x = b.param(0);
            b.mul(x, x)
        },
        |b| {
            let x = b.param(0);
            let three = b.c(3.0);
            let t = b.mul(three, x);
            let one = b.c(1.0);
            b.add(t, one)
        },
    );
    let m = module(vec![b.ret(r)]);
    check(&m, "f", 0, &[0.5], 1.0);
    check(&m, "f", 0, &[2.0], 3.0);
}

#[test]
fn if_arm_independent_of_x_has_zero_tangent() {
    // if x < 1 { 5 } else { x }  =>  0 or 1
    let mut b = B::new("f", vec![num()]);
    let x = b.param(0);
    let one = b.c(1.0);
    let c = b.cmp(CmpOp::Lt, x, one);
    let r = b.if_(c, |b| b.c(5.0), |b| b.param(0));
    let m = module(vec![b.ret(r)]);
    check(&m, "f", 0, &[0.5], 0.0);
    check(&m, "f", 0, &[2.0], 1.0);
}

/// `x^n` by repeated multiplication: `acc = 1; i = 0; while i < n { acc =
/// acc * x; i = i + 1 }`. Parameters `(x, n)`.
fn power_loop() -> Function {
    let mut b = B::new("pow_loop", vec![num(), num()]);
    let (x, n) = (b.param(0), b.param(1));
    let one = b.c(1.0);
    let zero = b.c(0.0);
    let (w, res) =
        b.b.op(OpKind::While, vec![one, zero], vec![num(), num()], sp());
    let outer = b.b.insertion_block();
    let before: RegionId = b.b.add_region(w);
    let bb = b.b.add_block(before, vec![num(), num()]);
    b.b.set_insertion_block(bb);
    let args = b.b.func().block(bb).args.clone();
    let c = b.cmp(CmpOp::Lt, args[1], n);
    b.b.op(OpKind::Condition, vec![c, args[0], args[1]], vec![], sp());
    let after = b.b.add_region(w);
    let ab = b.b.add_block(after, vec![num(), num()]);
    b.b.set_insertion_block(ab);
    let args = b.b.func().block(ab).args.clone();
    let acc = b.mul(args[0], x);
    let one = b.c(1.0);
    let i = b.add(args[1], one);
    b.b.op(OpKind::Yield, vec![acc, i], vec![], sp());
    b.b.set_insertion_block(outer);
    b.ret(res[0])
}

#[test]
fn while_loop_power() {
    let m = module(vec![power_loop()]);
    for (x, n) in [
        (2.0f64, 5.0f64),
        (1.5, 3.0),
        (-2.0, 4.0),
        (3.0, 0.0),
        (0.7, 1.0),
    ] {
        let exact = if n == 0.0 { 0.0 } else { n * x.powf(n - 1.0) };
        check(&m, "pow_loop", 0, &[x, n], exact);
    }
    // The counter and the bound do not depend on x: no tangent slot for
    // the counter, exactly one for the accumulator.
    let d = derivative(&m, "pow_loop", 0).unwrap();
    let w = d
        .ops
        .iter()
        .find(|o| o.kind == OpKind::While)
        .expect("a while");
    assert_eq!(w.operands.len(), 3);
    // Derivative with respect to the trip count parameter: zero.
    check_with(&m, "pow_loop", 1, &[2.0, 5.0], 0.0, false);
}

// ----- calls and recursion -----

/// `pow_rec(x, n) = if n <= 0 { 1 } else { x * pow_rec(x, n - 1) }`.
fn pow_rec() -> Function {
    let mut b = B::new("pow_rec", vec![num(), num()]);
    let (x, n) = (b.param(0), b.param(1));
    let zero = b.c(0.0);
    let c = b.cmp(CmpOp::Le, n, zero);
    let r = b.if_(
        c,
        |b| b.c(1.0),
        |b| {
            let (x, n) = (b.param(0), b.param(1));
            let one = b.c(1.0);
            let n1 = b.sub(n, one);
            let sub = b.op(
                OpKind::Call {
                    callee: "pow_rec".into(),
                },
                &[x, n1],
            );
            b.mul(x, sub)
        },
    );
    let _ = (x, n);
    b.ret(r)
}

#[test]
fn recursion_calls_its_own_derivative() {
    let m = module(vec![pow_rec()]);
    for (x, n) in [(2.0f64, 5.0f64), (1.5, 1.0), (3.0, 0.0), (-1.5, 4.0)] {
        let exact = if n == 0.0 { 0.0 } else { n * x.powf(n - 1.0) };
        check(&m, "pow_rec", 0, &[x, n], exact);
    }
    let all = derivative_functions(&m, "pow_rec", 0).unwrap();
    assert_eq!(all.len(), 1, "one derivative function, however deep");
    let calls_self = all[0]
        .ops
        .iter()
        .any(|o| matches!(&o.kind, OpKind::Call { callee } if callee == "dpow_rec/d0"));
    assert!(calls_self);
}

#[test]
fn calls_to_other_functions() {
    // h(x) = cube(x) + mul2(x, x)  =>  3x^2 + 2x
    let mut mul2 = B::new("mul2", vec![num(), num()]);
    let (a, b2) = (mul2.param(0), mul2.param(1));
    let r = mul2.mul(a, b2);
    let mul2 = mul2.ret(r);

    let mut h = B::new("h", vec![num()]);
    let x = h.param(0);
    let c = h.op(
        OpKind::Call {
            callee: "cube".into(),
        },
        &[x],
    );
    let d = h.op(
        OpKind::Call {
            callee: "mul2".into(),
        },
        &[x, x],
    );
    let r = h.add(c, d);
    let m = module(vec![cube(), mul2, h.ret(r)]);
    check(&m, "h", 0, &[1.5], 3.0 * 2.25 + 3.0);
    let all = derivative_functions(&m, "h", 0).unwrap();
    // dh/d0, dcube/d0, dmul2/d0, dmul2/d1
    assert_eq!(all.len(), 4);

    // A call whose arguments do not depend on x is copied as is.
    let mut k = B::new("k", vec![num(), num()]);
    let (x, y) = (k.param(0), k.param(1));
    let c = k.op(
        OpKind::Call {
            callee: "cube".into(),
        },
        &[y],
    );
    let r = k.mul(x, c);
    let m = module(vec![cube(), k.ret(r)]);
    check(&m, "k", 0, &[2.0, 3.0], 27.0);
    assert_eq!(derivative_functions(&m, "k", 0).unwrap().len(), 1);
}

#[test]
fn differentiate_into_is_idempotent() {
    let mut m = module(vec![cube()]);
    let a = differentiate_into(&mut m, "cube", 0).unwrap();
    let n = m.functions.len();
    let b = differentiate_into(&mut m, "cube", 0).unwrap();
    assert_eq!(a, b);
    assert_eq!(m.functions.len(), n);
}

// ----- units -----

/// `position(t) = x0 + v*t + a*t^2/2` with `t` a time, `v` a velocity and
/// `a` an acceleration constant.
fn position() -> Function {
    let len = dimty(Dimension::LENGTH);
    let time = dimty(Dimension::TIME);
    let vel = dimty(Dimension::VELOCITY);
    let acc = dimty(Dimension::from_ints([1, 0, -2, 0, 0, 0, 0]));
    let time_sq = dimty(Dimension::TIME.pow_int(2).unwrap());
    let mut b = FunctionBuilder::new("position", vec![time.clone()], vec![len.clone()], sp());
    let t = b.params()[0];
    b.set_name(t, "t");
    let x0 = b.op1(
        OpKind::ConstFloat {
            bits: 10.0f64.to_bits(),
        },
        vec![],
        len.clone(),
        sp(),
    );
    let v = b.op1(
        OpKind::ConstFloat {
            bits: 3.0f64.to_bits(),
        },
        vec![],
        vel,
        sp(),
    );
    let a = b.op1(
        OpKind::ConstFloat {
            bits: 4.0f64.to_bits(),
        },
        vec![],
        acc,
        sp(),
    );
    let vt = b.op1(OpKind::Mul, vec![v, t], len.clone(), sp());
    let t2 = b.op1(
        OpKind::PowConst {
            exponent: Rational::integer(2),
        },
        vec![t],
        time_sq,
        sp(),
    );
    let at2 = b.op1(OpKind::Mul, vec![a, t2], len.clone(), sp());
    let half = b.op1(
        OpKind::ConstRational(Rational::new(1, 2)),
        vec![],
        num(),
        sp(),
    );
    let half_at2 = b.op1(OpKind::Mul, vec![half, at2], len.clone(), sp());
    let s = b.op1(OpKind::Add, vec![x0, vt], len.clone(), sp());
    let r = b.op1(OpKind::Add, vec![s, half_at2], len, sp());
    b.op(OpKind::Return, vec![r], vec![], sp());
    b.finish()
}

fn number_type_of(t: &Type) -> &NumberType {
    t.as_number().expect("a number type")
}

#[test]
fn derivative_of_position_wrt_time_is_a_velocity() {
    let m = module(vec![position()]);
    let d = derivative(&m, "position", 0).unwrap();
    assert_eq!(d.name, "dposition/dt");
    assert_eq!(d.params, vec![dimty(Dimension::TIME)]);
    assert_eq!(d.results.len(), 2);
    assert_eq!(number_type_of(&d.results[0]).dim, Dimension::LENGTH);
    assert_eq!(number_type_of(&d.results[1]).dim, Dimension::VELOCITY);
    assert_eq!(number_type_of(&d.results[1]).repr, Repr::Unchosen);

    // The runtime dimensions (computed by the interpreter's unit
    // arithmetic from the operands' types) agree with the declared ones:
    // s(2) = 10 + 6 + 8 = 24 m, v(2) = 3 + 4*2 = 11 m/s.
    let mut full = m.clone();
    full.functions.push(d);
    let args = vec![Value::number(2.0, Dimension::TIME)];
    let out = sky_ast::with_deep_stack(|| call(&full, "dposition/dt", args, &mut Vec::new()))
        .expect("runs");
    assert_eq!(number(&out[0]), 24.0);
    assert_eq!(dim_of_value(&out[0]), Dimension::LENGTH);
    assert_eq!(number(&out[1]), 11.0);
    assert_eq!(dim_of_value(&out[1]), Dimension::VELOCITY);
}

#[test]
fn every_tangent_is_typed_dim_over_dim_x() {
    // f(t) = c / t  with c a length: df/dt has dimension length / time^2.
    let len = dimty(Dimension::LENGTH);
    let time = dimty(Dimension::TIME);
    let inv_time = dimty(Dimension::TIME.inverse().unwrap());
    let mut b = FunctionBuilder::new(
        "f",
        vec![time.clone()],
        vec![dimty(Dimension::VELOCITY)],
        sp(),
    );
    let t = b.params()[0];
    let c = b.op1(
        OpKind::ConstFloat {
            bits: 6.0f64.to_bits(),
        },
        vec![],
        len,
        sp(),
    );
    let it = b.op1(OpKind::Inverse, vec![t], inv_time, sp());
    let r = b.op1(OpKind::Mul, vec![c, it], dimty(Dimension::VELOCITY), sp());
    b.op(OpKind::Return, vec![r], vec![], sp());
    let m = module(vec![b.finish()]);
    let d = derivative(&m, "f", 0).unwrap();
    let accel = Dimension::from_ints([1, 0, -2, 0, 0, 0, 0]);
    assert_eq!(number_type_of(&d.results[1]).dim, accel);
    let mut full = m.clone();
    full.functions.push(d);
    let out = sky_ast::with_deep_stack(|| {
        call(
            &full,
            "df/d0",
            vec![Value::number(2.0, Dimension::TIME)],
            &mut Vec::new(),
        )
    })
    .expect("runs");
    // f = 6/t; f' = -6/t^2 = -1.5 m/s^2
    assert_eq!(number(&out[1]), -1.5);
    assert_eq!(dim_of_value(&out[1]), accel);
}

// ----- several results, and derivatives of derivatives -----

#[test]
fn a_function_with_several_results_returns_values_then_tangents() {
    // f(x) = (x^2, 3x): derivative results (x^2, 3x, 2x, 3).
    let mut b = FunctionBuilder::new("f", vec![num()], vec![num(), num()], sp());
    let x = b.params()[0];
    let sq = b.op1(
        OpKind::PowConst {
            exponent: Rational::integer(2),
        },
        vec![x],
        num(),
        sp(),
    );
    let three = b.op1(
        OpKind::ConstFloat {
            bits: 3.0f64.to_bits(),
        },
        vec![],
        num(),
        sp(),
    );
    let tx = b.op1(OpKind::Mul, vec![three, x], num(), sp());
    b.op(OpKind::Return, vec![sq, tx], vec![], sp());
    let mut m = module(vec![b.finish()]);
    let name = differentiate_into(&mut m, "f", 0).unwrap();
    let out = run(&m, &name, &[5.0]);
    let got: Vec<f64> = out.iter().map(number).collect();
    assert_eq!(got, [25.0, 15.0, 10.0, 3.0]);
}

#[test]
fn a_call_of_a_derivative_function_differentiates_again() {
    // g(x) = x^3; h(x) = the second result of dg/dx(x) = 3x^2, so
    // dh/dx = 6x. h calls a function with two results.
    let mut m = module(vec![cube()]);
    let d = differentiate_into(&mut m, "cube", 0).unwrap();
    let mut b = B::new("h", vec![num()]);
    let x = b.param(0);
    let (_, res) = b.b.op(
        OpKind::Call { callee: d },
        vec![x],
        vec![num(), num()],
        sp(),
    );
    let f = b.ret(res[1]);
    m.functions.push(f);
    check(&m, "h", 0, &[2.0], 12.0);
    // the second derivative also runs on the first: h itself is 3x^2
    assert_eq!(number(&run(&m, "h", &[2.0])[0]), 12.0);
}

// ----- errors -----

#[test]
fn wrt_out_of_range() {
    let m = module(vec![cube()]);
    let e = derivative(&m, "cube", 1).unwrap_err();
    assert_eq!(
        e,
        AdError::ParamOutOfRange {
            func: "cube".into(),
            wrt: 1,
            params: 1
        }
    );
    assert!(e.to_string().contains("parameter 1"));
    assert_eq!(
        derivative(&m, "nope", 0).unwrap_err(),
        AdError::NoSuchFunction("nope".into())
    );
}

#[test]
fn non_number_parameter() {
    let mut b = FunctionBuilder::new("f", vec![Type::Bool, num()], vec![num()], sp());
    let x = b.params()[1];
    b.op(OpKind::Return, vec![x], vec![], sp());
    let m = module(vec![b.finish()]);
    let e = derivative(&m, "f", 0).unwrap_err();
    assert_eq!(
        e,
        AdError::NonNumberParam {
            func: "f".into(),
            wrt: 0
        }
    );
    // The number parameter is fine, and the Bool is passed through.
    assert!(derivative(&m, "f", 1).is_ok());
}

#[test]
fn unknown_types_are_rejected_when_the_derivative_flows_through_them() {
    // wrt an Unknown parameter.
    let mut b = FunctionBuilder::new("f", vec![Type::Unknown], vec![num()], sp());
    let x = b.params()[0];
    b.op(OpKind::Return, vec![x], vec![], sp());
    let m = module(vec![b.finish()]);
    assert!(matches!(
        derivative(&m, "f", 0),
        Err(AdError::UnknownType { .. })
    ));

    // An Unknown intermediate on the derivative's path.
    let mut b = FunctionBuilder::new("g", vec![num()], vec![num()], sp());
    let x = b.params()[0];
    let u = b.op1(OpKind::Neg, vec![x], Type::Unknown, sp());
    let r = b.op1(OpKind::Neg, vec![u], num(), sp());
    b.op(OpKind::Return, vec![r], vec![], sp());
    let m = module(vec![b.finish()]);
    let e = derivative(&m, "g", 0).unwrap_err();
    assert!(matches!(e, AdError::UnknownType { .. }), "{e}");
    assert!(e.to_string().contains("unknown type"));

    // An Unknown value off the derivative's path is fine.
    let mut b = FunctionBuilder::new("h", vec![num(), Type::Unknown], vec![num()], sp());
    let x = b.params()[0];
    let y = b.params()[1];
    let _ = b.op1(OpKind::Neg, vec![y], Type::Unknown, sp());
    b.op(OpKind::Return, vec![x], vec![], sp());
    let m = module(vec![b.finish()]);
    assert!(derivative(&m, "h", 0).is_ok());
}

#[test]
fn bad_results_and_arrays() {
    let mut b = FunctionBuilder::new("none", vec![num()], vec![], sp());
    b.op(OpKind::Return, vec![], vec![], sp());
    let m = module(vec![b.finish()]);
    assert!(matches!(
        derivative(&m, "none", 0),
        Err(AdError::BadResult { .. })
    ));

    let mut b = FunctionBuilder::new("flag", vec![num()], vec![Type::Bool], sp());
    let x = b.params()[0];
    let c = b.op1(OpKind::Cmp(CmpOp::Lt), vec![x, x], Type::Bool, sp());
    b.op(OpKind::Return, vec![c], vec![], sp());
    let m = module(vec![b.finish()]);
    assert!(matches!(
        derivative(&m, "flag", 0),
        Err(AdError::BadResult { .. })
    ));

    // An array parameter.
    let vec3 = Type::Number(NumberType {
        shape: sky_types::Shape::vector(3u64),
        ..NumberType::scalar(Dimension::DIMENSIONLESS)
    });
    let mut b = FunctionBuilder::new("arr", vec![vec3], vec![num()], sp());
    let c = b.op1(OpKind::ConstFloat { bits: 0 }, vec![], num(), sp());
    b.op(OpKind::Return, vec![c], vec![], sp());
    let m = module(vec![b.finish()]);
    assert!(matches!(
        derivative(&m, "arr", 0),
        Err(AdError::NonScalar { .. })
    ));
}

#[test]
fn array_operations_are_unsupported_where_the_derivative_flows() {
    // f(x) = [x, 1][0]: the derivative would flow through `array.literal`
    // and `index`, which have no rules yet, and says so.
    let vec2 = Type::Number(NumberType {
        shape: sky_types::Shape::vector(2u64),
        ..NumberType::scalar(Dimension::DIMENSIONLESS)
    });
    let mut b = FunctionBuilder::new("pick", vec![num()], vec![num()], sp());
    let x = b.params()[0];
    let one = b.op1(
        OpKind::ConstFloat {
            bits: 1f64.to_bits(),
        },
        vec![],
        num(),
        sp(),
    );
    let v = b.op1(
        OpKind::ArrayLiteral { cols: None },
        vec![x, one],
        vec2,
        sp(),
    );
    let zero = b.op1(OpKind::ConstFloat { bits: 0 }, vec![], num(), sp());
    let e = b.op1(OpKind::Index, vec![v, zero], num(), sp());
    b.op(OpKind::Return, vec![e], vec![], sp());
    let m = module(vec![b.finish()]);
    let err = derivative(&m, "pick", 0).expect_err("no rule");
    assert!(matches!(err, AdError::Unsupported { .. }), "{err}");
    assert_eq!(
        err.to_string(),
        "cannot differentiate `array.literal` in `pick`: array differentiation is not supported yet"
    );

    // An array that does not depend on x is only data: its derivative is
    // never asked for.
    let mut b = FunctionBuilder::new("const_array", vec![num()], vec![num()], sp());
    let x = b.params()[0];
    let one = b.op1(
        OpKind::ConstFloat {
            bits: 1f64.to_bits(),
        },
        vec![],
        num(),
        sp(),
    );
    let vec2 = Type::Number(NumberType {
        shape: sky_types::Shape::vector(2u64),
        ..NumberType::scalar(Dimension::DIMENSIONLESS)
    });
    let v = b.op1(
        OpKind::ArrayLiteral { cols: None },
        vec![one, one],
        vec2,
        sp(),
    );
    let zero = b.op1(OpKind::ConstFloat { bits: 0 }, vec![], num(), sp());
    let e = b.op1(OpKind::Index, vec![v, zero], num(), sp());
    let r = b.op1(OpKind::Mul, vec![x, e], num(), sp());
    b.op(OpKind::Return, vec![r], vec![], sp());
    let m = module(vec![b.finish()]);
    assert!(derivative(&m, "const_array", 0).is_ok());
}

// ----- properties -----

/// `Σ c_k x^k` by Horner: `((c_n x + c_{n-1}) x + ...) + c_0`.
fn horner(coeffs: &[f64]) -> Function {
    let mut b = B::new("poly", vec![num()]);
    let x = b.param(0);
    let mut acc = b.c(*coeffs.last().unwrap());
    for c in coeffs.iter().rev().skip(1) {
        let m = b.mul(acc, x);
        let k = b.c(*c);
        acc = b.add(m, k);
    }
    b.ret(acc)
}

/// `Σ c_k * x^k` with `pow.const`.
fn power_form(coeffs: &[f64]) -> Function {
    let mut b = B::new("poly", vec![num()]);
    let x = b.param(0);
    let mut acc = b.c(coeffs[0]);
    for (k, c) in coeffs.iter().enumerate().skip(1) {
        let p = b.powc(x, k as i64, 1);
        let k = b.c(*c);
        let t = b.mul(k, p);
        acc = b.add(acc, t);
    }
    b.ret(acc)
}

fn poly_case(f: Function, coeffs: &[f64], x: f64) -> Result<(), TestCaseError> {
    let m = module(vec![f]);
    let mut m2 = m.clone();
    let name = differentiate_into(&mut m2, "poly", 0).unwrap();
    let out = run(&m2, &name, &[x]);
    let ad = number(&out[1]);
    // Exact derivative and the sum of the magnitudes of its terms, which
    // sets the scale for rounding error (cancellation is not AD's fault).
    let mut exact = 0.0;
    let mut mag = 0.0;
    for (k, c) in coeffs.iter().enumerate().skip(1) {
        let t = c * k as f64 * x.powi(k as i32 - 1);
        exact += t;
        mag += t.abs();
    }
    let n = coeffs.len() as f64;
    let tol = 4.0 * n * f64::EPSILON * mag.max(f64::MIN_POSITIVE);
    prop_assert!(
        (ad - exact).abs() <= tol,
        "AD {ad} vs exact {exact} ({} ulps of the term scale)",
        (ad - exact).abs() / (f64::EPSILON * mag)
    );
    Ok(())
}

proptest! {
    #[test]
    fn polynomial_derivatives_match_the_exact_ones(
        coeffs in proptest::collection::vec(-8i32..=8, 2..8),
        x in -3.0f64..3.0,
    ) {
        let coeffs: Vec<f64> = coeffs.into_iter().map(f64::from).collect();
        poly_case(horner(&coeffs), &coeffs, x)?;
        poly_case(power_form(&coeffs), &coeffs, x)?;
    }
}
