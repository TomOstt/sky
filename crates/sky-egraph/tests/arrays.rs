//! Rules on arrays: each exact rule is bit-exact for every input (random
//! values including `±0`, subnormals and huge ones, compared as bits), the
//! sign of zero is respected (`ALG-007` on arrays), the relaxed rules stay
//! behind their contract, and the extraction cost explains itself.

mod common;

use common::*;
use proptest::prelude::*;
use sky_egraph::{
    rewrite_function, rule_table, RewriteConfig, RewriteContract, RuleClass, COST_RULE,
};
use sky_interp::Value;
use sky_ir::{Function, NumberType, OpKind, Repr, Type, ValueId};
use sky_ir_interp::IrRuntimeError;
use sky_types::{Shape, ShapeDim, ShapeVar, TagSet};
use sky_units::Dimension;

fn repro() -> RewriteConfig {
    RewriteConfig::default()
}

fn relaxed() -> RewriteConfig {
    RewriteConfig::with_contract(RewriteContract::Relaxed)
}

fn num_ty(shape: Shape) -> NumberType {
    NumberType {
        dim: Dimension::DIMENSIONLESS,
        shape,
        tags: TagSet::EMPTY,
        repr: Repr::Unchosen,
    }
}

fn vec_ty(n: u64) -> Type {
    Type::Number(num_ty(Shape::vector(n)))
}

fn mat_ty(r: u64, c: u64) -> Type {
    Type::Number(num_ty(Shape::matrix(r, c)))
}

type Outcome = Result<Vec<Value>, IrRuntimeError>;

fn run_with(f: &Function, args: &[Value]) -> Outcome {
    let m = module(f);
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        sky_ir_interp::call(&m, "f", args.to_vec(), &mut out)
    })
}

/// Bit-identical values: unlike `==` on `f64`, `-0.0` and `0.0` differ.
fn bits_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (
            Value::Array {
                dims: da,
                data: xa,
                dim: ua,
            },
            Value::Array {
                dims: db,
                data: xb,
                dim: ub,
            },
        ) => {
            da == db
                && ua == ub
                && xa.len() == xb.len()
                && xa.iter().zip(xb).all(|(p, q)| p.to_bits() == q.to_bits())
        }
        (a, b) => same_value(a, b),
    }
}

/// The same bits, or the same error (message, spans, labels).
fn same(a: &Outcome, b: &Outcome) -> bool {
    match (a, b) {
        (Ok(x), Ok(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| bits_eq(p, q)),
        (Err(x), Err(y)) => x == y,
        _ => false,
    }
}

/// Random finite doubles with the awkward ones over-represented: both zeros,
/// subnormals, the extremes, huge values, and ordinary ones.
fn awkward() -> impl Strategy<Value = f64> {
    let sign = |x: f64, neg: bool| if neg { -x } else { x };
    prop_oneof![
        3 => Just(0.0),
        3 => Just(-0.0),
        2 => (1u64..(1u64 << 52), any::<bool>()).prop_map(move |(b, n)| sign(f64::from_bits(b), n)),
        1 => Just(f64::MIN_POSITIVE),
        1 => Just(f64::MAX),
        1 => Just(-f64::MAX),
        2 => (1.0e300..1.7e308f64, any::<bool>()).prop_map(move |(x, n)| sign(x, n)),
        4 => -1.0e6..1.0e6f64,
        2 => prop::sample::select(vec![1.0, -1.0, 2.0, 0.5, -0.5, 3.0]),
    ]
}

fn array_of(dims: Vec<usize>) -> impl Strategy<Value = Value> {
    let n: usize = dims.iter().product();
    prop::collection::vec(awkward(), n)
        .prop_map(move |data| Value::array(dims.clone(), data, Dimension::DIMENSIONLESS))
}

/// Rewrites a copy of `f` under `cfg`; returns it and the rules that fired
/// (cost explanations excluded).
fn rewritten(f: &Function, cfg: &RewriteConfig) -> (Function, Vec<&'static str>) {
    let mut g = f.clone();
    let fired = rewrite_function(&mut g, cfg)
        .iter()
        .map(|r| r.rule)
        .filter(|r| *r != COST_RULE)
        .collect();
    (g, fired)
}

fn neg(t: &mut T, a: ValueId, ty: &Type) -> ValueId {
    t.op(OpKind::Neg, &[a], ty.clone())
}

fn pow2(t: &mut T, a: ValueId, ty: &Type) -> ValueId {
    t.op(
        OpKind::PowConst {
            exponent: sky_units::Rational::integer(2),
        },
        &[a],
        ty.clone(),
    )
}

const N: usize = 3;

// ---------------------------------------------------------------------
// ALG-007: -(-x) on numbers and arrays
// ---------------------------------------------------------------------

#[test]
fn alg_007_scalar_negation_is_an_involution_for_every_bit() {
    let mut t = T::new(vec![num()], vec![num()]);
    let x = t.params()[0];
    let y = t.un(OpKind::Neg, x);
    let z = t.un(OpKind::Neg, y);
    let f = t.ret(&[z]);
    let (g, fired) = rewritten(&f, &repro());
    assert_eq!(fired, ["ALG-007"]);
    assert_eq!(kinds(&g), ["return"]);
    for x in [0.0, -0.0, 1.5, -2.0, f64::MIN_POSITIVE / 4.0, f64::MAX] {
        let args = [Value::scalar(x)];
        assert!(same(&run_with(&f, &args), &run_with(&g, &args)), "x = {x}");
    }
}

/// The regression: an array's negation is `0 - x`, so `-(-x)` turns `-0`
/// into `+0`, and the rewrite to `x` used to change the bits.
#[test]
fn alg_007_array_double_negation_keeps_the_sign_of_zero() {
    let ty = vec_ty(N as u64);
    let mut t = T::new(vec![ty.clone()], vec![ty.clone()]);
    let v = t.params()[0];
    let a = neg(&mut t, v, &ty);
    let b = neg(&mut t, a, &ty);
    let f = t.ret(&[b]);
    let (g, fired) = rewritten(&f, &repro());
    assert!(fired.is_empty(), "fired {fired:?}");
    assert_eq!(kinds(&g), ["neg", "neg", "return"]);
    let args = [Value::array(
        vec![N],
        vec![-0.0, 0.0, -1.0],
        Dimension::DIMENSIONLESS,
    )];
    let out = run_with(&g, &args).expect("runs");
    let Value::Array { data, .. } = &out[0] else {
        panic!("not an array")
    };
    // `-0` came back as `+0`: which is exactly why `x` is not the answer.
    assert_eq!(data[0].to_bits(), 0.0f64.to_bits());
    assert!(same(&run_with(&f, &args), &run_with(&g, &args)));
    // And the same holds under the other contracts: no contract may make
    // this rewrite, it is not valid over the reals either (`-0 = +0` there,
    // but the bits differ).
    for contract in [RewriteContract::Relaxed, RewriteContract::Fast] {
        let (_, fired) = rewritten(&f, &RewriteConfig::with_contract(contract));
        assert!(fired.is_empty(), "{contract:?} fired {fired:?}");
    }
}

#[test]
fn alg_007_array_double_negation_fires_when_no_entry_can_be_minus_zero() {
    // `v^2` has entries `>= +0`, never `-0`.
    let ty = vec_ty(N as u64);
    let mut t = T::new(vec![ty.clone()], vec![ty.clone()]);
    let v = t.params()[0];
    let sq = pow2(&mut t, v, &ty);
    let a = neg(&mut t, sq, &ty);
    let b = neg(&mut t, a, &ty);
    let f = t.ret(&[b]);
    let (g, fired) = rewritten(&f, &repro());
    assert_eq!(fired, ["ALG-007"]);
    assert_eq!(kinds(&g), ["pow.const", "return"]);
    let args = [Value::array(
        vec![N],
        vec![-0.0, 0.0, -1.5],
        Dimension::DIMENSIONLESS,
    )];
    assert!(same(&run_with(&f, &args), &run_with(&g, &args)));
}

proptest! {
    #[test]
    fn alg_007_on_proven_arrays_is_bit_exact(v in array_of(vec![N])) {
        let ty = vec_ty(N as u64);
        let mut t = T::new(vec![ty.clone()], vec![ty.clone()]);
        let p = t.params()[0];
        let sq = pow2(&mut t, p, &ty);
        let a = neg(&mut t, sq, &ty);
        let b = neg(&mut t, a, &ty);
        let f = t.ret(&[b]);
        let (g, fired) = rewritten(&f, &repro());
        prop_assert_eq!(fired, vec!["ALG-007"]);
        prop_assert!(same(&run_with(&f, &[v.clone()]), &run_with(&g, &[v])));
    }

    #[test]
    fn alg_007_on_numbers_is_bit_exact(x in awkward()) {
        let mut t = T::new(vec![num()], vec![num()]);
        let p = t.params()[0];
        let y = t.un(OpKind::Neg, p);
        let z = t.un(OpKind::Neg, y);
        let f = t.ret(&[z]);
        let (g, _) = rewritten(&f, &repro());
        let args = [Value::scalar(x)];
        prop_assert!(same(&run_with(&f, &args), &run_with(&g, &args)));
    }
}

// ---------------------------------------------------------------------
// ALG-011 / ALG-012: adding or subtracting a negation
// ---------------------------------------------------------------------

/// `f(x, y) = build(x, y)` on numbers.
fn scalar2(build: impl FnOnce(&mut T, ValueId, ValueId) -> ValueId) -> Function {
    let mut t = T::new(vec![num(), num()], vec![num()]);
    let p = t.params();
    let r = build(&mut t, p[0], p[1]);
    t.ret(&[r])
}

/// `f(v, w) = build(v, w)` on vectors.
fn vector2(build: impl FnOnce(&mut T, ValueId, ValueId) -> ValueId) -> Function {
    let ty = vec_ty(N as u64);
    let mut t = T::new(vec![ty.clone(), ty.clone()], vec![ty]);
    let p = t.params();
    let r = build(&mut t, p[0], p[1]);
    t.ret(&[r])
}

#[test]
fn alg_011_and_012_on_numbers() {
    let add_neg = scalar2(|t, x, y| {
        let ny = t.un(OpKind::Neg, y);
        t.bin(OpKind::Add, x, ny)
    });
    let neg_add = scalar2(|t, x, y| {
        let ny = t.un(OpKind::Neg, y);
        t.bin(OpKind::Add, ny, x)
    });
    let sub_neg = scalar2(|t, x, y| {
        let ny = t.un(OpKind::Neg, y);
        t.bin(OpKind::Sub, x, ny)
    });
    for (f, rule, after) in [
        (&add_neg, "ALG-011", ["sub", "return"]),
        (&neg_add, "ALG-011", ["sub", "return"]),
        (&sub_neg, "ALG-012", ["add", "return"]),
    ] {
        let (g, fired) = rewritten(f, &repro());
        assert_eq!(fired, [rule]);
        assert_eq!(kinds(&g), after);
    }
}

proptest! {
    #[test]
    fn alg_011_and_012_on_numbers_are_bit_exact(x in awkward(), y in awkward()) {
        let fs = [
            scalar2(|t, x, y| { let n = t.un(OpKind::Neg, y); t.bin(OpKind::Add, x, n) }),
            scalar2(|t, x, y| { let n = t.un(OpKind::Neg, y); t.bin(OpKind::Add, n, x) }),
            scalar2(|t, x, y| { let n = t.un(OpKind::Neg, y); t.bin(OpKind::Sub, x, n) }),
        ];
        let args = [Value::scalar(x), Value::scalar(y)];
        for f in &fs {
            let (g, fired) = rewritten(f, &repro());
            prop_assert_eq!(fired.len(), 1);
            prop_assert!(same(&run_with(f, &args), &run_with(&g, &args)));
        }
    }
}

/// On arrays the rewrite is not exact for `x = -0`, `y = +0` (`neg` is
/// `0 - y`), so the rules must not fire on an unproven `x` and must fire on
/// a proven one.
#[test]
fn alg_011_and_012_on_arrays_need_x_proven_not_minus_zero() {
    let ty = vec_ty(N as u64);
    let build_add = |proven: bool| {
        vector2(|t, x, y| {
            let x = if proven { pow2(t, x, &ty) } else { x };
            let ny = neg(t, y, &ty);
            t.op(OpKind::Add, &[x, ny], ty.clone())
        })
    };
    let build_sub = |proven: bool| {
        vector2(|t, x, y| {
            let x = if proven { pow2(t, x, &ty) } else { x };
            let ny = neg(t, y, &ty);
            t.op(OpKind::Sub, &[x, ny], ty.clone())
        })
    };
    let witness = [
        Value::array(vec![N], vec![-0.0, -0.0, 0.0], Dimension::DIMENSIONLESS),
        Value::array(vec![N], vec![0.0, -0.0, 0.0], Dimension::DIMENSIONLESS),
    ];
    for (mk, rule) in [
        (&build_add as &dyn Fn(bool) -> Function, "ALG-011"),
        (&build_sub, "ALG-012"),
    ] {
        let unproven = mk(false);
        let (g, fired) = rewritten(&unproven, &repro());
        assert!(fired.is_empty(), "{rule} fired on an unproven array");
        assert_eq!(g, unproven);
        let proven = mk(true);
        let (g, fired) = rewritten(&proven, &repro());
        assert_eq!(fired, [rule]);
        assert!(same(&run_with(&proven, &witness), &run_with(&g, &witness)));
    }
    // The witness that the unproven rewrite would change bits: `-0 + -(+0)`
    // is `+0` (the negation is `0 - 0 = +0`, and `-0 + +0 = +0`), while
    // `-0 - (+0)` is `-0`.
    let plain_add = vector2(|t, x, y| {
        let ny = neg(t, y, &ty);
        t.op(OpKind::Add, &[x, ny], ty.clone())
    });
    let naive = vector2(|t, x, y| t.op(OpKind::Sub, &[x, y], ty.clone()));
    let a = run_with(&plain_add, &witness);
    let b = run_with(&naive, &witness);
    assert!(!same(&a, &b), "the naive rewrite would have been exact");
}

proptest! {
    #[test]
    fn alg_011_and_012_on_proven_arrays_are_bit_exact(
        v in array_of(vec![N]),
        w in array_of(vec![N]),
    ) {
        let ty = vec_ty(N as u64);
        for (op, rule) in [(OpKind::Add, "ALG-011"), (OpKind::Sub, "ALG-012")] {
            for swap in [false, true] {
                if swap && op == OpKind::Sub {
                    continue;
                }
                let f = vector2(|t, x, y| {
                    let x = pow2(t, x, &ty);
                    let ny = neg(t, y, &ty);
                    let (a, b) = if swap { (ny, x) } else { (x, ny) };
                    t.op(op.clone(), &[a, b], ty.clone())
                });
                let (g, fired) = rewritten(&f, &repro());
                prop_assert_eq!(fired, vec![rule]);
                let args = [v.clone(), w.clone()];
                prop_assert!(same(&run_with(&f, &args), &run_with(&g, &args)));
            }
        }
    }
}

// ---------------------------------------------------------------------
// ALG-013: (-x) * (-y) on numbers only
// ---------------------------------------------------------------------

proptest! {
    #[test]
    fn alg_013_negated_factors_cancel_bit_for_bit(x in awkward(), y in awkward()) {
        let f = scalar2(|t, x, y| {
            let nx = t.un(OpKind::Neg, x);
            let ny = t.un(OpKind::Neg, y);
            t.bin(OpKind::Mul, nx, ny)
        });
        let (g, fired) = rewritten(&f, &repro());
        prop_assert_eq!(fired, vec!["ALG-013"]);
        prop_assert_eq!(kinds(&g), vec!["mul", "return"]);
        let args = [Value::scalar(x), Value::scalar(y)];
        prop_assert!(same(&run_with(&f, &args), &run_with(&g, &args)));
    }
}

#[test]
fn alg_013_leaves_arrays_alone() {
    // `(-s) * (-v)` with an array `v`: `(-s) * (0 - v)` and `s * v` differ
    // in the sign of a zero (`v = +0`, `s > 0`).
    let vty = vec_ty(N as u64);
    let mut t = T::new(vec![num(), vty.clone()], vec![vty.clone()]);
    let p = t.params();
    let ns = t.un(OpKind::Neg, p[0]);
    let nv = neg(&mut t, p[1], &vty);
    let r = t.op(OpKind::Mul, &[ns, nv], vty.clone());
    let f = t.ret(&[r]);
    let (_, fired) = rewritten(&f, &repro());
    assert!(fired.is_empty(), "fired {fired:?}");
}

// ---------------------------------------------------------------------
// ALG-015: v + v
// ---------------------------------------------------------------------

proptest! {
    #[test]
    fn alg_015_doubling_an_array_is_bit_exact(v in array_of(vec![N])) {
        let ty = vec_ty(N as u64);
        let mut t = T::new(vec![ty.clone()], vec![ty.clone()]);
        let p = t.params()[0];
        let r = t.op(OpKind::Add, &[p, p], ty.clone());
        let f = t.ret(&[r]);
        let (g, fired) = rewritten(&f, &repro());
        prop_assert_eq!(fired, vec!["ALG-015"]);
        prop_assert_eq!(kinds(&g), vec!["const.rational", "mul", "return"]);
        let args = [v];
        // Including the overflowing inputs: the same error, at the same span.
        prop_assert!(same(&run_with(&f, &args), &run_with(&g, &args)));
    }
}

#[test]
fn alg_015_does_not_touch_numbers() {
    let f = scalar2(|t, x, _| t.bin(OpKind::Add, x, x));
    let (_, fired) = rewritten(&f, &repro());
    assert!(fired.is_empty());
}

// ---------------------------------------------------------------------
// LA-002, LA-004, LA-005: transposes
// ---------------------------------------------------------------------

proptest! {
    #[test]
    fn la_002_double_transpose_is_bit_exact(a in array_of(vec![2, 3])) {
        let (m, mt) = (mat_ty(2, 3), mat_ty(3, 2));
        let mut t = T::new(vec![m.clone()], vec![m.clone()]);
        let p = t.params()[0];
        let x = t.op(OpKind::Transpose, &[p], mt);
        let y = t.op(OpKind::Transpose, &[x], m);
        let f = t.ret(&[y]);
        let (g, fired) = rewritten(&f, &repro());
        prop_assert_eq!(fired, vec!["LA-002"]);
        prop_assert_eq!(kinds(&g), vec!["return"]);
        let args = [a];
        prop_assert!(same(&run_with(&f, &args), &run_with(&g, &args)));
    }

    #[test]
    fn la_004_and_005_transposed_sums_are_bit_exact(
        a in array_of(vec![2, 3]),
        b in array_of(vec![2, 3]),
    ) {
        for (op, rule) in [(OpKind::Add, "LA-004"), (OpKind::Sub, "LA-005")] {
            let (m, mt) = (mat_ty(2, 3), mat_ty(3, 2));
            let mut t = T::new(vec![m.clone(), m.clone()], vec![mt.clone()]);
            let p = t.params();
            let x = t.op(OpKind::Transpose, &[p[0]], mt.clone());
            let y = t.op(OpKind::Transpose, &[p[1]], mt.clone());
            let r = t.op(op.clone(), &[x, y], mt.clone());
            let f = t.ret(&[r]);
            let (g, fired) = rewritten(&f, &repro());
            prop_assert_eq!(fired, vec![rule]);
            prop_assert_eq!(kinds(&g).len(), 3); // add/sub, transpose, return
            let args = [a.clone(), b.clone()];
            // Overflowing inputs give the same error at the same span.
            prop_assert!(same(&run_with(&f, &args), &run_with(&g, &args)));
        }
    }
}

/// The new `A + B` is a new failing operation; the write-back check that
/// keeps the order of failures accepts it because it fails where the old one
/// did, and only there.
#[test]
fn la_004_keeps_the_failure_behaviour() {
    let (m, mt) = (mat_ty(2, 2), mat_ty(2, 2));
    let mut t = T::new(vec![m.clone(), m.clone()], vec![mt.clone()]);
    let p = t.params();
    let x = t.op(OpKind::Transpose, &[p[0]], mt.clone());
    let y = t.op(OpKind::Transpose, &[p[1]], mt.clone());
    let r = t.op(OpKind::Add, &[x, y], mt.clone());
    let f = t.ret(&[r]);
    let (g, fired) = rewritten(&f, &repro());
    assert_eq!(fired, ["LA-004"]);
    let big = |x: f64| Value::array(vec![2, 2], vec![x; 4], Dimension::DIMENSIONLESS);
    let args = [big(f64::MAX), big(f64::MAX)];
    let (a, b) = (run_with(&f, &args), run_with(&g, &args));
    assert!(a.is_err());
    assert!(same(&a, &b));
}

// ---------------------------------------------------------------------
// The relaxed rules
// ---------------------------------------------------------------------

fn close(a: f64, b: f64, scale: f64) -> bool {
    (a - b).abs() <= 1e-12 * scale.max(1e-300)
}

fn values_close(a: &Value, b: &Value, scale: f64) -> bool {
    match (a, b) {
        (Value::Array { data: x, .. }, Value::Array { data: y, .. }) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| close(*p, *q, scale))
        }
        (Value::Number { value: x, .. }, Value::Number { value: y, .. }) => close(*x, *y, scale),
        _ => false,
    }
}

fn moderate() -> impl Strategy<Value = f64> {
    -1.0e3..1.0e3f64
}

#[test]
fn relaxed_rules_are_not_enabled_by_default() {
    let table = rule_table();
    let relaxed_ids: Vec<_> = table
        .iter()
        .filter(|r| r.class == RuleClass::Relaxed)
        .map(|r| r.id)
        .collect();
    for id in ["ALG-016", "ALG-017", "ALG-018", "LA-006", "LA-007"] {
        assert!(relaxed_ids.contains(&id), "{id} is not a Relaxed rule");
    }
    // Only the exact rules run by default: no rule of a relaxed class in the
    // Reproducible set, whatever the program.
    let ty = vec_ty(N as u64);
    let mut t = T::new(vec![num(), ty.clone(), ty.clone()], vec![ty.clone()]);
    let p = t.params();
    let a = t.op(OpKind::Mul, &[p[0], p[1]], ty.clone());
    let b = t.op(OpKind::Mul, &[p[0], p[2]], ty.clone());
    let s = t.op(OpKind::Add, &[a, b], ty.clone());
    let f = t.ret(&[s]);
    assert!(rewritten(&f, &repro()).1.is_empty());
    assert_eq!(rewritten(&f, &relaxed()).1, ["ALG-016"]);
    assert_eq!(
        rewritten(&f, &RewriteConfig::with_contract(RewriteContract::Fast)).1,
        ["ALG-016"]
    );
}

proptest! {
    #[test]
    fn alg_016_and_017_distribute_a_scaling(
        a in moderate(),
        v in prop::collection::vec(moderate(), N),
        w in prop::collection::vec(moderate(), N),
    ) {
        let ty = vec_ty(N as u64);
        for (op, rule) in [(OpKind::Add, "ALG-016"), (OpKind::Sub, "ALG-017")] {
            for swap in [false, true] {
                let mut t = T::new(vec![num(), ty.clone(), ty.clone()], vec![ty.clone()]);
                let p = t.params();
                let (l, r) = if swap {
                    (t.op(OpKind::Mul, &[p[1], p[0]], ty.clone()), t.op(OpKind::Mul, &[p[2], p[0]], ty.clone()))
                } else {
                    (t.op(OpKind::Mul, &[p[0], p[1]], ty.clone()), t.op(OpKind::Mul, &[p[0], p[2]], ty.clone()))
                };
                let s = t.op(op.clone(), &[l, r], ty.clone());
                let f = t.ret(&[s]);
                let (g, fired) = rewritten(&f, &relaxed());
                prop_assert_eq!(fired, vec![rule]);
                prop_assert_eq!(kinds(&g), vec![if op == OpKind::Add { "add" } else { "sub" }, "mul", "return"]);
                let args = [
                    Value::scalar(a),
                    Value::array(vec![N], v.clone(), Dimension::DIMENSIONLESS),
                    Value::array(vec![N], w.clone(), Dimension::DIMENSIONLESS),
                ];
                let (x, y) = (run_with(&f, &args).expect("runs"), run_with(&g, &args).expect("runs"));
                prop_assert!(values_close(&x[0], &y[0], a.abs() * 2.0e3 + 1.0));
            }
        }
    }

    #[test]
    fn alg_018_rescales_once(
        s in moderate(),
        t2 in moderate(),
        v in prop::collection::vec(moderate(), N),
    ) {
        let ty = vec_ty(N as u64);
        let mut t = T::new(vec![num(), num(), ty.clone()], vec![ty.clone()]);
        let p = t.params();
        let inner = t.op(OpKind::Mul, &[p[1], p[2]], ty.clone());
        let outer = t.op(OpKind::Mul, &[p[0], inner], ty.clone());
        let f = t.ret(&[outer]);
        let (g, fired) = rewritten(&f, &relaxed());
        prop_assert_eq!(fired, vec!["ALG-018"]);
        prop_assert!(rewritten(&f, &repro()).1.is_empty());
        let args = [
            Value::scalar(s),
            Value::scalar(t2),
            Value::array(vec![N], v, Dimension::DIMENSIONLESS),
        ];
        let (x, y) = (run_with(&f, &args).expect("runs"), run_with(&g, &args).expect("runs"));
        prop_assert!(values_close(&x[0], &y[0], (s * t2).abs() * 1.0e3 + 1.0));
    }
}

// ---------------------------------------------------------------------
// Matrix products: only what is provably cheaper, and only when relaxed
// ---------------------------------------------------------------------

fn sized(names: &[(&str, u64)]) -> Vec<ShapeDim> {
    names
        .iter()
        .map(|(n, id)| ShapeDim::Var(ShapeVar::from_parts(*id, *n)))
        .collect()
}

fn sym_mat(r: &ShapeDim, c: &ShapeDim) -> Type {
    Type::Number(num_ty(Shape::matrix(r.clone(), c.clone())))
}

fn sym_vec(n: &ShapeDim) -> Type {
    Type::Number(num_ty(Shape::vector(n.clone())))
}

/// `(A @ B) @ v` with `A`, `B` of `N x N` and `v` of `N`, named.
fn left_chain() -> Function {
    let n = sized(&[("N", 1)]).remove(0);
    let (m, v) = (sym_mat(&n, &n), sym_vec(&n));
    let mut t = T::new(vec![m.clone(), m.clone(), v.clone()], vec![v.clone()]);
    let p = t.params();
    for (id, name) in p.iter().zip(["A", "B", "v"]) {
        t.b.set_name(*id, name);
    }
    let ab = t.op(OpKind::MatMul, &[p[0], p[1]], m);
    let r = t.op(OpKind::MatMul, &[ab, p[2]], v);
    t.ret(&[r])
}

/// `A @ (B @ v)`.
fn right_chain() -> Function {
    let n = sized(&[("N", 1)]).remove(0);
    let (m, v) = (sym_mat(&n, &n), sym_vec(&n));
    let mut t = T::new(vec![m.clone(), m, v.clone()], vec![v.clone()]);
    let p = t.params();
    for (id, name) in p.iter().zip(["A", "B", "v"]) {
        t.b.set_name(*id, name);
    }
    let bv = t.op(OpKind::MatMul, &[p[1], p[2]], v.clone());
    let r = t.op(OpKind::MatMul, &[p[0], bv], v);
    t.ret(&[r])
}

#[test]
fn matmul_chain_is_reassociated_only_under_relaxed_and_explained_with_the_costs() {
    let f = left_chain();
    // Reproducible: matrix products are not exactly associative.
    let mut g = f.clone();
    assert!(rewrite_function(&mut g, &repro()).is_empty());
    assert_eq!(g, f);

    let mut g = f.clone();
    let rw = rewrite_function(&mut g, &relaxed());
    let rules: Vec<_> = rw.iter().map(|r| r.rule).collect();
    assert_eq!(rules, ["LA-006", "EG-COST-001"]);
    let why = &rw[1];
    assert_eq!(why.before, "(A @ B) @ v");
    assert_eq!(why.after, "A @ (B @ v)");
    assert_eq!(
        why.because,
        "chose `A @ (B @ v)` over `(A @ B) @ v`: 4·N² vs 2·N³ + 2·N² flops, \
         16·N² + 32·N vs 32·N² + 16·N bytes (cheaper for every size)"
    );
    assert_eq!(kinds(&g), ["matmul", "matmul", "return"]);
}

#[test]
fn a_chain_that_is_already_cheapest_is_left_alone() {
    // The checker plans `A @ (B @ v)`; the e-graph must not fight it, under
    // any contract.
    let f = right_chain();
    for cfg in [
        repro(),
        relaxed(),
        RewriteConfig::with_contract(RewriteContract::Fast),
    ] {
        let mut g = f.clone();
        assert!(rewrite_function(&mut g, &cfg).is_empty());
        assert_eq!(g, f);
    }
}

#[test]
fn a_chain_the_model_cannot_order_for_every_size_is_left_alone() {
    // `(A @ B) @ C` with `A: n x k`, `B: k x m`, `C: m x p`: which order is
    // cheaper depends on the sizes, so the checker's choice stands.
    let d = sized(&[("n", 1), ("k", 2), ("m", 3), ("p", 4)]);
    let (a, b, c) = (
        sym_mat(&d[0], &d[1]),
        sym_mat(&d[1], &d[2]),
        sym_mat(&d[2], &d[3]),
    );
    let mut t = T::new(
        vec![a.clone(), b.clone(), c.clone()],
        vec![sym_mat(&d[0], &d[3])],
    );
    let p = t.params();
    let ab = t.op(OpKind::MatMul, &[p[0], p[1]], sym_mat(&d[0], &d[2]));
    let r = t.op(OpKind::MatMul, &[ab, p[2]], sym_mat(&d[0], &d[3]));
    let f = t.ret(&[r]);
    let mut g = f.clone();
    assert!(rewrite_function(&mut g, &relaxed()).is_empty());
    assert_eq!(g, f);
}

#[test]
fn matmul_reassociation_with_known_sizes_agrees_numerically() {
    // 64 x 64 matrices: constant costs, so the order is decided.
    let (m, v) = (mat_ty(4, 4), vec_ty(4));
    let mut t = T::new(vec![m.clone(), m.clone(), v.clone()], vec![v.clone()]);
    let p = t.params();
    let ab = t.op(OpKind::MatMul, &[p[0], p[1]], m);
    let r = t.op(OpKind::MatMul, &[ab, p[2]], v);
    let f = t.ret(&[r]);
    let (g, fired) = rewritten(&f, &relaxed());
    assert_eq!(fired, ["LA-006"]);
    let mat = |k: f64| {
        Value::array(
            vec![4, 4],
            (0..16).map(|i| (i as f64) * 0.25 + k).collect(),
            Dimension::DIMENSIONLESS,
        )
    };
    let args = [
        mat(1.0),
        mat(-2.0),
        Value::array(vec![4], vec![1.0, 2.0, 3.0, 4.0], Dimension::DIMENSIONLESS),
    ];
    let (x, y) = (
        run_with(&f, &args).expect("runs"),
        run_with(&g, &args).expect("runs"),
    );
    assert!(values_close(&x[0], &y[0], 1.0e4));
}

// ---------------------------------------------------------------------
// Explanations
// ---------------------------------------------------------------------

#[test]
fn a_cheaper_array_form_is_chosen_and_explained_under_the_default_contract() {
    // `Aᵀ + Bᵀ` copies twice; `(A + B)ᵀ` once.
    let (m, mt) = (mat_ty(8, 8), mat_ty(8, 8));
    let mut t = T::new(vec![m.clone(), m.clone()], vec![mt.clone()]);
    let p = t.params();
    for (id, name) in p.iter().zip(["A", "B"]) {
        t.b.set_name(*id, name);
    }
    let x = t.op(OpKind::Transpose, &[p[0]], mt.clone());
    let y = t.op(OpKind::Transpose, &[p[1]], mt.clone());
    let r = t.op(OpKind::Add, &[x, y], mt);
    let f = t.ret(&[r]);
    let mut g = f.clone();
    let rw = rewrite_function(&mut g, &repro());
    assert_eq!(
        rw.iter().map(|r| r.rule).collect::<Vec<_>>(),
        ["LA-004", "EG-COST-001"]
    );
    assert_eq!(rw[0].before, "Aᵀ + Bᵀ");
    assert_eq!(rw[0].after, "(A + B)ᵀ");
    assert_eq!(rw[1].before, "Aᵀ + Bᵀ");
    assert_eq!(rw[1].after, "(A + B)ᵀ");
    assert_eq!(
        rw[1].because,
        "chose `(A + B)ᵀ` over `Aᵀ + Bᵀ`: 64 vs 64 flops, 2560 vs 3584 bytes, \
         2 vs 3 allocations (weighted cost 512 vs 704)"
    );
}

// ---------------------------------------------------------------------
// Where a rule-made failing operation reports its failure
// ---------------------------------------------------------------------

/// `((-x) * (-x))^1`: the class of the product also holds the `^1`, whose
/// span is wider. The product a rule makes must report an overflow at the
/// original `*`, not at the class's joined span.
#[test]
fn a_rule_made_failing_op_fails_at_the_span_of_the_original_op() {
    let big = [Value::scalar(1e200), Value::scalar(0.0)];
    // The counterexample of the differential proptest, as it was found.
    let f = scalar2(|t, x, _| {
        let nx = t.un(OpKind::Neg, x);
        let m = t.bin(OpKind::Mul, nx, nx);
        t.pow(m, 1, 1)
    });
    let (g, fired) = rewritten(&f, &repro());
    assert!(fired.contains(&"ALG-013"), "{fired:?}");
    let (a, b) = (run_with(&f, &big), run_with(&g, &big));
    assert!(a.is_err());
    assert!(same(&a, &b), "{a:?}\nvs\n{b:?}");

    // `(x + (-y))^1`, `(x - (-y))^1`, `(-x * -y)^1` on overflowing numbers.
    let wrap = |build: fn(&mut T, ValueId, ValueId) -> ValueId| {
        scalar2(move |t, x, y| {
            let r = build(t, x, y);
            t.pow(r, 1, 1)
        })
    };
    let cases: [(Function, &str); 3] = [
        (
            wrap(|t, x, y| {
                let ny = t.un(OpKind::Neg, y);
                t.bin(OpKind::Add, x, ny)
            }),
            "ALG-011",
        ),
        (
            wrap(|t, x, y| {
                let ny = t.un(OpKind::Neg, y);
                t.bin(OpKind::Sub, x, ny)
            }),
            "ALG-012",
        ),
        (
            wrap(|t, x, y| {
                let nx = t.un(OpKind::Neg, x);
                let ny = t.un(OpKind::Neg, y);
                t.bin(OpKind::Mul, nx, ny)
            }),
            "ALG-013",
        ),
    ];
    for (f, rule) in &cases {
        // `x + (-y)` overflows for `y = -MAX`; the others for `y = MAX`.
        let y = if *rule == "ALG-011" {
            -f64::MAX
        } else {
            f64::MAX
        };
        let huge = [Value::scalar(f64::MAX), Value::scalar(y)];
        let (g, fired) = rewritten(f, &repro());
        assert!(fired.contains(rule), "{rule}: {fired:?}");
        let (a, b) = (run_with(f, &huge), run_with(&g, &huge));
        assert!(a.is_err(), "{rule}");
        assert!(same(&a, &b), "{rule}: {a:?}\nvs\n{b:?}");
    }
}

#[test]
fn rule_made_failing_array_ops_fail_at_the_original_span_too() {
    let ty = vec_ty(N as u64);
    let pow1 = |t: &mut T, v: ValueId, ty: &Type| {
        t.op(
            OpKind::PowConst {
                exponent: sky_units::Rational::integer(1),
            },
            &[v],
            ty.clone(),
        )
    };
    let big = Value::array(vec![N], vec![f64::MAX; N], Dimension::DIMENSIONLESS);
    // `(v + v)^1` (ALG-015).
    let mut t = T::new(vec![ty.clone()], vec![ty.clone()]);
    let p = t.params()[0];
    let s = t.op(OpKind::Add, &[p, p], ty.clone());
    let r = pow1(&mut t, s, &ty);
    let f = t.ret(&[r]);
    let (g, fired) = rewritten(&f, &repro());
    assert!(fired.contains(&"ALG-015"), "{fired:?}");
    let args = [big.clone()];
    assert!(run_with(&f, &args).is_err());
    assert!(same(&run_with(&f, &args), &run_with(&g, &args)));

    // `(Aᵀ + Bᵀ)^1` (LA-004) on matrices.
    let (m, mt) = (mat_ty(2, 3), mat_ty(3, 2));
    let mut t = T::new(vec![m.clone(), m.clone()], vec![mt.clone()]);
    let p = t.params();
    let x = t.op(OpKind::Transpose, &[p[0]], mt.clone());
    let y = t.op(OpKind::Transpose, &[p[1]], mt.clone());
    let s = t.op(OpKind::Add, &[x, y], mt.clone());
    let r = pow1(&mut t, s, &mt);
    let f = t.ret(&[r]);
    let (g, fired) = rewritten(&f, &repro());
    assert!(fired.contains(&"LA-004"), "{fired:?}");
    let bigm = Value::array(vec![2, 3], vec![f64::MAX; 6], Dimension::DIMENSIONLESS);
    let args = [bigm.clone(), bigm];
    assert!(run_with(&f, &args).is_err());
    assert!(same(&run_with(&f, &args), &run_with(&g, &args)));
}

/// `(-y) * (-y)` and `y * y` are one class once ALG-013 fires, and the
/// program's own `y * y` (written later) may be what is extracted. The
/// overflow is still the first product's, at its span.
#[test]
fn an_extracted_op_reports_the_first_failing_op_of_its_class() {
    let mut t = T::new(vec![num(), num()], vec![num(), num()]);
    let y = t.params()[1];
    let ny = t.un(OpKind::Neg, y);
    let first = t.bin(OpKind::Mul, ny, ny);
    let second = t.bin(OpKind::Mul, y, y);
    let f = t.ret(&[first, second]);
    let (g, fired) = rewritten(&f, &repro());
    assert!(fired.contains(&"ALG-013"), "{fired:?}");
    let args = [Value::scalar(0.0), Value::scalar(1e200)];
    let (a, b) = (run_with(&f, &args), run_with(&g, &args));
    assert!(a.is_err());
    assert!(same(&a, &b), "{a:?}\nvs\n{b:?}");
}
