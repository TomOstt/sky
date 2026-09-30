//! Tests of the roundoff analysis (`sky_ir::roundoff`).
//!
//! The analysis claims, for every scalar value, a bound on how far the
//! `f64` the interpreter computes is from the exact real result. The tests
//! that matter therefore compare with an independent high-precision
//! evaluation ([`crate::roundoff_ref`], double-double, about 32 digits):
//!
//! * [`bounds_hold_for_constant_programs`]: random programs over constants
//!   (points): every intermediate value's computed `f64` is within its
//!   claimed absolute and relative bound of the reference, and never
//!   closer than its claimed lower bound. Run with and without the
//!   double-double enclosure (`Options::high_precision`), so the
//!   first-order machinery is tested on its own.
//! * [`bounds_hold_over_ranges`]: the same with two parameters that the
//!   analysis knows only as ranges, and the program run at a point inside.
//! * the hand-written tests pin what the analysis says about well-known
//!   cases (`0.1 + 0.2`, `sqrt(x^2 + 1) - x`, `log(1 + x)`), control flow,
//!   loops and calls.

use proptest::prelude::*;
use sky_diag::Span;
use sky_interp::arith::{arith, pow, reciprocal, ArithOp};
use sky_interp::Value;
use sky_ir::exact::Range;
use sky_ir::roundoff::{self, Fact, Goal, Options, Tolerance, Verdict};
use sky_ir::{FunctionBuilder, MathFn, Module, OpKind, ValueId};
use sky_units::Rational;

use crate::roundoff_ref::{self as reference, dd, Dd};
use crate::testutil::{module_of, num};

fn sp() -> Span {
    Span::point(0)
}

fn scalar(x: f64) -> Value {
    Value::scalar(x)
}

fn number(v: &Value) -> f64 {
    match v {
        Value::Number { value, .. } => *value,
        other => panic!("not a number: {other:?}"),
    }
}

// ------------------------------------------------------------ programs

#[derive(Clone, Debug)]
enum E {
    X,
    Y,
    Float(f64),
    Rat(i64, i64),
    Neg(Box<E>),
    Bin(u8, Box<E>, Box<E>),
    Pow(usize, Box<E>),
    Inv(Box<E>),
    Math1(usize, Box<E>),
    Math2(usize, Box<E>, Box<E>),
    /// `a ^ b`, with both computed.
    PowV(Box<E>, Box<E>),
}

const POWERS: [(i64, i64); 9] = [
    (2, 1),
    (3, 1),
    (-1, 1),
    (0, 1),
    (1, 2),
    (-2, 1),
    (1, 3),
    (5, 2),
    (7, 1),
];

const POOL: [f64; 18] = [
    0.1,
    0.2,
    0.3,
    1.0,
    2.0,
    3.0,
    0.5,
    1e-8,
    1e8,
    12345.678,
    7.0,
    1e-3,
    1e5,
    1e10,
    1e15,
    0.999_999_999_999,
    1.5,
    -2.5,
];

fn arity(n: usize) -> Vec<MathFn> {
    MathFn::ALL
        .into_iter()
        .filter(|f| f.arity() == n && *f != MathFn::Erf)
        .collect()
}

fn leaf(with_params: bool) -> BoxedStrategy<E> {
    let consts = prop_oneof![
        4 => prop::sample::select(POOL.to_vec()).prop_map(E::Float),
        2 => (-100.0f64..100.0).prop_map(E::Float),
        2 => (-9i32..12).prop_map(|k| E::Float(10f64.powi(k))),
        2 => prop::sample::select(vec![(1, 2), (2, 1), (3, 1), (7, 2), (1, 3), (0, 1), (-1, 1), (10, 1), (1, 7)])
            .prop_map(|(n, d)| E::Rat(n, d)),
    ];
    // `(big + small) - big`: a value with a large relative error (the
    // small part is rounded to a multiple of the ulp of `big`), so the
    // second-order terms of the bounds matter.
    let noisy = (
        prop::sample::select(vec![1e14, 1e15, 4.5e15, 1e16 / 4.0]),
        0.05f64..0.95,
    )
        .prop_map(|(big, small)| {
            E::Bin(
                1,
                Box::new(E::Bin(
                    0,
                    Box::new(E::Float(big)),
                    Box::new(E::Float(small)),
                )),
                Box::new(E::Float(big)),
            )
        });
    if with_params {
        prop_oneof![3 => Just(E::X), 2 => Just(E::Y), 3 => consts, 2 => noisy].boxed()
    } else {
        prop_oneof![5 => consts, 2 => noisy].boxed()
    }
}

fn expr(with_params: bool) -> impl Strategy<Value = E> {
    leaf(with_params).prop_recursive(4, 24, 3, |inner| {
        prop_oneof![
            2 => inner.clone().prop_map(|e| E::Neg(Box::new(e))),
            7 => (0u8..4, inner.clone(), inner.clone())
                .prop_map(|(k, a, b)| E::Bin(k, Box::new(a), Box::new(b))),
            2 => (0..POWERS.len(), inner.clone()).prop_map(|(i, e)| E::Pow(i, Box::new(e))),
            1 => inner.clone().prop_map(|e| E::Inv(Box::new(e))),
            4 => (0..arity(1).len(), inner.clone()).prop_map(|(i, e)| E::Math1(i, Box::new(e))),
            2 => (0..arity(2).len(), inner.clone(), inner.clone())
                .prop_map(|(i, a, b)| E::Math2(i, Box::new(a), Box::new(b))),
            1 => (inner.clone(), inner)
                .prop_map(|(a, b)| E::PowV(Box::new(a), Box::new(b))),
        ]
    })
}

/// One evaluated node of a program.
struct Node {
    value: ValueId,
    /// What the interpreter computes.
    computed: f64,
    /// The reference (`None`: not modelled).
    exact: Option<Dd>,
    /// The largest magnitude among the reference values this node depends
    /// on: the reference's own rounding is a few 1e-32 of this.
    scale: f64,
}

struct Gen {
    b: FunctionBuilder,
    next: u32,
    nodes: Vec<Node>,
    /// `None`: X and Y are parameters; `Some`: their values, as constants.
    fixed: Option<(f64, f64)>,
    xy: (f64, f64),
}

type Ev = Option<(ValueId, f64, Option<Dd>, f64)>;

impl Gen {
    fn op(&mut self, kind: OpKind, args: &[ValueId]) -> ValueId {
        self.next += 1;
        let span = Span::new(self.next * 10, self.next * 10 + 5);
        self.b.op1(kind, args.to_vec(), num(), span)
    }

    fn done(&mut self, v: ValueId, c: f64, r: Option<Dd>, scale: f64) -> Ev {
        // A reference that overflowed is not a reference.
        let r = r.filter(|r| r.hi.is_finite() && r.lo.is_finite());
        let scale = match r {
            Some(r) => scale.max(r.to_f64().abs()).max(c.abs()),
            None => scale,
        };
        self.nodes.push(Node {
            value: v,
            computed: c,
            exact: r,
            scale,
        });
        Some((v, c, r, scale))
    }

    /// Builds `e` and evaluates it with the interpreter's own functions and
    /// with the reference. `None` if the interpreter fails.
    fn visit(&mut self, e: &E, px: ValueId, py: ValueId) -> Ev {
        match e {
            E::X | E::Y => {
                let val = if matches!(e, E::X) {
                    self.xy.0
                } else {
                    self.xy.1
                };
                let v = if self.fixed.is_some() {
                    self.op(
                        OpKind::ConstFloat {
                            bits: val.to_bits(),
                        },
                        &[],
                    )
                } else if matches!(e, E::X) {
                    px
                } else {
                    py
                };
                self.done(v, val, Some(dd(val)), 0.0)
            }
            E::Float(x) => {
                let v = self.op(OpKind::ConstFloat { bits: x.to_bits() }, &[]);
                self.done(v, *x, Some(dd(*x)), 0.0)
            }
            E::Rat(n, d) => {
                let v = self.op(OpKind::ConstRational(Rational::new(*n, *d)), &[]);
                let c = *n as f64 / *d as f64;
                let r = dd(*n as f64).div(dd(*d as f64));
                self.done(v, c, r, 0.0)
            }
            E::Neg(a) => {
                let (va, ca, ra, sa) = self.visit(a, px, py)?;
                let v = self.op(OpKind::Neg, &[va]);
                self.done(v, -ca, ra.map(Dd::neg), sa)
            }
            E::Bin(k, a, b) => {
                let (va, ca, ra, sa) = self.visit(a, px, py)?;
                let (vb, cb, rb, sb) = self.visit(b, px, py)?;
                let (op, kind) = [
                    (ArithOp::Add, OpKind::Add),
                    (ArithOp::Sub, OpKind::Sub),
                    (ArithOp::Mul, OpKind::Mul),
                    (ArithOp::Div, OpKind::Div),
                ][*k as usize]
                    .clone();
                let c = arith(op, &scalar(ca), &scalar(cb), sp(), sp(), sp()).ok()?;
                let c = number(&c);
                let v = self.op(kind, &[va, vb]);
                let r = match (ra, rb) {
                    (Some(x), Some(y)) => match k {
                        0 => Some(x.add(y)),
                        1 => Some(x.sub(y)),
                        2 => Some(x.mul(y)),
                        _ => x.div(y),
                    },
                    _ => None,
                };
                self.done(v, c, r, sa.max(sb))
            }
            E::Pow(i, a) => {
                let (va, ca, ra, sa) = self.visit(a, px, py)?;
                let (n, d) = POWERS[*i];
                let q = Rational::new(n, d);
                let exp = scalar(q.numer() as f64 / q.denom() as f64);
                let c = pow(&scalar(ca), &exp, || Some(q), sp(), sp(), sp()).ok()?;
                let c = number(&c);
                let v = self.op(OpKind::PowConst { exponent: q }, &[va]);
                let r = ra.and_then(|x| {
                    if q.is_integer() {
                        x.powi(i32::try_from(q.numer()).ok()?)
                    } else if x.hi > 0.0 {
                        let p = dd(q.numer() as f64).div(dd(q.denom() as f64))?;
                        reference::exp(reference::ln(x)?.mul(p))
                    } else {
                        None
                    }
                });
                self.done(v, c, r, sa)
            }
            E::Inv(a) => {
                let (va, ca, ra, sa) = self.visit(a, px, py)?;
                let c = reciprocal("⁻¹", &scalar(ca), sp(), sp()).ok()?;
                let c = number(&c);
                let v = self.op(OpKind::Inverse, &[va]);
                self.done(v, c, ra.and_then(|x| dd(1.0).div(x)), sa)
            }
            E::Math1(i, a) => {
                let f = arity(1)[*i];
                let (va, ca, ra, sa) = self.visit(a, px, py)?;
                let c = sky_interp::math::apply(f, &[scalar(ca)], sp(), &[sp()]).ok()?;
                let c = number(&c);
                let v = self.op(OpKind::Math(f), &[va]);
                self.done(v, c, ra.and_then(|x| reference::math(f, &[x])), sa)
            }
            E::PowV(a, b) => {
                let (va, ca, ra, sa) = self.visit(a, px, py)?;
                let (vb, cb, rb, sb) = self.visit(b, px, py)?;
                let c = pow(&scalar(ca), &scalar(cb), || None, sp(), sp(), sp()).ok()?;
                let c = number(&c);
                let v = self.op(OpKind::Pow, &[va, vb]);
                let r = match (ra, rb) {
                    (Some(x), Some(y)) if x.hi > 0.0 => {
                        reference::ln(x).and_then(|l| reference::exp(l.mul(y)))
                    }
                    _ => None,
                };
                self.done(v, c, r, sa.max(sb))
            }
            E::Math2(i, a, b) => {
                let f = arity(2)[*i];
                let (va, ca, ra, sa) = self.visit(a, px, py)?;
                let (vb, cb, rb, sb) = self.visit(b, px, py)?;
                let c = sky_interp::math::apply(f, &[scalar(ca), scalar(cb)], sp(), &[sp(), sp()])
                    .ok()?;
                let c = number(&c);
                let v = self.op(OpKind::Math(f), &[va, vb]);
                let r = match (ra, rb) {
                    (Some(x), Some(y)) => reference::math(f, &[x, y]),
                    _ => None,
                };
                self.done(v, c, r, sa.max(sb))
            }
        }
    }
}

struct Program {
    module: Module,
    nodes: Vec<Node>,
}

/// `f`, a function returning the value of `e`, and the evaluation of every
/// node. `params`: X and Y are parameters (else constants of the same
/// values). `None` if the interpreter fails somewhere.
fn program(e: &E, xy: (f64, f64), params: bool) -> Option<Program> {
    let ptypes = if params { vec![num(), num()] } else { vec![] };
    let mut g = Gen {
        b: FunctionBuilder::new("f", ptypes, vec![num()], Span::new(0, 5)),
        next: 0,
        nodes: Vec::new(),
        fixed: (!params).then_some(xy),
        xy,
    };
    let ps = g.b.params();
    let (px, py) = if params {
        (ps[0], ps[1])
    } else {
        (ValueId(0), ValueId(0))
    };
    let (v, _, _, _) = g.visit(e, px, py)?;
    g.b.op(OpKind::Return, vec![v], vec![], sp());
    let nodes = std::mem::take(&mut g.nodes);
    let module = module_of(g.b.finish());
    Some(Program { module, nodes })
}

/// Checks every node against the analysis. Returns how many nodes had a
/// finite bound and how many were checked.
fn check_nodes(
    p: &Program,
    args: &[Range],
    opts: &Options,
    what: &str,
) -> Result<(usize, usize), TestCaseError> {
    let a = roundoff::analyze_function(&p.module, "f", args, opts);
    let (mut bounded, mut checked) = (0, 0);
    for node in &p.nodes {
        let Some(exact) = node.exact else { continue };
        let n = match a.fact("f", node.value) {
            Fact::Known(n) => *n,
            other => {
                return Err(TestCaseError::fail(format!(
                    "{what}: value {} not known: {other:?}",
                    node.value
                )))
            }
        };
        prop_assert!(
            n.range.contains(node.computed),
            "{what}: computed {:e} outside range {:?}",
            node.computed,
            n.range
        );
        // The reference's own rounding, and the slack for it.
        let slack = 1e-26 * node.scale + 1e-300;
        let err = dd(node.computed).sub(exact).to_f64().abs();
        let mag = exact.to_f64().abs();
        prop_assert!(
            err <= n.abs + slack,
            "{what}: |computed - exact| = {err:e} exceeds the claimed absolute bound {:e} (computed {:e}, exact {:e})",
            n.abs,
            node.computed,
            exact.to_f64()
        );
        if n.rel.is_finite() {
            prop_assert!(
                err <= n.rel * mag + slack,
                "{what}: |computed - exact| = {err:e} exceeds the claimed relative bound {:e} of {mag:e} (computed {:e}, exact {:e})",
                n.rel,
                node.computed,
                exact.to_f64()
            );
        }
        prop_assert!(
            n.lower_abs <= err + slack,
            "{what}: error {err:e} is below the claimed lower bound {:e}",
            n.lower_abs
        );
        prop_assert!(
            n.lower_rel * mag <= err + slack,
            "{what}: error {err:e} is below the claimed relative lower bound {:e} of {mag:e}",
            n.lower_rel
        );
        checked += 1;
        bounded += usize::from(n.abs.is_finite() || n.rel.is_finite());
    }
    Ok((bounded, checked))
}

fn options(hp: bool) -> Options {
    Options {
        high_precision: hp,
        ..Options::default()
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 1500, max_global_rejects: 10_000_000, ..ProptestConfig::default() })]

    #[test]
    fn bounds_hold_for_constant_programs(
        e in expr(true),
        x in prop::sample::select(POOL.to_vec()),
        y in -50.0f64..50.0,
    ) {
        let Some(p) = program(&e, (x, y), false) else {
            return Err(TestCaseError::reject("the interpreter fails"));
        };
        for hp in [false, true] {
            check_nodes(&p, &[], &options(hp), &format!("constants, hp={hp}, {e:?}"))?;
        }
    }

    #[test]
    fn bounds_hold_over_ranges(
        e in expr(true),
        x in prop::sample::select(POOL.to_vec()),
        y in -50.0f64..50.0,
        wx in prop::sample::select(vec![0.0, 0.25, 1.0, 4.0]),
        wy in prop::sample::select(vec![0.0, 0.25, 1.0, 4.0]),
    ) {
        let Some(p) = program(&e, (x, y), true) else {
            return Err(TestCaseError::reject("the interpreter fails"));
        };
        let width = |v: f64, w: f64| {
            let d = if v == 0.0 { w } else { v.abs() * w };
            Range::new(v - d, v + d)
        };
        let args = [width(x, wx), width(y, wy)];
        for hp in [false, true] {
            check_nodes(&p, &args, &options(hp), &format!("ranges {args:?}, hp={hp}, {e:?}"))?;
        }
    }
}

/// The properties above are only worth something if a good share of the
/// values they check have a finite bound (a vacuous "unbounded" is always
/// sound).
#[test]
fn the_generator_reaches_bounded_values() {
    use proptest::strategy::ValueTree;
    use proptest::test_runner::TestRunner;

    let mut runner = TestRunner::deterministic();
    let strategy = (
        expr(true),
        prop::sample::select(POOL.to_vec()),
        -50.0f64..50.0,
    );
    let (mut bounded, mut checked, mut programs) = (0, 0, 0);
    for _ in 0..800 {
        let (e, x, y) = strategy.new_tree(&mut runner).expect("a tree").current();
        let Some(p) = program(&e, (x, y), false) else {
            continue;
        };
        let (b, c) = check_nodes(&p, &[], &options(false), "coverage").expect("sound");
        bounded += b;
        checked += c;
        programs += 1;
    }
    assert!(programs > 100, "only {programs} programs ran");
    assert!(
        bounded * 10 > checked * 6,
        "only {bounded} of {checked} values have a bound"
    );
}

// ------------------------------------------------------------ examples

fn main_fn(build: impl FnOnce(&mut FunctionBuilder) -> ValueId) -> (Module, ValueId) {
    let mut b = FunctionBuilder::new(Module::MAIN, vec![], vec![], sp());
    let v = build(&mut b);
    b.op(OpKind::Return, vec![], vec![], sp());
    (module_of(b.finish()), v)
}

fn c(b: &mut FunctionBuilder, x: f64) -> ValueId {
    b.op1(
        OpKind::ConstFloat { bits: x.to_bits() },
        vec![],
        num(),
        sp(),
    )
}

fn op(b: &mut FunctionBuilder, k: OpKind, args: &[ValueId]) -> ValueId {
    b.op1(k, args.to_vec(), num(), sp())
}

fn known(m: &Module, v: ValueId, hp: bool) -> roundoff::Num {
    let a = roundoff::analyze(m, &options(hp));
    match a.fact(Module::MAIN, v) {
        Fact::Known(n) => *n,
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_error_of_point_zero_one_plus_point_zero_two_is_exact() {
    let (m, v) = main_fn(|b| {
        let (x, y) = (c(b, 0.1), c(b, 0.2));
        op(b, OpKind::Add, &[x, y])
    });
    for hp in [false, true] {
        let n = known(&m, v, hp);
        // fl(0.1 + 0.2) = 0.30000000000000004; the exact sum of the two
        // doubles is 0.3000000000000000166..., so the error is 2.7756e-17.
        assert!(n.abs >= 2.77e-17 && n.abs < 2.78e-17, "abs {:e}", n.abs);
        assert!(n.rel < 1e-16, "rel {:e}", n.rel);
    }
}

#[test]
fn exact_operations_have_no_error() {
    let (m, v) = main_fn(|b| {
        let (x, y) = (c(b, 0.5), c(b, 0.25));
        let s = op(b, OpKind::Add, &[x, y]);
        let h = c(b, 2.0);
        op(b, OpKind::Div, &[s, h])
    });
    let n = known(&m, v, false);
    assert!(n.is_exact(), "{n:?}");
    assert_eq!(n.range.as_point(), Some(0.375));
}

#[test]
fn subtracting_a_value_from_itself_is_exactly_zero() {
    let (m, v) = main_fn(|b| {
        let x = c(b, 0.1);
        let r = op(b, OpKind::Math(MathFn::Sqrt), &[x]);
        op(b, OpKind::Sub, &[r, r])
    });
    let n = known(&m, v, false);
    assert!(n.is_exact());
}

/// `sqrt(x^2 + 1) - x`, the example of cancellation.
fn sqrt_minus_x(b: &mut FunctionBuilder, x: ValueId) -> ValueId {
    let xx = op(b, OpKind::Mul, &[x, x]);
    let one = c(b, 1.0);
    let s = op(b, OpKind::Add, &[xx, one]);
    let r = op(b, OpKind::Math(MathFn::Sqrt), &[s]);
    op(b, OpKind::Sub, &[r, x])
}

#[test]
fn cancellation_is_violated_for_large_x() {
    let (m, v) = main_fn(|b| {
        let x = c(b, 1e8);
        sqrt_minus_x(b, x)
    });
    // The interpreter computes 0; the true value is 5e-9.
    let n = known(&m, v, true);
    assert_eq!(n.range.as_point(), Some(0.0));
    assert!(n.lower_abs > 4.9e-9 && n.lower_abs < 5.1e-9, "{n:?}");
    assert!(n.lower_rel > 0.99, "{n:?}");
    // Without the enclosure, the analysis proves nothing about the ratio.
    let n = known(&m, v, false);
    assert!(n.rel.is_infinite(), "{n:?}");
    assert!(n.abs > 1e-9, "{n:?}");
}

#[test]
fn cancellation_for_a_range_of_x_is_unbounded_relative() {
    let mut b = FunctionBuilder::new("f", vec![num()], vec![num()], sp());
    let x = b.params()[0];
    let r = sqrt_minus_x(&mut b, x);
    b.op(OpKind::Return, vec![r], vec![], sp());
    let m = module_of(b.finish());
    let a = roundoff::analyze_function(&m, "f", &[Range::new(1e6, 1e9)], &Options::default());
    let n = a.num("f", r).unwrap();
    // The absolute error is bounded (about 1e-7); the relative one is not:
    // the result may be as small as 5e-10.
    assert!(n.abs.is_finite() && n.abs < 1e-6, "{n:?}");
    // 5e-10 result, 1e-7 error: no relative guarantee better than 100%.
    assert!(n.rel.is_infinite(), "{n:?}");
}

#[test]
fn the_conjugate_form_is_accurate() {
    // 1 / (sqrt(x^2 + 1) + x): no subtraction.
    let mut b = FunctionBuilder::new("f", vec![num()], vec![num()], sp());
    let x = b.params()[0];
    let xx = op(&mut b, OpKind::Mul, &[x, x]);
    let one = c(&mut b, 1.0);
    let s = op(&mut b, OpKind::Add, &[xx, one]);
    let r = op(&mut b, OpKind::Math(MathFn::Sqrt), &[s]);
    let d = op(&mut b, OpKind::Add, &[r, x]);
    let q = op(&mut b, OpKind::Div, &[one, d]);
    b.op(OpKind::Return, vec![q], vec![], sp());
    let m = module_of(b.finish());
    let a = roundoff::analyze_function(&m, "f", &[Range::new(1e6, 1e9)], &Options::default());
    let n = a.num("f", q).unwrap();
    assert!(n.rel < 1e-15, "{n:?}");
}

#[test]
fn log_of_one_plus_small_loses_digits() {
    let (m, v) = main_fn(|b| {
        let (one, x) = (c(b, 1.0), c(b, 1e-10));
        let s = op(b, OpKind::Add, &[one, x]);
        op(b, OpKind::Math(MathFn::Log), &[s])
    });
    let n = known(&m, v, true);
    assert!(n.rel > 1e-8, "{n:?}");
    let (m, v) = main_fn(|b| {
        let x = c(b, 1e-10);
        op(b, OpKind::Math(MathFn::Log1p), &[x])
    });
    let n = known(&m, v, true);
    assert!(n.rel < 1e-14, "{n:?}");
}

fn goal(v: ValueId, t: Tolerance) -> Goal {
    Goal {
        function: Module::MAIN.to_string(),
        value: v,
        name: "y".to_string(),
        tolerance: t,
        span: sp(),
    }
}

#[test]
fn the_three_outcomes() {
    // Proven: a product of two constants, one rounding.
    let (m, v) = main_fn(|b| {
        let (x, y) = (c(b, 1.1), c(b, 2.3));
        op(b, OpKind::Mul, &[x, y])
    });
    let a = roundoff::analyze(&m, &Options::default());
    assert!(matches!(
        a.check(&m, &goal(v, Tolerance::Relative(1e-15))),
        Verdict::Proven { .. }
    ));
    // Violated: the cancellation.
    let (m, v) = main_fn(|b| {
        let x = c(b, 1e8);
        sqrt_minus_x(b, x)
    });
    let a = roundoff::analyze(&m, &Options::default());
    assert!(matches!(
        a.check(&m, &goal(v, Tolerance::Relative(1e-12))),
        Verdict::Violated { .. }
    ));
    // Unproven: log(1 + x) with a library function in the way, no lower
    // bound, and an upper bound too weak.
    let (m, v) = main_fn(|b| {
        let (one, x) = (c(b, 1.0), c(b, 1e-10));
        let s = op(b, OpKind::Add, &[one, x]);
        op(b, OpKind::Math(MathFn::Log), &[s])
    });
    let a = roundoff::analyze(&m, &Options::default());
    let Verdict::Unproven { blame, .. } = a.check(&m, &goal(v, Tolerance::Relative(1e-12))) else {
        panic!("expected unproven");
    };
    assert!(!blame.is_empty());
}

// ------------------------------------------------------------ control flow

fn lower(src: &str) -> Module {
    sky_lower::lower_source(src).expect("lowers")
}

/// The value bound to the last assignment of `name` in `<main>`.
fn named(m: &Module, name: &str) -> ValueId {
    let f = m.function(Module::MAIN).unwrap();
    (0..f.values.len() as u32)
        .rev()
        .map(ValueId)
        .find(|v| f.value(*v).name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no value named {name}"))
}

#[test]
fn a_counted_loop_is_unrolled() {
    let m = lower("s = 0.0\nfor i in range(10):\n    s = s + 0.1\n");
    let a = roundoff::analyze(&m, &Options::default());
    let n = a.num(Module::MAIN, named(&m, "s")).unwrap();
    // Ten additions of 0.1: the computed sum is 0.9999999999999999.
    assert!(n.abs.is_finite() && n.abs > 0.0 && n.abs < 1e-15, "{n:?}");
    assert!(n.range.as_point().is_some(), "{n:?}");
}

#[test]
fn a_loop_of_unknown_length_is_unbounded() {
    let m = lower(
        "def f(n):\n    s = 0.0\n    i = 0.0\n    while i < n:\n        s = s + 0.1\n        i = i + 1\n    return s\n",
    );
    let a = roundoff::analyze(&m, &Options::default());
    let f = m.function("f").unwrap();
    let ret = f
        .block(f.entry())
        .ops
        .last()
        .map(|o| f.op(*o).operands[0])
        .unwrap();
    match a.fact("f", ret) {
        Fact::Known(n) => assert!(!n.abs.is_finite() && !n.rel.is_finite(), "{n:?}"),
        Fact::Unknown(_) => {}
        Fact::Unreached => panic!("unreached"),
    }
}

#[test]
fn a_branch_that_rounding_can_flip_is_unbounded() {
    // A comparison the rounding error cannot change: bounded, and only the
    // taken arm counts.
    let m = lower("a = 0.1 + 0.2\nif a > 0.5:\n    r = 1.0\nelse:\n    r = 2.0\n");
    let a = roundoff::analyze(&m, &Options::default());
    let n = a.num(Module::MAIN, named(&m, "r")).unwrap();
    assert!(n.is_exact(), "{n:?}");
    assert_eq!(n.range.as_point(), Some(2.0));
    // A comparison decided by the rounding of the operand: 1e16 + 1 rounds
    // to 1e16, which is not greater than 1e16, but the exact sum is. The
    // exact computation takes the other arm, so nothing bounds `r`.
    let m = lower("a = 1e16 + 1.0\nif a > 1e16:\n    r = 1.0\nelse:\n    r = 2.0\n");
    let a = roundoff::analyze(&m, &Options::default());
    let n = a.num(Module::MAIN, named(&m, "r")).unwrap();
    assert!(!n.abs.is_finite(), "{n:?}");
    assert_eq!(n.range, Range::new(1.0, 2.0));
}

#[test]
fn calls_are_analysed_in_context() {
    let m = lower("def half(x: m) -> m:\n    return x / 2\ny = half(3 m)\n");
    let a = roundoff::analyze(&m, &Options::default());
    let n = a.num(Module::MAIN, named(&m, "y")).unwrap();
    assert!(n.is_exact(), "{n:?}");
}

#[test]
fn unknown_types_and_arrays_give_no_bound() {
    let m = lower("v = [1.0, 2.0]\nx = v[0] / 3.0\n");
    let a = roundoff::analyze(&m, &Options::default());
    match a.fact(Module::MAIN, named(&m, "x")) {
        Fact::Known(n) => assert!(!n.abs.is_finite()),
        Fact::Unknown(_) => {}
        Fact::Unreached => panic!(),
    }
}

// ------------------------------------------------------------ budgets in source

fn lower_goals(src: &str) -> (Module, Vec<Goal>) {
    sky_ast::with_deep_stack(|| {
        let lexed = sky_syntax::lex(src).expect("lexes");
        let (ast, errors) = sky_ast::parse_module(&lexed.tokens, src);
        assert!(errors.is_empty(), "{errors:?}");
        let checked = sky_types::check_module(&ast, src);
        assert!(!checked.has_errors(), "{:?}", checked.diagnostics);
        sky_lower::lower_with_accuracy(&ast, src, &checked).expect("lowers")
    })
}

fn verdicts(src: &str) -> Vec<Verdict> {
    let (m, goals) = lower_goals(src);
    let a = roundoff::analyze(&m, &Options::default());
    goals.iter().map(|g| a.check(&m, g)).collect()
}

#[test]
fn a_goal_is_about_the_value_at_its_statement() {
    // The first `x` is exact; the budget is stated before it is rebound to
    // the cancellation.
    let v = verdicts(
        "x = 0.5\naccuracy x within 1e-15\nx = sqrt(1e16 + 1) - 1e8\naccuracy x within 1e-3\n",
    );
    assert!(matches!(v[0], Verdict::Proven { .. }), "{:?}", v[0]);
    assert!(matches!(v[1], Verdict::Violated { .. }), "{:?}", v[1]);
}

#[test]
fn a_goal_in_a_function_is_analysed_with_the_arguments_of_its_calls() {
    let src = "def f(x: Scalar) -> Scalar:\n    y = sqrt(x^2 + 1) - x\n    accuracy y within 1e-3\n    return y\nr = f(3.0)\n";
    // Called with 3: no cancellation, the budget holds.
    let v = verdicts(src);
    assert!(matches!(v[0], Verdict::Proven { .. }), "{:?}", v[0]);
    // Never called: nothing is known about x.
    let v = verdicts(
        "def f(x: Scalar) -> Scalar:\n    y = sqrt(x^2 + 1) - x\n    accuracy y within 1e-3\n    return y\n",
    );
    assert!(matches!(v[0], Verdict::Unproven { .. }), "{:?}", v[0]);
}

#[test]
fn an_absolute_budget_is_in_si_units() {
    let v = verdicts("d = 1e8 m + 1 mm\naccuracy d within 10 nm\naccuracy d within 1 pm\n");
    assert!(matches!(v[0], Verdict::Proven { .. }), "{:?}", v[0]);
    // 1 pm = 1e-12 m is far below the spacing of an f64 at 1e8: the sum is
    // off by about 2e-9 m.
    assert!(matches!(v[1], Verdict::Violated { .. }), "{:?}", v[1]);
}

#[test]
fn the_flagship_example_names_the_subtraction() {
    let (m, goals) = lower_goals("x = 1e8\ny = sqrt(x^2 + 1) - x\naccuracy y within 1e-12\n");
    let a = roundoff::analyze(&m, &Options::default());
    let Verdict::Violated {
        blame,
        computed,
        exact,
        ..
    } = a.check(&m, &goals[0])
    else {
        panic!("expected a violation");
    };
    assert_eq!(computed, Some(0.0));
    assert!((exact.unwrap() - 5e-9).abs() < 1e-15, "{exact:?}");
    assert_eq!(blame.first().map(|b| b.mnemonic), Some("sub"));
    assert!(matches!(
        blame[0].kind,
        roundoff::BlameKind::Cancellation { .. }
    ));
}
