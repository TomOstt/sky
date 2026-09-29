//! Tests of the exactness analysis (`sky_ir::exact`) against the reference
//! interpreter, and of what the passes do with it.
//!
//! The analysis makes claims about the `f64` the interpreter computes, so
//! the tests that matter run the interpreter:
//!
//! * [`ranges_enclose_the_interpreter`]: for random ranges and points inside
//!   them, the interpreter's result (when it succeeds) is inside the range
//!   the transfer function gives, and when the transfer function says "cannot
//!   fail" the interpreter does not fail. Covers every arithmetic operation
//!   and every `MathFn`.
//! * [`no_proven_operation_ever_fails`]: on random programs over two
//!   parameters, run on inputs full of edge cases, no failure is ever
//!   reported at an operation the analysis proved infallible.
//! * [`the_pipeline_keeps_behaviour_with_parameters`]: the same programs
//!   before and after the whole default pipeline (`fold`, `egraph`, `dce`,
//!   `cholesky`), which now uses the analysis in `dce` and in the e-graph:
//!   the same values bit for bit, the same output, the same failure at the
//!   same operation. A failing program stays failing.

use std::collections::HashSet;

use proptest::prelude::*;
use sky_diag::Span;
use sky_interp::arith::{arith, pow, reciprocal, ArithOp};
use sky_interp::Value;
use sky_ir::exact::{self, range, Range};
use sky_ir::{FunctionBuilder, MathFn, Module, OpId, OpKind, PassManager, ValueId};
use sky_units::Rational;

use crate::default_passes;
use crate::testutil::{module_of, num, verify};

/// Values that sit on the edges of the interesting cases.
const EDGES: [f64; 24] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    2.0,
    -2.0,
    0.5,
    -0.5,
    3.0,
    -2.5,
    1e-310,
    5e-324,
    -5e-324,
    1e-160,
    1e150,
    -1e150,
    1e200,
    1.7e308,
    -1.7e308,
    709.0,
    710.0,
    -710.0,
    0.999_999_999_999_999_9,
    1.000_000_000_000_000_2,
];

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

/// A range from two edge values, and points inside it.
fn range_and_points() -> impl Strategy<Value = (Range, Vec<f64>)> {
    (0..EDGES.len(), 0..EDGES.len(), 0.0f64..=1.0).prop_map(|(i, j, t)| {
        let (lo, hi) = (EDGES[i].min(EDGES[j]), EDGES[i].max(EDGES[j]));
        let mid = lo / 2.0 + hi / 2.0;
        let lerp = (lo + (hi / 2.0 - lo / 2.0) * 2.0 * t).clamp(lo, hi);
        let mut pts = vec![lo, hi, mid, lerp];
        pts.extend(EDGES.iter().copied().filter(|e| *e >= lo && *e <= hi));
        // Halving a subnormal rounds, so keep only points inside the range.
        pts.retain(|p| lo <= *p && *p <= hi);
        (Range::new(lo, hi), pts)
    })
}

/// The interpreter's answer, or `None` for "fails".
fn interp_binary(op: ArithOp, a: f64, b: f64) -> Option<f64> {
    arith(op, &scalar(a), &scalar(b), sp(), sp(), sp())
        .ok()
        .map(|v| number(&v))
}

fn contains(r: Range, x: f64) -> bool {
    r.lo <= x && x <= r.hi
}

fn check_binary(
    name: &str,
    step: range::Step,
    a: &[f64],
    b: &[f64],
    f: impl Fn(f64, f64) -> Option<f64>,
) -> Result<(), TestCaseError> {
    for &x in a {
        for &y in b {
            match f(x, y) {
                Some(v) => prop_assert!(
                    contains(step.range, v),
                    "{name}({x:e}, {y:e}) = {v:e} outside {:?}",
                    step.range
                ),
                None => prop_assert!(
                    step.may_fail,
                    "{name}({x:e}, {y:e}) failed, but the analysis said it cannot"
                ),
            }
        }
    }
    Ok(())
}

fn check_unary(
    name: &str,
    step: range::Step,
    a: &[f64],
    f: impl Fn(f64) -> Option<f64>,
) -> Result<(), TestCaseError> {
    for &x in a {
        match f(x) {
            Some(v) => prop_assert!(
                contains(step.range, v),
                "{name}({x:e}) = {v:e} outside {:?}",
                step.range
            ),
            None => prop_assert!(
                step.may_fail,
                "{name}({x:e}) failed, but the analysis said it cannot"
            ),
        }
    }
    Ok(())
}

fn interp_math(f: MathFn, args: &[f64]) -> Option<f64> {
    let vals: Vec<Value> = args.iter().map(|x| scalar(*x)).collect();
    let spans = vec![sp(); vals.len()];
    sky_interp::math::apply(f, &vals, sp(), &spans)
        .ok()
        .map(|v| number(&v))
}

fn interp_pow(x: f64, e: Rational) -> Option<f64> {
    let exp = scalar(e.numer() as f64 / e.denom() as f64);
    pow(&scalar(x), &exp, || Some(e), sp(), sp(), sp())
        .ok()
        .map(|v| number(&v))
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 600, ..ProptestConfig::default() })]

    #[test]
    fn ranges_enclose_the_interpreter(
        (ra, pa) in range_and_points(),
        (rb, pb) in range_and_points(),
    ) {
        check_binary("add", range::add(ra, rb), &pa, &pb, |x, y| interp_binary(ArithOp::Add, x, y))?;
        check_binary("sub", range::sub(ra, rb), &pa, &pb, |x, y| interp_binary(ArithOp::Sub, x, y))?;
        check_binary("mul", range::mul(ra, rb), &pa, &pb, |x, y| interp_binary(ArithOp::Mul, x, y))?;
        check_binary("div", range::div(ra, rb), &pa, &pb, |x, y| interp_binary(ArithOp::Div, x, y))?;
        check_unary("neg", range::neg(ra), &pa, |x| Some(-x))?;
        check_unary("inverse", range::recip(ra), &pa, |x| {
            reciprocal("⁻¹", &scalar(x), sp(), sp()).ok().map(|v| number(&v))
        })?;
        for (n, d) in [(0, 1), (1, 1), (2, 1), (3, 1), (-1, 1), (-2, 1), (-3, 1), (64, 1), (65, 1),
                       (1, 2), (1, 3), (-1, 2), (5, 2), (100, 1), (-100, 1)] {
            let e = Rational::new(n, d);
            check_unary(
                &format!("pow {e}"),
                range::pow_const(ra, n as f64 / d as f64),
                &pa,
                |x| interp_pow(x, e),
            )?;
        }
        for f in MathFn::ALL {
            if f.arity() == 1 {
                check_unary(f.name(), range::math(f, &[ra]), &pa, |x| interp_math(f, &[x]))?;
            } else {
                check_binary(f.name(), range::math(f, &[ra, rb]), &pa, &pb, |x, y| {
                    interp_math(f, &[x, y])
                })?;
            }
        }
    }
}

// ------------------------------------------------------------ programs

#[derive(Clone, Debug)]
enum E {
    X,
    Y,
    Float(usize),
    Rat(i64, i64),
    Neg(Box<E>),
    Bin(u8, Box<E>, Box<E>),
    Pow(usize, Box<E>),
    Inv(Box<E>),
    Math1(usize, Box<E>),
    Math2(usize, Box<E>, Box<E>),
}

const POWERS: [(i64, i64); 8] = [
    (2, 1),
    (3, 1),
    (-1, 1),
    (0, 1),
    (1, 2),
    (-2, 1),
    (1, 3),
    (5, 2),
];

fn arity(n: usize) -> Vec<MathFn> {
    MathFn::ALL.into_iter().filter(|f| f.arity() == n).collect()
}

fn leaf() -> impl Strategy<Value = E> {
    prop_oneof![
        3 => Just(E::X),
        2 => Just(E::Y),
        3 => (0..EDGES.len()).prop_map(E::Float),
        2 => prop::sample::select(vec![(1, 2), (2, 1), (3, 1), (7, 2), (1, 3), (0, 1), (-1, 1), (10, 1)])
            .prop_map(|(n, d)| E::Rat(n, d)),
    ]
}

fn expr() -> impl Strategy<Value = E> {
    leaf().prop_recursive(4, 24, 3, |inner| {
        prop_oneof![
            2 => inner.clone().prop_map(|e| E::Neg(Box::new(e))),
            6 => (0u8..4, inner.clone(), inner.clone())
                .prop_map(|(k, a, b)| E::Bin(k, Box::new(a), Box::new(b))),
            2 => (0..POWERS.len(), inner.clone()).prop_map(|(i, e)| E::Pow(i, Box::new(e))),
            2 => inner.clone().prop_map(|e| E::Inv(Box::new(e))),
            4 => (0..arity(1).len(), inner.clone()).prop_map(|(i, e)| E::Math1(i, Box::new(e))),
            2 => (0..arity(2).len(), inner.clone(), inner)
                .prop_map(|(i, a, b)| E::Math2(i, Box::new(a), Box::new(b))),
        ]
    })
}

struct Gen {
    b: FunctionBuilder,
    next: u32,
}

impl Gen {
    fn op(&mut self, kind: OpKind, args: &[ValueId]) -> ValueId {
        self.next += 1;
        // Every operation gets its own span, so an error names it.
        let span = Span::new(self.next * 10, self.next * 10 + 5);
        self.b.op1(kind, args.to_vec(), num(), span)
    }

    fn build(&mut self, e: &E, x: ValueId, y: ValueId) -> ValueId {
        match e {
            E::X => x,
            E::Y => y,
            E::Float(i) => self.op(
                OpKind::ConstFloat {
                    bits: EDGES[*i].to_bits(),
                },
                &[],
            ),
            E::Rat(n, d) => self.op(OpKind::ConstRational(Rational::new(*n, *d)), &[]),
            E::Neg(a) => {
                let a = self.build(a, x, y);
                self.op(OpKind::Neg, &[a])
            }
            E::Bin(k, a, b) => {
                let (a, b) = (self.build(a, x, y), self.build(b, x, y));
                let kind =
                    [OpKind::Add, OpKind::Sub, OpKind::Mul, OpKind::Div][*k as usize].clone();
                self.op(kind, &[a, b])
            }
            E::Pow(i, a) => {
                let a = self.build(a, x, y);
                let (n, d) = POWERS[*i];
                self.op(
                    OpKind::PowConst {
                        exponent: Rational::new(n, d),
                    },
                    &[a],
                )
            }
            E::Inv(a) => {
                let a = self.build(a, x, y);
                self.op(OpKind::Inverse, &[a])
            }
            E::Math1(i, a) => {
                let a = self.build(a, x, y);
                self.op(OpKind::Math(arity(1)[*i]), &[a])
            }
            E::Math2(i, a, b) => {
                let (a, b) = (self.build(a, x, y), self.build(b, x, y));
                self.op(OpKind::Math(arity(2)[*i]), &[a, b])
            }
        }
    }
}

/// `f(x, y)`: computes `e1`, prints it, computes `e2` and `e3`, and returns
/// `e2`. `e3` is unused, so `dce` may remove it if it cannot fail.
fn program(e1: &E, e2: &E, e3: &E) -> Module {
    let mut g = Gen {
        b: FunctionBuilder::new("f", vec![num(), num()], vec![num()], Span::new(0, 5)),
        next: 0,
    };
    let [x, y] = g.b.params()[..] else {
        unreachable!()
    };
    let r1 = g.build(e1, x, y);
    g.next += 1;
    let s = Span::new(g.next * 10, g.next * 10 + 5);
    g.b.op(OpKind::Print, vec![r1], vec![], s);
    let r2 = g.build(e2, x, y);
    let _unused = g.build(e3, x, y);
    g.next += 1;
    let s = Span::new(g.next * 10, g.next * 10 + 5);
    g.b.op(OpKind::Return, vec![r2], vec![], s);
    let mut m = module_of(g.b.finish());
    let _ = sky_ir::effects::infer_effects(&mut m);
    m
}

type Outcome = (Result<Vec<Value>, sky_ir_interp::IrRuntimeError>, String);

fn run(m: &Module, x: f64, y: f64) -> Outcome {
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let r = sky_ir_interp::call(m, "f", vec![scalar(x), scalar(y)], &mut out);
        (r, String::from_utf8_lossy(&out).into_owned())
    })
}

fn same_outcome(a: &Outcome, b: &Outcome) -> bool {
    let same_value = |p: &Value, q: &Value| match (p, q) {
        (Value::Number { value: u, dim: du }, Value::Number { value: v, dim: dv }) => {
            u.to_bits() == v.to_bits() && du == dv
        }
        (p, q) => p == q,
    };
    a.1 == b.1
        && match (&a.0, &b.0) {
            (Ok(u), Ok(v)) => u.len() == v.len() && u.iter().zip(v).all(|(p, q)| same_value(p, q)),
            (Err(u), Err(v)) => u == v,
            _ => false,
        }
}

/// Pairs of parameter values: every edge against a rotation of the edges.
fn inputs() -> Vec<(f64, f64)> {
    let n = EDGES.len();
    (0..n)
        .map(|i| (EDGES[i], EDGES[(i * 7 + 3) % n]))
        .chain([(1.0, 1.0), (0.0, 0.0), (2.5, -3.0)])
        .collect()
}

/// How many reachable operations can fail by their kind.
fn fallible_ops(m: &Module) -> usize {
    let f = &m.functions[0];
    crate::util::reachable_ops(f)
        .into_iter()
        .filter(|o| {
            f.op(*o)
                .kind
                .properties()
                .effects
                .contains(sky_ir::Effect::Fail)
        })
        .count()
}

/// The properties above are only worth something if the generator reaches
/// the cases they are about: operations proved infallible, `dce` removing
/// operations that can fail by kind, and the e-graph firing.
#[test]
fn the_generator_reaches_the_interesting_cases() {
    use proptest::strategy::ValueTree;
    use proptest::test_runner::TestRunner;

    let mut runner = TestRunner::deterministic();
    let strategy = (expr(), expr(), expr());
    let (mut with_proofs, mut dce_removed_fallible, mut egraph_changed, mut total) = (0, 0, 0, 0);
    for _ in 0..600 {
        let (e1, e2, e3) = strategy.new_tree(&mut runner).expect("a tree").current();
        let m = program(&e1, &e2, &e3);
        let facts = exact::analyze(&m.functions[0]);
        let proofs = (0..m.functions[0].ops.len())
            .filter(|i| facts.proof(OpId(*i as u32)).is_some())
            .count();
        with_proofs += usize::from(proofs > 0);
        let mut d = m.clone();
        crate::dce(&mut d);
        dce_removed_fallible += usize::from(fallible_ops(&d) < fallible_ops(&m));
        let mut g = m.clone();
        sky_egraph::rewrite_module(&mut g, &sky_egraph::RewriteConfig::default());
        egraph_changed += usize::from(g != m);
        total += 1;
    }
    assert!(
        with_proofs > total / 4,
        "{with_proofs} of {total} have a proof"
    );
    assert!(
        dce_removed_fallible > total / 20,
        "dce removed a fallible operation in {dce_removed_fallible} of {total}"
    );
    assert!(egraph_changed > total / 10, "{egraph_changed} of {total}");
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 500, ..ProptestConfig::default() })]

    #[test]
    fn no_proven_operation_ever_fails(e1 in expr(), e2 in expr(), e3 in expr()) {
        let m = program(&e1, &e2, &e3);
        verify(&m).map_err(TestCaseError::fail)?;
        let f = &m.functions[0];
        let facts = exact::analyze(f);
        let proven: HashSet<Span> = f
            .ops
            .iter()
            .enumerate()
            .filter(|(i, _)| facts.proof(OpId(*i as u32)).is_some())
            .map(|(_, op)| op.span)
            .collect();
        for (x, y) in inputs() {
            if let (Err(e), _) = run(&m, x, y) {
                prop_assert!(
                    !proven.contains(&e.span),
                    "f({x:e}, {y:e}) failed with `{}` at an operation proved infallible",
                    e.message
                );
            }
        }
    }

    #[test]
    fn the_pipeline_keeps_behaviour_with_parameters(e1 in expr(), e2 in expr(), e3 in expr()) {
        let before = program(&e1, &e2, &e3);
        let mut after = before.clone();
        default_passes(PassManager::for_tests())
            .run(&mut after)
            .map_err(|e| TestCaseError::fail(format!("{e:?}")))?;
        verify(&after).map_err(TestCaseError::fail)?;
        for (x, y) in inputs() {
            let (a, b) = (run(&before, x, y), run(&after, x, y));
            prop_assert!(
                same_outcome(&a, &b),
                "f({x:e}, {y:e}):\n before {a:?}\n after  {b:?}"
            );
        }
    }
}
