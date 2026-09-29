//! Differential property tests: `fold` and `dce` must not change what a
//! program does. Random scalar programs (straight-line and with nested
//! `if`) are run with `sky_ir_interp::run` before and after, and must give
//! the same output and either both succeed or both fail with the same
//! message at the same span.
//!
//! Every operation gets its own span, so "the same span" pins the error to
//! the same operation. The generator deliberately produces failing programs
//! (division by zero, overflow, dimension mismatches, non-real powers) and
//! floats whose sums round (`0.1 + 0.2`), so exactness and error placement
//! are both exercised.

use proptest::prelude::*;
use sky_diag::Span;
use sky_ir::{CmpOp, FunctionBuilder, Module, OpKind, Type, ValueId};
use sky_units::{Dimension, Rational};

use crate::testutil::{module_of, verify};
use crate::{dce, fold, fold_with, FoldMode};

#[derive(Clone, Debug)]
enum Lit {
    Int(i64),
    Ratio(i64, i64),
    Float(usize),
}

const FLOATS: [f64; 12] = [
    0.0, -0.0, 0.1, 0.2, 0.3, 1.0, 2.5, -3.0, 1e300, -1e300, 1e-300, 1.7e308,
];

#[derive(Clone, Debug)]
enum Step {
    Lit(Lit, bool),
    Bin(u8, usize, usize),
    Neg(usize),
    PowConst(usize, i64, i64),
    Pow(usize, usize),
    Inv(usize),
    Cmp(u8, usize, usize),
    Not(usize),
    If(usize, Vec<Step>, Vec<Step>, usize, usize),
    Print(usize),
}

fn lit() -> impl Strategy<Value = Lit> {
    prop_oneof![
        (-6i64..7).prop_map(Lit::Int),
        (-12i64..13, 1i64..13).prop_map(|(n, d)| Lit::Ratio(n, d)),
        Just(Lit::Ratio(i64::MAX, 1)),
        Just(Lit::Ratio(i64::MAX, 3)),
        Just(Lit::Ratio(1, i64::MAX)),
        (0..FLOATS.len()).prop_map(Lit::Float),
    ]
}

fn step(depth: u32) -> BoxedStrategy<Step> {
    let idx = 0usize..64;
    let leaf = prop_oneof![
        3 => (lit(), any::<bool>()).prop_map(|(l, len)| Step::Lit(l, len)),
        5 => (0u8..4, idx.clone(), idx.clone()).prop_map(|(o, a, b)| Step::Bin(o, a, b)),
        1 => idx.clone().prop_map(Step::Neg),
        1 => (idx.clone(), -4i64..5, 1i64..3).prop_map(|(a, n, d)| Step::PowConst(a, n, d)),
        1 => (idx.clone(), idx.clone()).prop_map(|(a, b)| Step::Pow(a, b)),
        1 => idx.clone().prop_map(Step::Inv),
        2 => (0u8..6, idx.clone(), idx.clone()).prop_map(|(o, a, b)| Step::Cmp(o, a, b)),
        1 => idx.clone().prop_map(Step::Not),
        2 => idx.clone().prop_map(Step::Print),
    ];
    if depth == 0 {
        leaf.boxed()
    } else {
        let arms = || proptest::collection::vec(step(depth - 1), 0..5);
        prop_oneof![
            8 => leaf,
            2 => (idx.clone(), arms(), arms(), idx.clone(), idx)
                .prop_map(|(c, t, e, yt, ye)| Step::If(c, t, e, yt, ye)),
        ]
        .boxed()
    }
}

fn program() -> impl Strategy<Value = Vec<Step>> {
    proptest::collection::vec(step(2), 0..24)
}

/// The values in scope: numbers with their dimension, and Bools.
#[derive(Clone, Default)]
struct Pool {
    nums: Vec<(ValueId, Dimension)>,
    bools: Vec<ValueId>,
}

struct Gen {
    b: FunctionBuilder,
    next_span: u32,
}

fn ty(d: Dimension) -> Type {
    Type::scalar(d)
}

impl Gen {
    fn span(&mut self) -> Span {
        self.next_span += 1;
        Span::new(self.next_span, self.next_span + 1)
    }

    fn op1(&mut self, kind: OpKind, args: Vec<ValueId>, t: Type) -> ValueId {
        let s = self.span();
        self.b.op1(kind, args, t, s)
    }

    fn number(&mut self, l: &Lit, d: Dimension) -> ValueId {
        let kind = match l {
            Lit::Int(n) => OpKind::ConstRational(Rational::integer(*n)),
            Lit::Ratio(n, den) => OpKind::ConstRational(Rational::new(*n, *den)),
            Lit::Float(i) => OpKind::ConstFloat {
                bits: FLOATS[*i].to_bits(),
            },
        };
        self.op1(kind, vec![], ty(d))
    }

    fn emit(&mut self, steps: &[Step], pool: &mut Pool) {
        for st in steps {
            self.emit_step(st, pool);
        }
    }

    fn emit_step(&mut self, st: &Step, pool: &mut Pool) {
        // Seed: the pools are never empty.
        if pool.nums.is_empty() {
            let v = self.number(&Lit::Ratio(7, 2), Dimension::DIMENSIONLESS);
            pool.nums.push((v, Dimension::DIMENSIONLESS));
        }
        let pick = |p: &Pool, i: usize| p.nums[i % p.nums.len()];
        match st {
            Step::Lit(l, len) => {
                let d = if *len {
                    Dimension::LENGTH
                } else {
                    Dimension::DIMENSIONLESS
                };
                let v = self.number(l, d);
                pool.nums.push((v, d));
            }
            Step::Bin(o, i, j) => {
                let ((a, da), (b, db)) = (pick(pool, *i), pick(pool, *j));
                let (kind, d) = match o {
                    0 => (OpKind::Add, da),
                    1 => (OpKind::Sub, da),
                    2 => (OpKind::Mul, da.checked_mul(db).unwrap()),
                    _ => (OpKind::Div, da.checked_div(db).unwrap()),
                };
                let v = self.op1(kind, vec![a, b], ty(d));
                pool.nums.push((v, d));
            }
            Step::Neg(i) => {
                let (a, d) = pick(pool, *i);
                let v = self.op1(OpKind::Neg, vec![a], ty(d));
                pool.nums.push((v, d));
            }
            Step::PowConst(i, n, den) => {
                let (a, d) = pick(pool, *i);
                let e = Rational::new(*n, *den);
                let out = d.pow_rat(e).unwrap();
                let v = self.op1(OpKind::PowConst { exponent: e }, vec![a], ty(out));
                pool.nums.push((v, out));
            }
            Step::Pow(i, j) => {
                // A computed exponent needs dimensionless operands.
                let dimless: Vec<_> = pool
                    .nums
                    .iter()
                    .filter(|(_, d)| d.is_dimensionless())
                    .copied()
                    .collect();
                if dimless.is_empty() {
                    return;
                }
                let (a, _) = dimless[*i % dimless.len()];
                let (e, _) = dimless[*j % dimless.len()];
                let v = self.op1(OpKind::Pow, vec![a, e], ty(Dimension::DIMENSIONLESS));
                pool.nums.push((v, Dimension::DIMENSIONLESS));
            }
            Step::Inv(i) => {
                let (a, d) = pick(pool, *i);
                let out = d.inverse().unwrap();
                let v = self.op1(OpKind::Inverse, vec![a], ty(out));
                pool.nums.push((v, out));
            }
            Step::Cmp(o, i, j) => {
                // Well typed: same dimension on both sides.
                let (a, da) = pick(pool, *i);
                let same: Vec<_> = pool.nums.iter().filter(|(_, d)| *d == da).collect();
                let (b, _) = *same[*j % same.len()];
                let c = [
                    CmpOp::Eq,
                    CmpOp::Ne,
                    CmpOp::Lt,
                    CmpOp::Le,
                    CmpOp::Gt,
                    CmpOp::Ge,
                ][*o as usize];
                let v = self.op1(OpKind::Cmp(c), vec![a, b], Type::Bool);
                pool.bools.push(v);
            }
            Step::Not(i) => {
                let c = self.bool_of(pool, *i);
                let v = self.op1(OpKind::Not, vec![c], Type::Bool);
                pool.bools.push(v);
            }
            Step::Print(i) => {
                let (a, _) = pick(pool, *i);
                let s = self.span();
                self.b.op(OpKind::Print, vec![a], vec![], s);
            }
            Step::If(c, then_steps, else_steps, yt, ye) => {
                let cond = self.bool_of(pool, *c);
                let (_, d) = pick(pool, *yt);
                let s = self.span();
                let (iff, res) = self.b.op(OpKind::If, vec![cond], vec![ty(d)], s);
                let outer = self.b.insertion_block();
                for (steps, y) in [(then_steps, *yt), (else_steps, *ye)] {
                    let r = self.b.add_region(iff);
                    let blk = self.b.add_block(r, vec![]);
                    self.b.set_insertion_block(blk);
                    let mut inner = pool.clone();
                    self.emit(steps, &mut inner);
                    // Both arms yield the same dimension.
                    let same: Vec<_> = inner.nums.iter().filter(|(_, x)| *x == d).collect();
                    let v = if same.is_empty() {
                        self.number(&Lit::Int(1), d)
                    } else {
                        same[y % same.len()].0
                    };
                    let s = self.span();
                    self.b.op(OpKind::Yield, vec![v], vec![], s);
                }
                self.b.set_insertion_block(outer);
                pool.nums.push((res[0], d));
            }
        }
    }

    fn bool_of(&mut self, pool: &mut Pool, i: usize) -> ValueId {
        if pool.bools.is_empty() {
            let v = self.op1(OpKind::ConstBool(i % 2 == 0), vec![], Type::Bool);
            pool.bools.push(v);
        }
        pool.bools[i % pool.bools.len()]
    }
}

fn build(steps: &[Step]) -> Module {
    let s = Span::point(0);
    let mut g = Gen {
        b: FunctionBuilder::new(Module::MAIN, vec![], vec![], s),
        next_span: 0,
    };
    let mut pool = Pool::default();
    g.emit(steps, &mut pool);
    // Print everything at the top level, so results are observable.
    for (v, _) in pool.nums.clone() {
        let s = g.span();
        g.b.op(OpKind::Print, vec![v], vec![], s);
    }
    for v in pool.bools.clone() {
        let s = g.span();
        g.b.op(OpKind::Print, vec![v], vec![], s);
    }
    g.b.op(OpKind::Return, vec![], vec![], s);
    module_of(g.b.finish())
}

/// Output text and either success or the failure's message and span.
type Outcome = (String, Result<(), (String, Span)>);

fn outcome(m: &Module) -> Outcome {
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let r = sky_ir_interp::run(m, &mut out);
        (
            String::from_utf8(out).unwrap(),
            r.map_err(|e| (e.message.clone(), e.span)),
        )
    })
}

fn check(steps: &[Step], optimize: impl FnOnce(&mut Module)) -> Result<(), TestCaseError> {
    let before = build(steps);
    verify(&before).map_err(TestCaseError::fail)?;
    let mut after = before.clone();
    optimize(&mut after);
    verify(&after).map_err(TestCaseError::fail)?;
    prop_assert_eq!(outcome(&before), outcome(&after));
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 400, ..ProptestConfig::default() })]

    #[test]
    fn fold_then_dce_preserves_behaviour(steps in program()) {
        check(&steps, |m| {
            fold(m);
            dce(m);
        })?;
    }

    #[test]
    fn fold_alone_preserves_behaviour(steps in program()) {
        check(&steps, |m| {
            fold(m);
        })?;
    }

    #[test]
    fn dce_alone_preserves_behaviour(steps in program()) {
        check(&steps, |m| {
            dce(m);
        })?;
    }

    #[test]
    fn the_pipeline_is_idempotent(steps in program()) {
        let mut m = build(&steps);
        fold(&mut m);
        dce(&mut m);
        prop_assert!(fold(&mut m).is_empty());
        prop_assert!(dce(&mut m).is_empty());
    }

    /// Exact mode may change the last bit of a printed rational result, but
    /// it must never add or remove an error, or move it.
    #[test]
    fn exact_mode_keeps_errors_in_place(steps in program()) {
        let before = build(&steps);
        let mut after = before.clone();
        fold_with(&mut after, FoldMode::Exact);
        dce(&mut after);
        verify(&after).map_err(TestCaseError::fail)?;
        let (_, e0) = outcome(&before);
        let (_, e1) = outcome(&after);
        // A comparison of a rounded and an exact value may flip a branch,
        // so only require agreement when nothing branches on a float.
        if !steps.iter().any(|s| matches!(s, Step::If(..) | Step::Cmp(..))) {
            prop_assert_eq!(e0, e1);
        }
    }
}
