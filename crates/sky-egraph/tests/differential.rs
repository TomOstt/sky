//! Differential property tests: random pure scalar programs are run on the
//! reference interpreter before and after rewriting. Under the default
//! (reproducible) contract, only exact and conditional rules run, so the
//! outputs must be bit-identical and the errors identical (message, span,
//! labels), and so must the text printed before a failure.

mod common;

use common::*;
use proptest::prelude::*;
use sky_egraph::{rewrite_function, RewriteConfig, RewriteContract};
use sky_ir::{Function, MathFn, OpKind, ValueId};

#[derive(Clone, Debug)]
enum E {
    X,
    Y,
    Float(f64),
    Rat(i64, i64),
    Neg(Box<E>),
    Bin(u8, Box<E>, Box<E>),
    Pow(i64, i64, Box<E>),
    Math1(MathFn, Box<E>),
    Math2(MathFn, Box<E>, Box<E>),
    /// `e - e`, `e / e`, `e * e`: the same subterm twice, so the e-graph
    /// sees them as one class.
    SelfOp(u8, Box<E>),
}

fn leaf() -> impl Strategy<Value = E> {
    prop_oneof![
        Just(E::X),
        Just(E::Y),
        prop::sample::select(vec![0.0, -0.0, 1.0, -1.0, 2.0, 0.5, 1e200, -3.0]).prop_map(E::Float),
        prop::sample::select(vec![(0, 1), (1, 1), (1, 2), (2, 1), (-1, 1)])
            .prop_map(|(n, d)| E::Rat(n, d)),
    ]
}

fn expr() -> impl Strategy<Value = E> {
    leaf().prop_recursive(5, 40, 3, |inner| {
        prop_oneof![
            inner.clone().prop_map(|e| E::Neg(Box::new(e))),
            (0u8..4, inner.clone(), inner.clone()).prop_map(|(k, a, b)| E::Bin(
                k,
                Box::new(a),
                Box::new(b)
            )),
            (
                prop::sample::select(vec![(1, 1), (2, 1), (3, 1), (0, 1), (-1, 1), (1, 2)]),
                inner.clone()
            )
                .prop_map(|((n, d), e)| E::Pow(n, d, Box::new(e))),
            (
                prop::sample::select(vec![
                    MathFn::Sqrt,
                    MathFn::Abs,
                    MathFn::Exp,
                    MathFn::Log,
                    MathFn::Sin,
                    MathFn::Cos,
                    MathFn::Cbrt,
                    MathFn::Floor,
                ]),
                inner.clone()
            )
                .prop_map(|(f, e)| E::Math1(f, Box::new(e))),
            (
                prop::sample::select(vec![MathFn::Min, MathFn::Max, MathFn::Hypot, MathFn::Atan2]),
                inner.clone(),
                inner.clone()
            )
                .prop_map(|(f, a, b)| E::Math2(f, Box::new(a), Box::new(b))),
            (0u8..3, inner).prop_map(|(k, e)| E::SelfOp(k, Box::new(e))),
        ]
    })
}

fn build(t: &mut T, e: &E, x: ValueId, y: ValueId) -> ValueId {
    match e {
        E::X => x,
        E::Y => y,
        E::Float(v) => t.float(*v),
        E::Rat(n, d) => t.rat(*n, *d),
        E::Neg(a) => {
            let a = build(t, a, x, y);
            t.un(OpKind::Neg, a)
        }
        E::Bin(k, a, b) => {
            let a = build(t, a, x, y);
            let b = build(t, b, x, y);
            let kind = [OpKind::Add, OpKind::Sub, OpKind::Mul, OpKind::Div][*k as usize].clone();
            t.bin(kind, a, b)
        }
        E::Pow(n, d, a) => {
            let a = build(t, a, x, y);
            t.pow(a, *n, *d)
        }
        E::Math1(f, a) => {
            let a = build(t, a, x, y);
            t.math(*f, &[a])
        }
        E::Math2(f, a, b) => {
            let a = build(t, a, x, y);
            let b = build(t, b, x, y);
            t.math(*f, &[a, b])
        }
        E::SelfOp(k, a) => {
            let a1 = build(t, a, x, y);
            let a2 = build(t, a, x, y);
            let kind = [OpKind::Sub, OpKind::Div, OpKind::Mul][*k as usize].clone();
            t.bin(kind, a1, a2)
        }
    }
}

/// `f(x, y)`: computes `e1`, prints it, computes `e2`, returns both.
fn program(e1: &E, e2: &E) -> Function {
    let mut t = T::new(vec![num(), num()], vec![num(), num()]);
    let [x, y] = t.params()[..] else {
        unreachable!()
    };
    let r1 = build(&mut t, e1, x, y);
    t.print(&[r1]);
    let r2 = build(&mut t, e2, x, y);
    t.ret(&[r1, r2])
}

const INPUTS: &[f64] = &[
    0.0, -0.0, 1.0, -1.0, 2.5, -3.0, 0.5, 1e-200, 1e200, -1e200, 700.0,
];

fn no_malformed(f: &Function, args: &[f64]) -> bool {
    match run(f, args).0 {
        Err(e) => !e.message.starts_with("malformed IR"),
        Ok(_) => true,
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 800, failure_persistence: None, ..ProptestConfig::default() })]

    #[test]
    fn exact_rules_preserve_results_and_failures(e1 in expr(), e2 in expr()) {
        let original = program(&e1, &e2);
        let mut rewritten = original.clone();
        let cfg = RewriteConfig::default();
        assert_eq!(cfg.contract, RewriteContract::Reproducible);
        rewrite_function(&mut rewritten, &cfg);
        for &a in INPUTS {
            for &b in INPUTS {
                let before = run(&original, &[a, b]);
                let after = run(&rewritten, &[a, b]);
                prop_assert!(
                    same_outcome(&before, &after),
                    "x = {a}, y = {b}\nbefore: {before:?}\nafter: {after:?}\nrewritten: {:?}",
                    kinds(&rewritten)
                );
            }
        }
    }

    /// The fast contract may change bits and failures, but must never
    /// produce malformed IR.
    #[test]
    fn fast_rules_produce_well_formed_ir(e1 in expr(), e2 in expr()) {
        let mut rewritten = program(&e1, &e2);
        rewrite_function(&mut rewritten, &RewriteConfig::with_contract(RewriteContract::Fast));
        for &a in INPUTS {
            prop_assert!(no_malformed(&rewritten, &[a, 2.0]));
        }
    }
}

/// The property above would pass if rewriting never did anything, so check
/// that it does, on a program the exact rules improve.
#[test]
fn the_generator_reaches_the_rules() {
    // (x - x) + ((y * 1) - 0)
    let e = E::Bin(
        0,
        Box::new(E::SelfOp(0, Box::new(E::X))),
        Box::new(E::Bin(
            1,
            Box::new(E::Bin(2, Box::new(E::Y), Box::new(E::Rat(1, 1)))),
            Box::new(E::Rat(0, 1)),
        )),
    );
    let original = program(&e, &E::X);
    let mut rewritten = original.clone();
    let rw = rewrite_function(&mut rewritten, &RewriteConfig::default());
    assert!(!rw.is_empty());
    assert!(kinds(&rewritten).len() < kinds(&original).len());
    for &a in INPUTS {
        assert!(same_outcome(
            &run(&original, &[a, 2.0]),
            &run(&rewritten, &[a, 2.0])
        ));
    }
}
