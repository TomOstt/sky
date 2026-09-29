//! Saturation is bounded: iteration and node limits stop it, and hostile
//! inputs finish quickly and still mean the same thing.

mod common;

use std::time::{Duration, Instant};

use common::*;
use sky_egraph::{rewrite_function, rewrite_function_report, RewriteConfig, RewriteContract, Stop};
use sky_ir::{Function, MathFn, OpKind, ValueId};

/// `v = x; repeat n times { v = (v * 1) - 0 }; return v`
fn chain(n: usize) -> Function {
    let mut t = T::new(vec![num()], vec![num()]);
    let mut v = t.params()[0];
    for _ in 0..n {
        let one = t.rat(1, 1);
        let m = t.bin(OpKind::Mul, v, one);
        let z = t.rat(0, 1);
        v = t.bin(OpKind::Sub, m, z);
    }
    t.ret(&[v])
}

fn same_on(a: &Function, b: &Function, inputs: &[f64]) {
    for x in inputs {
        assert!(same_outcome(&run(a, &[*x]), &run(b, &[*x])), "x = {x}");
    }
}

#[test]
fn a_deep_chain_collapses_quickly() {
    let original = chain(3000);
    let mut f = original.clone();
    let start = Instant::now();
    let report = rewrite_function_report(&mut f, &RewriteConfig::default());
    assert!(
        start.elapsed() < Duration::from_secs(20),
        "{:?}",
        start.elapsed()
    );
    assert_eq!(kinds(&f), ["return"]);
    assert_eq!(report.segments.len(), 1);
    assert_eq!(report.segments[0].stop, Some(Stop::Saturated));
    assert!(!report.rewrites.is_empty());
    same_on(&original, &f, &[0.0, -0.0, 2.5, -1e300]);
}

#[test]
fn the_iteration_limit_stops_saturation_and_the_result_is_still_sound() {
    let original = chain(50);
    let mut f = original.clone();
    let cfg = RewriteConfig {
        iter_limit: 1,
        ..RewriteConfig::default()
    };
    let report = rewrite_function_report(&mut f, &cfg);
    assert_eq!(report.segments[0].iterations, 1);
    assert_eq!(report.segments[0].stop, Some(Stop::IterationLimit));
    same_on(&original, &f, &[0.0, 3.0, -7.5]);
}

#[test]
fn the_node_limit_stops_saturation_and_the_result_is_still_sound() {
    let original = chain(200);
    let mut f = original.clone();
    let cfg = RewriteConfig {
        node_limit: 10,
        ..RewriteConfig::default()
    };
    let report = rewrite_function_report(&mut f, &cfg);
    assert_eq!(report.segments[0].stop, Some(Stop::NodeLimit));
    same_on(&original, &f, &[0.0, 3.0, -7.5]);
}

#[test]
fn a_segment_over_the_size_cap_is_left_alone() {
    let original = chain(100);
    let mut f = original.clone();
    let cfg = RewriteConfig {
        max_segment_ops: 10,
        ..RewriteConfig::default()
    };
    let report = rewrite_function_report(&mut f, &cfg);
    assert_eq!(f, original);
    assert_eq!(report.segments[0].skipped, Some("segment too large"));
}

#[test]
fn a_pathological_power_tower_finishes_quickly() {
    // ((x^2)^2)^2 ... 100 levels: the exponent overflows a rational long
    // before the end, and every level offers a rewrite.
    let mut t = T::new(vec![num()], vec![num()]);
    let mut v = t.params()[0];
    for _ in 0..100 {
        v = t.pow(v, 2, 1);
    }
    let mut f = t.ret(&[v]);
    let start = Instant::now();
    let cfg = RewriteConfig::with_contract(RewriteContract::Fast);
    rewrite_function(&mut f, &cfg);
    assert!(
        start.elapsed() < Duration::from_secs(20),
        "{:?}",
        start.elapsed()
    );
    assert!(kinds(&f).contains(&"pow.const"));
}

#[test]
fn a_wide_program_with_many_identities_finishes_quickly() {
    // 300 independent sin^2 + cos^2 terms summed, under the fast contract.
    let mut t = T::new(vec![num()], vec![num()]);
    let x = t.params()[0];
    let mut acc: Option<ValueId> = None;
    for i in 0..300 {
        let k = t.float(i as f64 + 0.5);
        let a = t.bin(OpKind::Mul, x, k);
        let s = t.math(MathFn::Sin, &[a]);
        let c = t.math(MathFn::Cos, &[a]);
        let s2 = t.pow(s, 2, 1);
        let c2 = t.pow(c, 2, 1);
        let p = t.bin(OpKind::Add, s2, c2);
        acc = Some(match acc {
            None => p,
            Some(prev) => t.bin(OpKind::Add, prev, p),
        });
    }
    let mut f = t.ret(&[acc.unwrap()]);
    let start = Instant::now();
    let rw = rewrite_function(&mut f, &RewriteConfig::with_contract(RewriteContract::Fast));
    assert!(
        start.elapsed() < Duration::from_secs(20),
        "{:?}",
        start.elapsed()
    );
    assert_eq!(rw.iter().filter(|r| r.rule == "TRIG-001").count(), 300);
    assert!(!kinds(&f).contains(&"math.sin"));
    let (r, _) = run(&f, &[0.7]);
    let Ok(v) = r else { panic!("fails") };
    let sky_interp::Value::Number { value, .. } = &v[0] else {
        panic!()
    };
    assert!((value - 300.0).abs() < 1e-9);
}
