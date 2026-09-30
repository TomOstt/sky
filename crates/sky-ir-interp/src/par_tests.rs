//! Parallel kernels in the interpreter: the same bits and the same error on
//! every thread count and every chunking, and the first failure in
//! sequential order wins when several chunks fail.

use sky_diag::Span;
use sky_ir::parallel::Config;
use sky_ir::{parse, verify, Module};
use sky_units::Dimension;

use crate::{call_with, IrRuntimeError, Value};

fn module(text: &str) -> Module {
    let m = parse(&format!("sir edition 2026\n\n{text}")).unwrap_or_else(|e| panic!("{e}\n{text}"));
    verify(&m).unwrap_or_else(|e| panic!("{e:#?}\n{text}"));
    m
}

fn run(m: &Module, args: Vec<Value>, cfg: &Config) -> Result<Vec<Value>, IrRuntimeError> {
    sky_ast::with_deep_stack(|| call_with(m, "f", args, &mut Vec::new(), cfg))
}

fn vector(xs: &[f64]) -> Value {
    Value::array(vec![xs.len()], xs.to_vec(), Dimension::DIMENSIONLESS)
}

/// The value with its entries as bits, so `-0` and `0` differ.
fn bits(v: &Value) -> Vec<u64> {
    match v {
        Value::Array { data, .. } => data.iter().map(|x| x.to_bits()).collect(),
        Value::Number { value, .. } => vec![value.to_bits()],
        other => panic!("not a number: {other:?}"),
    }
}

/// Every configuration a result must be the same under: sequential, and
/// 2, 4 and 8 threads with the smallest chunks and with the defaults.
fn configs() -> Vec<Config> {
    let mut all = vec![Config::sequential()];
    for t in [2, 4, 8] {
        all.push(Config::eager(t));
        all.push(Config::with_threads(t));
        // Odd chunk sizes: chunk boundaries fall in different places.
        all.push(Config {
            chunk_min: 3,
            ..Config::eager(t)
        });
    }
    all
}

/// Runs `f` under every configuration and requires one result: the same
/// bits, or the same error (message, span, labels, notes).
fn same_everywhere(m: &Module, args: &[Value]) -> Result<Vec<Value>, IrRuntimeError> {
    let want = run(m, args.to_vec(), &Config::sequential());
    for cfg in configs() {
        let got = run(m, args.to_vec(), &cfg);
        match (&want, &got) {
            (Ok(a), Ok(b)) => assert_eq!(
                a.iter().map(bits).collect::<Vec<_>>(),
                b.iter().map(bits).collect::<Vec<_>>(),
                "{cfg:?}"
            ),
            (Err(a), Err(b)) => assert_eq!(a, b, "{cfg:?}"),
            _ => panic!("{cfg:?}: {want:?} vs {got:?}"),
        }
    }
    want
}

const CHAIN: &str =
    "fn @f(%a: num[1; (100)], %b: num[1; (100)], %k: num[1; ()]) -> (num[1; (100)]) effects(fail) {
  %r = fused %a, %b, %k : num[1; (100)] {
    ^1(%x: num[1; ()], %y: num[1; ()], %s: num[1; ()]):
      %t = add %x, %y : num[1; ()]
      %u = mul %t, %s : num[1; ()]
      %w = sub %u, %x : num[1; ()]
      %n = neg %w : num[1; ()]
      yield %n
  }
  return %r
}
";

fn ramp(n: usize, f: impl Fn(usize) -> f64) -> Vec<f64> {
    (0..n).map(f).collect()
}

#[test]
fn a_kernel_is_bit_identical_on_every_configuration() {
    let m = module(CHAIN);
    let a = vector(&ramp(100, |i| i as f64 * 0.1 - 3.3));
    let b = vector(&ramp(100, |i| (i as f64).sin()));
    let out = same_everywhere(&m, &[a, b, Value::scalar(0.37)]).unwrap();
    assert_eq!(bits(&out[0]).len(), 100);
}

#[test]
fn a_negated_zero_stays_positive_in_every_chunk() {
    // `neg` in a kernel is `0 - x`: `+0`, in whichever chunk the zero is.
    let m = module(
        "fn @f(%a: num[1; (64)]) -> (num[1; (64)]) {
  %r = fused %a : num[1; (64)] {
    ^1(%x: num[1; ()]):
      %n = neg %x : num[1; ()]
      yield %n
  }
  return %r
}
",
    );
    let out = same_everywhere(&m, &[vector(&[0.0; 64])]).unwrap();
    assert!(bits(&out[0]).iter().all(|&b| b == 0.0f64.to_bits()));
}

/// `%p / %q` at span 10 and `%r / %s` at span 20, then added: the failing
/// operation that comes first in the body is the error, whatever element and
/// whichever chunk it fails in.
const TWO_DIVISIONS: &str = "fn @f(%a: num[1; (64)], %b: num[1; (64)], %c: num[1; (64)], %d: num[1; (64)]) -> (num[1; (64)]) effects(fail) {
  %r = fused %a, %b, %c, %d : num[1; (64)] {
    ^1(%p: num[1; ()], %q: num[1; ()], %u: num[1; ()], %v: num[1; ()]):
      %first = div %p, %q : num[1; ()] @ 10..11
      %second = div %u, %v : num[1; ()] @ 20..21
      %sum = add %first, %second : num[1; ()] @ 30..31
      yield %sum @ 30..31
  } @ 10..31
  return %r
}
";

fn failure(r: Result<Vec<Value>, IrRuntimeError>) -> (String, Span) {
    let e = r.expect_err("fails");
    (e.message.clone(), e.span)
}

/// 64 ones with zeros at `at`.
fn ones_with_zeros(at: &[usize]) -> Value {
    vector(&ramp(64, |i| if at.contains(&i) { 0.0 } else { 1.0 }))
}

#[test]
fn the_earliest_failing_operation_wins_across_chunks() {
    let m = module(TWO_DIVISIONS);
    let ones = ones_with_zeros(&[]);
    // First division fails only at element 60 (a late chunk), the second at
    // element 3 (the first chunk): the first division is the error.
    let r = same_everywhere(
        &m,
        &[
            ones.clone(),
            ones_with_zeros(&[60]),
            ones.clone(),
            ones_with_zeros(&[3]),
        ],
    );
    assert_eq!(
        failure(r),
        ("division by zero".to_string(), Span::new(10, 11))
    );
    // Only the second fails, in several chunks: its error.
    let r = same_everywhere(
        &m,
        &[
            ones.clone(),
            ones.clone(),
            ones.clone(),
            ones_with_zeros(&[3, 20, 40, 63]),
        ],
    );
    assert_eq!(
        failure(r),
        ("division by zero".to_string(), Span::new(20, 21))
    );
    // The same operation fails in every chunk: one error, the same one.
    let all: Vec<usize> = (0..64).collect();
    let r = same_everywhere(
        &m,
        &[
            ones.clone(),
            ones_with_zeros(&all),
            ones.clone(),
            ones_with_zeros(&all),
        ],
    );
    assert_eq!(
        failure(r),
        ("division by zero".to_string(), Span::new(10, 11))
    );
    // Neither fails.
    let out = same_everywhere(&m, &[ones.clone(), ones.clone(), ones.clone(), ones]).unwrap();
    assert!(bits(&out[0]).iter().all(|&b| b == 2.0f64.to_bits()));
}

#[test]
fn an_overflow_in_a_late_chunk_does_not_hide_an_earlier_operation() {
    let m = module(
        "fn @f(%a: num[1; (64)], %b: num[1; (64)]) -> (num[1; (64)]) effects(fail) {
  %r = fused %a, %b : num[1; (64)] {
    ^1(%x: num[1; ()], %y: num[1; ()]):
      %first = add %x, %x : num[1; ()] @ 1..2
      %second = add %y, %y : num[1; ()] @ 3..4
      %sum = sub %first, %second : num[1; ()] @ 5..6
      yield %sum @ 5..6
  }
  return %r
}
",
    );
    let big = 1.7e308;
    let spike = |at: usize| vector(&ramp(64, |i| if i == at { big } else { 1.0 }));
    // `second` overflows at element 2 (first chunk), `first` at element 62
    // (last chunk): `first` is earlier in the body.
    let r = same_everywhere(&m, &[spike(62), spike(2)]);
    assert_eq!(
        failure(r),
        ("arithmetic overflow".to_string(), Span::new(1, 2))
    );
    // And the other way round.
    let r = same_everywhere(&m, &[spike(2), spike(62)]);
    assert_eq!(
        failure(r),
        ("arithmetic overflow".to_string(), Span::new(1, 2))
    );
}

#[test]
fn elementwise_array_operations_are_bit_identical_and_fail_alike() {
    // add, sub, mul by a scalar, div by a scalar, neg, all on arrays.
    let m = module(
        "fn @f(%a: num[1; (200)], %b: num[1; (200)], %k: num[1; ()]) -> (num[1; (200)]) effects(fail) {
  %s = add %a, %b : num[1; (200)] @ 1..2
  %d = sub %s, %b : num[1; (200)] @ 3..4
  %m = mul %d, %k : num[1; (200)] @ 5..6
  %q = div %m, %k : num[1; (200)] @ 7..8
  %n = neg %q : num[1; (200)] @ 9..10
  %p = mul %k, %n : num[1; (200)] @ 11..12
  return %p
}
",
    );
    let a = vector(&ramp(200, |i| i as f64 - 99.5));
    let b = vector(&ramp(200, |i| 1.0 / (1.0 + i as f64)));
    let out = same_everywhere(&m, &[a.clone(), b.clone(), Value::scalar(3.0)]).unwrap();
    assert_eq!(bits(&out[0]).len(), 200);
    // A zero divisor is the reference's error, not the loop's.
    let r = same_everywhere(&m, &[a.clone(), b.clone(), Value::scalar(0.0)]);
    assert_eq!(failure(r).0, "division by zero");
    // An overflow in one entry is one error for the operation.
    let big = vector(&ramp(200, |i| if i == 150 { 1.7e308 } else { 1.0 }));
    let r = same_everywhere(&m, &[big.clone(), big, Value::scalar(3.0)]);
    assert_eq!(
        failure(r),
        ("arithmetic overflow".to_string(), Span::new(1, 2))
    );
}

#[test]
fn a_matrix_and_a_dimensioned_array_keep_their_shape_and_unit() {
    let m = module(
        "fn @f(%a: num[L; (8, 9)], %t: num[T; ()]) -> (num[L·T⁻¹; (8, 9)]) effects(fail) {
  %v = div %a, %t : num[L·T⁻¹; (8, 9)]
  return %v
}
",
    );
    let a = Value::array(vec![8, 9], ramp(72, |i| i as f64), Dimension::LENGTH);
    let t = Value::number(4.0, Dimension::TIME);
    let want = run(&m, vec![a.clone(), t.clone()], &Config::sequential()).unwrap();
    let got = run(&m, vec![a, t], &Config::eager(4)).unwrap();
    assert_eq!(want, got);
}

#[test]
fn thresholds_decide_nothing_but_speed() {
    // Below the thresholds the sequential loop runs; the result is the same.
    let m = module(CHAIN);
    let a = vector(&ramp(100, |i| i as f64));
    let b = vector(&ramp(100, |i| -(i as f64)));
    let args = vec![a, b, Value::scalar(2.0)];
    let below = run(&m, args.clone(), &Config::with_threads(8)).unwrap();
    let above = run(&m, args, &Config::eager(8)).unwrap();
    assert_eq!(below, above);
}
