//! `fused` in the reference interpreter: each element runs the body, the
//! failure is the first failing operation's, and malformed kernels are
//! errors, never panics.

use sky_diag::Span;
use sky_ir::{parse, verify, Module};
use sky_units::Dimension;

use crate::{call, IrRuntimeError, Value};

fn module(text: &str) -> Module {
    let m = parse(&format!("sir edition 2026\n\n{text}")).unwrap_or_else(|e| panic!("{e}\n{text}"));
    verify(&m).unwrap_or_else(|e| panic!("{e:#?}\n{text}"));
    m
}

fn run(m: &Module, args: Vec<Value>) -> Result<Vec<Value>, IrRuntimeError> {
    sky_ast::with_deep_stack(|| call(m, "f", args, &mut Vec::new()))
}

fn vector(xs: &[f64]) -> Value {
    Value::array(vec![xs.len()], xs.to_vec(), Dimension::DIMENSIONLESS)
}

fn data(v: &Value) -> Vec<f64> {
    match v {
        Value::Array { data, .. } => data.clone(),
        other => panic!("not an array: {other:?}"),
    }
}

#[test]
fn every_element_runs_the_body_and_scalars_broadcast() {
    let m = module(
        "fn @f(%a: num[1; (3)], %b: num[1; (3)], %k: num[1; ()]) -> (num[1; (3)]) effects(fail) {
  %r = fused %a, %b, %k : num[1; (3)] {
    ^1(%x: num[1; ()], %y: num[1; ()], %s: num[1; ()]):
      %t = add %x, %y : num[1; ()]
      %u = mul %t, %s : num[1; ()]
      %w = sub %u, %x : num[1; ()]
      yield %w
  }
  return %r
}
",
    );
    let out = run(
        &m,
        vec![
            vector(&[1.0, 2.0, 3.0]),
            vector(&[4.0, 5.0, 6.0]),
            Value::scalar(2.0),
        ],
    )
    .unwrap();
    assert_eq!(data(&out[0]), [9.0, 12.0, 15.0]);
}

#[test]
fn the_result_has_the_dimension_of_the_body() {
    let m = module(
        "fn @f(%a: num[L; (2)], %t: num[T; ()]) -> (num[L·T⁻¹; (2)]) effects(fail) {
  %r = fused %a, %t : num[L·T⁻¹; (2)] {
    ^1(%x: num[L; ()], %s: num[T; ()]):
      %v = div %x, %s : num[L·T⁻¹; ()]
      yield %v
  }
  return %r
}
",
    );
    let metres = Value::array(vec![2], vec![10.0, 20.0], Dimension::LENGTH);
    let seconds = Value::number(5.0, Dimension::TIME);
    let out = run(&m, vec![metres, seconds]).unwrap();
    let Value::Array { data, dim, .. } = &out[0] else {
        panic!("{:?}", out[0])
    };
    assert_eq!(data, &[2.0, 4.0]);
    assert_eq!(
        *dim,
        Dimension::LENGTH.checked_div(Dimension::TIME).unwrap()
    );
}

#[test]
fn a_matrix_keeps_its_shape() {
    let m = module(
        "fn @f(%a: num[1; (2, 2)]) -> (num[1; (2, 2)]) {
  %r = fused %a : num[1; (2, 2)] {
    ^1(%x: num[1; ()]):
      %n = neg %x : num[1; ()]
      yield %n
  }
  return %r
}
",
    );
    let a = Value::array(
        vec![2, 2],
        vec![1.0, -2.0, 3.0, -4.0],
        Dimension::DIMENSIONLESS,
    );
    let out = run(&m, vec![a]).unwrap();
    let Value::Array { dims, data, .. } = &out[0] else {
        panic!("{:?}", out[0])
    };
    assert_eq!(
        (dims.as_slice(), data.as_slice()),
        (&[2, 2][..], &[-1.0, 2.0, -3.0, 4.0][..])
    );
}

#[test]
fn neg_is_zero_minus_x_as_on_arrays() {
    let m = module(
        "fn @f(%a: num[1; (2)]) -> (num[1; (2)]) {
  %r = fused %a : num[1; (2)] {
    ^1(%x: num[1; ()]):
      %n = neg %x : num[1; ()]
      yield %n
  }
  %u = neg %a : num[1; (2)]
  return %r
}
",
    );
    let out = run(&m, vec![vector(&[0.0, 5.0])]).unwrap();
    // `0 - 0` is `+0`: the array `neg`'s result, not the scalar `-0`.
    assert_eq!(data(&out[0])[0].to_bits(), 0.0f64.to_bits());
    let unfused = module(
        "fn @f(%a: num[1; (2)]) -> (num[1; (2)]) {
  %u = neg %a : num[1; (2)]
  return %u
}
",
    );
    let expected = run(&unfused, vec![vector(&[0.0, 5.0])]).unwrap();
    assert_eq!(
        data(&out[0])
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        data(&expected[0])
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
}

/// `%p / %q` at spans 10 and `%r / %s` at span 20, both on element
/// arguments, then added: which one fails is decided by their order in the
/// body, not by the element they fail at.
const TWO_DIVISIONS: &str = "fn @f(%a: num[1; (3)], %b: num[1; (3)], %c: num[1; (3)], %d: num[1; (3)]) -> (num[1; (3)]) effects(fail) {
  %r = fused %a, %b, %c, %d : num[1; (3)] {
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

#[test]
fn the_earliest_failing_operation_is_reported_whatever_the_element() {
    let m = module(TWO_DIVISIONS);
    let ones = vector(&[1.0, 1.0, 1.0]);
    // First division fails at element 2, second at element 0.
    let r = run(
        &m,
        vec![
            ones.clone(),
            vector(&[1.0, 1.0, 0.0]),
            ones.clone(),
            vector(&[0.0, 1.0, 1.0]),
        ],
    );
    assert_eq!(
        failure(r),
        ("division by zero".to_string(), Span::new(10, 11))
    );
    // Only the second fails: its error.
    let r = run(
        &m,
        vec![
            ones.clone(),
            ones.clone(),
            ones.clone(),
            vector(&[1.0, 0.0, 1.0]),
        ],
    );
    assert_eq!(
        failure(r),
        ("division by zero".to_string(), Span::new(20, 21))
    );
    // Both fail at the same element: the earlier one.
    let r = run(
        &m,
        vec![
            ones.clone(),
            vector(&[1.0, 0.0, 1.0]),
            ones.clone(),
            vector(&[1.0, 0.0, 1.0]),
        ],
    );
    assert_eq!(
        failure(r),
        ("division by zero".to_string(), Span::new(10, 11))
    );
    // Neither: the sum.
    let out = run(&m, vec![ones.clone(), ones.clone(), ones.clone(), ones]).unwrap();
    assert_eq!(data(&out[0]), [2.0, 2.0, 2.0]);
}

#[test]
fn an_overflow_in_a_later_operation_does_not_hide_an_earlier_one() {
    let m = module(
        "fn @f(%a: num[1; (3)], %b: num[1; (3)]) -> (num[1; (3)]) effects(fail) {
  %r = fused %a, %b : num[1; (3)] {
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
    // `second` overflows at element 0, `first` at element 2.
    let r = run(&m, vec![vector(&[1.0, 1.0, big]), vector(&[big, 1.0, 1.0])]);
    assert_eq!(
        failure(r),
        ("arithmetic overflow".to_string(), Span::new(1, 2))
    );
}

#[test]
fn malformed_kernels_are_errors_not_panics() {
    // No verification here: the interpreter must not rely on it.
    let build = |body: &str, args: &str| {
        parse(&format!(
            "sir edition 2026\n\nfn @f(%a: num[1; (3)], %b: num[1; (2)], %s: num[1; ()]) -> (num[1; (3)]) effects(fail) {{
  %r = fused {args} : num[1; (3)] {{
{body}
  }}
  return %r
}}
"
        ))
        .unwrap()
    };
    let a = vector(&[1.0, 2.0, 3.0]);
    let b = vector(&[1.0, 2.0]);
    let s = Value::scalar(1.0);
    let cases = [
        // No yield.
        (
            "    ^1(%x: num[1; ()]):
      %n = neg %x : num[1; ()]
      yield",
            "%a",
        ),
        // Wrong argument count.
        (
            "    ^1(%x: num[1; ()], %y: num[1; ()]):
      yield %x",
            "%a",
        ),
        // Arrays of different shapes.
        (
            "    ^1(%x: num[1; ()], %y: num[1; ()]):
      yield %x",
            "%a, %b",
        ),
        // Only scalars.
        (
            "    ^1(%x: num[1; ()]):
      yield %x",
            "%s",
        ),
        // A yield of a value that is not a number.
        (
            "    ^1(%x: num[1; ()]):
      %t = cmp.lt %x, %x : bool
      yield %t",
            "%a",
        ),
    ];
    for (body, args) in cases {
        let m = build(body, args);
        let e = run(&m, vec![a.clone(), b.clone(), s.clone()]).unwrap_err();
        assert!(e.message.starts_with("malformed IR"), "{}", e.message);
    }
}
