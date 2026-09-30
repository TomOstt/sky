//! Native kernels for `fused` against the interpreter's loop: the same
//! result to the bit and the same error, on random bodies and data that
//! overflows, divides by zero, leaves a function's domain, or is `-0`,
//! subnormal or huge.

use std::sync::atomic::Ordering;

use proptest::prelude::*;
use proptest::test_runner::{Config, TestRng, TestRunner};
use sky_ir::{parse, MathFn, Module};
use sky_units::Dimension;

use crate::native::{Jit, Native, NATIVE_RUNS};
use crate::{call_with_jit, IrRuntimeError, JitOptions, Value};

fn module(dims: bool, body: &str) -> Module {
    let (a, k) = if dims { ("L", "T") } else { ("1", "1") };
    let text = format!(
        "sir edition 2026\n\nfn @f(%a: num[{a}; (3)], %b: num[{a}; (3)], %k: num[{k}; ()]) -> (num[1; (3)]) effects(fail) {{
  %r = fused %a, %b, %k : num[1; (3)] {{
    ^1(%x: num[{a}; ()], %y: num[{a}; ()], %s: num[{k}; ()]):
{body}
  }}
  return %r
}}
"
    );
    parse(&text).unwrap_or_else(|e| panic!("{e}\n{text}"))
}

/// A value's bits, so `-0` and `+0` differ and a result compares exactly.
#[derive(Debug, PartialEq)]
enum Bits {
    Array(Vec<usize>, Vec<u64>, Dimension),
    Number(u64, Dimension),
    Other(String),
}

type Outcome = Result<Vec<Bits>, IrRuntimeError>;

fn bits(v: &Value) -> Bits {
    match v {
        Value::Array { dims, data, dim } => Bits::Array(
            dims.clone(),
            data.iter().map(|x| x.to_bits()).collect(),
            *dim,
        ),
        Value::Number { value, dim } => Bits::Number(value.to_bits(), *dim),
        other => Bits::Other(format!("{other:?}")),
    }
}

fn run(m: &Module, args: &[Value], jit: JitOptions) -> Outcome {
    sky_ast::with_deep_stack(|| {
        call_with_jit(m, "f", args.to_vec(), &mut Vec::new(), jit)
            .map(|vs| vs.iter().map(bits).collect())
    })
}

fn array(xs: &[f64], dim: Dimension) -> Value {
    Value::array(vec![xs.len()], xs.to_vec(), dim)
}

/// A body over `x`, `y`, `s`, from a list of choices: each step picks an
/// operation and operands among the values so far.
#[derive(Clone, Debug)]
struct Step {
    op: u8,
    a: usize,
    b: usize,
}

fn body_text(steps: &[Step]) -> String {
    let mut pool: Vec<String> = vec!["%x".into(), "%y".into(), "%s".into()];
    let mut text = String::new();
    let mut at = 100u32;
    for (i, st) in steps.iter().enumerate() {
        let name = format!("%t{i}");
        let (a, b) = (
            pool[st.a % pool.len()].clone(),
            pool[st.b % pool.len()].clone(),
        );
        let span = format!("@ {at}..{}", at + 5);
        at += 10;
        let line = match st.op {
            0 => format!(
                "{name} = const.float {:e} : num[1; ()]",
                [0.5, 2.0, -3.0, 0.1, 1e154, 1e-320, 0.0, 7.0][st.a % 8]
            ),
            1 => format!("{name} = const.rational {}/3 : num[1; ()]", 1 + st.a % 5),
            2 | 3 => format!("{name} = neg {a} : num[1; ()] {span} (1..2)"),
            4..=9 => {
                let op = ["add", "sub", "mul", "div", "add", "mul"][st.op as usize - 4];
                format!("{name} = {op} {a}, {b} : num[1; ()] {span} (3..4, 5..6)")
            }
            k => {
                let f = MathFn::ALL[(k as usize + st.a) % MathFn::ALL.len()];
                if f.arity() == 2 {
                    format!("{name} = {} {a}, {b} : num[1; ()] {span}", f.mnemonic())
                } else {
                    format!("{name} = {} {a} : num[1; ()] {span}", f.mnemonic())
                }
            }
        };
        text += &format!("      {line}\n");
        pool.push(name);
    }
    let last = pool.last().cloned().unwrap_or_else(|| "%x".into());
    text += &format!("      yield {last}");
    text
}

fn steps() -> impl Strategy<Value = Vec<Step>> {
    let step = (0u8..40, 0usize..64, 0usize..64).prop_map(|(op, a, b)| Step { op, a, b });
    proptest::collection::vec(step, 1..10)
}

/// Finite numbers that stress the operations.
fn number() -> impl Strategy<Value = f64> {
    prop_oneof![
        2 => Just(0.0),
        2 => Just(-0.0),
        4 => (-8i32..9).prop_map(f64::from),
        4 => -10.0..10.0f64,
        2 => -1e3..1e3f64,
        1 => (1u32..1000).prop_map(|k| f64::from(k) * 5e-324),
        1 => (1.0..2.0f64).prop_map(|x| x * 1e154),
        1 => (-2.0..-1.0f64).prop_map(|x| x * 1e300),
        1 => (1.0..2.0f64).prop_map(|x| x * 1e-300),
    ]
}

fn case(dims: bool, steps: &[Step], xs: &[f64], ys: &[f64], s: f64) -> Result<(), TestCaseError> {
    let m = module(dims, &body_text(steps));
    let (l, t) = if dims {
        (Dimension::LENGTH, Dimension::TIME)
    } else {
        (Dimension::DIMENSIONLESS, Dimension::DIMENSIONLESS)
    };
    let args = [array(xs, l), array(ys, l), Value::number(s, t)];
    let want = run(&m, &args, JitOptions::OFF);
    let got = run(&m, &args, JitOptions::ALWAYS);
    prop_assert_eq!(&got, &want, "\n{}", body_text(steps));
    Ok(())
}

#[test]
fn random_bodies_run_natively_exactly_as_the_interpreter_runs_them() {
    let before = NATIVE_RUNS.load(Ordering::Relaxed);
    let mut runner = TestRunner::new_with_rng(
        Config {
            cases: 500,
            failure_persistence: None,
            ..Config::default()
        },
        TestRng::deterministic_rng(proptest::test_runner::RngAlgorithm::ChaCha),
    );
    let failed = std::cell::Cell::new(0usize);
    let strategy = (
        any::<bool>(),
        steps(),
        1usize..40,
        proptest::collection::vec(number(), 40),
        proptest::collection::vec(number(), 40),
        number(),
    );
    runner
        .run(&strategy, |(dims, steps, n, xs, ys, s)| {
            // Small arrays too, so the scalar tail and the vector loop meet.
            case(dims, &steps, &xs[..n], &ys[..n], s)?;
            // A run that fails is one of the cases worth having.
            let m = module(dims, &body_text(&steps));
            let args = [
                array(&xs[..n], Dimension::DIMENSIONLESS),
                array(&ys[..n], Dimension::DIMENSIONLESS),
                Value::scalar(s),
            ];
            if run(&m, &args, JitOptions::OFF).is_err() {
                failed.set(failed.get() + 1);
            }
            Ok(())
        })
        .unwrap_or_else(|e| panic!("{e}"));
    let native = NATIVE_RUNS.load(Ordering::Relaxed) - before;
    assert!(native >= 250, "only {native} runs were native");
    assert!(failed.get() >= 60, "only {} runs failed", failed.get());
}

#[test]
fn the_ops_the_fuse_pass_makes_run_natively_at_every_length() {
    // `a * x + y` and a chain with a division, like the benchmarks'.
    let body = "      %t = mul %x, %s : num[1; ()] @ 10..15
      %u = add %t, %y : num[1; ()] @ 10..20
      %v = div %u, %s : num[1; ()] @ 10..25 (10..20, 24..25)
      yield %v";
    let m = module(false, body);
    for n in [1, 2, 3, 7, 8, 9, 15, 16, 17, 100, 1000, 4097] {
        let xs: Vec<f64> = (0..n).map(|i| i as f64 * 0.37 - 5.0).collect();
        let ys: Vec<f64> = (0..n).map(|i| 1.0 / (1.0 + i as f64)).collect();
        for s in [3.0, 0.0, -0.0] {
            let args = [
                array(&xs, Dimension::DIMENSIONLESS),
                array(&ys, Dimension::DIMENSIONLESS),
                Value::scalar(s),
            ];
            let want = run(&m, &args, JitOptions::OFF);
            assert_eq!(run(&m, &args, JitOptions::ALWAYS), want, "n={n} s={s}");
            if s == 0.0 {
                assert!(want.is_err());
            }
        }
    }
}

/// Bodies made of the entry-by-entry operations a user writes over arrays:
/// products and quotients of two arrays' entries, and math functions of
/// arrays and of an array and a number.
const ENTRYWISE_BODIES: [&str; 7] = [
    // sqrt(x*x + y*y) * s + x
    "      %xx = mul %x, %x : num[1; ()] @ 10..15 (10..11, 12..13)
      %yy = mul %y, %y : num[1; ()] @ 20..25 (20..21, 22..23)
      %q = add %xx, %yy : num[1; ()] @ 30..35 (10..15, 20..25)
      %rt = math.sqrt %q : num[1; ()] @ 40..45 (30..35)
      %c = mul %rt, %s : num[1; ()] @ 50..55 (40..45, 52..53)
      %d = add %c, %x : num[1; ()] @ 60..65 (50..55, 62..63)
      yield %d",
    "      %t = math.atan2 %y, %x : num[1; ()] @ 10..15 (10..11, 12..13)
      %u = math.hypot %t, %s : num[1; ()] @ 20..25 (20..21, 22..23)
      yield %u",
    "      %t = math.min %x, %s : num[1; ()] @ 10..15
      %u = math.max %y, %t : num[1; ()] @ 20..25
      %v = mul %u, %y : num[1; ()] @ 30..35 (30..31, 32..33)
      yield %v",
    "      %t = math.floor %x : num[1; ()] @ 10..15
      %u = math.round %y : num[1; ()] @ 20..25
      %v = math.sign %u : num[1; ()] @ 30..35
      %w = sub %t, %v : num[1; ()] @ 40..45 (40..41, 42..43)
      yield %w",
    "      %t = math.exp %x : num[1; ()] @ 10..15 (10..11)
      %u = math.log %y : num[1; ()] @ 20..25 (20..21)
      %v = div %t, %u : num[1; ()] @ 30..35 (30..31, 32..33)
      yield %v",
    "      %z = sub %y, %y : num[1; ()] @ 10..15 (10..11, 12..13)
      %t = div %x, %z : num[1; ()] @ 20..25 (20..21, 22..23)
      %u = math.abs %t : num[1; ()] @ 30..35
      yield %u",
    "      %t = math.sqrt %x : num[1; ()] @ 10..15 (10..11)
      %u = math.log %y : num[1; ()] @ 20..25 (20..21)
      %v = add %t, %u : num[1; ()] @ 30..35 (30..31, 32..33)
      yield %v",
];

#[test]
fn entrywise_bodies_run_natively_at_long_lengths_with_a_bad_entry_anywhere() {
    let before = NATIVE_RUNS.load(Ordering::Relaxed);
    let bad = [1e200, 0.0, -1.0, 1000.0, 1e308];
    let mut failing = 0;
    let mut total = 0;
    for (k, body) in ENTRYWISE_BODIES.iter().enumerate() {
        let m = module(false, body);
        for n in [1usize, 9, 513, 1025, 4099] {
            let clean_x: Vec<f64> = (0..n).map(|i| 0.25 + (i % 17) as f64 * 0.5).collect();
            let clean_y: Vec<f64> = (0..n).map(|i| 1.5 + (i % 11) as f64 * 0.75).collect();
            // No bad entry, then one planted at the first, last, second to
            // last and middle positions, in either array.
            let mut cases = vec![(clean_x.clone(), clean_y.clone())];
            for p in [0, n - 1, n / 2] {
                for (j, b) in bad.iter().enumerate() {
                    let (mut xs, mut ys) = (clean_x.clone(), clean_y.clone());
                    if (j + p + k) % 2 == 0 {
                        xs[p] = *b;
                    } else {
                        ys[p] = *b;
                    }
                    cases.push((xs, ys));
                }
            }
            for (xs, ys) in cases {
                for s in [0.5, 0.0] {
                    let args = [
                        array(&xs, Dimension::DIMENSIONLESS),
                        array(&ys, Dimension::DIMENSIONLESS),
                        Value::scalar(s),
                    ];
                    let want = run(&m, &args, JitOptions::OFF);
                    assert_eq!(run(&m, &args, JitOptions::ALWAYS), want, "body {k} n={n}");
                    total += 1;
                    failing += usize::from(want.is_err());
                }
            }
        }
    }
    let native = NATIVE_RUNS.load(Ordering::Relaxed) - before;
    assert!(
        native >= total / 2,
        "only {native} of {total} runs were native"
    );
    assert!(
        failing >= total / 10,
        "only {failing} of {total} runs failed"
    );
}

#[test]
fn the_earliest_failing_operation_wins_over_the_lowest_element_with_math() {
    // sqrt (span 10..15) fails only at the last element, log (20..25) at the
    // first: sqrt's error is the unfused chain's, natively too.
    let m = module(false, ENTRYWISE_BODIES[6]);
    for n in [3, 9, 512, 1001, 4100] {
        let mut xs = vec![4.0; n];
        let mut ys = vec![1.0; n];
        xs[n - 1] = -1.0;
        ys[0] = 0.0;
        let args = [
            array(&xs, Dimension::DIMENSIONLESS),
            array(&ys, Dimension::DIMENSIONLESS),
            Value::scalar(1.0),
        ];
        let want = run(&m, &args, JitOptions::OFF);
        let e = want.as_ref().unwrap_err();
        assert!(e.message.contains("`sqrt` of a negative number"), "{e}");
        assert_eq!((e.span.start, e.span.end), (10, 15));
        assert_eq!(run(&m, &args, JitOptions::ALWAYS), want, "n={n}");
        // Only the later operation fails: its lowest element decides.
        xs[n - 1] = 4.0;
        ys[n / 2] = -1.0;
        let args = [
            array(&xs, Dimension::DIMENSIONLESS),
            array(&ys, Dimension::DIMENSIONLESS),
            Value::scalar(1.0),
        ];
        let want = run(&m, &args, JitOptions::OFF);
        assert!(want.as_ref().unwrap_err().message.contains("`log` of zero"));
        assert_eq!(run(&m, &args, JitOptions::ALWAYS), want, "n={n}");
    }
}

#[test]
fn a_dimensioned_product_and_root_of_arrays_run_natively() {
    // sqrt(x*x + y*y) over lengths is a length; x * y is an area.
    let text = "sir edition 2026\n\nfn @f(%a: num[L; (3)], %b: num[L; (3)]) -> (num[L; (3)]) effects(fail) {
  %r = fused %a, %b : num[L; (3)] {
    ^1(%x: num[L; ()], %y: num[L; ()]):
      %xx = mul %x, %x : num[L²; ()] @ 10..15
      %yy = mul %y, %y : num[L²; ()] @ 20..25
      %q = add %xx, %yy : num[L²; ()] @ 30..35
      %rt = math.sqrt %q : num[L; ()] @ 40..45
      yield %rt
  }
  return %r
}
";
    let m = parse(text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    for n in [3, 1000, 4097] {
        let xs: Vec<f64> = (0..n).map(|i| 3.0 + i as f64).collect();
        let ys: Vec<f64> = (0..n).map(|i| 4.0 + 2.0 * i as f64).collect();
        let args = [array(&xs, Dimension::LENGTH), array(&ys, Dimension::LENGTH)];
        let want = run(&m, &args, JitOptions::OFF);
        assert!(want.is_ok());
        assert_eq!(run(&m, &args, JitOptions::ALWAYS), want, "n={n}");
    }
}

#[test]
fn a_run_below_the_threshold_or_with_the_jit_off_never_compiles() {
    let m = module(false, "      %t = add %x, %y : num[1; ()]\n      yield %t");
    let f = &m.functions[0];
    let op = f
        .ops
        .iter()
        .find(|o| o.kind == sky_ir::OpKind::Fused)
        .unwrap();
    let inputs = [
        array(&[1.0; 100], Dimension::DIMENSIONLESS),
        array(&[2.0; 100], Dimension::DIMENSIONLESS),
        Value::scalar(1.0),
    ];
    for options in [
        JitOptions::OFF,
        JitOptions {
            enabled: true,
            min_elements: 101,
        },
    ] {
        let mut jit = Jit::new(options);
        assert!(matches!(jit.run(f, op, &inputs, 100), Native::Declined));
        assert_eq!(jit.compiles, 0);
    }
    // At the threshold it compiles once and reuses the kernel, and
    // recompiles only if the operands' kinds change.
    let mut jit = Jit::new(JitOptions {
        enabled: true,
        min_elements: 100,
    });
    for _ in 0..3 {
        let Native::Done(data, _) = jit.run(f, op, &inputs, 100) else {
            panic!("not native")
        };
        assert_eq!(data, vec![3.0; 100]);
    }
    assert_eq!(jit.compiles, 1);
    let metres = [
        array(&[1.0; 100], Dimension::LENGTH),
        array(&[2.0; 100], Dimension::LENGTH),
        Value::scalar(1.0),
    ];
    let Native::Done(_, dim) = jit.run(f, op, &metres, 100) else {
        panic!("not native")
    };
    assert_eq!(dim, Dimension::LENGTH);
    assert_eq!(jit.compiles, 2);
}

#[test]
fn a_body_the_kernel_refuses_is_run_by_the_interpreter_and_asked_once() {
    // Adding a length to a time: the interpreter's error, not a kernel's.
    let m = module(
        true,
        "      %t = add %x, %s : num[1; ()] @ 10..15 (10..11, 14..15)\n      yield %t",
    );
    let args = [
        array(&[1.0; 20], Dimension::LENGTH),
        array(&[2.0; 20], Dimension::LENGTH),
        Value::number(1.0, Dimension::TIME),
    ];
    let want = run(&m, &args, JitOptions::OFF);
    assert!(want.is_err());
    assert_eq!(run(&m, &args, JitOptions::ALWAYS), want);
    let f = &m.functions[0];
    let op = f
        .ops
        .iter()
        .find(|o| o.kind == sky_ir::OpKind::Fused)
        .unwrap();
    let mut jit = Jit::new(JitOptions::ALWAYS);
    for _ in 0..3 {
        assert!(matches!(jit.run(f, op, &args, 20), Native::Declined));
    }
    assert_eq!(jit.compiles, 1);
}

#[test]
fn a_body_compiles_once_the_interpreter_has_spent_the_threshold_on_it() {
    let m = module(false, "      %t = add %x, %y : num[1; ()]\n      yield %t");
    let f = &m.functions[0];
    let op = f
        .ops
        .iter()
        .find(|o| o.kind == sky_ir::OpKind::Fused)
        .unwrap();
    let inputs = [
        array(&[1.0; 100], Dimension::DIMENSIONLESS),
        array(&[2.0; 100], Dimension::DIMENSIONLESS),
        Value::scalar(1.0),
    ];
    let mut jit = Jit::new(JitOptions {
        enabled: true,
        min_elements: 250,
    });
    // 100, 200 elements interpreted: not yet; the third call reaches 300.
    for _ in 0..2 {
        assert!(matches!(jit.run(f, op, &inputs, 100), Native::Declined));
    }
    assert_eq!(jit.compiles, 0);
    assert!(matches!(jit.run(f, op, &inputs, 100), Native::Done(..)));
    assert_eq!(jit.compiles, 1);
}

#[test]
fn the_environment_turns_native_kernels_off_and_sets_the_threshold() {
    let default = JitOptions::from_vars(None, None);
    assert_eq!(
        default,
        JitOptions {
            enabled: true,
            min_elements: sky_jit::tuning::kernel_threshold()
        }
    );
    for off in ["off", "OFF", " 0 ", "false", "no"] {
        assert!(!JitOptions::from_vars(Some(off), None).enabled, "{off}");
    }
    for on in ["on", "1", "", "yes"] {
        assert!(JitOptions::from_vars(Some(on), None).enabled, "{on}");
    }
    assert_eq!(JitOptions::from_vars(None, Some("0")).min_elements, 0);
    assert_eq!(JitOptions::from_vars(None, Some(" 77 ")).min_elements, 77);
    assert_eq!(
        JitOptions::from_vars(None, Some("many")).min_elements,
        sky_jit::tuning::kernel_threshold()
    );
}
