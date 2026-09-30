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
            min_elements: JitOptions::DEFAULT_THRESHOLD
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
        JitOptions::DEFAULT_THRESHOLD
    );
}
