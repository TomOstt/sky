//! Tests of compiled kernels: bit-for-bit against the reference path (which
//! is the interpreter's algorithm on `sky_interp`'s own functions), on
//! hand-written and random bodies and on data that overflows, divides by
//! zero, leaves a function's domain, or is `-0`, subnormal or huge.
//!
//! The end-to-end property test, against `sky-ir-interp` itself, is in that
//! crate (which depends on this one).

use sky_ir::{parse, Module, OpKind};
use sky_units::{Dimension, MathFn};

use super::*;

/// A small deterministic generator (xorshift64*), so a failure names its seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// A finite number, from ranges that stress the operations.
    fn number(&mut self) -> f64 {
        let unit = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        let sign = if self.next() & 1 == 0 { 1.0 } else { -1.0 };
        match self.below(12) {
            0 => 0.0,
            1 => -0.0,
            2 => sign * (self.below(9) as f64),
            3 | 4 => sign * unit * 10.0,
            5 => sign * unit * 1e3,
            6 => sign * 5e-324 * (1 + self.below(1000)) as f64,
            7 => sign * (1.0 + unit) * 1e-300,
            8 => sign * (1.0 + unit) * 1e154,
            9 => sign * (1.0 + unit) * 1e300,
            10 => sign * (1.0 + unit) * 1e-160,
            _ => sign * unit,
        }
    }
}

const HEAD: &str = "sir edition 2026\n\n";

fn module(body: &str) -> Module {
    let text = format!(
        "{HEAD}fn @f(%a: num[1; (3)], %b: num[1; (3)], %k: num[1; ()]) -> (num[1; (3)]) effects(fail) {{
  %r = fused %a, %b, %k : num[1; (3)] {{
    ^1(%x: num[1; ()], %y: num[1; ()], %s: num[1; ()]):
{body}
  }}
  return %r
}}
"
    );
    parse(&text).unwrap_or_else(|e| panic!("{e}\n{text}"))
}

fn fused_of(m: &Module) -> (&Function, &OpData) {
    let f = &m.functions[0];
    let op = f
        .ops
        .iter()
        .find(|o| o.kind == OpKind::Fused)
        .expect("a fused operation");
    (f, op)
}

const SIGS: [InputSig; 3] = [
    InputSig {
        dim: Dimension::DIMENSIONLESS,
        array: true,
    },
    InputSig {
        dim: Dimension::DIMENSIONLESS,
        array: true,
    },
    InputSig {
        dim: Dimension::DIMENSIONLESS,
        array: false,
    },
];

fn compile(body: &str, options: Options) -> Kernel {
    let m = module(body);
    let (f, op) = fused_of(&m);
    compile_kernel_with(f, op, &SIGS, options).unwrap_or_else(|e| panic!("{e}\n{body}"))
}

/// Runs `kernel` and the reference on the same data and demands they agree
/// to the bit, and that the fast path reports a failure exactly when the
/// reference fails (no false alarm, no miss).
fn agree(kernel: &Kernel, xs: &[f64], ys: &[f64], s: f64, what: &str) {
    let n = xs.len();
    let inputs = [Input::Array(xs), Input::Array(ys), Input::Scalar(s)];
    let mut out = vec![f64::NAN; n];
    let flagged = kernel.run_fast(&inputs, &mut out).unwrap();
    let oracle = kernel.reference.run(&inputs, n);
    match &oracle {
        Ok(want) => {
            assert!(!flagged, "false alarm: {what}\nx={xs:?}\ny={ys:?}\ns={s}");
            for (i, (g, w)) in out.iter().zip(want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "element {i}: {g:e} vs {w:e}: {what}\nx={xs:?}\ny={ys:?}\ns={s}"
                );
            }
        }
        Err(_) => assert!(
            flagged,
            "missed a failure: {what}\nx={xs:?}\ny={ys:?}\ns={s}"
        ),
    }
    let mut again = vec![0.0; n];
    let got = kernel.run(&inputs, &mut again);
    assert_eq!(
        got.map(|()| again.iter().map(|x| x.to_bits()).collect::<Vec<_>>()),
        oracle.map(|d| d.iter().map(|x| x.to_bits()).collect::<Vec<_>>()),
        "{what}"
    );
}

fn data(rng: &mut Rng, n: usize) -> (Vec<f64>, Vec<f64>, f64) {
    let xs = (0..n).map(|_| rng.number()).collect();
    let ys = (0..n).map(|_| rng.number()).collect();
    (xs, ys, rng.number())
}

#[test]
fn a_times_x_plus_y_is_exact_on_every_length() {
    let body = "      %t = mul %x, %s : num[1; ()]
      %u = add %t, %y : num[1; ()]
      yield %u";
    for options in [
        Options::default(),
        Options {
            simd: false,
            ..Options::default()
        },
        Options {
            unroll: 1,
            ..Options::default()
        },
        Options {
            unroll: 3,
            ..Options::default()
        },
        Options {
            fma_check: false,
            ..Options::default()
        },
    ] {
        let kernel = compile(body, options);
        let mut rng = Rng(7);
        for n in (0..70).chain([100, 129, 1000, 1001]) {
            let (xs, ys, s) = data(&mut rng, n);
            // Small data, so nothing overflows and the bits are compared.
            let xs: Vec<f64> = xs.iter().map(|x| x.clamp(-1e100, 1e100)).collect();
            let ys: Vec<f64> = ys.iter().map(|x| x.clamp(-1e100, 1e100)).collect();
            agree(
                &kernel,
                &xs,
                &ys,
                s.clamp(-1e100, 1e100),
                &format!("{options:?} n={n}"),
            );
        }
    }
}

#[test]
fn it_never_fuses_a_multiply_and_an_add() {
    // 1e16 + 1 rounds at every step; an FMA would keep the low bit of x*x.
    let x = 1.0 + f64::EPSILON;
    let kernel = compile(
        "      %t = mul %x, %x : num[1; ()]
      %u = sub %t, %y : num[1; ()]
      yield %u",
        Options::default(),
    );
    let xs = vec![x; 9];
    let ys = vec![x * x; 9];
    let mut out = vec![1.0; 9];
    kernel
        .run(
            &[Input::Array(&xs), Input::Array(&ys), Input::Scalar(0.0)],
            &mut out,
        )
        .unwrap();
    assert!(out.iter().all(|o| *o == 0.0), "{out:?}");
}

#[test]
fn zero_signs_follow_the_interpreter() {
    // `neg` is `0 - x`: `-(+0)` is `+0`. `sqrt(-0)` is `+0` after the
    // interpreter's `+ 0.0`. `abs` and `x * -1` for a zero.
    let cases = [
        "      %t = neg %x : num[1; ()]
      yield %t",
        "      %t = math.sqrt %x : num[1; ()]
      yield %t",
        "      %t = math.abs %x : num[1; ()]
      yield %t",
        "      %m = const.float -1 : num[1; ()]
      %t = mul %x, %m : num[1; ()]
      yield %t",
        "      %t = math.floor %x : num[1; ()]
      yield %t",
        "      %t = math.ceil %x : num[1; ()]
      %u = sub %t, %y : num[1; ()]
      yield %u",
        "      %t = math.round %x : num[1; ()]
      yield %t",
    ];
    let xs = [0.0, -0.0, -0.4, 0.4, 2.0, -2.5, 0.0, -0.0, 1.5, 3.0, -0.0];
    let ys = [0.0; 11];
    for body in cases {
        let kernel = compile(body, Options::default());
        agree(&kernel, &xs, &ys, 0.0, body);
    }
}

#[test]
fn failures_are_found_wherever_they_are_and_however_hidden() {
    let overflow = [
        1.0e200, 1.0, 1.0e200, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
    ];
    let ones = [1.0; 11];
    let cases = [
        // Overflow at the yield, and division by zero.
        ("      %t = mul %x, %y : num[1; ()]\n      yield %t", &overflow, &overflow),
        ("      %t = div %x, %y : num[1; ()]\n      yield %t", &ones, &[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        // The overflow of `x * x` is an infinity, which the division turns
        // into a finite zero: it must still be found.
        (
            "      %t = mul %x, %x : num[1; ()]\n      %u = div %y, %t : num[1; ()]\n      yield %u",
            &overflow,
            &ones,
        ),
        // The same, through a `neg` and a `sub`.
        (
            "      %t = mul %x, %y : num[1; ()]\n      %n = neg %t : num[1; ()]\n      %u = div %s, %n : num[1; ()]\n      %v = sub %u, %u : num[1; ()]\n      yield %v",
            &overflow,
            &overflow,
        ),
        // A dead operation that fails.
        (
            "      %t = mul %x, %y : num[1; ()]\n      %u = add %x, %y : num[1; ()]\n      yield %u",
            &overflow,
            &overflow,
        ),
        // Domain errors, of a call and of the instruction.
        ("      %t = math.sqrt %x : num[1; ()]\n      yield %t", &[4.0, -1.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0], &ones),
        ("      %t = math.log %x : num[1; ()]\n      yield %t", &[4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 0.0], &ones),
        // A math function that hides a non-finite argument (`atan(inf)`).
        (
            "      %t = mul %x, %y : num[1; ()]\n      %u = math.atan %t : num[1; ()]\n      yield %u",
            &overflow,
            &overflow,
        ),
        // A failure only in the last (tail) element, and only in the first.
        ("      %t = div %x, %y : num[1; ()]\n      yield %t", &ones, &[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    ];
    for (body, xs, ys) in cases {
        for options in [
            Options::default(),
            Options {
                simd: false,
                ..Options::default()
            },
            Options {
                fma_check: false,
                ..Options::default()
            },
        ] {
            let kernel = compile(body, options);
            for n in [1, 2, 3, 8, 9, 11] {
                agree(&kernel, &xs[..n], &ys[..n], 2.0, body);
            }
        }
    }
}

#[test]
fn the_first_failing_operation_is_reported_not_the_first_element() {
    // The `div` fails at element 0, the earlier `mul` at element 1: the
    // unfused chain reports the `mul` (span 10..15).
    let kernel = compile(
        "      %t = mul %x, %y : num[1; ()] @ 10..15
      %u = div %t, %s : num[1; ()] @ 20..25 (20..21, 22..23)
      yield %u",
        Options::default(),
    );
    let xs = [1.0, 1e200, 1.0, 1.0];
    let ys = [1.0, 1e200, 1.0, 1.0];
    let mut out = vec![0.0; 4];
    let e = kernel
        .run(
            &[Input::Array(&xs), Input::Array(&ys), Input::Scalar(0.0)],
            &mut out,
        )
        .unwrap_err();
    let d = e.to_diagnostic();
    assert_eq!(d.primary.span, Span::new(10, 15), "{d:?}");
    // With a finite product, the division is the failure.
    let e = kernel
        .run(
            &[
                Input::Array(&xs[..1]),
                Input::Array(&ys[..1]),
                Input::Scalar(0.0),
            ],
            &mut out[..1],
        )
        .unwrap_err();
    let d = e.to_diagnostic();
    assert_eq!(d.message, "division by zero");
    assert_eq!(d.primary.span, Span::new(20, 25));
    assert_eq!(d.secondary[0].span, Span::new(22, 23));
}

#[test]
fn what_the_kernel_cannot_do_is_refused() {
    let m = module(
        "      %t = cmp.lt %x, %y : bool
      yield %t",
    );
    let (f, op) = fused_of(&m);
    assert!(compile_kernel(f, op, &SIGS).is_err());
    // Dimensions that do not agree are the interpreter's error to report.
    let m = module("      %t = add %x, %y : num[1; ()]\n      yield %t");
    let (f, op) = fused_of(&m);
    let mut sigs = SIGS;
    sigs[1].dim = Dimension::LENGTH;
    let refusal = compile_kernel(f, op, &sigs).unwrap_err();
    assert!(refusal.0.contains("dimensions"), "{refusal}");
    // Only scalars.
    let mut sigs = SIGS;
    sigs[0].array = false;
    sigs[1].array = false;
    assert!(compile_kernel(f, op, &sigs).is_err());
}

#[test]
fn the_result_dimension_is_the_bodys() {
    let m = parse(&format!(
        "{HEAD}fn @f(%a: num[L; (3)], %t: num[T; ()]) -> (num[L·T⁻¹; (3)]) effects(fail) {{
  %r = fused %a, %t : num[L·T⁻¹; (3)] {{
    ^1(%x: num[L; ()], %s: num[T; ()]):
      %q = mul %x, %x : num[L²; ()]
      %w = math.sqrt %q : num[L; ()]
      %v = div %w, %s : num[L·T⁻¹; ()]
      yield %v
  }}
  return %r
}}
"
    ))
    .unwrap();
    let (f, op) = fused_of(&m);
    let sigs = [
        InputSig {
            dim: Dimension::LENGTH,
            array: true,
        },
        InputSig {
            dim: Dimension::TIME,
            array: false,
        },
    ];
    let kernel = compile_kernel(f, op, &sigs).unwrap();
    assert_eq!(
        kernel.output_dim(),
        Dimension::LENGTH.checked_div(Dimension::TIME).unwrap()
    );
    let xs = [10.0, 20.0, 30.0];
    let mut out = [0.0; 3];
    kernel
        .run(&[Input::Array(&xs), Input::Scalar(5.0)], &mut out)
        .unwrap();
    assert_eq!(out, [2.0, 4.0, 6.0]);
}

/// A random body over `x`, `y`, `s`, constants and earlier results.
fn random_body(rng: &mut Rng) -> String {
    let mut pool: Vec<String> = vec!["%x".into(), "%y".into(), "%s".into()];
    let mut text = String::new();
    let len = 1 + rng.below(9);
    let mut at = 100u32;
    for i in 0..len {
        let pick = |rng: &mut Rng, pool: &[String]| pool[rng.below(pool.len())].clone();
        let span = format!("@ {at}..{}", at + 5);
        at += 10;
        let name = format!("%t{i}");
        let line = match rng.below(14) {
            0 => {
                let c = [0.5, 2.0, -3.0, 0.1, 1e154, 1e-320, 0.0, 7.0][rng.below(8)];
                let line = format!("      {name} = const.float {c:e} : num[1; ()]\n");
                pool.push(name);
                text += &line;
                continue;
            }
            1 => {
                let line = format!("      {name} = const.rational 1/3 : num[1; ()]\n");
                pool.push(name);
                text += &line;
                continue;
            }
            2 | 3 => {
                let a = pick(rng, &pool);
                format!("{name} = neg {a} : num[1; ()] {span} (1..2)")
            }
            4..=6 => {
                let (a, b) = (pick(rng, &pool), pick(rng, &pool));
                let op = ["add", "sub", "mul", "div"][rng.below(4)];
                format!("{name} = {op} {a}, {b} : num[1; ()] {span} (3..4, 5..6)")
            }
            _ => {
                let f = MathFn::ALL[rng.below(MathFn::ALL.len())];
                let a = pick(rng, &pool);
                if f.arity() == 2 {
                    let b = pick(rng, &pool);
                    format!("{name} = {} {a}, {b} : num[1; ()] {span}", f.mnemonic())
                } else {
                    format!("{name} = {} {a} : num[1; ()] {span}", f.mnemonic())
                }
            }
        };
        pool.push(name);
        text += &format!("      {line}\n");
    }
    let last = pool.last().cloned().unwrap_or_else(|| "%x".into());
    text += &format!("      yield {last}");
    text
}

#[test]
fn random_bodies_agree_with_the_reference_bit_for_bit() {
    let mut compiled = 0;
    let mut failing = 0;
    for seed in 1..=400u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let body = random_body(&mut rng);
        let m = module(&body);
        let (f, op) = fused_of(&m);
        let options = match seed % 3 {
            0 => Options::default(),
            1 => Options {
                simd: false,
                fma_check: false,
                ..Options::default()
            },
            _ => Options {
                unroll: 2,
                ..Options::default()
            },
        };
        let Ok(kernel) = compile_kernel_with(f, op, &SIGS, options) else {
            continue;
        };
        compiled += 1;
        for round in 0..6 {
            let n = [1, 5, 8, 9, 16, 31][round];
            let (xs, ys, s) = data(&mut rng, n);
            let inputs = [Input::Array(&xs), Input::Array(&ys), Input::Scalar(s)];
            if kernel.reference.run(&inputs, n).is_err() {
                failing += 1;
            }
            agree(&kernel, &xs, &ys, s, &format!("seed {seed}\n{body}"));
        }
    }
    assert!(compiled >= 390, "only {compiled} bodies compiled");
    assert!(failing >= 100, "only {failing} runs failed");
}

/// The bodies of the entry-by-entry operations: a product and a quotient of
/// two arrays' entries, a chain with `sqrt`, and the math functions of one
/// and of two arrays (or of an array and a number).
const ENTRYWISE: [&str; 8] = [
    "      %xx = mul %x, %x : num[1; ()]
      %yy = mul %y, %y : num[1; ()]
      %q = add %xx, %yy : num[1; ()]
      %rt = math.sqrt %q : num[1; ()]
      %c = mul %rt, %s : num[1; ()]
      %d = add %c, %x : num[1; ()]
      yield %d",
    "      %t = div %x, %y : num[1; ()]
      %u = mul %t, %y : num[1; ()]
      yield %u",
    "      %t = math.atan2 %x, %y : num[1; ()]
      %u = math.hypot %t, %s : num[1; ()]
      yield %u",
    "      %t = math.min %x, %s : num[1; ()]
      %u = math.max %y, %t : num[1; ()]
      yield %u",
    "      %t = math.floor %x : num[1; ()]
      %u = math.round %y : num[1; ()]
      %v = math.ceil %s : num[1; ()]
      %w = sub %t, %u : num[1; ()]
      %z = math.sign %w : num[1; ()]
      %e = add %z, %v : num[1; ()]
      yield %e",
    "      %t = math.exp %x : num[1; ()]
      %u = math.log %y : num[1; ()]
      %v = div %t, %u : num[1; ()]
      yield %v",
    "      %z = sub %y, %y : num[1; ()]
      %t = div %x, %z : num[1; ()]
      %u = math.abs %t : num[1; ()]
      yield %u",
    "      %t = math.sqrt %x : num[1; ()]
      %u = math.log %y : num[1; ()]
      %v = math.sin %s : num[1; ()]
      %w = mul %t, %v : num[1; ()]
      %e = add %w, %u : num[1; ()]
      yield %e",
];

#[test]
fn entrywise_bodies_agree_on_long_arrays_and_every_tail() {
    let bad = [
        1e200, -1e200, 0.0, -0.0, -1.0, 1e-320, 1000.0, 1e308, 5e-324,
    ];
    for (k, body) in ENTRYWISE.iter().enumerate() {
        for options in [
            Options::default(),
            Options {
                simd: false,
                ..Options::default()
            },
            Options {
                unroll: 3,
                fma_check: false,
                ..Options::default()
            },
        ] {
            let kernel = compile(body, options);
            let mut rng = Rng(100 + k as u64);
            // Lengths around the unroll (8) and a kernel's 512 threshold.
            for n in [17usize, 25, 65, 511, 513, 1025] {
                let xs: Vec<f64> = (0..n)
                    .map(|_| 0.25 + (rng.below(64) as f64) * 0.125)
                    .collect();
                let ys: Vec<f64> = (0..n)
                    .map(|_| 0.5 + (rng.below(64) as f64) * 0.25)
                    .collect();
                let s = 0.75 + rng.below(4) as f64;
                let what = format!("body {k} {options:?} n={n}");
                agree(&kernel, &xs, &ys, s, &what);
                // One bad entry at the start, in a tail, in the middle, at the
                // end; in either array, and in the scalar.
                for p in [0, n / 2, n - 2, n - 1] {
                    let b = bad[(p + n + k) % bad.len()];
                    let (mut x2, mut y2) = (xs.clone(), ys.clone());
                    if (p + k) % 2 == 0 {
                        x2[p] = b;
                    } else {
                        y2[p] = b;
                    }
                    agree(&kernel, &x2, &y2, s, &format!("{what} p={p} bad={b:e}"));
                    // Two bad entries, so the earliest operation has to be told
                    // from the lowest element.
                    x2[n - 1] = -1.0;
                    y2[0] = 0.0;
                    agree(&kernel, &x2, &y2, s, &format!("{what} p={p} both"));
                }
                agree(&kernel, &xs, &ys, bad[(n + k) % bad.len()], &what);
            }
        }
    }
}

#[test]
fn the_earliest_failing_math_operation_is_reported_at_any_length() {
    // `sqrt` (span 10..15) fails only at the last entry, `log` (20..25) at
    // the first: sqrt's error is the unfused chain's.
    let kernel = compile(
        "      %t = math.sqrt %x : num[1; ()] @ 10..15 (10..11)
      %u = math.log %y : num[1; ()] @ 20..25 (20..21)
      %v = add %t, %u : num[1; ()] @ 30..35 (30..31, 32..33)
      yield %v",
        Options::default(),
    );
    for n in [2usize, 8, 9, 100, 513, 2000] {
        let mut xs = vec![4.0; n];
        let mut ys = vec![1.0; n];
        xs[n - 1] = -1.0;
        ys[0] = 0.0;
        let mut out = vec![0.0; n];
        let e = kernel
            .run(
                &[Input::Array(&xs), Input::Array(&ys), Input::Scalar(0.0)],
                &mut out,
            )
            .unwrap_err();
        let d = e.to_diagnostic();
        assert_eq!(d.primary.span, Span::new(10, 15), "n={n} {d:?}");
        assert!(d.message.contains("`sqrt` of a negative number"), "{d:?}");
        // Without the sqrt failure the log's lowest entry is the error.
        xs[n - 1] = 4.0;
        ys[n - 1] = -1.0;
        let e = kernel
            .run(
                &[Input::Array(&xs), Input::Array(&ys), Input::Scalar(0.0)],
                &mut out,
            )
            .unwrap_err();
        let d = e.to_diagnostic();
        assert_eq!(d.primary.span, Span::new(20, 25), "n={n}");
        assert!(d.message.contains("`log` of zero"), "{d:?}");
    }
}

#[test]
fn every_math_function_agrees_on_awkward_arguments() {
    let args = [
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.0,
        -1.0,
        2.0,
        10.0,
        -10.0,
        1e-300,
        5e-324,
        700.0,
        800.0,
        -800.0,
        1e300,
        -1e300,
        0.999999999,
        1.000000001,
        3.5,
        -3.5,
    ];
    for f in MathFn::ALL {
        let body = if f.arity() == 2 {
            format!(
                "      %t = {} %x, %y : num[1; ()]\n      yield %t",
                f.mnemonic()
            )
        } else {
            format!(
                "      %t = {} %x : num[1; ()]\n      yield %t",
                f.mnemonic()
            )
        };
        let kernel = compile(&body, Options::default());
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for x in args {
            for y in args {
                xs.push(x);
                ys.push(y);
            }
        }
        // One long run, and one element at a time, so failures are
        // attributed to their arguments.
        agree(&kernel, &xs, &ys, 0.0, &body);
        for (x, y) in xs.iter().zip(&ys) {
            agree(&kernel, &[*x], &[*y], 0.0, &body);
        }
    }
}

#[test]
fn a_constant_body_and_scalar_only_arithmetic_broadcast() {
    let kernel = compile(
        "      %c = const.float 2 : num[1; ()]
      %t = mul %s, %c : num[1; ()]
      %u = math.sqrt %t : num[1; ()]
      yield %u",
        Options::default(),
    );
    let (xs, ys) = (vec![0.0; 13], vec![0.0; 13]);
    agree(&kernel, &xs, &ys, 8.0, "scalar body");
    let mut out = vec![0.0; 13];
    kernel
        .run(
            &[Input::Array(&xs), Input::Array(&ys), Input::Scalar(8.0)],
            &mut out,
        )
        .unwrap();
    assert!(out.iter().all(|x| *x == 4.0));
    // The uniform value fails once, before the loop.
    agree(&kernel, &xs, &ys, -8.0, "scalar body, domain error");
}

#[test]
fn a_kernel_run_on_the_wrong_operands_is_an_error_not_a_crash() {
    let kernel = compile(
        "      %t = add %x, %y : num[1; ()]\n      yield %t",
        Options::default(),
    );
    let xs = [1.0, 2.0];
    let mut out = [0.0; 2];
    for inputs in [
        vec![Input::Array(&xs)],
        vec![Input::Array(&xs), Input::Scalar(1.0), Input::Scalar(1.0)],
        vec![
            Input::Array(&xs),
            Input::Array(&xs[..1]),
            Input::Scalar(1.0),
        ],
    ] {
        let e = kernel.run(&inputs, &mut out).unwrap_err();
        assert!(e.to_diagnostic().message.starts_with("malformed IR"));
    }
}

// ----- relaxed numerics: contraction -----

fn contracting() -> Options {
    Options {
        contract: true,
        ..Options::default()
    }
}

fn host_fuses() -> bool {
    has_fma(&isa().expect("a code generator"))
}

const MUL_ADD: &str =
    "      %t = mul %x, %s : num[1; ()]\n      %u = add %t, %y : num[1; ()]\n      yield %u";

/// A body `a*x + y` (in the shapes a sum with a product takes) becomes one
/// fused multiply-add per element, on every length (vector body and tail):
/// the bits are `mul_add`'s.
#[test]
fn contracted_kernels_are_fused_multiply_adds() {
    if !host_fuses() {
        return;
    }
    type Want = fn(f64, f64, f64) -> f64;
    let bodies: [(&str, Want); 4] = [
        (MUL_ADD, |x, y, s| x.mul_add(s, y)),
        (
            "      %t = mul %x, %s : num[1; ()]\n      %u = add %y, %t : num[1; ()]\n      yield %u",
            |x, y, s| x.mul_add(s, y),
        ),
        (
            "      %t = mul %x, %s : num[1; ()]\n      %u = sub %t, %y : num[1; ()]\n      yield %u",
            |x, y, s| x.mul_add(s, -y),
        ),
        (
            "      %t = mul %x, %s : num[1; ()]\n      %u = sub %y, %t : num[1; ()]\n      yield %u",
            |x, y, s| (-x).mul_add(s, y),
        ),
    ];
    let mut rng = Rng(7);
    for (body, want) in bodies {
        let kernel = compile(body, contracting());
        let plain = compile(body, Options::default());
        let mut differed = false;
        for n in [1, 2, 3, 7, 8, 9, 16, 33, 100, 1001] {
            let xs: Vec<f64> = (0..n).map(|_| rng.number().clamp(-1e100, 1e100)).collect();
            let ys: Vec<f64> = (0..n).map(|_| rng.number().clamp(-1e100, 1e100)).collect();
            let s = 0.1 + (rng.below(1000) as f64) / 7.0;
            let inputs = [Input::Array(&xs), Input::Array(&ys), Input::Scalar(s)];
            let got = kernel.run_alloc(&inputs, n).expect("finite data");
            let unfused = plain.run_alloc(&inputs, n).expect("finite data");
            for i in 0..n {
                let w = want(xs[i], ys[i], s);
                assert_eq!(got[i].to_bits(), w.to_bits(), "n={n} i={i}: {body}");
                differed |= got[i].to_bits() != unfused[i].to_bits();
            }
        }
        assert!(differed, "fusing changed nothing (is it fusing?): {body}");
    }
}

/// A product used twice, and a product that does not depend on the element,
/// are not contracted: the bits are the interpreter's.
#[test]
fn a_product_with_other_uses_or_no_element_is_left_alone() {
    let twice = "      %t = mul %x, %s : num[1; ()]\n      %u = add %t, %y : num[1; ()]\n      %v = add %u, %t : num[1; ()]\n      yield %v";
    let uniform =
        "      %t = mul %s, %s : num[1; ()]\n      %u = add %t, %y : num[1; ()]\n      yield %u";
    let mut rng = Rng(8);
    for body in [twice, uniform] {
        let kernel = compile(body, contracting());
        for n in [3, 9, 200] {
            let (xs, ys, s) = data(&mut rng, n);
            agree(&kernel, &xs, &ys, s, body);
        }
    }
}

/// Contracted kernels still fail exactly where the interpreter does, with
/// its error: an overflowing product, an overflowing sum.
#[test]
fn contracted_kernels_report_the_interpreters_errors() {
    let kernel = compile(MUL_ADD, contracting());
    let plain = compile(MUL_ADD, Options::default());
    let cases: [(Vec<f64>, Vec<f64>, f64); 2] = [
        // The product overflows; the addend is finite.
        (vec![1.0, 1e200, 2.0], vec![0.0, 5.0, 1.0], 1e200),
        // The sum overflows; the product does not.
        (vec![1.0, 1e308, 2.0], vec![0.0, 1e308, 1.0], 1.0),
    ];
    for (xs, ys, s) in cases {
        let inputs = [Input::Array(&xs), Input::Array(&ys), Input::Scalar(s)];
        let want = plain.run_alloc(&inputs, xs.len()).map(|_| ());
        assert!(want.is_err());
        let got = kernel.run_alloc(&inputs, xs.len()).map(|_| ());
        assert_eq!(got, want, "the same error, at the same operation");
    }
}

/// The documented relaxation: a product that overflows while the fused sum
/// is finite is accepted (the value is the fused one).
#[test]
fn a_product_rescued_by_its_addend_is_accepted_when_contracting() {
    if !host_fuses() {
        return;
    }
    let body =
        "      %t = mul %x, %s : num[1; ()]\n      %u = sub %t, %y : num[1; ()]\n      yield %u";
    let (xs, ys, s) = ([2e154], [1.7e308], 1e154);
    let inputs = [Input::Array(&xs), Input::Array(&ys), Input::Scalar(s)];
    let plain = compile(body, Options::default());
    assert!(
        plain.run_alloc(&inputs, 1).is_err(),
        "2e154 * 1e154 overflows"
    );
    let kernel = compile(body, contracting());
    let got = kernel
        .run_alloc(&inputs, 1)
        .expect("the fused sum is finite");
    assert_eq!(got[0], 2e154_f64.mul_add(1e154, -1.7e308));
    assert!(got[0] > 2.9e307 && got[0] < 3.1e307, "{}", got[0]);
}

// ----- chunks on the pool -----

/// The configurations a parallel run must agree with the sequential one
/// under: every thread count, the smallest chunks, odd chunk sizes, the
/// defaults.
fn par_configs() -> Vec<Config> {
    let mut all = Vec::new();
    for t in [2, 3, 4, 8] {
        all.push(Config::eager(t));
        all.push(Config::with_threads(t));
        all.push(Config {
            chunk_min: 3,
            ..Config::eager(t)
        });
        all.push(Config {
            chunk_min: 7,
            ..Config::eager(t)
        });
    }
    all
}

/// The bits of a result, or the whole diagnostic of the error.
fn outcome(r: Result<Vec<f64>, RuntimeError>) -> Result<Vec<u64>, String> {
    r.map(|d| d.iter().map(|x| x.to_bits()).collect())
        .map_err(|e| format!("{:?}", e.to_diagnostic()))
}

#[test]
fn chunks_give_the_sequential_bits_and_error_on_random_bodies() {
    let (mut ok, mut failed) = (0, 0);
    for seed in 1..=120u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let body = random_body(&mut rng);
        let m = module(&body);
        let (f, op) = fused_of(&m);
        let Ok(kernel) = compile_kernel_with(f, op, &SIGS, Options::default()) else {
            continue;
        };
        // Lengths around chunk boundaries; some data fails in several chunks.
        for n in [2, 9, 64, 257, 1000] {
            let (xs, ys, s) = data(&mut rng, n);
            let inputs = [Input::Array(&xs), Input::Array(&ys), Input::Scalar(s)];
            let want = outcome(kernel.run_alloc(&inputs, n));
            if want.is_ok() {
                ok += 1;
            } else {
                failed += 1;
            }
            for cfg in par_configs() {
                let got = outcome(kernel.run_alloc_par(&inputs, n, &cfg));
                assert_eq!(got, want, "seed {seed}, n {n}, {cfg:?}\n{body}");
                // The slice entry point is the same computation.
                let mut out = vec![f64::NAN; n];
                let got = kernel
                    .run_par(&inputs, &mut out, &cfg)
                    .map(|()| out.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
                    .map_err(|e| format!("{:?}", e.to_diagnostic()));
                assert_eq!(got, want, "run_par: seed {seed}, n {n}, {cfg:?}\n{body}");
            }
        }
    }
    assert!(ok > 50 && failed > 50, "{ok} succeeded, {failed} failed");
}

#[test]
fn the_failing_operation_first_in_the_body_wins_across_chunks() {
    // The `mul` (span 10..15) comes before the `div` (20..25) in the body.
    // The division fails at element 0, in the first chunk; the product only
    // at the last element, in the last chunk. The body's first failing
    // operation is reported (the `mul`), as in a sequential run.
    let kernel = compile(
        "      %t = mul %x, %y : num[1; ()] @ 10..15
      %u = div %t, %s : num[1; ()] @ 20..25 (20..21, 22..23)
      yield %u",
        Options::default(),
    );
    let n = 640;
    let mut xs = vec![1.0; n];
    let mut ys = vec![1.0; n];
    // Element 0: the product is finite, the division by `s = 0` fails.
    xs[0] = 2.0;
    // The last element: the product overflows.
    xs[n - 1] = 1e200;
    ys[n - 1] = 1e200;
    let inputs = [Input::Array(&xs), Input::Array(&ys), Input::Scalar(0.0)];
    let want = kernel.run_alloc(&inputs, n).unwrap_err().to_diagnostic();
    assert_eq!(want.primary.span, Span::new(10, 15));
    for cfg in par_configs() {
        let got = kernel
            .run_alloc_par(&inputs, n, &cfg)
            .unwrap_err()
            .to_diagnostic();
        assert_eq!(got, want, "{cfg:?}");
    }
    // Without the overflow the division is the failure, at every thread
    // count.
    xs[n - 1] = 1.0;
    ys[n - 1] = 1.0;
    let inputs = [Input::Array(&xs), Input::Array(&ys), Input::Scalar(0.0)];
    let want = kernel.run_alloc(&inputs, n).unwrap_err().to_diagnostic();
    assert_eq!(want.message, "division by zero");
    for cfg in par_configs() {
        let got = kernel
            .run_alloc_par(&inputs, n, &cfg)
            .unwrap_err()
            .to_diagnostic();
        assert_eq!(got, want, "{cfg:?}");
    }
}

#[test]
fn a_run_under_the_threshold_or_on_one_thread_is_not_split() {
    let kernel = compile(
        "      %t = add %x, %y : num[1; ()]\n      yield %t",
        Options::default(),
    );
    assert!(!kernel.splits(1_000_000, &Config::sequential()));
    assert!(!kernel.splits(1, &Config::eager(4)));
    assert!(!kernel.splits(100, &Config::with_threads(4)));
    assert!(kernel.splits(4096, &Config::with_threads(4)));
    assert!(kernel.splits(2, &Config::eager(2)));
}

#[test]
fn a_chunked_run_on_the_wrong_operands_is_an_error_not_a_crash() {
    let kernel = compile(
        "      %t = add %x, %y : num[1; ()]\n      yield %t",
        Options::default(),
    );
    let xs = vec![1.0; 100];
    let short = vec![1.0; 10];
    let cfg = Config::eager(4);
    let e = kernel
        .run_alloc_par(
            &[Input::Array(&xs), Input::Array(&short), Input::Scalar(0.0)],
            100,
            &cfg,
        )
        .unwrap_err();
    assert!(e.message.contains("malformed IR"), "{}", e.message);
    let e = kernel
        .run_alloc_par(&[Input::Array(&xs)], 100, &cfg)
        .unwrap_err();
    assert!(e.message.contains("malformed IR"), "{}", e.message);
}
