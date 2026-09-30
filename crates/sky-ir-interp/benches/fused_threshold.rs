//! Where a native kernel starts to pay: the interpreter's loop against a
//! native kernel, for a `fused` chain of `n` elements.
//!
//! `cargo bench -p sky-ir-interp --bench fused_threshold`
//!
//! Each row times one *call* of a function holding a `fused` of `n`
//! elements, in microseconds:
//!
//! * `interp`: the interpreter's loop (`JitOptions::OFF`);
//! * `native, cold`: native kernels on, in a fresh run, so the call
//!   includes compiling the body (what a run pays for the first long
//!   `fused` it meets);
//! * `native, warm`: the same call when the kernel is already compiled (a
//!   second `fused` of the same body in one run, timed by running the call
//!   twice and subtracting the cold call).
//!
//! `JitOptions::DEFAULT_THRESHOLD` is chosen where `native, cold` is first
//! clearly cheaper than `interp`: below it a single use loses the compile
//! time, above it the kernel wins even if used once.

use std::hint::black_box;
use std::time::Instant;

use sky_ir::parse;
use sky_ir_interp::{call_with_jit, JitOptions, Value};
use sky_units::Dimension;

const BODIES: [(&str, &str); 2] = [
    (
        "a*x+y (2 ops)",
        "      %t = mul %x, %s : num[1; ()]
      %u = add %t, %y : num[1; ()]
      yield %u",
    ),
    (
        "6 ops with a division",
        "      %t1 = mul %x, %s : num[1; ()]
      %t2 = add %t1, %y : num[1; ()]
      %t3 = mul %x, %x : num[1; ()]
      %c = const.float 1.5 : num[1; ()]
      %t4 = add %t3, %c : num[1; ()]
      %t5 = div %t2, %t4 : num[1; ()]
      %t6 = sub %t5, %y : num[1; ()]
      yield %t6",
    ),
];

fn best_us(reps: usize, mut f: impl FnMut()) -> f64 {
    f();
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e6
        })
        .fold(f64::MAX, f64::min)
}

fn main() {
    for (name, body) in BODIES {
        let text = format!(
            "sir edition 2026\n\nfn @f(%a: num[1; (3)], %b: num[1; (3)], %k: num[1; ()]) -> (num[1; (3)]) effects(fail) {{
  %r = fused %a, %b, %k : num[1; (3)] {{
    ^1(%x: num[1; ()], %y: num[1; ()], %s: num[1; ()]):
{body}
  }}
  return %r
}}
"
        );
        let m = parse(&text).expect("the IR parses");
        println!("\n{name}");
        println!(
            "{:>8} | {:>10} {:>14} {:>14} | {:>9}",
            "n", "interp us", "native cold us", "native warm us", "cold/interp"
        );
        for n in [
            16usize, 64, 128, 256, 512, 1024, 2048, 4096, 16_384, 131_072,
        ] {
            let xs: Vec<f64> = (0..n).map(|i| i as f64 * 0.01 + 0.5).collect();
            let ys: Vec<f64> = (0..n).map(|i| 1.0 / (1.0 + i as f64)).collect();
            let args = vec![
                Value::array(vec![n], xs, Dimension::DIMENSIONLESS),
                Value::array(vec![n], ys, Dimension::DIMENSIONLESS),
                Value::scalar(2.5),
            ];
            let reps = (2_000_000 / n).clamp(5, 2000);
            let run = |jit| {
                black_box(
                    call_with_jit(&m, "f", args.clone(), &mut Vec::new(), jit).expect("no failure"),
                );
            };
            let interp = best_us(reps, || run(JitOptions::OFF));
            let cold = best_us(reps, || run(JitOptions::ALWAYS));
            // The clone of the arguments is in both; the native kernel's own
            // cost with a warm cache is what `native, warm` estimates: the
            // cold call minus the compile time, measured separately.
            let compile = compile_us(&m, &args);
            let warm = (cold - compile).max(0.0);
            println!(
                "{n:>8} | {interp:>10.1} {cold:>14.1} {warm:>14.1} | {:>8.2}x",
                cold / interp
            );
        }
    }
}

/// The time to compile the body: a native call of a one-element array
/// (which the kernel handles as a scalar tail) minus the same call with the
/// kernel already compiled is not separable from outside, so this compiles
/// through the public kernel API.
fn compile_us(m: &sky_ir::Module, args: &[Value]) -> f64 {
    let f = &m.functions[0];
    let op = f
        .ops
        .iter()
        .find(|o| o.kind == sky_ir::OpKind::Fused)
        .expect("a fused op");
    let sigs: Vec<sky_jit::kernel::InputSig> = args
        .iter()
        .map(|v| match v {
            Value::Array { dim, .. } => sky_jit::kernel::InputSig {
                dim: *dim,
                array: true,
            },
            Value::Number { dim, .. } => sky_jit::kernel::InputSig {
                dim: *dim,
                array: false,
            },
            _ => unreachable!("numbers and arrays only"),
        })
        .collect();
    best_us(20, || {
        black_box(sky_jit::kernel::compile_kernel(f, op, &sigs).expect("compiles"));
    })
}
