//! `sky-bench par --native`: how much automatic parallelism buys on native
//! code (`sky_jit`), and where it starts to pay: the numbers behind
//! `sky_ir::parallel::NATIVE_CALL_COST_SCALE` and the speedup tables of
//! `docs/design/ir.md`.
//!
//! ```text
//! cargo run --release -p sky-bench -- par --native [--threads 1,2,4,8] [--reps N] [--quick]
//! ```
//!
//! Tables: the compiled 4-operation kernel over 10^4 to 10^7 elements
//! (output buffer reused, and a fresh output as the runtime calls it);
//! elementwise `add` of two arrays; two and four independent calls of a loop
//! over how much each does; and the price of starting a call against what
//! native code spends per operation, which is the break-even that the native
//! call threshold is set from. Each number is the fastest of `--reps` runs;
//! every measurement first checks that the threaded run gave the bits of the
//! sequential one.

use std::io::sink;
use std::time::Instant;

use sky_diag::Span;
use sky_interp::Value;
use sky_ir::parallel::{Config, NATIVE_CALL_COST_SCALE};
use sky_ir::{parse, OpKind};
use sky_jit::kernel::{compile_kernel, Input, InputSig, Kernel};
use sky_units::Dimension;

use crate::par::Options;

/// The fastest of `reps` runs of `f`, in nanoseconds.
fn fastest(reps: usize, mut f: impl FnMut()) -> u64 {
    f();
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos() as u64
        })
        .min()
        .unwrap_or(0)
}

fn ms(ns: u64) -> String {
    if ns >= 10_000_000 {
        format!("{:.1}", ns as f64 / 1e6)
    } else if ns >= 100_000 {
        format!("{:.2}", ns as f64 / 1e6)
    } else {
        format!("{:.3}", ns as f64 / 1e6)
    }
}

/// A configuration that splits whatever is big enough to have two chunks, and
/// starts every call.
fn always(threads: usize) -> Config {
    Config {
        fused_work: 0,
        array_elements: 1,
        call_cost: 0,
        ..Config::with_threads(threads)
    }
}

fn header(o: &Options, first: &str) -> String {
    let mut s = format!("{first:>12}");
    for t in &o.threads {
        s.push_str(&format!(" {:>9}", format!("{t}t ms")));
    }
    for t in o.threads.iter().skip(1) {
        s.push_str(&format!(" {:>7}", format!("x{t}")));
    }
    s
}

fn row(label: &str, ts: &[u64]) -> String {
    let mut s = format!("{label:>12}");
    for t in ts {
        s.push_str(&format!(" {:>9}", ms(*t)));
    }
    for t in ts.iter().skip(1) {
        s.push_str(&format!(" {:>7.2}", ts[0] as f64 / (*t).max(1) as f64));
    }
    s
}

/// The 4-operation kernel of `sky-bench par`, compiled for two array operands
/// and a scalar.
fn kernel(n: usize) -> Kernel {
    let text = format!(
        "sir edition 2026\n\nfn @f(%a: num[1; ({n})], %b: num[1; ({n})], %k: num[1; ()]) -> (num[1; ({n})]) effects(fail) {{
  %r = fused %a, %b, %k : num[1; ({n})] {{
    ^1(%x: num[1; ()], %y: num[1; ()], %s: num[1; ()]):
      %t = add %x, %y : num[1; ()]
      %u = mul %t, %s : num[1; ()]
      %w = sub %u, %x : num[1; ()]
      %v = add %w, %y : num[1; ()]
      yield %v
  }}
  return %r
}}
"
    );
    let m = parse(&text).expect("the kernel parses");
    let f = &m.functions[0];
    let op = f
        .ops
        .iter()
        .find(|o| o.kind == OpKind::Fused)
        .expect("a fused operation");
    let dimensionless = Dimension::DIMENSIONLESS;
    let sigs = [
        InputSig {
            dim: dimensionless,
            array: true,
        },
        InputSig {
            dim: dimensionless,
            array: true,
        },
        InputSig {
            dim: dimensionless,
            array: false,
        },
    ];
    compile_kernel(f, op, &sigs).expect("the kernel compiles")
}

fn bits(xs: &[f64]) -> Vec<u64> {
    xs.iter().map(|x| x.to_bits()).collect()
}

/// The compiled kernel over `n` elements under every thread count; `fresh`
/// allocates the output on every run (as the runtime does), else it is reused.
fn kernel_times(o: &Options, n: usize, fresh: bool) -> Vec<u64> {
    let k = kernel(n);
    let a: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
    let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
    let inputs = [Input::Array(&a), Input::Array(&b), Input::Scalar(0.37)];
    let want = k.run_alloc(&inputs, n).expect("runs");
    let mut out = vec![0.0; n];
    o.threads
        .iter()
        .map(|&t| {
            let cfg = always(t);
            let got = k.run_alloc_par(&inputs, n, &cfg).expect("runs");
            assert_eq!(bits(&got), bits(&want), "the threaded kernel differs");
            if fresh {
                fastest(o.reps, || {
                    let _ = std::hint::black_box(k.run_alloc_par(&inputs, n, &cfg));
                })
            } else {
                fastest(o.reps, || {
                    k.run_par(&inputs, &mut out, &cfg).expect("runs");
                    std::hint::black_box(&out);
                })
            }
        })
        .collect()
}

fn kernel_table(o: &Options, title: &str, fresh: bool) {
    println!("{title}");
    println!("{}", header(o, "elements"));
    let sizes: &[usize] = if o.quick {
        &[10_000, 100_000]
    } else {
        &[10_000, 100_000, 1_000_000, 10_000_000]
    };
    for &n in sizes {
        println!("{}", row(&n.to_string(), &kernel_times(o, n, fresh)));
    }
    println!();
}

/// The smallest size, over a sweep, at which `threads` threads run the
/// compiled kernel at least 10% faster than one (the output buffer reused):
/// the break-even that `fused_work` (in element-operations: four per element
/// here) is set from for native code.
fn kernel_break_even(o: &Options, threads: usize) {
    let mut n = 1024usize;
    let limit = if o.quick { 1 << 16 } else { 1 << 24 };
    let mut found = None;
    let mut sweep = Vec::new();
    while n <= limit {
        let k = kernel(n);
        let a: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
        let inputs = [Input::Array(&a), Input::Array(&b), Input::Scalar(0.37)];
        let mut out = vec![0.0; n];
        let seq = fastest(o.reps, || {
            k.run(&inputs, &mut out).expect("runs");
            std::hint::black_box(&out);
        });
        let cfg = always(threads);
        let par = fastest(o.reps, || {
            k.run_par(&inputs, &mut out, &cfg).expect("runs");
            std::hint::black_box(&out);
        });
        sweep.push(format!("{n}: {}/{} us", seq / 1000, par / 1000));
        if found.is_none() && (par as f64) < seq as f64 * 0.9 {
            found = Some(n);
        }
        n *= 2;
    }
    println!(
        "compiled kernel: {threads} threads first win by 10% at {} elements ({} element-operations)",
        found.map_or("(never in the sweep)".to_string(), |n| n.to_string()),
        found.map_or("-".to_string(), |n| (4 * n).to_string()),
    );
    println!("  sequential/parallel per size: {}", sweep.join(", "));
    println!();
}

fn elementwise_table(o: &Options) {
    println!("elementwise `add` of two arrays on native code (ms; the runtime's loop)");
    println!("{}", header(o, "elements"));
    let sizes: &[usize] = if o.quick {
        &[100_000, 1_000_000]
    } else {
        &[100_000, 1_000_000, 4_000_000]
    };
    for &n in sizes {
        let vector = |f: fn(usize) -> f64| {
            Value::array(
                vec![n],
                (0..n).map(f).collect(),
                sky_units::Dimension::DIMENSIONLESS,
            )
        };
        let (a, b) = (vector(|i| i as f64), vector(|i| 2.0 * i as f64));
        let span = Span::new(0, 0);
        let run = |cfg: &Config| {
            sky_jit::par::elementwise(&OpKind::Add, &[&a, &b], span, cfg)
                .expect("a big add is split")
                .expect("no overflow")
        };
        let want = run(&always(2));
        let ts: Vec<u64> = o
            .threads
            .iter()
            .map(|&t| {
                let cfg = always(t);
                if t == 1 {
                    // One thread is the sequential loop, as `sky_interp` runs it.
                    return fastest(o.reps, || {
                        let _ = std::hint::black_box(sky_interp::arith::arith(
                            sky_interp::arith::ArithOp::Add,
                            &a,
                            &b,
                            span,
                            span,
                            span,
                        ));
                    });
                }
                assert_eq!(run(&cfg), want, "the threaded add differs");
                fastest(o.reps, || {
                    let _ = std::hint::black_box(run(&cfg));
                })
            })
            .collect();
        println!("{}", row(&n.to_string(), &ts));
    }
    println!();
}

/// A program with `calls` independent calls of a loop of `iterations`.
fn calls_program(calls: usize, iterations: u64) -> String {
    let mut src = String::from(
        "def work(n):\n    total = 0\n    i = 0\n    while i < n:\n        total = total + sqrt(i + 1) / (i + 1)\n        i = i + 1\n    return total\n",
    );
    let mut sum = String::new();
    for k in 0..calls {
        src.push_str(&format!("r{k} = work({})\n", iterations + k as u64));
        if k > 0 {
            sum.push_str(" + ");
        }
        sum.push_str(&format!("r{k}"));
    }
    src.push_str(&format!("print({sum})\n"));
    src
}

fn compile_ir(src: &str) -> sky_ir::Module {
    let mut ir = sky_lower::lower_source(src).expect("lowers");
    sky_opt::default_pipeline()
        .verify_each(false)
        .run(&mut ir)
        .expect("the pipeline runs");
    ir
}

/// Native code for `ir` under `cfg`, and what it prints.
fn native_run(ir: &sky_ir::Module, cfg: &Config) -> (sky_jit::Compiled, Vec<u8>) {
    let code = sky_jit::compile_par(ir, sky_jit::Numerics::Reproducible, sky_jit::Slp::Auto, cfg)
        .expect("compiles");
    let mut out = Vec::new();
    code.run(&mut out).expect("runs");
    (code, out)
}

/// Times one run of the compiled program (compiled once, outside the timing).
fn time_run(reps: usize, code: &sky_jit::Compiled) -> u64 {
    fastest(reps, || {
        code.run(&mut sink()).expect("runs");
    })
}

fn calls_table(o: &Options) {
    for &k in &[2usize, 4] {
        println!(
            "{k} independent calls of a loop of N iterations on native code (ms; fastest of {})",
            o.reps
        );
        println!("{}", header(o, "iterations"));
        let sizes: &[u64] = if o.quick {
            &[1000, 100_000]
        } else {
            &[100, 1000, 10_000, 100_000, 1_000_000, 10_000_000]
        };
        for &n in sizes {
            let ir = compile_ir(&calls_program(k, n));
            let (_, want) = native_run(&ir, &Config::sequential());
            let ts: Vec<u64> = o
                .threads
                .iter()
                .map(|&t| {
                    let (code, got) = native_run(&ir, &always(t));
                    assert_eq!(got, want, "the threaded output differs");
                    time_run(o.reps, &code)
                })
                .collect();
            println!("{}", row(&n.to_string(), &ts));
        }
        println!();
    }
}

/// The price of starting calls on workers, against native code's price per
/// operation: their ratio is how many operations a call must contain to be
/// worth it, which is what `NATIVE_CALL_COST_SCALE` times the interpreter's
/// `call_cost` is set to cover.
fn call_cost(o: &Options) {
    // Native code's price per operation: a loop of 7 operations.
    let iterations = 2_000_000u64;
    let ir = compile_ir(&calls_program(1, iterations));
    let (code, _) = native_run(&ir, &Config::sequential());
    let seq = time_run(o.reps, &code);
    let per_op_ns = seq as f64 / (iterations as f64 * 7.0);
    // The fixed price: groups of two calls of a body that does almost
    // nothing, started against called in place.
    let groups = 40_000u64;
    let tiny = compile_ir(&format!(
        "def w(x):\n    return x + 1\ntotal = 0\ni = 0\nwhile i < {groups}:\n    total = total + w(i) + w(i + 1)\n    i = i + 1\nprint(total)\n"
    ));
    let (seq_code, _) = native_run(&tiny, &Config::sequential());
    let (par_code, _) = native_run(&tiny, &always(4));
    let s = time_run(o.reps, &seq_code);
    let p = time_run(o.reps, &par_code);
    let overhead = p.saturating_sub(s) as f64 / groups as f64;
    println!("start-up price of a two-call group and the price per operation on native code");
    println!(
        "  per operation: {per_op_ns:.2} ns (a loop body of ~7 operations, {iterations} iterations)"
    );
    println!(
        "  {groups} groups of two trivial calls: {} ms in place, {} ms as groups: {:.0} ns extra per group",
        ms(s),
        ms(p),
        overhead
    );
    let break_even = overhead / per_op_ns.max(0.001);
    let threshold = Config::with_threads(2).for_native().call_cost;
    println!(
        "  break-even: {break_even:.0} operations per call; the native threshold is {threshold} estimated operations ({NATIVE_CALL_COST_SCALE} times the interpreter's)"
    );
    println!();
}

/// Everything runs on one deep-stack thread: native code is compiled and run
/// on the same thread (`sky_jit::Compiled` is not `Send`).
pub fn main(o: &Options) {
    sky_ast::with_deep_stack(|| run_all(o));
}

fn run_all(o: &Options) {
    if cfg!(debug_assertions) {
        eprintln!(
            "warning: a debug build; use `cargo run --release -p sky-bench -- par --native`\n"
        );
    }
    println!(
        "automatic parallelism on native code: {} cores available; the fastest of {} runs\n",
        std::thread::available_parallelism().map_or(1, usize::from),
        o.reps
    );
    kernel_table(
        o,
        "compiled fused kernel of 4 operations, output buffer reused (ms)",
        false,
    );
    kernel_table(
        o,
        "compiled fused kernel of 4 operations, fresh output as the runtime allocates it (ms)",
        true,
    );
    kernel_break_even(o, 2);
    kernel_break_even(o, 4);
    elementwise_table(o);
    calls_table(o);
    call_cost(o);
}
