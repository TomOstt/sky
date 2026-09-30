//! `sky-bench par`: how much automatic parallelism (`sky_ir::parallel`) buys,
//! and where it starts to pay: the numbers behind the default thresholds of
//! `sky_ir::parallel::Config`.
//!
//! ```text
//! cargo run --release -p sky-bench -- par [--threads 1,2,4,8] [--reps N] [--quick]
//! ```
//!
//! Four tables: the speedup of a fused kernel and of a plain elementwise
//! operation over threads and sizes; the size at which each starts to win
//! (the break-even that `fused_work` and `array_elements` are set from);
//! the speedup of two and four independent calls over how much work each does;
//! and the fixed price of starting a call elsewhere against what the
//! interpreter spends per operation (the break-even of `call_cost`).
//!
//! Each number is the fastest of `--reps` runs, which is the steadiest on a
//! busy machine. Nothing here changes any output: every measurement
//! first checks that the parallel run printed what the sequential one did.

use std::io::sink;
use std::time::Instant;

use sky_ir::parallel::Config;
use sky_ir::{parse, Module};
use sky_ir_interp::{call_with, Value};
use sky_units::Dimension;

/// The options of `sky-bench par`.
pub struct Options {
    pub threads: Vec<usize>,
    pub reps: usize,
    pub quick: bool,
}

pub fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        threads: vec![1, 2, 4, 8],
        reps: 7,
        quick: false,
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--threads" => {
                let v = it.next().ok_or("--threads needs a list like 1,2,4,8")?;
                o.threads = v
                    .split(',')
                    .map(|t| {
                        t.trim()
                            .parse::<usize>()
                            .ok()
                            .filter(|&n| n >= 1)
                            .ok_or_else(|| format!("`{t}` is not a thread count"))
                    })
                    .collect::<Result<_, _>>()?;
                if !o.threads.contains(&1) {
                    o.threads.insert(0, 1);
                }
            }
            "--reps" => {
                let v = it.next().ok_or("--reps needs a number")?;
                o.reps = v
                    .parse::<usize>()
                    .ok()
                    .filter(|&n| n >= 1)
                    .ok_or_else(|| format!("`{v}` is not a repetition count"))?;
            }
            "--quick" => o.quick = true,
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    Ok(o)
}

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

fn vector(n: usize, f: impl Fn(usize) -> f64) -> Value {
    Value::array(vec![n], (0..n).map(f).collect(), Dimension::DIMENSIONLESS)
}

fn kernel_module(n: usize) -> Module {
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
    parse(&text).expect("the kernel parses")
}

fn add_module(n: usize) -> Module {
    let text = format!(
        "sir edition 2026\n\nfn @f(%a: num[1; ({n})], %b: num[1; ({n})]) -> (num[1; ({n})]) effects(fail) {{
  %r = add %a, %b : num[1; ({n})]
  return %r
}}
"
    );
    parse(&text).expect("the add parses")
}

/// A configuration that splits whatever is big enough to have two chunks.
fn always(threads: usize) -> Config {
    Config {
        fused_work: 0,
        array_elements: 1,
        call_cost: 0,
        ..Config::with_threads(threads)
    }
}

fn run_f(m: &Module, args: &[Value], cfg: &Config) -> Vec<Value> {
    call_with(m, "f", args.to_vec(), &mut sink(), cfg).expect("runs")
}

/// What passing `args` costs (the interpreter takes them by value), which
/// every timing of `f` has in it and this takes out.
fn clone_cost(o: &Options, args: &[Value]) -> u64 {
    fastest(o.reps, || {
        std::hint::black_box(args.to_vec());
    })
}

/// Times `f` under every thread count, without the cost of passing the
/// arguments; the results must agree.
fn times(o: &Options, m: &Module, args: &[Value]) -> Vec<u64> {
    let want = run_f(m, args, &Config::sequential());
    let clone = clone_cost(o, args);
    o.threads
        .iter()
        .map(|&t| {
            let cfg = always(t);
            assert_eq!(run_f(m, args, &cfg), want, "parallel result differs");
            fastest(o.reps, || {
                let _ = run_f(m, args, &cfg);
            })
            .saturating_sub(clone)
        })
        .collect()
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

fn speedups(o: &Options, title: &str, make: impl Fn(usize) -> (Module, Vec<Value>)) {
    println!("{title}");
    println!("{}", header(o, "elements"));
    let sizes: &[usize] = if o.quick {
        &[10_000, 100_000]
    } else {
        &[10_000, 100_000, 1_000_000, 4_000_000]
    };
    for &n in sizes {
        let (m, args) = make(n);
        println!("{}", row(&n.to_string(), &times(o, &m, &args)));
    }
    println!();
}

/// The smallest size, over a sweep, at which `threads` threads beat one by at
/// least 10%.
fn break_even(
    o: &Options,
    title: &str,
    threads: usize,
    make: impl Fn(usize) -> (Module, Vec<Value>),
) {
    let mut n = 128usize;
    let limit = if o.quick { 4096 } else { 1 << 23 };
    let mut found = None;
    let mut sweep = Vec::new();
    while n <= limit {
        let (m, args) = make(n);
        let clone = clone_cost(o, &args);
        let seq = fastest(o.reps, || {
            let _ = run_f(&m, &args, &Config::sequential());
        })
        .saturating_sub(clone);
        let cfg = always(threads);
        let par = fastest(o.reps, || {
            let _ = run_f(&m, &args, &cfg);
        })
        .saturating_sub(clone);
        sweep.push(format!("{n}: {}/{} us", seq / 1000, par / 1000));
        if found.is_none() && (par as f64) < seq as f64 * 0.9 {
            found = Some(n);
        }
        n *= 2;
    }
    println!(
        "{title}: {threads} threads first win by 10% at {} elements",
        found.map_or("(never in the sweep)".to_string(), |n| n.to_string())
    );
    println!("  sequential/parallel per size: {}", sweep.join(", "));
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

fn run_program(ir: &Module, cfg: &Config) -> Vec<u8> {
    let mut out = Vec::new();
    sky_ast::with_deep_stack(|| sky_ir_interp::run_with(ir, &mut out, cfg)).expect("runs");
    out
}

fn compile(src: &str) -> Module {
    let mut ir = sky_lower::lower_source(src).expect("lowers");
    sky_opt::default_pipeline()
        .verify_each(false)
        .run(&mut ir)
        .expect("the pipeline runs");
    ir
}

fn calls(o: &Options) {
    for &k in &[2usize, 4] {
        println!(
            "{k} independent calls of a loop of N iterations (ms; fastest of {})",
            o.reps
        );
        println!("{}", header(o, "iterations"));
        let sizes: &[u64] = if o.quick {
            &[10, 1000]
        } else {
            &[1, 10, 30, 100, 300, 1000, 10_000, 100_000]
        };
        for &n in sizes {
            let ir = compile(&calls_program(k, n));
            let want = run_program(&ir, &Config::sequential());
            let ts: Vec<u64> = o
                .threads
                .iter()
                .map(|&t| {
                    let cfg = always(t);
                    assert_eq!(run_program(&ir, &cfg), want, "parallel output differs");
                    fastest(o.reps, || {
                        let _ = run_program(&ir, &cfg);
                    })
                })
                .collect();
            println!("{}", row(&n.to_string(), &ts));
        }
        println!();
    }
}

/// The price of starting calls elsewhere, against the interpreter's price per
/// operation: their ratio is how many operations a call must contain to
/// be worth it.
fn call_cost(o: &Options) {
    // The interpreter's price per operation: a loop of 7 operations.
    let iterations = 20_000u64;
    let ir = compile(&calls_program(1, iterations));
    let seq = fastest(o.reps, || {
        let _ = run_program(&ir, &Config::sequential());
    });
    let per_op_ns = seq as f64 / (iterations as f64 * 7.0);
    // The fixed price: a loop of 4000 groups of two calls of a body that does
    // almost nothing, in parallel against in sequence.
    let groups = 4000u64;
    let tiny = compile(&format!(
        "def w(x):\n    return x + 1\ntotal = 0\ni = 0\nwhile i < {groups}:\n    total = total + w(i) + w(i + 1)\n    i = i + 1\nprint(total)\n"
    ));
    let cfg = always(4);
    let s = fastest(o.reps, || {
        let _ = run_program(&tiny, &Config::sequential());
    });
    let p = fastest(o.reps, || {
        let _ = run_program(&tiny, &cfg);
    });
    let overhead = p.saturating_sub(s) as f64 / groups as f64;
    println!("start-up price of a two-call group and the price per interpreted operation");
    println!("  per operation: {per_op_ns:.1} ns (a loop body of ~7 operations, {iterations} iterations)");
    println!(
        "  {groups} groups of two trivial calls: {} ms sequential, {} ms as groups: {:.0} ns extra per group",
        ms(s),
        ms(p),
        overhead
    );
    println!(
        "  break-even: {:.0} operations per call (a call must cost more than that to be started elsewhere)",
        overhead / per_op_ns.max(0.001)
    );
    println!();
}

pub fn main(o: &Options) {
    if cfg!(debug_assertions) {
        eprintln!("warning: a debug build; use `cargo run --release -p sky-bench -- par`\n");
    }
    println!(
        "automatic parallelism: {} cores available; the fastest of {} runs\n",
        std::thread::available_parallelism().map_or(1, usize::from),
        o.reps
    );
    speedups(o, "fused kernel of 4 operations (ms)", |n| {
        (
            kernel_module(n),
            vec![
                vector(n, |i| i as f64 * 0.5),
                vector(n, |i| (i % 7) as f64),
                Value::scalar(0.37),
            ],
        )
    });
    speedups(o, "elementwise `add` of two arrays (ms)", |n| {
        (
            add_module(n),
            vec![vector(n, |i| i as f64), vector(n, |i| 2.0 * i as f64)],
        )
    });
    break_even(o, "fused kernel", 4, |n| {
        (
            kernel_module(n),
            vec![
                vector(n, |i| i as f64 * 0.5),
                vector(n, |i| (i % 7) as f64),
                Value::scalar(0.37),
            ],
        )
    });
    break_even(o, "elementwise add", 4, |n| {
        (
            add_module(n),
            vec![vector(n, |i| i as f64), vector(n, |i| 2.0 * i as f64)],
        )
    });
    calls(o);
    call_cost(o);
}
