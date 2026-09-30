//! Compiled fused kernels against hand-written loops (ROADMAP Phase 4:
//! "fused elementwise code in Sky within 2x of hand-tuned").
//!
//! `cargo bench -p sky-jit --bench kernels`
//!
//! Three kernels (`a*x + y`, `sqrt(x*x + y*y)`, and a six-operation chain
//! with a division) on 1e3, 1e5 and 1e7 elements, each timed as the best of
//! nine batches, in nanoseconds per element, with the output array
//! preallocated (as the baselines' is):
//!
//! * `sky`: the shipping kernel (128-bit vectors, four per iteration, and
//!   Sky's finiteness check);
//! * `nochk`: the same with the check compiled out, which is *not* Sky's
//!   semantics (an overflow would leave an infinity in the output): it
//!   isolates what the check costs;
//! * `scalar`: no vectors, the check kept: what SIMD buys;
//! * `rust`, `c`: hand-written loops in Rust (`rustc -C opt-level=3 -C
//!   target-cpu=native`) and in C (`clang -O3 -march=native
//!   -ffp-contract=off -fno-math-errno`), built from `benches/baselines/`
//!   when the compilers exist, in two forms: plain arithmetic, and with the
//!   same finiteness checks Sky makes (`+chk`).
//!
//! The ratios that matter are `sky` and `nochk` over the best plain
//! baseline (the 2x target is for arithmetic kernels), and `sky` over the
//! best checked one (the same work).

use std::collections::BTreeMap;
use std::hint::black_box;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use sky_ir::{parse, OpKind};
use sky_jit::kernel::{compile_kernel_with, Input, InputSig, Kernel, Options};
use sky_units::Dimension;

const SIZES: [usize; 3] = [1_000, 100_000, 10_000_000];

const KERNELS: [(&str, &str); 3] = [
    (
        "axpy",
        "      %t = mul %x, %s : num[1; ()]
      %u = add %t, %y : num[1; ()]
      yield %u",
    ),
    (
        "hypot",
        "      %p = mul %x, %x : num[1; ()]
      %q = mul %y, %y : num[1; ()]
      %c = add %p, %q : num[1; ()]
      %w = math.sqrt %c : num[1; ()]
      yield %w",
    ),
    (
        "chain6",
        "      %t1 = mul %x, %s : num[1; ()]
      %t2 = add %t1, %y : num[1; ()]
      %t3 = mul %x, %x : num[1; ()]
      %c15 = const.float 1.5 : num[1; ()]
      %t4 = add %t3, %c15 : num[1; ()]
      %t5 = div %t2, %t4 : num[1; ()]
      %t6 = sub %t5, %y : num[1; ()]
      yield %t6",
    ),
];

fn compile(body: &str, options: Options) -> (Kernel, f64) {
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
    let m = parse(&text).expect("the benchmark's IR parses");
    let f = &m.functions[0];
    let op = f
        .ops
        .iter()
        .find(|o| o.kind == OpKind::Fused)
        .expect("a fused op");
    let sig = |array| InputSig {
        dim: Dimension::DIMENSIONLESS,
        array,
    };
    let sigs = [sig(true), sig(true), sig(false)];
    let start = Instant::now();
    let kernel = compile_kernel_with(f, op, &sigs, options).expect("the kernel compiles");
    (kernel, start.elapsed().as_secs_f64() * 1e3)
}

fn time(n: usize, mut f: impl FnMut()) -> f64 {
    let reps = (20_000_000 / n).max(1);
    f();
    let mut best = f64::MAX;
    for _ in 0..9 {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        best = best.min(t.elapsed().as_secs_f64() / (reps * n) as f64);
    }
    best * 1e9
}

/// Builds and runs a baseline, returning `(kernel, variant, n) -> ns`.
fn baseline(
    tool: &str,
    build: &mut Command,
    out: &std::path::Path,
) -> BTreeMap<(String, String, usize), f64> {
    let mut results = BTreeMap::new();
    match build.status() {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("({tool} is not available: no {tool} baseline)");
            return results;
        }
    }
    let Ok(run) = Command::new(out).output() else {
        return results;
    };
    for line in String::from_utf8_lossy(&run.stdout).lines() {
        let w: Vec<&str> = line.split_whitespace().collect();
        if let [_, kernel, variant, n, ns] = w[..] {
            if let (Ok(n), Ok(ns)) = (n.parse(), ns.parse()) {
                results.insert((kernel.to_string(), variant.to_string(), n), ns);
            }
        }
    }
    results
}

fn main() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/baselines");
    let tmp = std::env::var_os("CARGO_TARGET_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let _ = std::fs::create_dir_all(&tmp);
    let (rs_exe, c_exe) = (tmp.join("sky_baseline_rs"), tmp.join("sky_baseline_c"));
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let rust = baseline(
        "rustc",
        Command::new(rustc)
            .args(["-C", "opt-level=3", "-C", "target-cpu=native"])
            .arg(dir.join("baseline.rs"))
            .arg("-o")
            .arg(&rs_exe),
        &rs_exe,
    );
    let c = baseline(
        "clang",
        Command::new("clang")
            .args([
                "-O3",
                "-march=native",
                "-ffp-contract=off",
                "-fno-math-errno",
            ])
            .arg(dir.join("baseline.c"))
            .arg("-o")
            .arg(&c_exe)
            .arg("-lm"),
        &c_exe,
    );

    let variants = [
        ("sky", Options::default()),
        (
            "nochk",
            Options {
                check_finite: false,
                ..Options::default()
            },
        ),
        (
            "scalar",
            Options {
                simd: false,
                ..Options::default()
            },
        ),
    ];
    println!(
        "ns per element (best of 9 batches); host {}",
        std::env::consts::ARCH
    );
    println!(
        "{:<8} {:>9} | {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7} {:>7} | {:>9} {:>9} {:>9}",
        "kernel",
        "n",
        "sky",
        "nochk",
        "scalar",
        "rust",
        "rust+chk",
        "c",
        "c+chk",
        "sky/best",
        "nochk/best",
        "sky/chk"
    );
    let mut compile_ms = Vec::new();
    for (name, body) in KERNELS {
        let kernels: Vec<(Kernel, f64)> = variants.iter().map(|(_, o)| compile(body, *o)).collect();
        // Compile time: the median of five (the first includes warming up).
        let mut times: Vec<f64> = (0..5)
            .map(|_| compile(body, Options::default()).1)
            .collect();
        times.sort_by(f64::total_cmp);
        compile_ms.push((name, times[2]));
        for n in SIZES {
            let x: Vec<f64> = (0..n).map(|i| (i % 1000) as f64 * 0.001 + 0.5).collect();
            let y: Vec<f64> = (0..n).map(|i| (i % 777) as f64 * 0.002 - 0.7).collect();
            let mut out = vec![0.0; n];
            let mut own = Vec::new();
            for (kernel, _) in &kernels {
                let ns = time(n, || {
                    kernel
                        .run(
                            &[
                                Input::Array(black_box(&x)),
                                Input::Array(black_box(&y)),
                                Input::Scalar(black_box(2.5)),
                            ],
                            &mut out,
                        )
                        .expect("no element fails");
                });
                own.push(ns);
            }
            let get = |m: &BTreeMap<(String, String, usize), f64>, variant: &str| {
                m.get(&(name.to_string(), variant.to_string(), n)).copied()
            };
            let best = |variant: &str| {
                [get(&rust, variant), get(&c, variant)]
                    .into_iter()
                    .flatten()
                    .min_by(f64::total_cmp)
            };
            let (plain, checked) = (best("plain"), best("checked"));
            let cell = |v: Option<f64>| v.map_or("-".to_string(), |v| format!("{v:.3}"));
            let ratio =
                |a: f64, b: Option<f64>| b.map_or("-".to_string(), |b| format!("{:.2}x", a / b));
            println!(
                "{:<8} {:>9} | {:>7.3} {:>7.3} {:>7.3} | {:>7} {:>7} {:>7} {:>7} | {:>9} {:>9} {:>9}",
                name,
                n,
                own[0],
                own[1],
                own[2],
                cell(get(&rust, "plain")),
                cell(get(&rust, "checked")),
                cell(get(&c, "plain")),
                cell(get(&c, "checked")),
                ratio(own[0], plain),
                ratio(own[1], plain),
                ratio(own[0], checked),
            );
        }
    }
    println!();
    for (name, ms) in compile_ms {
        println!("compile time of `{name}`: {ms:.3} ms (median of 5)");
    }
}
