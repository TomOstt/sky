//! What specialization buys the SIR interpreter, on programs whose helpers
//! are written for any size or any exponent and called with a fixed one.
//!
//! The ordinary test runs each program at a small size and requires the same
//! output with and without the pass. The measurement is `#[ignore]`d (it
//! needs `--release` and a quiet machine):
//!
//! ```text
//! cargo test --release -p sky-opt --test specialize_speed -- --ignored --nocapture
//! ```
//!
//! It runs each module (lowered and optimized by the pipeline with and
//! without `specialize`; the interpreter is the same) nine times, the two
//! alternating, and prints the fastest and the median time of each; the
//! speedup is the ratio of the fastest.

use std::time::{Duration, Instant};

use sky_ir::{Module, PassManager};
use sky_opt::{default_passes, default_passes_without_specialize};

/// The 3-vector physics loop of `benches/programs/vec3_physics.sky`.
const VEC3_PHYSICS: &str = include_str!("../../../benches/programs/vec3_physics.sky");

/// Recursion on a constant exponent: `power(x, 6)` with a computed `x`
/// becomes a chain of six multiplications, no calls.
const POWER: &str = "\
N = 200000

def power(x: Scalar, n: Scalar) -> Scalar:
    if n <= 0:
        return 1
    return x * power(x, n - 1)

s = 0
for i in range(N):
    s = s + power(1 + i * 1e-9, 6) * 1e-6
print(s)
";

/// A dot product over a 4-vector, its loop over `len` unrolled.
const DOT4: &str = "\
N = 200000

def dot(a: Vector[K], b: Vector[K]) -> Scalar:
    s = 0
    i = 0
    while i < len(a):
        s += a[i] * b[i]
        i += 1
    return s

a = [1, 2, 3, 4]
b = [0.5, -0.25, 0.125, 2]
s = 0
for i in range(N):
    s = s + dot(a, b) * 1e-6
print(s)
";

fn programs() -> [(&'static str, &'static str, &'static str); 3] {
    [
        ("vec3_physics", VEC3_PHYSICS, "N = 20000"),
        ("power", POWER, "N = 200000"),
        ("dot4", DOT4, "N = 200000"),
    ]
}

fn build(src: &str, specialize: bool) -> Module {
    let mut m = sky_lower::lower_source(src).unwrap_or_else(|e| panic!("{src}\n{e}"));
    let passes = if specialize {
        default_passes(PassManager::new().verify_each(true))
    } else {
        default_passes_without_specialize(PassManager::new().verify_each(true))
    };
    passes.run(&mut m).expect("valid IR");
    m
}

fn run(m: &Module) -> (String, Duration) {
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let start = Instant::now();
        sky_ir_interp::run(m, &mut out).expect("runs");
        (String::from_utf8(out).expect("UTF-8"), start.elapsed())
    })
}

fn quick(src: &str, full: &str) -> String {
    let n: String = full.replace("N = ", "");
    let small = (n.parse::<u64>().unwrap() / 1000).max(20);
    src.replacen(full, &format!("N = {small}"), 1)
}

#[test]
fn the_specialized_programs_print_the_same() {
    for (name, src, full) in programs() {
        let src = quick(src, full);
        let (plain, special) = (build(&src, false), build(&src, true));
        assert_ne!(plain, special, "{name}: nothing was specialized");
        assert_eq!(run(&plain).0, run(&special).0, "{name}");
    }
}

#[test]
#[ignore = "a measurement: run with --release --ignored --nocapture"]
fn specialization_speeds_up_the_sir_interpreter() {
    let reps = 9;
    println!(
        "{:<14}{:>16}{:>16}{:>10}{:>16}{:>16}",
        "program", "generic ms", "specialized ms", "speedup", "generic med", "specialized med"
    );
    for (name, src, _) in programs() {
        let (plain, special) = (build(src, false), build(src, true));
        let mut times: [Vec<Duration>; 2] = [Vec::new(), Vec::new()];
        let mut outputs = [String::new(), String::new()];
        for _ in 0..reps {
            for (i, m) in [&plain, &special].into_iter().enumerate() {
                let (out, t) = run(m);
                outputs[i] = out;
                times[i].push(t);
            }
        }
        assert_eq!(outputs[0], outputs[1], "{name}");
        // The fastest run is the one the least disturbed by other work on
        // the machine; the median is shown for the spread.
        let ms = |d: &Duration| d.as_secs_f64() * 1e3;
        let stats = |v: &mut Vec<Duration>| {
            v.sort();
            (ms(&v[0]), ms(&v[v.len() / 2]))
        };
        let ((a, am), (b, bm)) = (stats(&mut times[0]), stats(&mut times[1]));
        println!(
            "{name:<14}{a:>16.1}{b:>16.1}{:>9.2}x{am:>16.1}{bm:>16.1}",
            a / b
        );
        assert!(
            b < a,
            "{name}: specialization did not help ({a} ms vs {b} ms)"
        );
    }
}
