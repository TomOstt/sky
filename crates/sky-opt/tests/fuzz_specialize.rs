//! Specialization must not change what a program does: for random programs,
//! the default pipeline with `specialize` and without it (and the lowered
//! program) give the same stdout and the same error, compared as the whole
//! rendered diagnostic (message, spans, labels, secondary labels, notes).
//!
//! Two sources of programs:
//!
//! * the differential fuzzer's generator (`fuzz/harness/differential.rs`),
//!   as in `fuzz_pipeline.rs`: scalar programs with functions called with
//!   constants, contracts, loops, recursion and run-time failures;
//! * a generator of array programs in this file, because that one has no
//!   arrays: functions over `Vector[N]` (loops over `len`, indexing,
//!   contracts on sizes, failures inside loops), called with vectors of
//!   random sizes and with constants, so shapes, constants, signs and loops
//!   of constant trip count are all in play.
//!
//! `SKY_FUZZ_FROM` and `SKY_FUZZ_SEEDS` run a longer campaign (use
//! `--release`), and the test prints how often the pass did something, so a
//! campaign that tests nothing is visible:
//!
//! ```text
//! SKY_FUZZ_SEEDS=50000 cargo test --release -p sky-opt --test fuzz_specialize -- --nocapture
//! ```

#[path = "../../../fuzz/harness/differential.rs"]
mod harness;

use std::sync::atomic::{AtomicUsize, Ordering};

use sky_diag::{Diagnostic, SourceFile};
use sky_ir::{Module, PassManager};
use sky_opt::{default_passes, default_passes_without_specialize};

/// A deterministic byte stream for a seed: xorshift64*.
fn bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 56) as u8
        })
        .collect()
}

fn env_number(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

// ----- array programs -----

/// The choices of an array program, read from a seed.
struct Rng(Vec<u8>, usize);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(bytes(seed ^ 0xA5A5_5A5A, 256), 0)
    }

    fn below(&mut self, n: usize) -> usize {
        let b = self.0[self.1 % self.0.len()] as usize;
        self.1 += 1;
        b % n
    }

    fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[self.below(xs.len())]
    }
}

const FUNCTIONS: &[(&str, &str)] = &[
    (
        "dot",
        "def dot(a: Vector[N], b: Vector[N]) -> Scalar:\n    s = 0\n    i = 0\n    while i < len(a):\n        s += a[i] * b[i]\n        i += 1\n    return s\n",
    ),
    (
        "recips",
        "def recips(a: Vector[N], c: Scalar) -> Scalar:\n    s = 0\n    for i in range(len(a)):\n        s += 1 / (a[i] - c)\n    return s\n",
    ),
    (
        "pick",
        "def pick(v: Vector[N], i: Scalar) -> Scalar:\n    return v[i]\n",
    ),
    (
        "axpy",
        "def axpy(k: Scalar, x: Vector[N], y: Vector[N]) -> Vector[N]:\n    return x * k + y\n",
    ),
    (
        "checked",
        "def checked(v: Vector[N], lo: Scalar) -> Scalar:\n    requires len(v) >= 2, \"needs two entries\"\n    requires lo >= 0\n    return v[0] + v[1] * lo\n",
    ),
    (
        "power",
        "def power(x: Scalar, n: Scalar) -> Scalar:\n    if n <= 0:\n        return 1\n    return x * power(x, n - 1)\n",
    ),
    (
        "norm2",
        "def norm2(v: Vector[N]) -> Scalar:\n    return v @ v\n",
    ),
    (
        "both",
        "def both(a: Vector[N]) -> Scalar:\n    return norm2(a) + dot(a, a) + power(2, len(a))\n",
    ),
    (
        "safe",
        "def safe(x: Scalar) -> Scalar:\n    if x > 0:\n        return sqrt(x)\n    return 0\n",
    ),
    (
        "acc",
        "def acc(n: Scalar, x: Scalar) -> Scalar:\n    s = 0\n    i = 0\n    while i < n:\n        s = s * 0.5 + x / (n - i - 1)\n        i += 1\n    return s\n",
    ),
    (
        "first",
        "def first(v: Vector[N]) -> Scalar:\n    total = v[0]\n    i = 1\n    while i < len(v):\n        total = total + v[i]\n        i += 1\n    return total\n",
    ),
    (
        "unused",
        "def unused(x: Scalar, y: Scalar) -> Scalar:\n    spare = x / y\n    return x * 2\n",
    ),
];

/// Functions the calls of each function need defined before it.
fn needs(name: &str) -> &'static [&'static str] {
    match name {
        "both" => &["norm2", "dot", "power"],
        _ => &[],
    }
}

const NUMBERS: &[&str] = &["0", "1", "-1", "2", "0.5", "3", "-2.5", "100", "7", "1e-3"];
const COUNTS: &[&str] = &["0", "1", "2", "3", "4", "5", "8", "9", "12"];

fn array_program(seed: u64) -> String {
    let mut r = Rng::new(seed);
    let mut wanted: Vec<&str> = Vec::new();
    for _ in 0..(2 + r.below(5)) {
        let (name, _) = FUNCTIONS[r.below(FUNCTIONS.len())];
        for dep in needs(name) {
            if !wanted.contains(dep) {
                wanted.push(dep);
            }
        }
        if !wanted.contains(&name) {
            wanted.push(name);
        }
    }
    // Definitions in the table's order, so dependencies come first.
    let mut src = String::new();
    let order = ["dot", "power", "norm2"];
    for (name, text) in FUNCTIONS {
        if wanted.contains(name) && order.contains(name) {
            src.push_str(text);
            src.push('\n');
        }
    }
    for (name, text) in FUNCTIONS {
        if wanted.contains(name) && !order.contains(name) {
            src.push_str(text);
            src.push('\n');
        }
    }
    // Two vector sizes, each with two vectors of it.
    let sizes = [1 + r.below(5), 1 + r.below(5)];
    for (k, n) in sizes.iter().enumerate() {
        for name in ["p", "q"] {
            let entries: Vec<&str> = (0..*n).map(|_| r.pick(NUMBERS)).collect();
            src.push_str(&format!("{name}{k} = [{}]\n", entries.join(", ")));
        }
    }
    let vec = |r: &mut Rng, k: usize| format!("{}{k}", r.pick(&["p", "q"]));
    for _ in 0..(2 + r.below(5)) {
        let name = wanted[r.below(wanted.len())];
        let k = r.below(2);
        let call = match name {
            "dot" => format!("dot({}, {})", vec(&mut r, k), vec(&mut r, k)),
            "recips" => format!("recips({}, {})", vec(&mut r, k), r.pick(NUMBERS)),
            "pick" => format!("pick({}, {})", vec(&mut r, k), r.pick(COUNTS)),
            "axpy" => format!(
                "axpy({}, {}, {})",
                r.pick(NUMBERS),
                vec(&mut r, k),
                vec(&mut r, k)
            ),
            "checked" => format!("checked({}, {})", vec(&mut r, k), r.pick(NUMBERS)),
            "power" => format!("power({}, {})", r.pick(NUMBERS), r.pick(COUNTS)),
            "norm2" => format!("norm2({})", vec(&mut r, k)),
            "both" => format!("both({})", vec(&mut r, k)),
            "safe" => format!("safe({})", r.pick(NUMBERS)),
            "acc" => format!("acc({}, {})", r.pick(COUNTS), r.pick(NUMBERS)),
            "first" => format!("first({})", vec(&mut r, k)),
            _ => format!("unused({}, {})", r.pick(NUMBERS), r.pick(NUMBERS)),
        };
        src.push_str(&format!("print({call})\n"));
    }
    src
}

// ----- the oracle -----

/// Stdout, and the error as (message, rendered diagnostic).
type Run = (String, Option<(String, String)>);

fn render(d: &Diagnostic, src: &str) -> String {
    d.render_to_string(&SourceFile::new("program.sky", src))
}

/// The two resource-limit errors compare as one (see the harness).
fn normalized(r: Run) -> (String, Option<String>) {
    let error = r.1.map(|(message, rendered)| {
        if message.starts_with("recursion is too deep")
            || message == "the program nests too deeply for the interpreter"
        {
            "<resource limit>".to_string()
        } else {
            rendered
        }
    });
    (r.0, error)
}

fn run_sir(m: &Module, src: &str) -> Run {
    let mut out = Vec::new();
    let r = sky_ir_interp::run(m, &mut out);
    (
        String::from_utf8_lossy(&out).into_owned(),
        r.err()
            .map(|e| (e.message.clone(), render(&e.to_diagnostic(), src))),
    )
}

/// How the pass fared on the programs so far.
#[derive(Default)]
struct Stats {
    programs: AtomicUsize,
    skipped: AtomicUsize,
    specialized_programs: AtomicUsize,
    clones: AtomicUsize,
    unrolled: AtomicUsize,
    failing: AtomicUsize,
}

/// Runs one program through the three pipelines and compares. `Err` is a
/// difference in behaviour or a broken pass (or a program the checker
/// rejects, which the caller skips); `stats` counts what the pass did.
fn check(src: &str, stats: &Stats) -> Result<(), String> {
    let lexed = sky_syntax::lex(src).map_err(|_| "the lexer rejects the program".to_string())?;
    let (ast, errors) = sky_ast::parse_module(&lexed.tokens, src);
    if !errors.is_empty() {
        return Err(format!("parse errors: {errors:?}"));
    }
    let checked = sky_types::check_module(&ast, src);
    if checked.has_errors() {
        return Err("sky check rejects the program".to_string());
    }
    let lowered = sky_lower::lower(&ast, src, &checked).map_err(|e| format!("lowering: {e}"))?;

    let mut without = lowered.clone();
    default_passes_without_specialize(PassManager::new().verify_each(true))
        .run(&mut without)
        .map_err(|e| format!("a pass broke the IR (without specialize): {e}"))?;
    let mut with = lowered.clone();
    let report = default_passes(PassManager::new().verify_each(true))
        .run(&mut with)
        .map_err(|e| format!("a pass broke the IR (with specialize): {e}"))?;

    let want = normalized(run_sir(&lowered, src));
    for (what, m) in [("without specialize", &without), ("with specialize", &with)] {
        let got = normalized(run_sir(m, src));
        if got != want {
            return Err(format!(
                "the SIR {what} differs from the lowered SIR\n--- lowered ---\n{}{}\n--- {what} ---\n{}{}\n--- SIR text ---\n{}",
                want.0,
                want.1.as_deref().unwrap_or("(ok)"),
                got.0,
                got.1.as_deref().unwrap_or("(ok)"),
                sky_ir::print(m),
            ));
        }
    }
    let n = report
        .explanations
        .iter()
        .filter(|x| x.rule == sky_opt::specialize::RULE_SPECIALIZED)
        .count();
    let loops = report
        .explanations
        .iter()
        .filter(|x| x.rule == sky_opt::specialize::RULE_UNROLLED)
        .count();
    stats.programs.fetch_add(1, Ordering::Relaxed);
    stats.clones.fetch_add(n, Ordering::Relaxed);
    stats.unrolled.fetch_add(loops, Ordering::Relaxed);
    if n > 0 {
        stats.specialized_programs.fetch_add(1, Ordering::Relaxed);
    }
    if want.1.is_some() {
        stats.failing.fetch_add(1, Ordering::Relaxed);
    }
    Ok(())
}

/// Runs `first..first+count` of `program(seed)` on all cores; returns the
/// failures.
fn campaign(
    first: u64,
    count: u64,
    stats: &Stats,
    program: impl Fn(u64) -> String + Sync,
) -> Vec<String> {
    let lanes = std::thread::available_parallelism().map_or(1, |n| n.get().min(8)) as u64;
    let keep_going = std::env::var("SKY_FUZZ_KEEP_GOING").is_ok();
    std::thread::scope(|scope| {
        let program = &program;
        let handles: Vec<_> = (0..lanes)
            .map(|lane| {
                scope.spawn(move || {
                    let mut failures = Vec::new();
                    let mut seed = first + lane;
                    while seed < first + count {
                        let src = program(seed);
                        match sky_ast::with_deep_stack(|| check(&src, stats)) {
                            Ok(()) => {}
                            Err(msg)
                                if msg.starts_with("sky check rejects")
                                    || msg.starts_with("lowering:") =>
                            {
                                stats.skipped.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(msg) => {
                                failures
                                    .push(format!("seed {seed}: {msg}\n--- program ---\n{src}"));
                                if !keep_going {
                                    break;
                                }
                            }
                        }
                        seed += lanes;
                    }
                    failures
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a lane panicked"))
            .collect()
    })
}

fn report(what: &str, stats: &Stats) {
    eprintln!(
        "fuzz_specialize {what}: {} programs ({} skipped as unchecked), {} with a specialization \
         ({} clones, {} loops unrolled), {} ending in a run-time error",
        stats.programs.load(Ordering::Relaxed),
        stats.skipped.load(Ordering::Relaxed),
        stats.specialized_programs.load(Ordering::Relaxed),
        stats.clones.load(Ordering::Relaxed),
        stats.unrolled.load(Ordering::Relaxed),
        stats.failing.load(Ordering::Relaxed),
    );
}

fn assert_no_failures(failures: &[String]) {
    let summary: Vec<&str> = failures
        .iter()
        .map(|f| f.lines().take(2).last().unwrap_or(""))
        .collect();
    assert!(
        failures.is_empty(),
        "{} failure(s): {summary:#?}\nfirst:\n{}",
        failures.len(),
        failures[0]
    );
}

#[test]
fn specializing_keeps_the_whole_diagnostic_on_generated_programs() {
    let first = env_number("SKY_FUZZ_FROM").unwrap_or(1);
    let count = env_number("SKY_FUZZ_SEEDS").unwrap_or(1500);
    let stats = Stats::default();
    let failures = campaign(first, count, &stats, |seed| {
        let len = 150 + (seed.wrapping_mul(2_654_435_761) % 1400) as usize;
        harness::generate(&bytes(seed, len))
    });
    report("differential generator", &stats);
    assert_no_failures(&failures);
}

#[test]
fn specializing_keeps_the_whole_diagnostic_on_array_programs() {
    let first = env_number("SKY_FUZZ_FROM").unwrap_or(1);
    let count = env_number("SKY_FUZZ_SEEDS").unwrap_or(1500);
    let stats = Stats::default();
    let failures = campaign(first, count, &stats, array_program);
    report("array programs", &stats);
    // The generator must exercise the pass, or the test says nothing.
    let done = stats.programs.load(Ordering::Relaxed);
    assert!(
        stats.specialized_programs.load(Ordering::Relaxed) * 2 > done,
        "the array programs rarely specialize: {done} programs"
    );
    assert_no_failures(&failures);
}

/// A tool for a fuzzing session: `SKY_SEED=123` prints the array program of
/// that seed. Does nothing unless set.
#[test]
fn print_an_array_program() {
    if let Some(seed) = env_number("SKY_SEED") {
        println!("{}", array_program(seed));
    }
}
