//! Determinism of automatic parallelism (`sky_ir::parallel`) on generated
//! programs: the same stdout and the same error, compared as the whole
//! rendered diagnostic, on 1, 2, 4 and 8 threads, with every threshold at its
//! minimum (so that small programs take the parallel paths) and with the
//! defaults, for the lowered SIR and for the optimized SIR.
//!
//! Three generators feed it:
//!
//! * the differential fuzzer's (`fuzz/harness/differential.rs`): scalar
//!   programs with functions, loops, recursion, contracts and run-time
//!   failures (few of them have two independent calls, so the next one
//!   exists);
//! * one here for call programs: small functions that fail for some
//!   arguments, print, or call each other, combined in a main by
//!   independent calls, dependent calls, a failing operation between calls,
//!   prints between calls and loops, which exercise call groups and their
//!   error order;
//! * one here for array programs: vectors with entries chosen to overflow at
//!   some elements only, scaled by zero or by huge numbers, in chains that
//!   fuse, which exercise kernels, and several chunks failing at once.
//!
//! `SKY_FUZZ_FROM` and `SKY_FUZZ_SEEDS` run a longer campaign (use
//! `--release`): `SKY_FUZZ_SEEDS=50000 cargo test --release -p sky-opt --test
//! fuzz_parallel`.

#[path = "../../../fuzz/harness/differential.rs"]
mod harness;

use sky_diag::SourceFile;
use sky_ir::parallel::Config;
use sky_ir::{Module, PassManager};
use sky_opt::default_passes;

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

/// Stdout and the rendered error.
type Outcome = (String, Option<String>);

fn run(ir: &Module, src: &str, cfg: &Config) -> Outcome {
    let mut out = Vec::new();
    let r = sky_ir_interp::run_with(ir, &mut out, cfg);
    (
        String::from_utf8_lossy(&out).into_owned(),
        r.err().map(|e| {
            e.to_diagnostic()
                .render_to_string(&SourceFile::new("p.sky", src))
        }),
    )
}

/// The configurations every program must give one outcome under.
fn configs() -> Vec<Config> {
    let mut all = Vec::new();
    for t in [2, 4, 8] {
        all.push(Config::eager(t));
        all.push(Config::with_threads(t));
    }
    // Chunk boundaries in different places.
    all.push(Config {
        chunk_min: 3,
        ..Config::eager(4)
    });
    all
}

/// The `--explain` notes for `t` threads, with the number of threads taken
/// out: what is parallel, and why not, must not depend on it.
fn notes(m: &Module, t: usize) -> Vec<String> {
    let words = format!("{t} threads");
    sky_ir::parallel::explain(m, &Config::eager(t))
        .iter()
        .map(|n| {
            format!(
                "{} {:?} {} {:?}",
                n.rule,
                n.span,
                n.headline.replace(&words, "N threads"),
                n.because
            )
        })
        .collect()
}

/// Lowers `src`, optimizes it, and requires the same outcome from every
/// configuration for both, and the same notes.
fn check(src: &str) -> Result<(), String> {
    let mut ir = sky_lower::lower_source(src).map_err(|e| format!("lowering: {e}"))?;
    let lowered = ir.clone();
    default_passes(PassManager::new().verify_each(true))
        .run(&mut ir)
        .map_err(|e| format!("a pass broke the IR: {e}"))?;
    for (what, m) in [("lowered", &lowered), ("optimized", &ir)] {
        let want = run(m, src, &Config::sequential());
        let plan = notes(m, 2);
        for t in [4, 8, 64] {
            if notes(m, t) != plan {
                return Err(format!(
                    "the {what} SIR is explained differently on {t} threads"
                ));
            }
        }
        for cfg in configs() {
            let got = run(m, src, &cfg);
            if got != want {
                return Err(format!(
                    "the {what} SIR differs with {cfg:?}\n--- 1 thread ---\n{}{}\n--- {} threads ---\n{}{}\n--- SIR text ---\n{}",
                    want.0,
                    want.1.as_deref().unwrap_or("(ok)"),
                    cfg.threads,
                    got.0,
                    got.1.as_deref().unwrap_or("(ok)"),
                    sky_ir::print(m),
                ));
            }
        }
    }
    Ok(())
}

/// A small deterministic generator for the array programs.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() >> 33) as usize % n
    }

    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len())]
    }
}

/// An entry: mostly modest, sometimes zero, sometimes one that overflows when
/// added to itself or scaled.
fn entry(r: &mut Rng, risky: bool) -> &'static str {
    const MODEST: &[&str] = &["0", "1", "-1", "0.5", "3", "7", "2.5", "-4", "10", "1e10"];
    const RISKY: &[&str] = &["1e300", "-1e300", "1.7e308", "-1.7e308", "1e-300", "0"];
    if risky && r.below(6) == 0 {
        r.pick(RISKY)
    } else {
        r.pick(MODEST)
    }
}

/// A scalar factor or divisor.
fn scalar(r: &mut Rng, risky: bool) -> &'static str {
    const MODEST: &[&str] = &["2", "0.5", "-3", "1", "7", "0.25"];
    const RISKY: &[&str] = &["0", "1e200", "1e-200", "1e300", "-1e308"];
    if risky && r.below(3) == 0 {
        r.pick(RISKY)
    } else {
        r.pick(MODEST)
    }
}

/// A vector expression over `vars`.
fn expr(r: &mut Rng, vars: &[String], risky: bool, depth: u32) -> String {
    if depth == 0 || r.below(4) == 0 {
        return r
            .pick(&vars.iter().map(String::as_str).collect::<Vec<_>>())
            .to_string();
    }
    match r.below(6) {
        0 => format!(
            "({} + {})",
            expr(r, vars, risky, depth - 1),
            expr(r, vars, risky, depth - 1)
        ),
        1 => format!(
            "({} - {})",
            expr(r, vars, risky, depth - 1),
            expr(r, vars, risky, depth - 1)
        ),
        2 => format!(
            "({} * {})",
            expr(r, vars, risky, depth - 1),
            scalar(r, risky)
        ),
        3 => format!(
            "({} * {})",
            scalar(r, risky),
            expr(r, vars, risky, depth - 1)
        ),
        4 => format!(
            "({} / {})",
            expr(r, vars, risky, depth - 1),
            scalar(r, risky)
        ),
        _ => format!("(-{})", expr(r, vars, risky, depth - 1)),
    }
}

/// An array program for `seed`: a few vectors of one length, chains of
/// elementwise operations over them (some fuse, some are shared and stay
/// separate), and prints of the results and of single entries.
fn array_program(seed: u64) -> String {
    let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    // Mostly short vectors (chunks of a few elements), some past the default
    // thresholds.
    let len = match r.below(10) {
        0 => 2100 + r.below(400),
        1..=2 => 100 + r.below(200),
        _ => 2 + r.below(40),
    };
    let risky = r.below(2) == 0;
    let mut src = String::new();
    let mut vars = Vec::new();
    for i in 0..(2 + r.below(3)) {
        let entries: Vec<&str> = (0..len).map(|_| entry(&mut r, risky)).collect();
        src.push_str(&format!("v{i} = [{}]\n", entries.join(", ")));
        vars.push(format!("v{i}"));
    }
    for i in 0..(2 + r.below(4)) {
        let depth = 1 + r.below(4) as u32;
        let e = expr(&mut r, &vars, risky, depth);
        src.push_str(&format!("w{i} = {e}\n"));
        vars.push(format!("w{i}"));
        if r.below(3) == 0 {
            src.push_str(&format!("print(w{i}[{}])\n", r.below(len)));
        }
    }
    let last = vars.last().cloned().unwrap_or_default();
    src.push_str(&format!("print({last})\n"));
    src
}

/// The body of `f{i}`: pure, or (rarely) printing, that fails for some
/// arguments and may call the functions before it.
fn function_body(r: &mut Rng, i: usize) -> String {
    let k = r.below(4);
    let n = 2 + r.below(4);
    let (a, b) = (1 + r.below(5), r.below(4));
    let callee = |r: &mut Rng| r.below(i.max(1));
    match r.below(16) {
        0..=3 => format!("    return x * {a} + {b}\n"),
        4..=5 => format!("    return x - {b}\n"),
        6 => format!("    return 1 / (x - {k})\n"),
        7 => format!("    return sqrt(x - {k})\n"),
        8 => "    return x * 1e200 * 1e200\n".to_string(),
        9 => format!(
            "    total = 0\n    i = 0\n    while i < {n}:\n        total = total + x / (i + {})\n        i = i + 1\n    return total\n",
            1 + r.below(3)
        ),
        10..=11 if i > 0 => {
            let (j, l) = (callee(r), callee(r));
            format!("    return f{j}(x) + f{l}(x + 1)\n")
        }
        12 if i > 0 => {
            let j = callee(r);
            format!("    return f{j}(x) * {a}\n")
        }
        13 => format!("    assert x < {}\n    return x + {b}\n", 3 + r.below(6)),
        14 if r.below(2) == 0 => format!("    print(x)\n    return x + {b}\n"),
        _ => format!("    return x * {a} - {b}\n"),
    }
}

/// A program of a few small functions and a main that combines calls of them
/// in every way that matters to `sky_ir::parallel`: independent calls in one
/// expression, calls that use each other's results, an operation between
/// calls that fails, prints between calls, calls in a loop. Most functions
/// are pure and cheap to read, some fail for some arguments, a few print.
fn call_program(seed: u64) -> String {
    let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut src = String::new();
    let nf = 2 + r.below(4);
    for i in 0..nf {
        src.push_str(&format!("def f{i}(x):\n{}", function_body(&mut r, i)));
    }
    let mut vars: Vec<String> = Vec::new();
    let arg = |r: &mut Rng, vars: &[String]| -> String {
        if !vars.is_empty() && r.below(3) == 0 {
            vars[r.below(vars.len())].clone()
        } else {
            r.below(6).to_string()
        }
    };
    let ops = ["+", "-", "*"];
    for s in 0..(3 + r.below(8)) {
        let v = format!("v{s}");
        match r.below(8) {
            0 | 1 => {
                let (a, b) = (r.below(nf), r.below(nf));
                let (x, y) = (arg(&mut r, &vars), arg(&mut r, &vars));
                let op = ops[r.below(3)];
                src.push_str(&format!("{v} = f{a}({x}) {op} f{b}({y})\n"));
            }
            2 => {
                let (a, b, c) = (r.below(nf), r.below(nf), r.below(nf));
                let (x, y, z) = (arg(&mut r, &vars), arg(&mut r, &vars), arg(&mut r, &vars));
                src.push_str(&format!("{v} = f{a}({x}) * f{b}({y}) - f{c}({z})\n"));
            }
            3 => {
                let (a, b, c) = (r.below(nf), r.below(nf), r.below(nf));
                let (x, y) = (arg(&mut r, &vars), arg(&mut r, &vars));
                src.push_str(&format!("{v} = f{a}({x}) + f{b}(f{c}({y}))\n"));
            }
            4 => {
                // An operation that can fail between two calls.
                let (a, b) = (r.below(nf), r.below(nf));
                let (x, y) = (arg(&mut r, &vars), arg(&mut r, &vars));
                let d = arg(&mut r, &vars);
                src.push_str(&format!(
                    "{v} = f{a}({x})\nw{s} = 1 / ({d} - {})\nu{s} = f{b}({y})\n",
                    r.below(4)
                ));
                vars.push(format!("w{s}"));
                vars.push(format!("u{s}"));
            }
            5 => {
                let (a, b) = (r.below(nf), r.below(nf));
                src.push_str(&format!(
                    "s{s} = 0\ni{s} = 0\nwhile i{s} < {}:\n    s{s} = s{s} + f{a}(i{s}) + f{b}(i{s} + 1)\n    i{s} = i{s} + 1\n{v} = s{s}\n",
                    2 + r.below(3)
                ));
            }
            6 => {
                // Prints between calls keep them apart.
                let (a, b) = (r.below(nf), r.below(nf));
                let (x, y) = (arg(&mut r, &vars), arg(&mut r, &vars));
                src.push_str(&format!("{v} = f{a}({x})\nprint({v})\nq{s} = f{b}({y})\n"));
                vars.push(format!("q{s}"));
            }
            _ => {
                let a = r.below(nf);
                let x = arg(&mut r, &vars);
                src.push_str(&format!("{v} = f{a}({x})\nprint({v})\n"));
            }
        }
        vars.push(v);
    }
    let last = vars.last().cloned().unwrap_or_else(|| "0".to_string());
    src.push_str(&format!("print({last})\n"));
    src
}

fn campaign(what: &str, first: u64, count: u64, program: impl Fn(u64) -> String + Sync) {
    let lanes = std::thread::available_parallelism().map_or(1, |n| n.get().min(8)) as u64;
    let keep_going = std::env::var("SKY_FUZZ_KEEP_GOING").is_ok();
    let program = &program;
    let failures: Vec<String> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..lanes)
            .map(|lane| {
                scope.spawn(move || {
                    let mut failures = Vec::new();
                    let mut seed = first + lane;
                    while seed < first + count {
                        let src = program(seed);
                        if let Err(msg) = sky_ast::with_deep_stack(|| check(&src)) {
                            failures
                                .push(format!("{what} seed {seed}: {msg}\n--- program ---\n{src}"));
                            if !keep_going {
                                break;
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
    });
    let summary: Vec<&str> = failures
        .iter()
        .map(|f| f.lines().next().unwrap_or(""))
        .collect();
    assert!(
        failures.is_empty(),
        "{} failure(s): {summary:#?}\nfirst:\n{}",
        failures.len(),
        failures[0]
    );
}

#[test]
fn parallel_runs_agree_on_the_differential_fuzzers_programs() {
    let first = env_number("SKY_FUZZ_FROM").unwrap_or(1);
    let count = env_number("SKY_FUZZ_SEEDS").unwrap_or(300);
    campaign("scalar", first, count, |seed| {
        let len = 150 + (seed.wrapping_mul(2_654_435_761) % 1400) as usize;
        harness::generate(&bytes(seed, len))
    });
}

#[test]
fn parallel_runs_agree_on_generated_array_programs() {
    let first = env_number("SKY_FUZZ_FROM").unwrap_or(1);
    let count = env_number("SKY_FUZZ_SEEDS").unwrap_or(100);
    campaign("array", first, count, array_program);
}

/// The array generator makes programs that check, fuse, and fail in some
/// (not all) of the cases, or it would test nothing.
#[test]
fn the_array_generator_reaches_kernels_and_failures() {
    let (mut fused, mut failed, mut ok) = (0, 0, 0);
    for seed in 1..=100 {
        let src = array_program(seed);
        let mut ir = sky_lower::lower_source(&src).unwrap_or_else(|e| panic!("{e}\n{src}"));
        default_passes(PassManager::new().verify_each(true))
            .run(&mut ir)
            .expect("the pipeline verifies");
        if sky_ir::print(&ir).contains("fused") {
            fused += 1;
        }
        let r = sky_ast::with_deep_stack(|| run(&ir, &src, &Config::sequential()));
        if r.1.is_some() {
            failed += 1;
        } else {
            ok += 1;
        }
    }
    assert!(fused > 20, "only {fused} of 100 programs fuse");
    assert!(failed > 10, "only {failed} of 100 programs fail");
    assert!(ok > 10, "only {ok} of 100 programs succeed");
}

#[test]
fn parallel_runs_agree_on_generated_call_programs() {
    let first = env_number("SKY_FUZZ_FROM").unwrap_or(1);
    let count = env_number("SKY_FUZZ_SEEDS").unwrap_or(300);
    campaign("calls", first, count, call_program);
}

/// The call generator makes programs that have call groups, and that fail in
/// some (not all) of the cases, or it would test nothing.
#[test]
fn the_call_generator_reaches_groups_and_failures() {
    let (mut grouped, mut failed_groups, mut ok_groups) = (0, 0, 0);
    for seed in 1..=200 {
        let src = call_program(seed);
        let mut ir = sky_lower::lower_source(&src).unwrap_or_else(|e| panic!("{e}\n{src}"));
        default_passes(PassManager::new().verify_each(true))
            .run(&mut ir)
            .expect("the pipeline verifies");
        if sky_ir::parallel::analyze(&ir, &Config::eager(4)).has_groups() {
            grouped += 1;
            let r = sky_ast::with_deep_stack(|| run(&ir, &src, &Config::sequential()));
            if r.1.is_some() {
                failed_groups += 1;
            } else {
                ok_groups += 1;
            }
        }
    }
    eprintln!(
        "{grouped} of 200 programs have call groups: {failed_groups} fail, {ok_groups} succeed"
    );
    assert!(
        grouped > 100,
        "only {grouped} of 200 programs have call groups"
    );
    assert!(
        failed_groups > 20,
        "only {failed_groups} grouped programs fail"
    );
    assert!(ok_groups > 20, "only {ok_groups} grouped programs succeed");
}
