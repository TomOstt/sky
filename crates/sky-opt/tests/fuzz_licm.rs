//! Loops with invariant operations under the default pipeline (`gvn` and
//! `licm` in it): the tree-walking interpreter, the lowered SIR and the SIR
//! after the pipeline (with and without `specialize`) must print the same
//! bytes and stop with the same error, compared as the whole rendered
//! diagnostic (message, spans, labels, notes).
//!
//! The differential fuzzer's generator (`fuzz/harness/differential.rs`)
//! makes loops whose bodies rarely have operations that do not depend on the
//! loop. This one is about exactly those: every program is a function
//! `run(n, p, q, r)` whose parameters are dimensionless numbers the loops
//! cannot change, so `p / q`, `sqrt(p)`, `log(q - r)`, `v[p]`, `inv(M)`,
//! `Mᵀ` and calls of `g(p)` are invariant in them, next to statements that
//! depend on the loop (counters, accumulators). The invariants can fail
//! (`q` is often `0`, `p` is often negative or out of bounds, `M` is often
//! singular) and sit before and after `print`s, before and after other
//! failing operations, in `while` and `for` loops that run zero times, once
//! or a few times, in `if` arms, and in nested loops. `run` is called with
//! several argument lists, so one program covers zero-trip and failing
//! cases, and because the parameters are not constants inside `run` (a
//! call's constants are what `specialize` makes a clone from), the loop's
//! trip count is unknown to the pass that hoists.
//!
//! Every program type-checks by construction. Loops always terminate (each
//! has a counter the body cannot touch, or `range`).
//!
//! `SKY_FUZZ_FROM` and `SKY_FUZZ_SEEDS` run a longer campaign (use
//! `--release`); `SKY_FUZZ_KEEP_GOING=1` lists every failing seed;
//! `SKY_SEED=n` prints the program for seed `n`.

use std::sync::atomic::{AtomicUsize, Ordering};

use sky_diag::{Diagnostic, SourceFile};
use sky_ir::PassManager;
use sky_opt::{default_passes, default_passes_without_specialize};

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }

    fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[self.below(xs.len())]
    }
}

/// The parameters: numbers the loops cannot change.
const OUTER: [&str; 3] = ["p", "q", "r"];
/// The scalars the loops assign.
const INNER: [&str; 3] = ["x", "y", "z"];
/// Scalars the loops assign from invariant expressions.
const TEMPS: [&str; 2] = ["t", "s"];

struct Gen {
    rng: Rng,
    out: String,
    /// Loop counters in use, so nested loops get their own.
    counters: usize,
    in_loop: usize,
}

impl Gen {
    fn line(&mut self, indent: usize, text: &str) {
        for _ in 0..indent {
            self.out.push_str("    ");
        }
        self.out.push_str(text);
        self.out.push('\n');
    }

    fn constant(&mut self) -> String {
        self.rng
            .pick(&["0", "1", "2", "3", "0.5", "1e200", "0.1", "5"])
            .to_string()
    }

    /// An expression of the parameters and constants only: invariant in
    /// every loop of `run`.
    fn invariant(&mut self, depth: usize) -> String {
        if depth == 0 || self.rng.chance(25) {
            return match self.rng.below(5) {
                0 => self.constant(),
                _ => self.rng.pick(&OUTER).to_string(),
            };
        }
        let a = self.invariant(depth - 1);
        let b = self.invariant(depth - 1);
        match self.rng.below(14) {
            0 | 1 => format!("({a} + {b})"),
            2 => format!("({a} - {b})"),
            3 | 4 => format!("({a} * {b})"),
            5 | 6 => format!("({a} / {b})"),
            7 => format!("sqrt({a})"),
            8 => format!("log({a})"),
            9 => format!("exp({a})"),
            10 => format!("abs({a})"),
            11 => format!("v[{}]", self.rng.pick(&["p", "q", "0", "1", "2", "3"])),
            12 => format!("g({a})"),
            _ => format!("({a} * {b} * {b})"),
        }
    }

    /// An expression that may use the loop's variables too.
    fn any(&mut self, depth: usize) -> String {
        if depth == 0 || self.rng.chance(25) {
            return match self.rng.below(4) {
                0 => self.constant(),
                1 => self.rng.pick(&OUTER).to_string(),
                _ => self.rng.pick(&INNER).to_string(),
            };
        }
        let mix = self.rng.chance(60);
        let a = if mix {
            self.any(depth - 1)
        } else {
            self.invariant(depth - 1)
        };
        let b = if mix {
            self.invariant(depth - 1)
        } else {
            self.any(depth - 1)
        };
        match self.rng.below(8) {
            0 | 1 => format!("({a} + {b})"),
            2 => format!("({a} - {b})"),
            3 => format!("({a} * {b})"),
            4 => format!("({a} / {b})"),
            5 => format!("sqrt({a})"),
            6 => format!("({a} * 0.5)"),
            _ => self.rng.pick(&INNER).to_string(),
        }
    }

    fn condition(&mut self) -> String {
        let a = self.any(1);
        let b = self.any(1);
        let op = self.rng.pick(&["<", ">", "<=", ">=", "==", "!="]);
        format!("{a} {op} {b}")
    }

    fn matrix_stmt(&mut self, ind: usize) {
        match self.rng.below(5) {
            0 => self.line(ind, "W = inv(M)"),
            1 => self.line(ind, "T = Mᵀ"),
            2 => self.line(ind, "u = W @ w"),
            3 => self.line(ind, "u = Mᵀ @ w"),
            _ => self.line(ind, "u = u + M @ w"),
        }
    }

    fn stmt(&mut self, ind: usize, depth: usize) {
        match self.rng.below(16) {
            0..=2 => {
                let t = self.rng.pick(&TEMPS);
                let e = self.invariant(2);
                self.line(ind, &format!("{t} = {e}"));
            }
            3..=5 => {
                let x = self.rng.pick(&INNER);
                let e = self.any(2);
                self.line(ind, &format!("{x} = {e}"));
            }
            6..=7 => {
                let mut items = Vec::new();
                for _ in 0..1 + self.rng.below(2) {
                    items.push(if self.rng.chance(50) {
                        self.rng.pick(&INNER).to_string()
                    } else {
                        self.rng.pick(&TEMPS).to_string()
                    });
                }
                self.line(ind, &format!("print({})", items.join(", ")));
            }
            8 => {
                let c = self.condition();
                self.line(ind, &format!("if {c}:"));
                let n = 1 + self.rng.below(2);
                self.block(ind + 1, depth + 1, n);
                if self.rng.chance(40) {
                    self.line(ind, "else:");
                    self.block(ind + 1, depth + 1, 1);
                }
            }
            9 => self.matrix_stmt(ind),
            10 => {
                let x = self.rng.pick(&INNER);
                let idx = self.rng.pick(&["p", "q", "0", "1", "2"]);
                self.line(ind, &format!("{x} = v[{idx}]"));
            }
            11 if depth < 2 => self.looped(ind, depth + 1),
            12 if self.in_loop > 0 => {
                let c = self.condition();
                self.line(ind, &format!("if {c}:"));
                self.line(ind + 1, "break");
            }
            13 => {
                let a = self.invariant(1);
                let c = self.rng.pick(&["<", ">"]);
                self.line(ind, &format!("assert {a} {c} 100"));
            }
            14 => {
                let t = self.rng.pick(&TEMPS);
                let e = self.invariant(1);
                self.line(ind, &format!("{t} = h({e})"));
            }
            _ => {
                let x = self.rng.pick(&INNER);
                let t = self.rng.pick(&TEMPS);
                self.line(ind, &format!("{x} = {x} + {t}"));
            }
        }
    }

    fn block(&mut self, ind: usize, depth: usize, n: usize) {
        for _ in 0..n {
            self.stmt(ind, depth);
        }
    }

    fn looped(&mut self, ind: usize, depth: usize) {
        let bound = match self.rng.below(4) {
            0 => "n".to_string(),
            1 => "n".to_string(),
            2 => "2".to_string(),
            _ => "0".to_string(),
        };
        self.in_loop += 1;
        let body = 1 + self.rng.below(5);
        if self.rng.chance(35) {
            let k = format!("k{}", self.counters);
            self.counters += 1;
            self.line(ind, &format!("for {k} in range({bound}):"));
            self.block(ind + 1, depth, body);
        } else {
            let c = format!("c{}", self.counters);
            self.counters += 1;
            self.line(ind, &format!("{c} = 0"));
            self.line(ind, &format!("while {c} < {bound}:"));
            // The counter's increment goes first or last, never skipped by
            // a `break` that ends the loop anyway.
            let first = self.rng.chance(50);
            if first {
                self.line(ind + 1, &format!("{c} += 1"));
            }
            self.block(ind + 1, depth, body);
            if !first {
                self.line(ind + 1, &format!("{c} += 1"));
            }
        }
        self.in_loop -= 1;
    }
}

fn program(seed: u64) -> String {
    let mut g = Gen {
        rng: Rng::new(seed),
        out: String::new(),
        counters: 0,
        in_loop: 0,
    };
    g.line(0, "def g(a):");
    g.line(1, "return 1 / a");
    g.line(0, "def h(a):");
    g.line(1, "print(a)");
    g.line(1, "return a");
    g.line(
        0,
        "def run(n: Scalar, p: Scalar, q: Scalar, r: Scalar) -> Scalar:",
    );
    g.line(1, "v = [1, 2, 3]");
    let m = g.rng.pick(&[
        "[[2, 1], [1, 3]]",
        "[[1, 2], [2, 4]]",
        "[[0, 0], [0, 0]]",
        "[[4, 0], [0, 0.5]]",
    ]);
    g.line(1, &format!("M = {m}"));
    g.line(1, "w = [3, 4]");
    g.line(1, "u = w");
    g.line(1, "W = M");
    g.line(1, "T = M");
    g.line(1, "x = 1");
    g.line(1, "y = 2");
    g.line(1, "z = 0.5");
    g.line(1, "t = 0");
    g.line(1, "s = 0");
    for _ in 0..1 + g.rng.below(3) {
        if g.rng.chance(70) {
            g.looped(1, 0);
        } else {
            g.stmt(1, 0);
        }
        if g.rng.chance(30) {
            g.line(1, "print(x, y, z, u)");
        }
    }
    g.line(1, "print(x, y, z, t, s, u, W, T)");
    g.line(1, "return x + y + z");
    let args = [
        ("3", "2", "4", "1"),
        ("0", "0", "0", "0"),
        ("1", "1", "0", "2"),
        ("2", "-1", "2", "0.5"),
        ("2", "3", "1", "1"),
        ("1", "2", "1", "1"),
    ];
    for i in 0..3 {
        let (n, p, q, r) = args[(g.rng.below(args.len()) + i) % args.len()];
        g.line(0, &format!("print(run({n}, {p}, {q}, {r}))"));
    }
    g.out
}

/// Stdout, and the error as (message, rendered diagnostic).
type Run = (String, Option<(String, String)>);

fn render(d: &Diagnostic, src: &str) -> String {
    d.render_to_string(&SourceFile::new("program.sky", src))
}

/// The two resource-limit errors compare as one.
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

/// How many programs the pipeline changed by `gvn` and by `licm`.
static LICM_CHANGED: AtomicUsize = AtomicUsize::new(0);
static GVN_CHANGED: AtomicUsize = AtomicUsize::new(0);
static GUARDED: AtomicUsize = AtomicUsize::new(0);
static REJECTED: AtomicUsize = AtomicUsize::new(0);

fn check(src: &str) -> Result<(), String> {
    let lexed = sky_syntax::lex(src).map_err(|_| "the lexer rejects the program".to_string())?;
    let (ast, errors) = sky_ast::parse_module(&lexed.tokens, src);
    if !errors.is_empty() {
        return Err(format!("parse errors: {errors:?}"));
    }
    let checked = sky_types::check_module(&ast, src);
    if checked.has_errors() {
        REJECTED.fetch_add(1, Ordering::Relaxed);
        return Ok(());
    }
    let lowered_ir = sky_lower::lower(&ast, src, &checked).map_err(|e| format!("lowering: {e}"))?;

    let walker: Run = {
        let mut env = sky_interp::RuntimeEnv::with_prelude();
        let mut out = Vec::new();
        let r = sky_interp::run_module(&ast, src, &mut env, &mut out);
        (
            String::from_utf8_lossy(&out).into_owned(),
            r.err()
                .map(|e| (e.message.clone(), render(&e.to_diagnostic(), src))),
        )
    };
    let run_sir = |m: &sky_ir::Module| -> Run {
        let mut out = Vec::new();
        let r = sky_ir_interp::run(m, &mut out);
        (
            String::from_utf8_lossy(&out).into_owned(),
            r.err()
                .map(|e| (e.message.clone(), render(&e.to_diagnostic(), src))),
        )
    };
    let want = normalized(walker);
    let lowered = normalized(run_sir(&lowered_ir));
    if lowered != want {
        return Err(format!(
            "the lowered SIR differs from `sky run` (not a pass's doing)\n--- sky run ---\n{}{}\n--- SIR ---\n{}{}",
            want.0,
            want.1.as_deref().unwrap_or("(ok)"),
            lowered.0,
            lowered.1.as_deref().unwrap_or("(ok)"),
        ));
    }
    for (name, with_specialize) in [("optimized", true), ("optimized without specialize", false)] {
        let mut ir = lowered_ir.clone();
        let manager = PassManager::new().verify_each(true);
        let manager = if with_specialize {
            default_passes(manager)
        } else {
            default_passes_without_specialize(manager)
        };
        let report = manager
            .run(&mut ir)
            .map_err(|e| format!("a pass broke the IR ({name}): {e}"))?;
        if !with_specialize {
            for (pass, changed) in &report.ran {
                if changed.is_yes() && *pass == "licm" {
                    LICM_CHANGED.fetch_add(1, Ordering::Relaxed);
                }
                if changed.is_yes() && *pass == "gvn" {
                    GVN_CHANGED.fetch_add(1, Ordering::Relaxed);
                }
            }
            if report.explanations.iter().any(|x| x.rule == "LICM-002") {
                GUARDED.fetch_add(1, Ordering::Relaxed);
            }
        }
        let got = normalized(run_sir(&ir));
        if got != want {
            return Err(format!(
                "the {name} SIR differs from `sky run`\n--- sky run ---\n{}{}\n--- SIR ---\n{}{}\n--- SIR text ---\n{}",
                want.0,
                want.1.as_deref().unwrap_or("(ok)"),
                got.0,
                got.1.as_deref().unwrap_or("(ok)"),
                sky_ir::print(&ir),
            ));
        }
    }
    Ok(())
}

fn env_number(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

#[test]
fn hoisting_keeps_the_whole_diagnostic_on_loops_with_invariants() {
    if let Some(seed) = env_number("SKY_SEED") {
        println!("{}", program(seed));
        return;
    }
    let first = env_number("SKY_FUZZ_FROM").unwrap_or(1);
    let count = env_number("SKY_FUZZ_SEEDS").unwrap_or(1500);
    let lanes = std::thread::available_parallelism().map_or(1, |n| n.get().min(8)) as u64;
    let keep_going = std::env::var("SKY_FUZZ_KEEP_GOING").is_ok();
    let failures: Vec<String> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..lanes)
            .map(|lane| {
                scope.spawn(move || {
                    let mut failures = Vec::new();
                    let mut seed = first + lane;
                    while seed < first + count {
                        let src = program(seed);
                        if let Err(msg) = sky_ast::with_deep_stack(|| check(&src)) {
                            failures.push(format!("seed {seed}: {msg}\n--- program ---\n{src}"));
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
            .flat_map(|h| h.join().expect("fuzz lane panicked"))
            .collect()
    });
    let rejected = REJECTED.load(Ordering::Relaxed);
    println!(
        "{count} programs from seed {first}: gvn changed {}, licm changed {} ({} guarded), the checker rejected {rejected}",
        GVN_CHANGED.load(Ordering::Relaxed),
        LICM_CHANGED.load(Ordering::Relaxed),
        GUARDED.load(Ordering::Relaxed),
    );
    assert!(
        failures.is_empty(),
        "{} failing seed(s):\n{}",
        failures.len(),
        failures.join("\n\n")
    );
    // The fuzzer must exercise the passes, or it proves nothing.
    if count >= 500 {
        assert!(
            (rejected as u64) * 10 < count,
            "the checker rejects too many generated programs: {rejected}"
        );
        assert!(LICM_CHANGED.load(Ordering::Relaxed) as u64 * 5 > count);
        assert!(GUARDED.load(Ordering::Relaxed) > 0);
    }
}
