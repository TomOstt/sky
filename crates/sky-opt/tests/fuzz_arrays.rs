//! Array programs under the engines: the tree-walker, the lowered SIR,
//! and the SIR after the default pipeline must print the same bytes and stop
//! with the same error.
//!
//! The differential fuzzer's generator (`fuzz/harness/differential.rs`) makes
//! scalar programs. This one is about what buffer reuse could get wrong, so
//! it makes programs full of *aliasing*: a few vectors and matrices that are
//! copied into one another (`d = a`), reassigned from themselves
//! (`a = a + a`, `a += b`), scaled, negated, passed to functions that return
//! them or update them, updated in loops (`for` and `while`, with `break`
//! and `continue`) and in the arms of `if`, and read again afterwards. Every
//! program prints its variables along the way and at the end, so a value
//! that a reused buffer corrupted shows up as different output.
//!
//! Programs are generated from a seed and type-check by construction. All
//! values are dimensionless (arithmetic on them cannot mismatch), of two
//! shapes: `Vector[3]` and `Matrix[2, 2]`, kept apart.
//!
//! `SKY_FUZZ_FROM` and `SKY_FUZZ_SEEDS` run a longer campaign (use
//! `--release`); `SKY_FUZZ_KEEP_GOING=1` lists every failing seed;
//! `SKY_SEED=n` prints the program for seed `n`.

use sky_diag::SourceFile;
use sky_ir::PassManager;
use sky_opt::default_passes;

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

const VECS: [&str; 4] = ["a", "b", "c", "d"];
const MATS: [&str; 2] = ["M", "N"];

struct Gen {
    rng: Rng,
    out: String,
    /// Loop counters in use, so nested `while` loops get their own.
    counters: usize,
    /// Nesting of loops (`break` and `continue` need one).
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

    fn vec(&mut self) -> &'static str {
        VECS[self.rng.below(VECS.len())]
    }

    fn mat(&mut self) -> &'static str {
        MATS[self.rng.below(MATS.len())]
    }

    /// A scalar expression.
    fn scalar(&mut self) -> String {
        match self.rng.below(8) {
            0 => "s".to_string(),
            1 => "t".to_string(),
            2 => "2".to_string(),
            3 => "0.5".to_string(),
            4 => format!("{}[{}]", self.vec(), self.rng.below(3)),
            5 => format!(
                "{}[{}, {}]",
                self.mat(),
                self.rng.below(2),
                self.rng.below(2)
            ),
            6 => "(s * t)".to_string(),
            _ => "(s + 1)".to_string(),
        }
    }

    /// A vector expression.
    fn vector(&mut self, depth: usize) -> String {
        if depth == 0 {
            return self.vec().to_string();
        }
        match self.rng.below(24) {
            0 | 1 => self.vec().to_string(),
            2 => format!("{} + {}", self.vector(depth - 1), self.vector(depth - 1)),
            3 => format!("{} - {}", self.vector(depth - 1), self.vec()),
            4 => format!("{} * {}", self.vec(), self.scalar()),
            5 => format!("{} * {}", self.scalar(), self.vector(depth - 1)),
            6 => format!("{} / 2", self.vector(depth - 1)),
            7 => format!("-{}", self.vec()),
            8 => format!("({} + {}) * {}", self.vec(), self.vec(), self.scalar()),
            9 => format!("{} + {} * {}", self.vec(), self.scalar(), self.vec()),
            10 => {
                let v = self.vec();
                format!("{v} + {v}")
            }
            11 => format!(
                "add2({}, {})",
                self.vector(depth - 1),
                self.vector(depth - 1)
            ),
            12 => format!("keep({})", self.vector(depth - 1)),
            13 => {
                let v = self.vec();
                format!("add2({v}, {v})")
            }
            14 => format!("[{}, {}, {}]", self.scalar(), self.scalar(), self.scalar()),
            // Entry by entry: products, quotients and math functions of
            // arrays, with operands that alias each other and the target.
            15 => format!("{} * {}", self.vector(depth - 1), self.vec()),
            16 => {
                let v = self.vec();
                format!("{v} * {v}")
            }
            17 => format!(
                "{} / hypot({}, {})",
                self.vector(depth - 1),
                self.vec(),
                self.vec()
            ),
            18 => format!("sqrt(abs({}))", self.vector(depth - 1)),
            19 => format!("max({}, {})", self.vector(depth - 1), self.vec()),
            20 => format!("hypot({}, {})", self.vec(), self.vector(depth - 1)),
            21 => format!("atan2({}, {}) * {}", self.vec(), self.vec(), self.scalar()),
            22 => {
                let v = self.vec();
                format!("min({v}, {}) / hypot({v}, {v})", self.scalar())
            }
            _ => format!("{} - {}", self.vec(), self.vector(depth - 1)),
        }
    }

    /// A matrix expression.
    fn matrix(&mut self, depth: usize) -> String {
        if depth == 0 {
            return self.mat().to_string();
        }
        match self.rng.below(12) {
            0 | 1 => self.mat().to_string(),
            2 => format!("{} + {}", self.matrix(depth - 1), self.mat()),
            3 => format!("{} - {}", self.mat(), self.matrix(depth - 1)),
            4 => format!("{} * {}", self.mat(), self.scalar()),
            5 => format!("{}ᵀ", self.mat()),
            6 => format!("{} @ {}", self.mat(), self.mat()),
            7 => {
                let m = self.mat();
                format!("{m} + {m}")
            }
            8 => format!("{} * {}", self.matrix(depth - 1), self.mat()),
            9 => {
                let m = self.mat();
                format!("{m} * {m}")
            }
            10 => format!("sqrt(abs({}))", self.matrix(depth - 1)),
            _ => format!("-{}", self.mat()),
        }
    }

    fn statement(&mut self, indent: usize, depth: usize) {
        let n = if depth == 0 { 9 } else { 15 };
        match self.rng.below(n) {
            0..=2 => {
                let v = self.vec();
                let e = self.vector(2);
                self.line(indent, &format!("{v} = {e}"));
            }
            3 => {
                // A reassignment from itself: the reuse case.
                let v = self.vec();
                let e = match self.rng.below(4) {
                    0 => format!("{v} + {}", self.vec()),
                    1 => format!("{v} * {}", self.scalar()),
                    2 => format!("{v} + {} * {}", self.scalar(), self.vec()),
                    _ => format!("{v} - {v}"),
                };
                self.line(indent, &format!("{v} = {e}"));
            }
            4 => {
                let v = self.vec();
                let op = self.rng.pick(&["+=", "-="]);
                let e = self.vector(1);
                self.line(indent, &format!("{v} {op} {e}"));
            }
            5 => {
                let m = self.mat();
                let e = self.matrix(2);
                self.line(indent, &format!("{m} = {e}"));
            }
            6 => {
                let (x, y) = (self.vec(), self.vec());
                self.line(indent, &format!("{x} = {y}"));
            }
            7 => {
                let e = self.scalar();
                let target = self.rng.pick(&["s", "t"]);
                self.line(indent, &format!("{target} = {e} * 0.5 + 0.25"));
            }
            8 => {
                let v = self.vec();
                let text = if self.rng.chance(50) {
                    format!("print({v})")
                } else {
                    let e = self.scalar();
                    format!("print({e}, {v}, {})", self.mat())
                };
                self.line(indent, &text);
            }
            9 | 10 => {
                let cond = match self.rng.below(3) {
                    0 => "s > t".to_string(),
                    1 => format!("{}[0] < {}[1]", self.vec(), self.vec()),
                    _ => "s < 100".to_string(),
                };
                self.line(indent, &format!("if {cond}:"));
                self.block(indent + 1, depth - 1);
                if self.rng.chance(60) {
                    self.line(indent, "else:");
                    self.block(indent + 1, depth - 1);
                }
            }
            11 | 12 => {
                let k = 1 + self.rng.below(3);
                self.line(indent, &format!("for i in range({k}):"));
                self.in_loop += 1;
                self.block(indent + 1, depth - 1);
                self.in_loop -= 1;
            }
            13 => {
                let counter = format!("n{}", self.counters);
                self.counters += 1;
                let k = 1 + self.rng.below(3);
                self.line(indent, &format!("{counter} = 0"));
                self.line(indent, &format!("while {counter} < {k}:"));
                self.line(indent + 1, &format!("{counter} = {counter} + 1"));
                self.in_loop += 1;
                self.block(indent + 1, depth - 1);
                self.in_loop -= 1;
            }
            _ => {
                if self.in_loop > 0 {
                    let word = self.rng.pick(&["break", "continue"]);
                    self.line(indent, "if s > 0.75:");
                    self.line(indent + 1, word);
                } else {
                    let v = self.vec();
                    self.line(indent, &format!("{v} = bump({v})"));
                }
            }
        }
    }

    fn block(&mut self, indent: usize, depth: usize) {
        let n = 1 + self.rng.below(3);
        for _ in 0..n {
            self.statement(indent, depth);
        }
    }
}

/// The program for `seed`.
fn generate(seed: u64) -> String {
    let mut g = Gen {
        rng: Rng::new(seed),
        out: String::new(),
        counters: 0,
        in_loop: 0,
    };
    g.line(0, "def add2(u: Vector[3], w: Vector[3]) -> Vector[3]:");
    g.line(1, "return u + w");
    g.line(0, "def keep(u: Vector[3]) -> Vector[3]:");
    g.line(1, "return u");
    g.line(0, "def bump(u: Vector[3]) -> Vector[3]:");
    g.line(1, "u = u + u");
    g.line(1, "return u");
    g.line(0, "a = [1, 2, 3]");
    g.line(0, "b = [0.5, 0.25, 4]");
    g.line(0, "c = [2, 2, 2]");
    g.line(0, "d = a");
    g.line(0, "M = [[1, 2], [3, 4]]");
    g.line(0, "N = M");
    g.line(0, "s = 2");
    g.line(0, "t = 0.5");
    let top = 4 + g.rng.below(8);
    for _ in 0..top {
        g.statement(0, 2);
    }
    g.line(0, "print(a, b, c, d)");
    g.line(0, "print(M, N, s, t)");
    g.out
}

/// Stdout, and the error as (message, rendered diagnostic).
type Run = (String, Option<(String, String)>);

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

fn check(src: &str) -> Result<(), String> {
    let lexed = sky_syntax::lex(src).map_err(|_| "the lexer rejects the program".to_string())?;
    let (ast, errors) = sky_ast::parse_module(&lexed.tokens, src);
    if !errors.is_empty() {
        return Err(format!("generator bug: parse errors: {errors:?}"));
    }
    let checked = sky_types::check_module(&ast, src);
    if checked.has_errors() {
        let ms: Vec<&str> = checked
            .diagnostics
            .iter()
            .filter(|d| d.severity == sky_diag::Severity::Error)
            .map(|d| d.message.as_str())
            .collect();
        return Err(format!(
            "generator bug: sky check rejects the program: {ms:?}"
        ));
    }
    let mut ir = sky_lower::lower(&ast, src, &checked).map_err(|e| format!("lowering: {e}"))?;
    let render =
        |d: &sky_diag::Diagnostic| d.render_to_string(&SourceFile::new("program.sky", src));

    let walker: Run = {
        let mut env = sky_interp::RuntimeEnv::with_prelude();
        let mut out = Vec::new();
        let r = sky_interp::run_module(&ast, src, &mut env, &mut out);
        (
            String::from_utf8_lossy(&out).into_owned(),
            r.err()
                .map(|e| (e.message.clone(), render(&e.to_diagnostic()))),
        )
    };
    let run_with = |m: &sky_ir::Module, cfg: &sky_ir::parallel::Config| -> Run {
        let mut out = Vec::new();
        let r = sky_ir_interp::run_with(m, &mut out, cfg);
        (
            String::from_utf8_lossy(&out).into_owned(),
            r.err()
                .map(|e| (e.message.clone(), render(&e.to_diagnostic()))),
        )
    };
    // The default configuration (`SKY_THREADS`), and the most parallel one
    // (every threshold at its minimum, so independent calls and kernels of
    // these small programs run on the pool): both must give the sequential
    // run's output.
    let run_sir = |m: &sky_ir::Module| run_with(m, &sky_ir::parallel::Config::from_env());
    let run_eager = |m: &sky_ir::Module| run_with(m, &sky_ir::parallel::Config::eager(4));
    let lowered = run_sir(&ir);
    let lowered_eager = run_eager(&ir);
    default_passes(PassManager::new().verify_each(true))
        .run(&mut ir)
        .map_err(|e| format!("a pass broke the IR: {e}"))?;
    let optimized = run_sir(&ir);
    let optimized_eager = run_eager(&ir);

    let want = normalized(walker);
    for (what, got) in [
        ("lowered", lowered),
        ("optimized", optimized),
        ("lowered, eager parallel", lowered_eager),
        ("optimized, eager parallel", optimized_eager),
    ] {
        let got = normalized(got);
        if got != want {
            return Err(format!(
                "the {what} SIR differs from `sky run`\n--- sky run ---\n{}{}\n--- SIR ---\n{}{}\n--- SIR text ---\n{}",
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
fn aliasing_array_programs_agree_in_all_three_engines() {
    if let Some(seed) = env_number("SKY_SEED") {
        println!("{}", generate(seed));
        return;
    }
    let first = env_number("SKY_FUZZ_FROM").unwrap_or(1);
    let count = env_number("SKY_FUZZ_SEEDS").unwrap_or(1500);
    let lanes = std::thread::available_parallelism().map_or(1, |n| n.get().min(8)) as u64;
    let keep_going = std::env::var("SKY_FUZZ_KEEP_GOING").is_ok();
    let results: Vec<(Vec<String>, u64, u64)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..lanes)
            .map(|lane| {
                scope.spawn(move || {
                    let mut failures = Vec::new();
                    let (mut ran, mut errors) = (0u64, 0u64);
                    let mut seed = first + lane;
                    while seed < first + count {
                        let src = generate(seed);
                        match sky_ast::with_deep_stack(|| check(&src)) {
                            Ok(()) => ran += 1,
                            Err(msg) => {
                                failures
                                    .push(format!("seed {seed}: {msg}\n--- program ---\n{src}"));
                                errors += 1;
                                if !keep_going {
                                    break;
                                }
                            }
                        }
                        seed += lanes;
                    }
                    (failures, ran, errors)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("a lane panicked"))
            .collect()
    });
    let ran: u64 = results.iter().map(|r| r.1).sum();
    let failures: Vec<&String> = results.iter().flat_map(|r| &r.0).collect();
    eprintln!("{ran} array programs agree, {} disagree", failures.len());
    assert!(
        failures.is_empty(),
        "{} failure(s), first:\n{}",
        failures.len(),
        failures[0]
    );
}
