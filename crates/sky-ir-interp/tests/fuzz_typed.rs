//! Type specialization under fuzzing: programs made of *unannotated*
//! functions, which is where `specialize` clones a function for the types of
//! its arguments (`f[a:Scalar(length)]`) and types the body again.
//!
//! The differential fuzzer's generator rarely calls an unannotated function
//! with arguments whose types are known, so this file has its own. Each
//! program defines a few functions (dimensionless arithmetic, same-dimension
//! arithmetic, loops over vectors, Bool logic, recursion, early returns,
//! calls to earlier functions, division that may fail) and calls each with
//! arguments of several kinds: numbers, numbers with units, Bools, strings,
//! vectors, and the results of other calls (unknown at the call). Most calls
//! fit the function; some do not, so the errors the generic function reports
//! ("`+` needs numbers", a dimension mismatch, a Bool where a number is
//! needed) are compared too.
//!
//! The oracle is `fuzz/harness/differential.rs`'s `check_source`: the
//! tree-walker, the lowered SIR, the SIR after the default pipeline and the
//! native engine on both must print the same and fail the same, as the whole
//! rendered diagnostic.
//!
//! `SKY_FUZZ_FROM` and `SKY_FUZZ_SEEDS` run a longer campaign (use
//! `--release`); the test reports how many programs got a typed clone.

#[path = "../../../fuzz/harness/differential.rs"]
mod harness;

use std::sync::atomic::{AtomicUsize, Ordering};

/// xorshift64*.
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

/// The kind of value a function is written for.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// Dimensionless numbers.
    Plain,
    /// Numbers of one dimension, whatever it is.
    Same,
    /// A vector (of any dimension) and a dimensionless index.
    Vector,
    /// Bools.
    Logic,
}

struct Func {
    name: String,
    kind: Kind,
    params: usize,
}

const UNITS: [&str; 6] = ["m", "s", "kg", "m/s", "m^2", "N"];

/// A literal a function of `kind` accepts (mostly).
fn arg_for(r: &mut Rng, kind: Kind, earlier: &[Func]) -> String {
    let wrong = r.chance(12);
    let kind = if wrong {
        [Kind::Plain, Kind::Same, Kind::Vector, Kind::Logic][r.below(4)]
    } else {
        kind
    };
    let n = |r: &mut Rng| ["0", "1", "2", "3", "5", "-4", "0.5", "7", "10"][r.below(9)];
    match kind {
        Kind::Plain => {
            if r.chance(10) {
                call_of(r, earlier).unwrap_or_else(|| n(r).to_string())
            } else {
                n(r).to_string()
            }
        }
        Kind::Same => format!("{} {}", n(r), UNITS[r.below(UNITS.len())]),
        Kind::Vector => {
            let len = 1 + r.below(12);
            let unit = if r.chance(50) { "" } else { " m" };
            let xs: Vec<String> = (0..len).map(|i| format!("{}{unit}", i + 1)).collect();
            format!("[{}]", xs.join(", "))
        }
        Kind::Logic => {
            if r.chance(90) {
                r.pick(&["true", "false"]).to_string()
            } else {
                "\"s\"".to_string()
            }
        }
    }
}

/// A call to an earlier function whose result is unknown at the call.
fn call_of(r: &mut Rng, earlier: &[Func]) -> Option<String> {
    let plain: Vec<&Func> = earlier.iter().filter(|f| f.kind == Kind::Plain).collect();
    if plain.is_empty() {
        return None;
    }
    let f = plain[r.below(plain.len())];
    let args: Vec<String> = (0..f.params)
        .map(|_| ["1", "2", "3"][r.below(3)].to_string())
        .collect();
    Some(format!("{}({})", f.name, args.join(", ")))
}

fn plain_expr(r: &mut Rng, vars: &[&str], depth: usize) -> String {
    if depth == 0 || r.chance(30) {
        return if r.chance(65) {
            vars[r.below(vars.len())].to_string()
        } else {
            ["1", "2", "3", "0.5", "10"][r.below(5)].to_string()
        };
    }
    let a = plain_expr(r, vars, depth - 1);
    let b = plain_expr(r, vars, depth - 1);
    match r.below(9) {
        0 | 1 => format!("({a} + {b})"),
        2 | 3 => format!("({a} - {b})"),
        4 => format!("({a} * {b})"),
        5 => format!("({a} / ({b} + 0.5))"),
        6 => format!("abs({a})"),
        7 => format!("max({a}, {b})"),
        _ => format!("-{a}"),
    }
}

fn same_expr(r: &mut Rng, vars: &[&str], depth: usize) -> String {
    if depth == 0 || r.chance(35) {
        return vars[r.below(vars.len())].to_string();
    }
    let a = same_expr(r, vars, depth - 1);
    let b = same_expr(r, vars, depth - 1);
    match r.below(7) {
        0 | 1 => format!("({a} + {b})"),
        2 | 3 => format!("({a} - {b})"),
        4 => format!("({a} * 2)"),
        5 => format!("abs({a})"),
        _ => format!("min({a}, {b})"),
    }
}

fn cond(r: &mut Rng, kind: Kind, vars: &[&str]) -> String {
    let v = |r: &mut Rng| vars[r.below(vars.len())].to_string();
    match kind {
        Kind::Plain => {
            let op = r.pick(&["<", ">", "<=", ">=", "==", "!="]);
            format!(
                "{} {op} {}",
                plain_expr(r, vars, 1),
                ["0", "1", "2", "5"][r.below(4)]
            )
        }
        Kind::Same => {
            let op = r.pick(&["<", ">", "<=", ">=", "==", "!="]);
            format!("{} {op} {}", v(r), v(r))
        }
        Kind::Logic => match r.below(3) {
            0 => format!("{} and {}", v(r), v(r)),
            1 => format!("{} or {}", v(r), v(r)),
            _ => format!("not {}", v(r)),
        },
        Kind::Vector => "true".to_string(),
    }
}

fn expr(r: &mut Rng, kind: Kind, vars: &[&str]) -> String {
    match kind {
        Kind::Plain => plain_expr(r, vars, 2),
        Kind::Same => same_expr(r, vars, 2),
        Kind::Logic => cond(r, Kind::Logic, vars),
        Kind::Vector => vars[0].to_string(),
    }
}

/// One function.
fn function(r: &mut Rng, index: usize, earlier: &[Func]) -> (Func, String) {
    let name = format!("f{index}");
    let kind = [
        Kind::Plain,
        Kind::Plain,
        Kind::Same,
        Kind::Vector,
        Kind::Logic,
    ][r.below(5)];
    let mut out = String::new();
    let params = match kind {
        Kind::Vector => 1,
        _ => 1 + r.below(2),
    };
    let vars: Vec<&str> = ["a", "b"][..params].to_vec();
    let plist = vars.join(", ");
    out.push_str(&format!("def {name}({plist}):\n"));
    match kind {
        Kind::Vector => {
            let shape = r.below(3);
            out.push_str("    total = a[0]\n    i = 1\n    while i < len(a):\n");
            match shape {
                0 => out.push_str("        total += a[i]\n"),
                1 => out.push_str("        if a[i] > total:\n            total = a[i]\n"),
                _ => out.push_str("        total = total + a[i] - a[i - 1]\n"),
            }
            out.push_str("        i += 1\n");
            if r.chance(30) {
                out.push_str("    if len(a) > 4:\n        return total\n");
            }
            out.push_str("    return total\n");
        }
        _ => {
            let recursive = kind == Kind::Plain && params == 1 && r.chance(30);
            if recursive {
                let k = 1 + r.below(3);
                let combine = r.pick(&["+", "+", "*"]);
                // A bound, so an argument that is the result of another call
                // does not start an exponential recursion.
                out.push_str(&format!(
                    "    if a > 14:\n        return a\n    if a < {k}:\n        return a\n"
                ));
                let calls = if r.chance(50) {
                    format!("{name}(a - 1) {combine} {name}(a - 2)")
                } else {
                    format!("a {combine} {name}(a - 1)")
                };
                out.push_str(&format!("    return {calls}\n"));
            } else {
                let mut vars = vars.clone();
                let stmts = 1 + r.below(4);
                let mut locals = 0;
                for _ in 0..stmts {
                    match r.below(6) {
                        0 => {
                            let l = format!("t{locals}");
                            locals += 1;
                            out.push_str(&format!("    {l} = {}\n", expr(r, kind, &vars)));
                            vars.push(match l.as_str() {
                                "t0" => "t0",
                                "t1" => "t1",
                                "t2" => "t2",
                                _ => "t3",
                            });
                        }
                        1 => {
                            let c = cond(r, kind, &vars);
                            out.push_str(&format!(
                                "    if {c}:\n        return {}\n",
                                expr(r, kind, &vars)
                            ));
                        }
                        2 => {
                            let c = cond(r, kind, &vars);
                            out.push_str(&format!(
                                "    if {c}:\n        return {}\n    elif {}:\n        return {}\n",
                                expr(r, kind, &vars),
                                cond(r, kind, &vars),
                                expr(r, kind, &vars)
                            ));
                        }
                        3 if kind == Kind::Plain => {
                            let l = format!("t{locals}");
                            locals += 1;
                            out.push_str(&format!(
                                "    {l} = 0\n    k = 0\n    while k < {}:\n        {l} += {}\n        k += 1\n",
                                1 + r.below(4),
                                plain_expr(r, &vars, 1)
                            ));
                            vars.push(match l.as_str() {
                                "t0" => "t0",
                                "t1" => "t1",
                                "t2" => "t2",
                                _ => "t3",
                            });
                        }
                        4 if kind == Kind::Plain && r.chance(50) => {
                            out.push_str(&format!(
                                "    if {}:\n        return 1 / {}\n",
                                cond(r, kind, &vars),
                                plain_expr(r, &vars, 1)
                            ));
                        }
                        5 if r.chance(20) => {
                            out.push_str(&format!("    print({})\n", vars[r.below(vars.len())]));
                        }
                        _ => {
                            if let (true, Some(c)) = (kind == Kind::Plain, call_of(r, earlier)) {
                                out.push_str(&format!(
                                    "    if {}:\n        return {c}\n",
                                    cond(r, kind, &vars)
                                ));
                            }
                        }
                    }
                }
                out.push_str(&format!("    return {}\n", expr(r, kind, &vars)));
            }
        }
    }
    (Func { name, kind, params }, out)
}

fn program(seed: u64) -> String {
    let mut r = Rng::new(seed);
    let mut funcs: Vec<Func> = Vec::new();
    let mut src = String::new();
    for i in 0..1 + r.below(4) {
        let (f, text) = function(&mut r, i, &funcs);
        src.push_str(&text);
        src.push('\n');
        funcs.push(f);
    }
    for _ in 0..2 + r.below(5) {
        let f = &funcs[r.below(funcs.len())];
        let args: Vec<String> = (0..f.params)
            .map(|i| {
                // The arguments of one call mostly share a kind (and a
                // dimension), so `a + b` is meaningful.
                let _ = i;
                arg_for(&mut r, f.kind, &funcs[..])
            })
            .collect();
        let args = if f.kind == Kind::Same && f.params == 2 && r.chance(80) {
            let one = arg_for(&mut r, Kind::Same, &funcs[..]);
            let unit = one.split_once(' ').map_or("m", |(_, u)| u).to_string();
            vec![one, format!("{} {unit}", ["1", "2", "3"][r.below(3)])]
        } else {
            args
        };
        src.push_str(&format!("print({}({}))\n", f.name, args.join(", ")));
    }
    src
}

fn env_number(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

/// Programs that got a typed clone in the default pipeline.
fn typed_clones(src: &str) -> usize {
    let Ok(mut m) = sky_lower::lower_source(src) else {
        return 0;
    };
    let Ok(report) = sky_opt::default_pipeline().run(&mut m) else {
        return 0;
    };
    report
        .explanations
        .iter()
        .filter(|x| x.rule == sky_opt::specialize::RULE_RETYPED)
        .count()
}

#[test]
fn typed_clones_agree_in_every_engine() {
    // `SKY_SEED=n` prints the program for seed `n` and stops.
    if let Some(seed) = env_number("SKY_SEED") {
        println!("{}", program(seed));
        return;
    }
    let first = env_number("SKY_FUZZ_FROM").unwrap_or(1);
    let count = env_number("SKY_FUZZ_SEEDS").unwrap_or(1500);
    let lanes = std::thread::available_parallelism().map_or(1, |n| n.get().min(8)) as u64;
    let keep_going = std::env::var("SKY_FUZZ_KEEP_GOING").is_ok();
    let programs = AtomicUsize::new(0);
    let with_clone = AtomicUsize::new(0);
    let clones = AtomicUsize::new(0);
    let rejected = AtomicUsize::new(0);
    let errors = AtomicUsize::new(0);
    let failures: Vec<String> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..lanes)
            .map(|lane| {
                let (programs, with_clone, clones, rejected, errors) =
                    (&programs, &with_clone, &clones, &rejected, &errors);
                scope.spawn(move || {
                    let mut failures = Vec::new();
                    let mut seed = first + lane;
                    while seed < first + count {
                        let src = program(seed);
                        let started = std::time::Instant::now();
                        let checked = harness::check_source(&src);
                        if started.elapsed().as_secs() >= 5 {
                            eprintln!("slow seed {seed}: {:?}", started.elapsed());
                        }
                        match checked {
                            Ok(o) => {
                                programs.fetch_add(1, Ordering::Relaxed);
                                errors.fetch_add(usize::from(o.error.is_some()), Ordering::Relaxed);
                                let n = sky_ast::with_deep_stack(|| typed_clones(&src));
                                clones.fetch_add(n, Ordering::Relaxed);
                                with_clone.fetch_add(usize::from(n > 0), Ordering::Relaxed);
                            }
                            // The checker rejected the program: a generator
                            // limit, not a finding.
                            Err(msg) if msg.starts_with("generator bug") => {
                                rejected.fetch_add(1, Ordering::Relaxed);
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
    });
    eprintln!(
        "fuzz_typed: {} programs ({} rejected by the checker), {} with a typed clone ({} clones), {} ending in a run-time error",
        programs.load(Ordering::Relaxed),
        rejected.load(Ordering::Relaxed),
        with_clone.load(Ordering::Relaxed),
        clones.load(Ordering::Relaxed),
        errors.load(Ordering::Relaxed),
    );
    assert!(
        failures.is_empty(),
        "{} failure(s):\n{}",
        failures.len(),
        failures.join("\n\n")
    );
    // A campaign that never reaches the pass proves nothing.
    assert!(
        with_clone.load(Ordering::Relaxed) * 4 >= programs.load(Ordering::Relaxed),
        "too few programs got a typed clone"
    );
}
