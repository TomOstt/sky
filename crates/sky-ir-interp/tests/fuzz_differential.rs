//! Runs the `differential` fuzz check with stable Rust, on a few thousand
//! programs generated from fixed seeds. The generator and the oracle are
//! `fuzz/harness/differential.rs`, the same file the fuzz target includes.
//!
//! Every program is well typed by construction and is run through the AST
//! interpreter and through lowering plus the SIR interpreter; stdout and the
//! error (or success) must be identical, the lowered SIR must verify, and the
//! SIR text format must round-trip (see the harness for the details).
//!
//! The tests at the end are the bugs this fuzzer found, each reduced to a few
//! lines; all are fixed (two are design limits of lowering, which the tests
//! pin). A bug found later is added there as an `#[ignore]`d test, and the
//! generator steers around it (a `GATE_*` constant in the harness) so the
//! main test stays green and the fuzzer keeps looking for new bugs.
//!
//! Tools for a fuzzing session (each does nothing unless its variable is
//! set): `SKY_SEED=n` prints the program for seed `n`; `SKY_FILE=x.sky` runs
//! the oracle on a file; `SKY_MINIMIZE=x.sky` reduces a failing program;
//! `SKY_FUZZ_FROM`/`SKY_FUZZ_SEEDS`/`SKY_FUZZ_LEN` run a longer campaign
//! (use `--release`), with `SKY_FUZZ_KEEP_GOING=1` to list every failing
//! seed instead of stopping at the first.

#[path = "../../../fuzz/harness/differential.rs"]
mod harness;

use std::collections::BTreeMap;

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

/// The input for `seed`; lengths vary so programs of every size appear.
fn input(seed: u64) -> Vec<u8> {
    if let Some(len) = env_number("SKY_FUZZ_LEN") {
        return bytes(seed, len as usize);
    }
    let len = 150 + (seed.wrapping_mul(2_654_435_761) % 1400) as usize;
    bytes(seed, len)
}

/// What a run of seeds saw.
#[derive(Default)]
struct Stats {
    programs: u64,
    failed: u64,
    limit: u64,
    printed: usize,
    known: BTreeMap<&'static str, u64>,
    /// Why native code refused programs, and how many times.
    native_refusals: BTreeMap<String, u64>,
    /// Programs on which native code deoptimized.
    native_deopts: u64,
    errors: BTreeMap<String, u64>,
    sources: String,
}

impl Stats {
    fn merge(&mut self, o: Stats) {
        self.programs += o.programs;
        self.failed += o.failed;
        self.limit += o.limit;
        self.printed += o.printed;
        for (k, v) in o.known {
            *self.known.entry(k).or_default() += v;
        }
        self.native_deopts += o.native_deopts;
        for (k, v) in o.native_refusals {
            *self.native_refusals.entry(k).or_default() += v;
        }
        for (k, v) in o.errors {
            *self.errors.entry(k).or_default() += v;
        }
        self.sources.push_str(&o.sources);
    }
}

/// Runs the seeds `first + lane`, `first + lane + lanes`, ... below `end`.
fn run_lane(first: u64, end: u64, lane: u64, lanes: u64, keep_sources: bool) -> Stats {
    let mut stats = Stats::default();
    let keep_going = std::env::var("SKY_FUZZ_KEEP_GOING").is_ok();
    let mut seed = first + lane;
    while seed < end {
        let src = harness::generate(&input(seed));
        match harness::check_source(&src) {
            Ok(o) => {
                stats.programs += 1;
                stats.failed += u64::from(o.failed);
                stats.limit += u64::from(o.limit);
                stats.printed += o.printed;
                if let Some(e) = o.error {
                    // The kind of error, without the operand values.
                    let kind: String = e.chars().take(34).collect();
                    *stats.errors.entry(kind).or_default() += 1;
                }
                stats.native_deopts += u64::from(o.native_deopt);
                if let Some(why) = o.native_refusal {
                    *stats.native_refusals.entry(why).or_default() += 1;
                }
                if let Some(id) = o.known {
                    *stats.known.entry(id).or_default() += 1;
                }
                if keep_sources {
                    stats.sources.push_str(&src);
                }
            }
            Err(msg) if keep_going => {
                eprintln!("FAIL seed {seed}: {}", msg.lines().next().unwrap_or(""));
            }
            Err(msg) => panic!("seed {seed}: {msg}\n--- program ---\n{src}"),
        }
        seed += lanes;
    }
    stats
}

#[test]
fn generated_programs_agree_in_both_engines() {
    let first = env_number("SKY_FUZZ_FROM").unwrap_or(1);
    let count = env_number("SKY_FUZZ_SEEDS").unwrap_or(3000);
    let default_run = count == 3000 && first == 1 && env_number("SKY_FUZZ_LEN").is_none();
    let lanes = std::thread::available_parallelism().map_or(1, |n| n.get().min(8)) as u64;
    let mut stats = Stats::default();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..lanes)
            .map(|lane| {
                scope.spawn(move || run_lane(first, first + count, lane, lanes, default_run))
            })
            .collect();
        for h in handles {
            match h.join() {
                Ok(s) => stats.merge(s),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
    });
    eprintln!(
        "{} programs: {} stop with a run-time error ({} of them resource limits), {} bytes printed; known bugs tolerated: {:?}",
        stats.programs, stats.failed, stats.limit, stats.printed, stats.known
    );
    eprintln!(
        "native code refused {} programs: {:#?}; deoptimized on {}",
        stats.native_refusals.values().sum::<u64>(),
        stats.native_refusals,
        stats.native_deopts
    );
    if !default_run {
        eprintln!("errors: {:#?}", stats.errors);
        return;
    }
    let total = stats.programs;

    // The comparison must mean something: most programs run to the end, the
    // error paths are exercised too, and the recursion limit is reached.
    assert!(
        stats.errors.len() >= 6,
        "too few kinds of run-time error: {:?}",
        stats.errors
    );
    // About 7% of the fixed seeds fail at run time since math is no longer
    // steered around (math calls take choices that riskier arithmetic used
    // to); 6% keeps the error paths meaningfully exercised.
    assert!(
        stats.failed * 100 >= total * 6,
        "too few failing programs: {}",
        stats.failed
    );
    assert!(
        stats.failed * 100 <= total * 55,
        "too many failing programs: {}",
        stats.failed
    );
    assert!(stats.limit >= 1, "no program hit the recursion limit");
    // Contracts that fail are part of the comparison: the messages (with the
    // values of the names) and the spans agree.
    for kind in ["assertion failed", "precondition of", "postcondition of"] {
        assert!(
            stats.errors.keys().any(|e| e.starts_with(kind)),
            "no program failed with `{kind}...`: {:?}",
            stats.errors.keys().collect::<Vec<_>>()
        );
    }
    assert!(
        stats.printed > 100_000,
        "only {} bytes printed",
        stats.printed
    );

    // The generator covers the language it claims to.
    for needle in [
        " km",
        " ms",
        " μm",
        " kg",
        " GeV",
        "m/s",
        "kg*m",
        "^(1/2)",
        "^0.5",
        "^-1",
        "^2",
        "^64",
        "pi",
        "g_n",
        "c ",
        "sqrt(",
        "cbrt(",
        "exp(",
        "sin(",
        "cos(",
        "log(",
        "atan2(",
        "hypot(",
        "floor(",
        "round(",
        "inv(",
        "⁻¹",
        "abs(",
        "min(",
        "max(",
        " and ",
        " or ",
        "not ",
        "if ",
        "elif ",
        "else:",
        "while ",
        "break",
        "continue",
        "def ",
        " -> ",
        "return",
        "print(",
        " else ",
        "+=",
        "*=",
        "/=",
        "-=",
        ": Scalar",
        "≤",
        "≠",
        "true",
        "false",
        "\"hi\"",
        "while true:",
        "def f",
        "assert ",
        "requires ",
        "ensures ",
        "result",
    ] {
        assert!(
            stats.sources.contains(needle),
            "the generator never produced {needle:?}"
        );
    }
}

/// The inputs a mutating fuzzer produces first: empty, constant, ramps,
/// short, and the cut-off prefixes of a generated program's bytes. Each must
/// still give a valid program on which the engines agree.
#[test]
fn degenerate_inputs_are_valid_programs() {
    let mut inputs: Vec<(String, Vec<u8>)> = vec![("empty".into(), Vec::new())];
    for len in [1, 2, 3, 8, 50, 300, 2000] {
        for fill in [0u8, 1, 7, 100, 200, 255] {
            inputs.push((format!("{len} x {fill}"), vec![fill; len]));
        }
        inputs.push((
            format!("ramp {len}"),
            (0..len).map(|i| (i % 251) as u8).collect(),
        ));
        inputs.push((
            format!("down {len}"),
            (0..len).map(|i| 255 - (i % 251) as u8).collect(),
        ));
        inputs.push((
            format!("alternating {len}"),
            (0..len).map(|i| if i % 2 == 0 { 0 } else { 255 }).collect(),
        ));
    }
    // Every prefix (in steps) of a few generated inputs, and single-byte
    // changes to one.
    for seed in [3u64, 11, 42] {
        let data = input(seed);
        for cut in (0..data.len()).step_by(13) {
            inputs.push((format!("seed {seed} cut at {cut}"), data[..cut].to_vec()));
        }
        for at in (0..data.len().min(400)).step_by(7) {
            let mut d = data.clone();
            d[at] = d[at].wrapping_add(97);
            inputs.push((format!("seed {seed} byte {at} changed"), d));
        }
    }
    for (name, data) in inputs {
        let src = harness::generate(&data);
        if let Err(msg) = harness::check_source(&src) {
            panic!("{name}: {msg}\n--- program ---\n{src}");
        }
    }
}

/// The checked-in seed corpus of the fuzz target passes the fuzz check, so
/// the seeds the fuzzer starts from are valid inputs.
#[test]
fn the_seed_corpus_passes_the_fuzz_check() {
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fuzz/corpus/differential");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .collect();
    files.sort();
    assert!(files.len() >= 10, "{} seeds", files.len());
    for path in files {
        let data = std::fs::read(&path).expect("seed file");
        harness::check(&data);
    }
}

#[test]
fn show_generated_program() {
    if let Some(seed) = env_number("SKY_SEED") {
        println!("{}", harness::generate(&input(seed)));
    }
}

#[test]
fn replay_a_source_file() {
    if let Ok(path) = std::env::var("SKY_FILE") {
        let src = std::fs::read_to_string(&path).expect("SKY_FILE is readable");
        let run = if std::env::var("SKY_LENIENT").is_ok() {
            harness::check_source
        } else {
            harness::check_source_strict
        };
        match run(&src) {
            Ok(o) => println!("agree: {o:?}"),
            Err(msg) => panic!("{msg}"),
        }
    }
}

/// The kind of failure, without the values that change as a program shrinks.
fn signature(msg: &str) -> Option<String> {
    if msg.starts_with("generator bug") {
        return None;
    }
    if msg.starts_with("the error differs") {
        // The messages of the two diagnostics, without the position in their
        // headers (`program.sky:3:7: error: division by zero`), which moves
        // as a program shrinks.
        let mut sig = vec!["the error differs".to_string()];
        let mut lines = msg.lines();
        while let Some(l) = lines.next() {
            if l == "--- sky run ---" || l == "--- SIR ---" {
                let header = lines.next().unwrap_or("");
                sig.push(
                    header
                        .split_once(": error: ")
                        .map_or(header, |(_, m)| m)
                        .to_string(),
                );
            }
        }
        return Some(sig.join("\n"));
    }
    Some(msg.lines().next().unwrap_or("").to_string())
}

/// Whether every `while` in `lines` still ends: generated loops count with
/// `kN`, and removing the counter line would make a reduced program run
/// forever.
fn loops_terminate(lines: &[String]) -> bool {
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        let Some(rest) = t.strip_prefix("while ") else {
            continue;
        };
        let indent = line.len() - line.trim_start().len();
        let body: Vec<&str> = lines[i + 1..]
            .iter()
            .take_while(|l| l.trim().is_empty() || l.len() - l.trim_start().len() > indent)
            .map(|l| l.trim())
            .collect();
        if rest == "true:" {
            let ok = body.windows(2).any(|w| {
                let counter = w[0]
                    .strip_prefix("if ")
                    .and_then(|c| c.strip_suffix(" > 3:"));
                counter.is_some_and(|k| {
                    w[1] == "break"
                        && body
                            .iter()
                            .any(|b| *b == format!("{k} = {k} + 1") || *b == format!("{k} += 1"))
                })
            });
            if !ok {
                return false;
            }
        } else {
            let counter = rest.split(' ').next().unwrap_or("");
            let step = body.first().copied().unwrap_or("");
            if !(step == format!("{counter} = {counter} + 1") || step == format!("{counter} += 1"))
            {
                return false;
            }
        }
    }
    true
}

/// A tool for reducing a failing program: `SKY_MINIMIZE=in.sky
/// SKY_OUT=out.sky cargo test -p sky-ir-interp --test fuzz_differential
/// minimize`. Deletes lines and blocks while the same kind of failure remains
/// and the program stays valid.
#[test]
fn minimize_a_source_file() {
    let Ok(path) = std::env::var("SKY_MINIMIZE") else {
        return;
    };
    let src = std::fs::read_to_string(&path).expect("SKY_MINIMIZE is readable");
    let target = harness::check_source_strict(&src)
        .err()
        .and_then(|m| signature(&m))
        .expect("the program must fail the check");
    let holds = |ls: &[String]| {
        if !loops_terminate(ls) {
            return false;
        }
        let text = ls.join("\n") + "\n";
        harness::check_source_strict(&text)
            .err()
            .and_then(|m| signature(&m))
            .is_some_and(|sig| sig == target)
    };
    let indent = |l: &str| l.len() - l.trim_start().len();
    let mut lines: Vec<String> = src.lines().map(str::to_string).collect();
    loop {
        let before = lines.clone();
        // Chunks of lines, halving the chunk size.
        let mut n = lines.len().max(2) / 2;
        while n >= 1 {
            let mut i = 0;
            while i < lines.len() {
                let end = (i + n).min(lines.len());
                let mut cand = lines.clone();
                cand.drain(i..end);
                if !cand.is_empty() && holds(&cand) {
                    lines = cand;
                } else {
                    i += n;
                }
            }
            n /= 2;
        }
        // A line with the block under it.
        let mut i = 0;
        while i < lines.len() {
            let ind = indent(&lines[i]);
            let mut end = i + 1;
            while end < lines.len() && (lines[end].trim().is_empty() || indent(&lines[end]) > ind) {
                end += 1;
            }
            let mut cand = lines.clone();
            cand.drain(i..end);
            if !cand.is_empty() && holds(&cand) {
                lines = cand;
            } else {
                i += 1;
            }
        }
        // A block header replaced by its body, dedented one level.
        let mut i = 0;
        while i < lines.len() {
            if lines[i].trim_end().ends_with(':') && !lines[i].trim_start().starts_with("def ") {
                let ind = indent(&lines[i]);
                let mut end = i + 1;
                while end < lines.len()
                    && (lines[end].trim().is_empty() || indent(&lines[end]) > ind)
                {
                    end += 1;
                }
                let mut cand = lines.clone();
                let body: Vec<String> = lines[i + 1..end]
                    .iter()
                    .map(|l| l.get(4..).unwrap_or(l).to_string())
                    .collect();
                cand.splice(i..end, body);
                if holds(&cand) {
                    lines = cand;
                    continue;
                }
            }
            i += 1;
        }
        if lines == before {
            break;
        }
    }
    let out = lines.join("\n") + "\n";
    if let Ok(dest) = std::env::var("SKY_OUT") {
        std::fs::write(dest, &out).expect("SKY_OUT is writable");
    }
    println!("{out}");
}

// ----- bugs the fuzzer found, reduced -----

/// Requires the whole oracle (strict: no known bug is tolerated) to pass.
fn assert_agree(src: &str) {
    if let Err(msg) = harness::check_source_strict(src) {
        panic!("{msg}\n--- program ---\n{src}");
    }
}

/// Found by the fuzzer: `sky_ir::print` wrote `math.sqrt`, which
/// `sky_ir::parse` used to reject, so a module with a math operation did not
/// survive the text format. Fixed; kept as a regression test.
#[test]
fn sir_text_parse_accepts_the_math_operations_the_printer_writes() {
    assert_agree("print(sqrt(4))\n");
}

/// The checker follows the arm that falls through: `t` is bound after the
/// `if`, since the other arm leaves the loop. Lowering used to build no `t`
/// for the merge ("cannot find `t` in this scope"). Fixed: a name is a result
/// of the merge when every arm that falls through binds it.
#[test]
fn lowering_loses_a_name_bound_beside_a_break_arm() {
    // An `elif` chain, and a name bound in a `while true` loop that the
    // breaking arm also binds.
    assert_agree(
        "k = 0\nwhile k < 4:\n    k += 1\n    if k > 2:\n        break\n    elif k > 1:\n        t = 5\n    else:\n        t = 6\n    print(t)\nprint(k)\n",
    );
    assert_agree(
        "k = 0\nwhile true:\n    k += 1\n    if k > 2:\n        u = 1\n        break\n    else:\n        t = 6\n    print(t)\nprint(u, k)\n",
    );
    assert_agree(
        "k = 0\nwhile k < 3:\n    k += 1\n    if k > 1:\n        break\n    else:\n        t = 5\n    print(t)\nprint(k)\n",
    );
    // `continue` is the same.
    assert_agree(
        "k = 0\nwhile k < 3:\n    k += 1\n    if k > 1:\n        continue\n    else:\n        t = 5\n    print(t)\nprint(k)\n",
    );
}

/// `sky run` reads a rebound constant as the new value after the `if`;
/// the lowered code used to read the constant's own value (`pi`, `c`, ...),
/// at module level, in a function, and in a loop. Fixed: a constant bound
/// inside a block is an ordinary variable in the scope, seeded with its
/// value.
#[test]
fn lowering_ignores_a_constant_rebound_in_a_block() {
    // A function that reads the module's constant, which a block rebound.
    assert_agree("def h():\n    return pi\nx = 1\nif x > 0:\n    pi = 2\nprint(h(), pi)\n");
    // A `for` variable named like a constant.
    assert_agree("for c in range(3):\n    print(c)\nprint(c)\n");
    assert_agree("x = 1\nif x > 0:\n    c = 2 m/s\nprint(c)\n");
    assert_agree("def f(n):\n    if n > 0:\n        pi = 2\n    return pi\nprint(f(1), f(0))\n");
    assert_agree("k = 0\nwhile k < 2:\n    k += 1\n    g_n = k * 1 m/s^2\nprint(g_n)\n");
    // A loop that first reads the constant, then rebinds it: the function
    // returns 0, then fails, instead of `pi`.
    assert_agree("def f():\n    while true:\n        y = pi\n        pi = 2\n        break\n    return y\nprint(f())\n");
}

/// `sky run` used to look a called name up when the call ran, lowering bound
/// each call to the definition the checker saw. They differed when the name
/// was defined again after the caller: a second `def`, a `def` that shadows a
/// builtin (`max`), or a rebinding to a number. Early (lexical) binding is
/// the language rule (spec, "Functions"), so the tree-walker now binds a
/// call in a function body to the definition in scope at the `def`.
#[test]
fn lowering_binds_a_call_to_the_definition_seen_at_check_time() {
    assert_agree(
        "def f(x):\n    return x + 1\ndef g(x):\n    return f(x)\nprint(g(1))\ndef f(x):\n    return x + 100\nprint(g(1), f(1))\n",
    );
    assert_agree(
        "def f(x):\n    return max(x, 2)\nprint(f(1))\ndef max(a, b):\n    return a\nprint(f(1))\n",
    );
    assert_agree("def f(x):\n    return x + 1\ndef g(x):\n    return f(x)\nf = 5\nprint(g(1))\n");
}

/// Found by the longer campaign after the fixes above (seed 275714): unary `+`
/// was lowered as its operand, so for a value the checker cannot type the
/// tree-walker's "`+` needs a number, found a Bool" was missing in SIR, which
/// said "cannot add a Bool and a Bool" (or nothing, if the value is only
/// returned). Fixed: an operand of unknown type gets an `expect.number`, like
/// the `expect.bool` of a condition; `inv(x)` of one too, for its wording.
#[test]
fn unary_plus_of_an_unknown_non_number_is_checked_in_sir() {
    assert_agree("def f(a):\n    return +a + a\nprint(f(true))\n");
    assert_agree("def f(a):\n    return +a\nprint(f(true))\n");
    assert_agree("def f(a):\n    return inv(a)\nprint(f(true))\n");
}

/// What `sky run` does with `src`: its output and the error that stopped it.
fn run_in_the_tree_walker(src: &str) -> (String, Option<String>) {
    let lexed = sky_syntax::lex(src).expect("lexes");
    let (ast, errors) = sky_ast::parse_module(&lexed.tokens, src);
    assert!(errors.is_empty(), "{errors:?}");
    let checked = sky_types::check_module(&ast, src);
    assert!(!checked.has_errors(), "{:?}", checked.diagnostics);
    sky_ast::with_deep_stack(|| {
        let mut env = sky_interp::RuntimeEnv::with_prelude();
        let mut out = Vec::new();
        let r = sky_interp::run_module(&ast, src, &mut env, &mut out);
        (
            String::from_utf8_lossy(&out).into_owned(),
            r.err().map(|e| e.message.clone()),
        )
    })
}

/// Design limit, not a bug: lowering refuses the whole program ("`f` is not a
/// function known at compile time") although `sky run` runs it, because `f`
/// is defined in both arms of an `if`: SIR calls name one function, fixed at
/// compile time, and SIR has no function values to pick one at run time.
/// Documented in sky-lower and in the spec ("Functions"); `sky run` falls back
/// to the tree-walker on a lowering error.
#[test]
fn lowering_rejects_a_function_defined_in_both_arms_of_an_if() {
    let src = "x = 1\nif x > 0:\n    def f(a):\n        return a + 1\nelse:\n    def f(a):\n        return a + 2\nprint(f(1))\n";
    assert_eq!(run_in_the_tree_walker(src), ("2\n".to_string(), None));
    let e = sky_lower::lower_source(src).expect_err("a design limit");
    assert!(
        e.contains("`f` is not a function known at compile time"),
        "{e}"
    );
}

/// Design limit, not a bug: a nested `def` reads its enclosing function's
/// local. `sky run` fails only if the inner function runs ("cannot find `a`",
/// its frame is not the enclosing one: there are no closures); lowering fails
/// the whole program, also when it never runs (here `f(0)` returns first),
/// because SIR functions have no access to another function's values.
#[test]
fn lowering_rejects_a_nested_def_that_reads_an_enclosing_local() {
    let src = "def f(a):\n    if a > 1:\n        def h(b):\n            return b + a\n        return h(1)\n    return 0\nprint(f(0))\n";
    assert_eq!(run_in_the_tree_walker(src), ("0\n".to_string(), None));
    let e = sky_lower::lower_source(src).expect_err("a design limit");
    assert!(e.contains("cannot find `a` in this scope"), "{e}");
    // The same program does fail in `sky run` once the inner function runs.
    let src = src.replace("print(f(0))", "print(f(2))");
    let (out, err) = run_in_the_tree_walker(&src);
    assert_eq!(out, "");
    assert_eq!(err.as_deref(), Some("cannot find `a` in this scope"));
}

/// A number literal too large for f64 (also once its unit is applied) is a
/// run-time error in `sky run`, only if evaluated; lowering used to reject the
/// program. It lowers to `literal.error` now, which fails when it runs.
#[test]
fn a_literal_too_large_is_an_error_only_where_it_is_evaluated() {
    assert_agree("print(1)\nif false:\n    x = 1e999\nprint(2)\n");
    assert_agree("print(1)\nif false:\n    x = 1e300 ms/nm^2/μg\nprint(2)\n");
    assert_agree("print(1)\nx = 1e300 ms/nm^2/μg\nprint(2)\n");
    assert_agree("print(1)\nx = 1e999\nprint(2)\n");
    assert_agree("print(1)\nx = 1e999 m\nprint(2)\n");
    assert_agree("def f(a):\n    return a * 1e999\nprint(1)\nprint(f(2))\n");
    // Not evaluated: a function that is never called, a short-circuited operand.
    assert_agree("def f(a):\n    return a * 1e999\nprint(1)\nprint(true or 1e999 > 1)\n");
}

/// A declaration without a value used to be a parameter of `<main>`: the SIR
/// interpreter stopped at entry, before anything was printed, while `sky run`
/// prints up to the declaration and stops there. It is an `input` operation
/// now, which fails where the declaration is.
#[test]
fn an_input_declaration_stops_the_sir_interpreter_before_any_output() {
    assert_agree("print(1)\nx: m\nprint(2)\n");
    // Inside a loop that runs first, and with the declared type in use.
    assert_agree(
        "k = 0\nwhile k < 2:\n    print(k)\n    k += 1\nA: Matrix[2, 2]\nprint(A[0, 0])\n",
    );
}

/// A function without annotations has an unknown result type, so the checker
/// lets `f(1)` stand where a Bool goes. `sky run` checks at run time and stops
/// ("`or` needs a Bool operand, found a number"); the SIR interpreter used to
/// check the left operand only (as an `if` condition) and let a number
/// through as the right operand: `false or f(1)` printed 1. Lowering now puts
/// an `expect.bool` on an operand of unknown type.
#[test]
fn sir_does_not_check_the_right_operand_of_and_or_for_bool() {
    assert_agree("def f(a):\n    return a\nprint(false or f(1))\n");
    assert_agree("def f(a):\n    return a\nprint(true and f(1))\n");
}

/// The same error in the left operand of `and` was worded as the `if` it is
/// lowered to.
#[test]
fn the_left_operand_of_and_is_worded_as_an_if_in_sir() {
    assert_agree("def f(a):\n    return a\nprint(f(1) and true)\n");
}
