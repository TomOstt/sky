//! The default pipeline on the fuzzer's programs: for every program the
//! `differential` fuzz harness generates (`fuzz/harness/differential.rs`, the
//! generator only), the tree-walking interpreter, the lowered SIR and the
//! lowered SIR after the default pipeline (`specialize`, `fold`, `egraph`, `gvn`,
//! `licm`, `contracts`, `dce`, `fuse`, `cholesky`, ...) must give
//! the same stdout and the same error, compared as the whole rendered
//! diagnostic: message, spans, labels, secondary labels and notes. That is
//! what pins the passes' handling of operand spans (`OpData::operand_spans`):
//! a rewrite that lost or misplaced one changes a label.
//!
//! `SKY_FUZZ_FROM` and `SKY_FUZZ_SEEDS` run a longer campaign (use
//! `--release`), as in sky-ir-interp's `fuzz_differential`.

#[path = "../../../fuzz/harness/differential.rs"]
mod harness;

use sky_diag::{Diagnostic, SourceFile};
use sky_ir::PassManager;
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

fn check(src: &str) -> Result<(), String> {
    let lexed = sky_syntax::lex(src).map_err(|_| "the lexer rejects the program".to_string())?;
    let (ast, errors) = sky_ast::parse_module(&lexed.tokens, src);
    if !errors.is_empty() {
        return Err(format!("parse errors: {errors:?}"));
    }
    let checked = sky_types::check_module(&ast, src);
    if checked.has_errors() {
        return Err("sky check rejects the program".to_string());
    }
    let mut ir = sky_lower::lower(&ast, src, &checked).map_err(|e| format!("lowering: {e}"))?;

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
    let lowered = run_sir(&ir);
    default_passes(PassManager::new().verify_each(true))
        .run(&mut ir)
        .map_err(|e| format!("a pass broke the IR: {e}"))?;
    let optimized = run_sir(&ir);

    let want = normalized(walker);
    for (what, got) in [("lowered", lowered), ("optimized", optimized)] {
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

#[test]
fn the_pipeline_keeps_the_whole_diagnostic_on_generated_programs() {
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
                        let len = 150 + (seed.wrapping_mul(2_654_435_761) % 1400) as usize;
                        let src = harness::generate(&bytes(seed, len));
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
            .flat_map(|h| h.join().expect("a lane panicked"))
            .collect()
    });
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

/// A tool for a fuzzing session: `SKY_FILE=x.sky cargo test -p sky-opt --test
/// fuzz_pipeline replay -- --nocapture` lowers a program, runs the default
/// passes one at a time and prints the SIR after each (and what the verifier
/// says about it). Does nothing unless `SKY_FILE` is set.
#[test]
fn replay_a_source_file() {
    let Ok(path) = std::env::var("SKY_FILE") else {
        return;
    };
    let src = std::fs::read_to_string(&path).expect("SKY_FILE is readable");
    sky_ast::with_deep_stack(|| {
        let mut ir = sky_lower::lower_source(&src).expect("lowers");
        println!("--- lowered ---\n{}", sky_ir::print(&ir));
        for pass in sky_opt::DEFAULT_PASSES {
            let one = match pass {
                "specialize" => PassManager::new().with(sky_opt::SpecializePass),
                "fold" => PassManager::new().with(sky_opt::FoldPass::default()),
                "egraph" => PassManager::new().with(sky_egraph::EgraphPass::default()),
                "gvn" => PassManager::new().with(sky_opt::GvnPass),
                "licm" => PassManager::new().with(sky_opt::LicmPass),
                "contracts" => PassManager::new().with(sky_opt::ContractsPass),
                "dce" => PassManager::new().with(sky_opt::DcePass),
                "powers" => PassManager::new().with(sky_opt::PowersPass),
                "fuse" => PassManager::new().with(sky_opt::FusePass),
                "transposes" => PassManager::new().with(sky_opt::TransposesPass),
                "structure" => PassManager::new().with(sky_opt::StructurePass),
                "algorithms" => PassManager::new().with(sky_opt::AlgorithmPass::default()),
                _ => PassManager::new().with(sky_opt::CholeskyPass),
            };
            let _ = one.verify_each(false).run(&mut ir);
            println!("--- after {pass} ---\n{}", sky_ir::print(&ir));
            if let Err(errors) = sky_ir::verify(&ir) {
                println!("INVALID: {errors:#?}");
                break;
            }
        }
    });
}
