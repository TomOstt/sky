//! The `contracts` pass (and the dead-code elimination that follows it in
//! the pipeline) on the differential fuzzer's programs: lowered, run before
//! and after the passes (verified after each), the runs must agree on stdout
//! and on the error, its message and where it happened. The whole pipeline
//! on the same programs is `fuzz_pipeline.rs`; this test isolates the
//! contracts pass and counts how often it fires, so a generator change that
//! stops writing contracts, or a pass that stops discharging them, fails
//! here.
//!
//! The generator (`fuzz/harness/differential.rs`) writes contracts too
//! (`assert`, `requires`, `ensures`), so this is also the soundness test of
//! the `contracts` pass on programs nobody chose: a check the pass removes
//! must be one that never fails, and one it keeps must fail with the same
//! message at the same span.

#[path = "../../../fuzz/harness/differential.rs"]
mod harness;

use sky_ir::PassManager;
use sky_opt::{ContractsPass, DcePass};

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

type Run = (String, Option<(String, u32, u32)>);

fn run(m: &sky_ir::Module) -> Run {
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let r = sky_ir_interp::run(m, &mut out);
        let error = r.err().map(|e| {
            // The two resource limits are one outcome (see the fuzzer).
            if e.message.starts_with("recursion is too deep")
                || e.message == "the program nests too deeply for the interpreter"
            {
                ("<resource limit>".to_string(), 0, 0)
            } else {
                (e.message.clone(), e.span.start, e.span.end)
            }
        });
        (String::from_utf8_lossy(&out).into_owned(), error)
    })
}

#[test]
fn the_contracts_pass_keeps_the_behaviour_of_generated_programs() {
    let count: u64 = std::env::var("SKY_FUZZ_SEEDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(800);
    let first: u64 = std::env::var("SKY_FUZZ_FROM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let (mut contracts, mut dropped) = (0u64, 0u64);
    for seed in first..first + count {
        let len = 150 + (seed.wrapping_mul(2_654_435_761) % 1400) as usize;
        let src = harness::generate(&bytes(seed, len));
        // Programs that do not lower (nested defs and the like) are the
        // fuzzer's business, not this test's.
        let Ok(mut m) = sky_lower::lower_source(&src) else {
            continue;
        };
        if std::env::var_os("SKY_FUZZ_VERBOSE").is_some() {
            eprintln!("seed {seed}");
        }
        let asserts_before = sky_ir::print(&m).matches("assert[").count();
        if asserts_before == 0 {
            continue;
        }
        let before = run(&m);
        let report = PassManager::for_tests()
            .with(ContractsPass)
            .with(DcePass)
            .run(&mut m)
            .unwrap_or_else(|e| panic!("seed {seed}: {e}\n{src}"));
        let after = run(&m);
        assert_eq!(
            after, before,
            "seed {seed}: the passes changed the behaviour of:\n{src}"
        );
        contracts += 1;
        dropped += report
            .explanations
            .iter()
            .filter(|x| x.rule.starts_with("CONTRACT-"))
            .count() as u64;
    }
    if count >= 800 {
        assert!(contracts > 20, "only {contracts} programs had contracts");
        assert!(dropped > 3, "only {dropped} checks were ever discharged");
    }
}
