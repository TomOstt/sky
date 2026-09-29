//! The default pipeline (`fold`, `egraph`, `dce`, `fuse`, `cholesky`) must not
//! change what a program does: lower it, run it in the reference
//! interpreter, run the pipeline (verified after every pass), run it again,
//! and the two runs must agree byte for byte on stdout and on the error (its
//! message and where it happened).
//!
//! Programs come from the repository's UI suites (every `ok_*.sky` that
//! lowers; to make them observable, each top-level `name = ...` is printed at
//! the end) and from the inline programs below, which are chosen so that
//! every pass has something to do, including failures the passes must not
//! remove or move.
//!
//! It also pins the adapters: each pass, run through the pass manager,
//! reports exactly the rules and texts its function form returns.

use std::path::{Path, PathBuf};

use sky_ir::{Explanation, Module, PassManager};
use sky_opt::{default_passes, default_pipeline, DEFAULT_PASSES};

/// What running a program did.
#[derive(Clone, Debug, PartialEq)]
struct Outcome {
    stdout: String,
    /// The message and byte range of the runtime error, if the run failed.
    error: Option<(String, u32, u32)>,
}

fn run(m: &Module) -> Outcome {
    sky_ast::with_deep_stack(|| {
        let mut out = Vec::new();
        let result = sky_ir_interp::run(m, &mut out);
        Outcome {
            stdout: String::from_utf8(out).expect("UTF-8 output"),
            error: result
                .err()
                .map(|e| (e.message.clone(), e.span.start, e.span.end)),
        }
    })
}

/// The pipeline, verifying after every pass and panicking (naming the pass)
/// if it breaks the IR.
fn optimize(m: &mut Module) -> sky_ir::PipelineReport {
    default_passes(PassManager::for_tests())
        .run(m)
        .expect("valid IR")
}

/// Lowers `src`, runs it before and after the pipeline, and asserts the
/// runs agree. Returns the shared outcome and the report.
fn same_before_and_after(src: &str) -> (Outcome, sky_ir::PipelineReport) {
    let mut m = sky_lower::lower_source(src).unwrap_or_else(|e| panic!("{src}\n{e}"));
    let before = run(&m);
    let report = optimize(&mut m);
    let after = run(&m);
    assert_eq!(
        after, before,
        "the pipeline changed the behaviour of:\n{src}"
    );
    (before, report)
}

// ----- every runnable UI program -----

fn ui_programs() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut out = Vec::new();
    for dir in [
        "tests/ui/check",
        "tests/ui/types",
        "crates/sky-types/tests/ui",
    ] {
        let mut files: Vec<PathBuf> = std::fs::read_dir(root.join(dir))
            .unwrap_or_else(|e| panic!("{dir}: {e}"))
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                n.starts_with("ok_") && n.ends_with(".sky")
            })
            .collect();
        files.sort();
        out.extend(files);
    }
    out
}

/// `src` with a `print(name)` for every top-level `name = ...` appended,
/// when that still lowers, so the run shows the values the program computed.
fn observable(src: &str) -> String {
    let names: Vec<&str> = src
        .lines()
        .filter_map(|l| {
            let (name, _) = l.split_once(" = ")?;
            let plain = !name.is_empty()
                && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                && !name.starts_with(|c: char| c.is_ascii_digit());
            plain.then_some(name)
        })
        .collect();
    let mut extended = src.to_string();
    if !extended.ends_with('\n') {
        extended.push('\n');
    }
    for n in names {
        extended.push_str(&format!("print({n})\n"));
    }
    if sky_lower::lower_source(&extended).is_ok() {
        extended
    } else {
        src.to_string()
    }
}

#[test]
fn every_runnable_ui_program_behaves_the_same_through_the_pipeline() {
    let files = ui_programs();
    assert!(files.len() >= 30, "found {} programs", files.len());
    let (mut ran, mut printed, mut changed) = (0, 0, 0);
    for path in &files {
        let src = std::fs::read_to_string(path).expect("reads");
        // Programs SIR cannot represent yet (arrays) have nothing to run.
        if sky_lower::lower_source(&src).is_err() {
            continue;
        }
        let src = observable(&src);
        let (outcome, report) = same_before_and_after(&src);
        ran += 1;
        printed += usize::from(!outcome.stdout.is_empty());
        changed += usize::from(report.ran.iter().any(|(_, c)| c.is_yes()));
    }
    assert!(ran >= 20, "only {ran} programs lowered");
    assert!(printed >= 5, "only {printed} programs printed anything");
    assert!(changed >= 1, "the pipeline never changed a program");
}

// ----- inline programs where the passes have work -----

#[test]
fn constants_fold_without_changing_a_printed_digit() {
    let (o, report) = same_before_and_after(
        "a = 0.1 + 0.2\nprint(a)\nprint(1/3 + 1/6, 2 ^ 10, 7 / 2)\n\
         v = 36 km / 1 h\nprint(v * 1, 2 m + 3 cm)\n",
    );
    assert_eq!(
        o.stdout,
        "0.30000000000000004\n0.5 1024 3.5\n10 m/s 2.03 m\n"
    );
    assert_eq!(o.error, None);
    assert!(report.explanations.iter().any(|x| x.pass == "fold"));
}

#[test]
fn folding_and_dce_keep_the_failure_where_it_was() {
    // The division fails at run time, is never used, and must still fail,
    // after the output before it and before the output after it.
    let (o, _) = same_before_and_after("print(1)\nunused = 1 / (2 - 2)\nprint(2)\n");
    assert_eq!(o.stdout, "1\n");
    assert_eq!(o.error.map(|e| e.0).as_deref(), Some("division by zero"));
    let (o, _) = same_before_and_after("x = 10 ^ 400\nprint(x)\n");
    assert_eq!(o.error.map(|e| e.0).as_deref(), Some("arithmetic overflow"));
    let (o, _) = same_before_and_after("print(0 - 8 ^ (1 / 3))\nprint((0 - 8) ^ (1 / 3))\n");
    assert_eq!(
        o.error.map(|e| e.0).as_deref(),
        Some("the result is not a real number")
    );
}

#[test]
fn dead_code_is_removed_and_explained_but_effects_stay() {
    let (o, report) = same_before_and_after(
        "kept = 5\nunused = kept * 4\ndef loud(n):\n    print(n)\n    return n\n\
         ignored = loud(7)\nprint(kept)\n",
    );
    assert_eq!(o.stdout, "7\n5\n");
    let dce: Vec<&Explanation> = report
        .explanations
        .iter()
        .filter(|x| x.pass == "dce")
        .collect();
    assert!(
        dce.iter().any(|x| x.because[0].contains("`unused`")),
        "{dce:#?}"
    );
    // The call to `loud` stays: it prints.
    assert!(dce.iter().all(|x| !x.before.contains("call")), "{dce:#?}");
}

#[test]
fn identities_rewrite_under_the_reproducible_contract_only() {
    // `x: m` is an input, so the run fails the same way before and after;
    // the point is the rules that fire (and that only exact ones do).
    let src = "x: m\nt: s\ny = x * 1\nz = -(-t)\nw = x + 0 m\nprint(y, z, w)\n";
    let (o, report) = same_before_and_after(src);
    assert!(o.error.is_some(), "inputs cannot be run");
    let rules: Vec<&str> = report.explanations.iter().map(|x| x.rule).collect();
    assert!(rules.contains(&"ALG-001"), "{rules:?}");
    assert!(rules.contains(&"ALG-007"), "{rules:?}");
    // Rules valid only over the reals (`@fast`) never fire by default.
    for x in &report.explanations {
        let table = sky_egraph::rule_table();
        if let Some(info) = table.iter().find(|r| r.id == x.rule) {
            assert_ne!(info.class, sky_egraph::RuleClass::RealOnly, "{}", x.rule);
        }
    }
}

#[test]
fn loops_functions_recursion_and_branches_are_preserved() {
    let (o, _) = same_before_and_after(
        "def fact(n):\n    if n <= 1:\n        return 1\n    return n * fact(n - 1)\n\
         total = 0\ni = 1\nwhile i <= 10:\n    total += i\n    i += 1\nprint(total, fact(5))\n\
         if 1 < 2:\n    print(\"yes\")\nelse:\n    print(\"no\")\n\
         k = 3 * 4\nif k > 10:\n    j = k * 1\nelse:\n    j = 0\nprint(j)\n",
    );
    assert_eq!(o.stdout, "55 120\nyes\n12\n");
    let (o, _) = same_before_and_after("def f(x):\n    return f(x)\nf(1)\n");
    assert!(o.error.is_some());
}

#[test]
fn math_functions_and_their_domain_errors_are_preserved() {
    let (o, _) = same_before_and_after(
        "x = 2 m * exp(-1 s / 2 s) * cos(2 * pi * 0.25 Hz * 1 s)\n\
         print(sqrt(9 m^2), hypot(3 m, 4 m), max(1 s, 2 s), round(2.5), x * 1)\n",
    );
    assert_eq!(o.error, None);
    for bad in ["log(0)", "sqrt(0 - 1)", "asin(2)", "exp(1000)"] {
        let (o, _) = same_before_and_after(&format!("y = {bad}\nprint(y)\n"));
        assert!(o.error.is_some(), "{bad}");
    }
}

// ----- kernel fusion -----

#[test]
fn elementwise_array_chains_fuse_and_are_explained() {
    let (o, report) = same_before_and_after(
        "a = [1, 2, 3]\nb = [4, 5, 6]\nd = [1, 1, 1]\nx = (a + b) * 2 - d\nprint(x)\n",
    );
    assert_eq!(o.stdout, "[9, 13, 17]\n");
    let fuse: Vec<&Explanation> = report
        .explanations
        .iter()
        .filter(|x| x.pass == "fuse")
        .collect();
    assert_eq!(fuse.len(), 1, "{fuse:#?}");
    assert_eq!(fuse[0].rule, "FUSE-001");
    assert_eq!(fuse[0].before, "(a + b) * 2 - d");
    assert_eq!(
        fuse[0].because[0],
        "fused 3 elementwise operations into one kernel; saves 2 temporary arrays of 3 elements"
    );
    // `fuse` is where the module changed to have a kernel.
    let fuse_ran = report.ran.iter().find(|(n, _)| *n == "fuse").unwrap();
    assert!(fuse_ran.1.is_yes());
}

#[test]
fn a_fused_kernel_fails_with_the_error_of_the_first_failing_operation() {
    // `big * big` overflows at the last element, `small * big` at the
    // first: the separate operations fail in the first one, and so does the
    // kernel.
    let src = "big = [1, 1, 1e200]\nsmall = [1e200, 1, 1]\nk = 1e200\n\
               y = big * k + small * k\nprint(y)\n";
    let (o, report) = same_before_and_after(src);
    let (message, start, end) = o.error.expect("overflows");
    assert_eq!(message, "arithmetic overflow");
    assert_eq!(&src[start as usize..end as usize], "big * k");
    assert!(report.explanations.iter().any(|x| x.rule == "FUSE-001"));
}

// ----- the pipeline and its adapters -----

#[test]
fn the_default_pipeline_runs_the_five_passes_in_order() {
    let mut m = sky_lower::lower_source("x = 2 * 3\nprint(x)\n").unwrap();
    let report = default_pipeline().verify_each(true).run(&mut m).unwrap();
    let names: Vec<&str> = report.ran.iter().map(|(n, _)| *n).collect();
    assert_eq!(names, ["fold", "egraph", "dce", "fuse", "cholesky"]);
    assert_eq!(names, DEFAULT_PASSES);
    // fold changed the module (`2 * 3`), dce removed the operands.
    assert!(report.ran[0].1.is_yes());
    assert!(!report.ran[1].1.is_yes());
    assert!(report.ran[2].1.is_yes());
    assert!(!report.ran[3].1.is_yes());
    assert!(!report.ran[4].1.is_yes());
    // A second run finds nothing left to do.
    let again = default_pipeline().verify_each(true).run(&mut m).unwrap();
    assert!(again.ran.iter().all(|(_, c)| !c.is_yes()), "{again:?}");
    assert!(again.explanations.is_empty());
}

#[test]
fn print_after_captures_each_pass() {
    let mut m = sky_lower::lower_source("x = 2 * 3\nprint(x)\n").unwrap();
    let report = default_pipeline()
        .verify_each(true)
        .print_after("fold")
        .run(&mut m)
        .unwrap();
    assert_eq!(report.printed.len(), 1);
    assert_eq!(report.printed[0].0, "fold");
    assert!(report.printed[0].1.contains("const.float 6"));
}

/// Explanations as a comparable tuple.
fn key(
    x: &Explanation,
) -> (
    &'static str,
    &'static str,
    u32,
    u32,
    String,
    String,
    Vec<String>,
) {
    (
        x.pass,
        x.rule,
        x.span.start,
        x.span.end,
        x.before.clone(),
        x.after.clone(),
        x.because.clone(),
    )
}

/// The inverse reaches `@` through a name (`Ai = A⁻¹`): a direct `A⁻¹ @ b`
/// is a site the checker proves, which lowering already emits as `solve`,
/// leaving the Cholesky pass nothing to do.
const ADAPTER_SRC: &str = "x: m\nA: PositiveDefinite Matrix[3, 3]\nb: Vector[3]\n\
    a = 2 * 3\nunused = a * 4\ny = x * 1\nAi = A⁻¹\ns = Ai @ b\nprint(a, y, s)\n";

#[test]
fn the_pass_adapters_report_the_records_of_the_function_forms() {
    let lowered = sky_lower::lower_source(ADAPTER_SRC).unwrap();

    // fold
    let mut direct = lowered.clone();
    let expected = sky_opt::fold(&mut direct);
    let mut via = lowered.clone();
    let report = PassManager::for_tests()
        .with(sky_opt::FoldPass::default())
        .run(&mut via)
        .unwrap();
    assert_eq!(via, direct);
    assert!(!expected.is_empty());
    let got: Vec<_> = report.explanations.iter().map(key).collect();
    let want: Vec<_> = expected
        .iter()
        .map(|r| {
            (
                r.pass,
                r.rule,
                r.span.start,
                r.span.end,
                r.before.clone(),
                r.after.clone(),
                r.because.clone(),
            )
        })
        .collect();
    assert_eq!(got, want);

    // egraph
    let mut direct = lowered.clone();
    let cfg = sky_egraph::RewriteConfig::default();
    let expected = sky_egraph::rewrite_module(&mut direct, &cfg);
    let mut via = lowered.clone();
    let report = PassManager::for_tests()
        .with(sky_egraph::EgraphPass::default())
        .run(&mut via)
        .unwrap();
    assert_eq!(via, direct);
    assert_eq!(expected.len(), 1, "{expected:?}");
    let got: Vec<_> = report.explanations.iter().map(key).collect();
    let want: Vec<_> = expected
        .iter()
        .map(|r| {
            (
                r.pass,
                r.rule,
                r.span.start,
                r.span.end,
                r.before.clone(),
                r.after.clone(),
                vec![r.because.clone()],
            )
        })
        .collect();
    assert_eq!(got, want);
    assert_eq!(got[0].1, "ALG-001");
    assert_eq!(got[0].0, "egraph");

    // Dead code only exists once `fold` has turned `a * 4` into a constant.
    let mut folded = lowered.clone();
    let _ = sky_opt::fold(&mut folded);
    adapter_matches("dce", sky_opt::DcePass, sky_opt::dce, &folded);
    adapter_matches(
        "cholesky",
        sky_opt::CholeskyPass,
        sky_opt::cholesky,
        &lowered,
    );
}

/// `pass`, run through the manager, changes the module like `direct` and
/// explains each of the records `direct` returns.
fn adapter_matches<P: sky_ir::Pass + 'static>(
    name: &str,
    pass: P,
    direct: fn(&mut Module) -> Vec<sky_opt::Rewrite>,
    lowered: &Module,
) {
    let mut by_function = lowered.clone();
    let expected = direct(&mut by_function);
    assert!(!expected.is_empty(), "{name}");
    let mut by_pass = lowered.clone();
    let report = PassManager::for_tests()
        .with(pass)
        .run(&mut by_pass)
        .unwrap();
    assert_eq!(by_pass, by_function, "{name}");
    assert_eq!(report.explanations.len(), expected.len(), "{name}");
    for (x, r) in report.explanations.iter().zip(&expected) {
        assert_eq!(
            (x.pass, x.rule, x.span, &x.before, &x.after, &x.because),
            (r.pass, r.rule, r.span, &r.before, &r.after, &r.because),
            "{name}"
        );
    }
}

#[test]
fn the_cholesky_explanation_joins_into_the_checkers_sentence() {
    let mut m = sky_lower::lower_source(ADAPTER_SRC).unwrap();
    let report = PassManager::for_tests()
        .with(sky_opt::CholeskyPass)
        .run(&mut m)
        .unwrap();
    let x = &report.explanations[0];
    assert_eq!((x.pass, x.rule), ("cholesky", "LA-CHOLESKY-001"));
    assert_eq!(
        sky_opt::cholesky_text(x).unwrap(),
        "rewrite `A⁻¹ @ b` to a Cholesky solve: A is Matrix[3,3] and PositiveDefinite, \
         b is Vector[3]. Factor A = L·Lᵀ (L lower-triangular), solve L·y = b, \
         then Lᵀ·x = y. x = A⁻¹·b is Vector[3], and A⁻¹ is never formed."
    );
}
