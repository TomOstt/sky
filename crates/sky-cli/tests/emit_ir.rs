//! `sky check --emit-ir[=<stage>]` and the pass half of `--explain`.
//!
//! Golden files live in `tests/ir/`, next to `NAME.sky`:
//!
//! * `NAME.sir`: stdout of `sky check --emit-ir NAME.sky` (empty when the
//!   program has no IR);
//! * `NAME.lower.sir`, if present: stdout of `--emit-ir=lower`;
//! * `NAME.stderr`, if present: stderr of `--emit-ir` (the note saying why
//!   there is no IR, warnings, and the `ok` line);
//! * `NAME.explain`, if present: stderr of `--explain`.
//!
//! Regenerate with `SKY_BLESS=1 cargo test -p sky-cli --test emit_ir` and
//! review the diff. Every non-empty `.sir` must also parse, verify and
//! reprint identically, because it is the text format a tool would read.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/sky-cli is two levels below the workspace root")
        .to_path_buf()
}

fn sky(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sky"))
        .arg("check")
        .args(args)
        .current_dir(root())
        .output()
        .expect("run sky")
}

fn out(o: &Output) -> String {
    String::from_utf8(o.stdout.clone()).expect("stdout is UTF-8")
}

fn err(o: &Output) -> String {
    String::from_utf8(o.stderr.clone()).expect("stderr is UTF-8")
}

fn sky_files() -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(root().join("tests/ir"))
        .expect("tests/ir exists")
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "sky"))
        .collect();
    files.sort();
    files
}

/// `tests/ir/NAME.sky` as the argument the goldens were made with.
fn rel(path: &Path) -> String {
    format!(
        "tests/ir/{}",
        path.file_name().and_then(|n| n.to_str()).expect("name")
    )
}

/// Compares `actual` with the golden `path`, or writes it when blessing.
fn expect_golden(path: &Path, actual: &str, what: &str, failures: &mut Vec<String>) {
    if std::env::var_os("SKY_BLESS").is_some() {
        fs::write(path, actual).expect("bless");
        return;
    }
    match fs::read_to_string(path) {
        Ok(expected) if expected == actual => {}
        Ok(expected) => failures.push(format!(
            "{what}: differs from {}\n--- expected\n{expected}--- actual\n{actual}---",
            path.display()
        )),
        Err(_) => failures.push(format!("{what}: missing {}", path.display())),
    }
}

#[test]
fn goldens() {
    let bless = std::env::var_os("SKY_BLESS").is_some();
    let files = sky_files();
    assert!(files.len() >= 3, "{files:?}");
    let mut failures = Vec::new();
    for sky_file in &files {
        let arg = rel(sky_file);
        let emitted = sky(&["--emit-ir", &arg]);
        assert_eq!(emitted.status.code(), Some(0), "{arg}: {}", err(&emitted));
        let sir = out(&emitted);
        expect_golden(
            &sky_file.with_extension("sir"),
            &sir,
            &format!("{arg} --emit-ir"),
            &mut failures,
        );
        if !sir.is_empty() {
            let m = sky_ir::parse(&sir).unwrap_or_else(|e| panic!("{arg}: {e}\n{sir}"));
            sky_ir::verify(&m).unwrap_or_else(|e| panic!("{arg}: {e:?}\n{sir}"));
            assert_eq!(sky_ir::print(&m), sir, "{arg}: the text does not reprint");
        }
        let lower = sky_file.with_extension("lower.sir");
        if lower.exists() || (bless && !sir.is_empty()) {
            let o = sky(&["--emit-ir=lower", &arg]);
            expect_golden(
                &lower,
                &out(&o),
                &format!("{arg} --emit-ir=lower"),
                &mut failures,
            );
        }
        let stderr = sky_file.with_extension("stderr");
        if stderr.exists() || (bless && sir.is_empty()) {
            expect_golden(
                &stderr,
                &err(&emitted),
                &format!("{arg} stderr"),
                &mut failures,
            );
        }
        let explain = sky_file.with_extension("explain");
        if explain.exists() || bless {
            let o = sky(&["--explain", &arg]);
            assert_eq!(o.status.code(), Some(0), "{arg}");
            expect_golden(
                &explain,
                &err(&o),
                &format!("{arg} --explain"),
                &mut failures,
            );
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn stdout_is_only_the_ir() {
    // `edition:` and `ok` go to stderr, so stdout can be a file.
    let o = sky(&["--emit-ir", "--explain", "tests/ir/pipeline.sky"]);
    assert_eq!(o.status.code(), Some(0));
    let text = out(&o);
    assert!(text.starts_with("sir edition 2026\n"), "{text}");
    sky_ir::parse(&text).expect("stdout parses as SIR");
    let stderr = err(&o);
    assert!(stderr.contains("edition: 2026"), "{stderr}");
    assert!(stderr.ends_with("tests/ir/pipeline.sky: ok\n"), "{stderr}");
    // Without `--emit-ir`, they stay on stdout as before.
    let o = sky(&["--explain", "tests/ir/pipeline.sky"]);
    assert_eq!(
        out(&o),
        "edition: 2026 (no sky.toml: the latest)\ntests/ir/pipeline.sky: ok\n"
    );
}

#[test]
fn the_stages_show_each_pass() {
    let file = "tests/ir/pipeline.sky";
    let text = |stage: &str| {
        let o = sky(&[&format!("--emit-ir={stage}"), file]);
        assert_eq!(o.status.code(), Some(0), "{stage}: {}", err(&o));
        out(&o)
    };
    let (lowered, fold, egraph, dce, cholesky) = (
        text("lower"),
        text("after:fold"),
        text("after:egraph"),
        text("after:dce"),
        text("after:cholesky"),
    );
    let final_ir = out(&sky(&["--emit-ir", file]));
    assert_eq!(final_ir, cholesky, "the default stage is the end");
    assert_eq!(text("final"), final_ir);
    // A bare pass name is the same as `after:<pass>`.
    assert_eq!(text("dce"), dce);

    let count = |ir: &str, mnemonic: &str| ir.matches(&format!(" {mnemonic} ")).count();
    // fold: `2 * 3` and `6 * 4` are constants (the dead ones stay for dce);
    // `x * 1` and `-(-t)` are for the e-graph.
    assert_eq!(count(&lowered, "mul"), 3, "{lowered}");
    assert_eq!(count(&fold, "mul"), 1, "{fold}");
    assert_eq!(count(&fold, "neg"), 2, "{fold}");
    assert!(fold.contains("const.float 24"), "{fold}");
    // egraph: both are rewritten away, and the function's uses follow.
    assert_eq!(count(&egraph, "mul") + count(&egraph, "neg"), 0, "{egraph}");
    assert!(egraph.contains("print %a, %x, %t"), "{egraph}");
    // dce: nothing dead is left, the inverse is still there.
    assert_eq!(count(&dce, "mul") + count(&dce, "neg"), 0, "{dce}");
    assert!(!dce.contains("const.float 24"), "{dce}");
    assert_eq!(count(&dce, "inverse"), 1, "{dce}");
    assert_eq!(count(&dce, "solve[cholesky]"), 0, "{dce}");
    // cholesky: the last pass.
    assert_eq!(count(&cholesky, "inverse"), 0, "{cholesky}");
    assert!(cholesky.contains("solve[cholesky] %A, %b"), "{cholesky}");
    // Every stage is valid, parseable SIR.
    for ir in [&lowered, &fold, &egraph, &dce, &cholesky] {
        let m = sky_ir::parse(ir).expect("parses");
        sky_ir::verify(&m).expect("verifies");
    }
}

#[test]
fn flags_may_come_before_or_after_the_file() {
    let a = sky(&["--emit-ir", "--explain", "tests/ir/pipeline.sky"]);
    let b = sky(&["tests/ir/pipeline.sky", "--explain", "--emit-ir"]);
    assert_eq!((out(&a), err(&a)), (out(&b), err(&b)));
}

#[test]
fn a_bad_stage_or_repeated_flag_is_a_usage_error() {
    for args in [
        vec!["--emit-ir=nonsense", "tests/ir/pipeline.sky"],
        vec!["--emit-ir=", "tests/ir/pipeline.sky"],
        vec!["--emit-ir=after:", "tests/ir/pipeline.sky"],
        vec!["--emit-ir=after:lower", "tests/ir/pipeline.sky"],
        vec!["--emit-ir", "--emit-ir=lower", "tests/ir/pipeline.sky"],
        vec!["--emit-ir=\u{1b}[31m", "tests/ir/pipeline.sky"],
    ] {
        let o = sky(&args);
        assert_eq!(o.status.code(), Some(2), "{args:?}");
        assert_eq!(out(&o), "", "{args:?}");
        let stderr = err(&o);
        assert!(stderr.starts_with("usage: sky check"), "{args:?}: {stderr}");
        assert!(!stderr.contains('\u{1b}'), "{args:?}: {stderr:?}");
    }
    let o = sky(&["--emit-ir=nonsense", "tests/ir/pipeline.sky"]);
    assert!(
        err(&o).contains("unknown IR stage `nonsense`"),
        "{}",
        err(&o)
    );
}

#[test]
fn a_program_without_ir_still_checks() {
    let file = "tests/ir/no_ir_array_literal.sky";
    let o = sky(&["--emit-ir", file]);
    assert_eq!(o.status.code(), Some(0));
    assert_eq!(out(&o), "", "no IR, so nothing on stdout");
    let stderr = err(&o);
    assert!(
        stderr.contains("note: no IR: array literals are not supported in SIR yet"),
        "{stderr}"
    );
    assert!(stderr.ends_with(&format!("{file}: ok\n")), "{stderr}");
    // Every stage says the same, and `--explain` alone is unchanged: the
    // checker's Cholesky note, and no mention of IR.
    for stage in ["lower", "after:fold", "final"] {
        let o = sky(&[&format!("--emit-ir={stage}"), file]);
        assert_eq!((o.status.code(), out(&o)), (Some(0), String::new()));
    }
    let o = sky(&["--explain", file]);
    assert_eq!(o.status.code(), Some(0));
    assert!(!err(&o).contains("IR"), "{}", err(&o));
    assert!(err(&o).contains("rewritten to a Cholesky solve"));
}

#[test]
fn a_program_with_errors_has_no_ir_and_fails_as_before() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"));
    let path = dir.join("emit_ir_error.sky");
    fs::write(&path, "x = 1 m + 2 s\n").expect("write");
    let path = path.to_str().expect("UTF-8 temp dir");
    let o = sky(&["--emit-ir", path]);
    assert_eq!(o.status.code(), Some(1));
    assert_eq!(out(&o), "");
    assert!(err(&o).contains("error:"), "{}", err(&o));
    assert!(!err(&o).contains("no IR"), "{}", err(&o));
}

#[test]
fn the_cholesky_note_is_shown_once_and_in_the_checkers_words() {
    let file = "tests/ir/pipeline.sky";
    let explained = err(&sky(&["--explain", file]));
    assert_eq!(
        explained.matches("rewritten to a Cholesky solve").count(),
        1,
        "{explained}"
    );
    // The checker's note, not the pass's, for a site both see.
    assert!(
        explained.contains("  |       - positive definite\n"),
        "{explained}"
    );
    assert!(!explained.contains("LA-CHOLESKY-001"), "{explained}");
    // One only the IR sees is reported by the pass, worded the same.
    let through = err(&sky(&["--explain", "tests/ir/cholesky_through_a_name.sky"]));
    assert_eq!(
        through.matches("rewritten to a Cholesky solve").count(),
        1,
        "{through}"
    );
    assert!(
        through.contains("applied by the `cholesky` pass"),
        "{through}"
    );
}

#[test]
fn explanations_come_in_source_order_with_the_checkers_notes() {
    let explained = err(&sky(&["--explain", "tests/ir/pipeline.sky"]));
    let lines: Vec<usize> = explained
        .lines()
        .filter(|l| l.contains(": note: "))
        .map(|l| {
            let mut parts = l.split(':').skip(1);
            parts.next().expect("line").parse().expect("a number")
        })
        .collect();
    assert!(lines.len() >= 5, "{explained}");
    assert!(lines.windows(2).all(|w| w[0] <= w[1]), "{lines:?}");
    for id in ["FOLD-FLOAT-001", "DCE-001", "ALG-001", "ALG-007"] {
        assert!(explained.contains(id), "{id} missing:\n{explained}");
    }
}
