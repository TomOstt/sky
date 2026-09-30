//! Golden tests: runs `sky check` on every `tests/ui/**/*.sky` file and
//! compares its stderr, byte for byte, with the `.stderr` file next to it
//! (an empty `.stderr` means the file must check cleanly).
//!
//! The exit status must be 1 if the expected output contains an error and
//! 0 otherwise. Paths are passed relative to the workspace root with `/`
//! separators, so goldens read `tests/ui/lexer/foo.sky:1:1: ...`.
//!
//! When `NAME.explain` exists next to `NAME.sky`, the file is also run with
//! `sky check --explain` and that stderr must equal `NAME.explain`. This
//! pins the Cholesky rewrite decisions end to end; an empty `.explain` pins
//! "no rewrite". A `.types` file belongs to the sky-types runner (see
//! `crates/sky-types/tests/ui_check.rs`) and is ignored here.
//!
//! To regenerate expectations after an intended change, run
//! `SKY_BLESS=1 cargo test -p sky-cli --test ui` and review the diff.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/sky-cli is two levels below the workspace root")
        .to_path_buf()
}

fn collect_sky_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_sky_files(&path, out);
        } else if path.extension().is_some_and(|x| x == "sky") {
            out.push(path);
        } else if path
            .extension()
            .is_some_and(|x| x == "stderr" || x == "explain" || x == "types")
            && !path.with_extension("sky").exists()
        {
            panic!("{} has no matching .sky file", path.display());
        }
    }
}

/// True if any diagnostic header in `stderr` has severity `error`. A header
/// is `<file>:<line>:<col>: <severity>: <message>`, and the file name may
/// itself contain `:`, so look for `:<digits>:<digits>: error:` anywhere.
fn has_error(stderr: &str) -> bool {
    stderr.lines().filter(|l| !l.starts_with(' ')).any(|l| {
        l.match_indices(": error:").any(|(i, _)| {
            let mut parts = l[..i].rsplitn(3, ':');
            let (col, line) = (parts.next(), parts.next());
            let digits = |s: Option<&str>| {
                s.is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
            };
            digits(col) && digits(line) && parts.next().is_some()
        })
    })
}

#[test]
fn has_error_reads_headers_with_colons_in_the_path() {
    assert!(has_error("a.sky:1:2: error: x\n"));
    assert!(has_error("dir:with:colons/a.sky:10:3: error: x\n"));
    assert!(!has_error("a.sky:1:2: warning: x\n"));
    assert!(!has_error("dir: error:/a.sky:1:2: warning: x\n"));
    assert!(!has_error("  = note: a.sky:1:2: error: quoted\n"));
}

#[test]
fn ui() {
    let root = workspace_root();
    let mut files = Vec::new();
    collect_sky_files(&root.join("tests/ui"), &mut files);
    assert!(!files.is_empty(), "no .sky files under tests/ui");

    let bless = std::env::var_os("SKY_BLESS").is_some();
    let mut failures = Vec::new();
    for sky in &files {
        let rel = sky.strip_prefix(&root).expect("under root");
        let rel_arg: Vec<_> = rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect();
        let rel_arg = rel_arg.join("/");
        let run = |explain: bool| {
            let mut cmd = Command::new(env!("CARGO_BIN_EXE_sky"));
            cmd.arg("check");
            if explain {
                cmd.arg("--explain");
            }
            // The parallelism notes speak of the thread count; the goldens
            // must not depend on the machine's cores.
            cmd.arg(&rel_arg)
                .env("SKY_THREADS", "8")
                .current_dir(&root)
                .output()
                .expect("run sky")
        };
        let output = run(false);
        let actual = String::from_utf8(output.stderr).expect("stderr is UTF-8");
        let expected_path = sky.with_extension("stderr");

        let explain_path = sky.with_extension("explain");
        if explain_path.exists() {
            let explained = String::from_utf8(run(true).stderr).expect("stderr is UTF-8");
            if bless {
                fs::write(&explain_path, &explained).expect("write .explain");
            } else {
                let expected = fs::read_to_string(&explain_path).expect("read .explain");
                if explained != expected {
                    failures.push(format!(
                        "{rel_arg}: --explain stderr mismatch\n--- expected\n{expected}--- actual\n{explained}---"
                    ));
                }
            }
        }

        if bless {
            fs::write(&expected_path, &actual).expect("write .stderr");
            continue;
        }
        let Ok(expected) = fs::read_to_string(&expected_path) else {
            failures.push(format!("{rel_arg}: missing {}", expected_path.display()));
            continue;
        };
        if actual != expected {
            failures.push(format!(
                "{rel_arg}: stderr mismatch\n--- expected\n{expected}--- actual\n{actual}---"
            ));
        }
        let want_code = if has_error(&expected) { 1 } else { 0 };
        if output.status.code() != Some(want_code) {
            failures.push(format!(
                "{rel_arg}: exit status {:?}, expected {want_code}",
                output.status.code()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} ui tests failed:\n\n{}",
        failures.len(),
        files.len(),
        failures.join("\n\n")
    );
}
