//! `sky check` runs the parser: stdout and exit status for the goldens in
//! `tests/ui/check/` (the ui harness compares only stderr), and nesting far
//! past the parser's depth limit ends in a clean error, not a stack overflow.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

fn sky_check(rel: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sky"))
        .arg("check")
        .arg(rel)
        .current_dir(root())
        .output()
        .expect("run sky")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

#[test]
fn ok_only_when_there_are_no_errors() {
    for f in [
        "ok_parses",
        "ok_depth_256",
        "ok_unit_mismatch_not_checked_yet",
        "ok_bool_literals",
    ] {
        let rel = format!("tests/ui/check/{f}.sky");
        let out = sky_check(&rel);
        assert_eq!(out.status.code(), Some(0), "{rel}");
        assert_eq!(stdout(&out), format!("{rel}: ok\n"));
    }
    for f in [
        "err_superscript_inverse_after_unit",
        "err_depth_257",
        "err_lexer_error_before_parse",
        "err_unit_after_bool",
        "err_while_not_supported_yet",
    ] {
        let rel = format!("tests/ui/check/{f}.sky");
        let out = sky_check(&rel);
        assert_eq!(out.status.code(), Some(1), "{rel}");
        assert_eq!(stdout(&out), "", "{rel}");
    }
}

#[test]
fn very_deep_nesting_is_a_clean_error() {
    // 100 000 levels: far past MAX_DEPTH (256). The parser stops at the
    // limit on the deep stack, so this is one diagnostic and exit 1, not a
    // crash (a signal would leave no exit code).
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("check-deep-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join(".git")).unwrap();
    let n = 100_000;
    let file = dir.join("deep.sky");
    fs::write(&file, format!("x = {}1{}\n", "(".repeat(n), ")".repeat(n))).unwrap();
    let out = sky_check(file.to_str().unwrap());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.contains("error: expression is nested too deeply (the limit is 256 levels)"),
        "{}",
        &stderr[..stderr.len().min(300)]
    );
    assert_eq!(stdout(&out), "");
    fs::remove_dir_all(&dir).unwrap();
}
