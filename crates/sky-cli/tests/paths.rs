//! Paths are untrusted text: `sky check` must escape terminal-unsafe
//! characters in them (spec/diagnostics.md) wherever it prints one, not only
//! in rendered diagnostics.

use std::fs;
use std::path::Path;
use std::process::Command;

fn sky_check(path: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_sky"))
        .arg("check")
        .arg(path)
        .output()
        .expect("run sky")
}

#[test]
fn unreadable_path_is_escaped() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"));
    let path = dir.join("missing-\u{1B}[31m-\u{202E}.sky");
    let path = path.to_str().expect("UTF-8 temp dir");
    let _ = fs::remove_file(path);
    let out = sky_check(path);
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8(out.stderr).expect("UTF-8");
    let dir = dir.to_str().unwrap();
    assert!(
        stderr.starts_with(&format!(
            "error: cannot read {dir}/missing-\\u{{001B}}[31m-\\u{{202E}}.sky: "
        )),
        "{stderr:?}"
    );
    assert!(!stderr.contains(['\u{1B}', '\u{202E}']), "{stderr:?}");
}

#[test]
fn ok_path_is_escaped() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"));
    let path = dir.join("ok-\u{1B}]0;title\u{7}.sky");
    fs::write(&path, "x = 1\n").expect("write temp file");
    let path = path.to_str().expect("UTF-8 temp dir");
    let out = sky_check(path);
    let _ = fs::remove_file(path);
    assert_eq!(out.status.code(), Some(0));
    assert!(out.stderr.is_empty(), "{:?}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8(out.stdout).expect("UTF-8");
    let dir = dir.to_str().unwrap();
    assert_eq!(
        stdout,
        format!("{dir}/ok-\\u{{001B}}]0;title\\u{{0007}}.sky: ok\n")
    );
}
