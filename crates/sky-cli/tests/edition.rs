//! `sky.toml`'s `edition` through the real binary: `check` and `run` read it
//! (an invalid manifest stops them with exit 1 and a diagnostic with a
//! span), `check --explain` shows it, and a lone file uses the latest.
#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A fresh, canonical temp dir with a `.git` boundary, so the walk stays in it.
struct Sandbox(PathBuf);

impl Sandbox {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("sky-cli-ed-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join(".git")).unwrap();
        Sandbox(dir.canonicalize().unwrap())
    }
    fn file(&self, rel: &str, text: &str) -> PathBuf {
        let p = self.0.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(&p, text).unwrap();
        p
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn sky(args: &[&str], file: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sky"))
        .args(args)
        .arg(file)
        .output()
        .unwrap()
}

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

#[test]
fn a_declared_edition_is_shown_by_explain() {
    let s = Sandbox::new("declared");
    let manifest = s.file("sky.toml", "edition = \"2026\"\n");
    let f = s.file("main.sky", "x = 1\n");
    let out = sky(&["check", "--explain"], &f);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(
        text(&out.stdout),
        format!(
            "edition: 2026 (declared in {})\n{}: ok\n",
            manifest.display(),
            f.display()
        )
    );
    // Without --explain nothing changes.
    let out = sky(&["check"], &f);
    assert_eq!(text(&out.stdout), format!("{}: ok\n", f.display()));
}

#[test]
fn a_lone_file_uses_the_latest_edition() {
    let s = Sandbox::new("lone");
    let f = s.file("main.sky", "x = 1\n");
    let out = sky(&["check", "--explain"], &f);
    assert_eq!(out.status.code(), Some(0));
    assert_eq!(
        text(&out.stdout),
        format!(
            "edition: 2026 (no sky.toml: the latest)\n{}: ok\n",
            f.display()
        )
    );
}

#[test]
fn a_newer_edition_stops_check_and_run_with_a_span() {
    let s = Sandbox::new("newer");
    let manifest = s.file("sky.toml", "# proj\nedition = \"2031\"\n");
    let f = s.file("main.sky", "print(1)\n");
    for cmd in ["check", "run"] {
        let out = sky(&[cmd], &f);
        assert_eq!(out.status.code(), Some(1), "{cmd}");
        assert_eq!(text(&out.stdout), "", "{cmd}: nothing ran");
        let err = text(&out.stderr);
        assert!(
            err.starts_with(&format!(
                "{}:2:11: error: edition 2031 is newer than this Sky (supports 2026)\n",
                manifest.display()
            )),
            "{cmd}: {err}"
        );
        assert!(err.contains("2 | edition = \"2031\""), "{err}");
    }
}

#[test]
fn a_manifest_with_an_unknown_key_is_an_error() {
    let s = Sandbox::new("key");
    s.file("sky.toml", "editon = \"2026\"\n");
    let f = s.file("main.sky", "x = 1\n");
    let out = sky(&["check", "--explain"], &f);
    assert_eq!(out.status.code(), Some(1));
    // An invalid manifest never prints an edition.
    assert_eq!(text(&out.stdout), "");
    let err = text(&out.stderr);
    assert!(err.contains("error: unknown key `editon`"), "{err}");
    assert!(err.contains("= help: did you mean `edition`?"), "{err}");
}

#[test]
fn a_valid_project_runs() {
    let s = Sandbox::new("runs");
    s.file("sky.toml", "edition = \"2026\"\n");
    let f = s.file("main.sky", "print(1 + 1)\n");
    let out = sky(&["run"], &f);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), "2\n");
}
