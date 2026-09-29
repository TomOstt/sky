//! `sky check` finds the project root only by walking up from the file.
#![cfg(unix)]

use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A fresh, canonical temp dir containing a `.git` boundary, so the real
/// binary's walk never leaves it.
struct Sandbox(PathBuf);

impl Sandbox {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("sky-cli-{}-{name}", std::process::id()));
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
    /// Plants a symlinked `sky.toml` in `rel`: if `sky check` ever treats that
    /// directory as a root candidate, it refuses and exits 1.
    fn trap(&self, rel: &str) -> PathBuf {
        let d = self.0.join(rel);
        fs::create_dir_all(&d).unwrap();
        symlink("/dev/null", d.join("sky.toml")).unwrap();
        d
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn sky_check(cwd: &Path, file: &Path, envs: &[(&str, &Path)]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_sky"));
    cmd.arg("check").arg(file).current_dir(cwd);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.output().unwrap()
}

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

#[test]
fn cwd_and_env_vars_are_never_a_root_source() {
    let s = Sandbox::new("cwd-env");
    let f = s.file("proj/main.sky", "x = 1\n"); // no sky.toml: single-file mode
    let evil = s.trap("evil");
    let envs: Vec<(&str, &Path)> = [
        "HOME",
        "PWD",
        "SKY_ROOT",
        "SKY_PROJECT_ROOT",
        "SKY_PATH",
        "SKY_HOME",
    ]
    .into_iter()
    .map(|k| (k, evil.as_path()))
    .collect();

    // Absolute path, cwd and env all pointing at the trap.
    let out = sky_check(&evil, &f, &envs);
    assert!(out.status.success(), "stderr: {}", text(&out.stderr));
    // Relative path, interpreted against the trap cwd.
    let out = sky_check(&evil, Path::new("../proj/main.sky"), &envs);
    assert!(out.status.success(), "stderr: {}", text(&out.stderr));
    assert!(text(&out.stdout).starts_with("../proj/main.sky: ok"));
}

#[test]
fn regular_sky_toml_in_a_parent_is_accepted() {
    let s = Sandbox::new("ok");
    s.file("sky.toml", "");
    let f = s.file("src/physics/kinematics.sky", "v = 10 km\n");
    let out = sky_check(&s.0, &f, &[]);
    assert!(out.status.success(), "stderr: {}", text(&out.stderr));
}

#[test]
fn symlinked_sky_toml_is_refused_with_a_diagnostic() {
    let s = Sandbox::new("symlink");
    let root = s.trap("proj");
    let f = s.file("proj/src/main.sky", "x = 1\n");
    let out = sky_check(&s.0, &f, &[]);
    assert_eq!(out.status.code(), Some(1));
    let m = root.join("sky.toml");
    assert_eq!(
        text(&out.stderr),
        format!(
            "\
{m}:1:1: error: refusing to use {m}: it is a symbolic link
  |
1 |
  | ^
  = help: replace the link with a regular sky.toml file owned by you
",
            m = m.display()
        )
    );
    assert!(out.stdout.is_empty());
}

#[test]
fn refusal_diagnostic_escapes_control_characters_in_the_path() {
    let s = Sandbox::new("escape");
    // A directory name carrying an ANSI "clear screen" and a bidi override.
    let dir = "proj\u{1b}[2J\u{202e}x";
    s.trap(dir);
    let f = s.file(&format!("{dir}/main.sky"), "x = 1\n");
    let out = sky_check(&s.0, &f, &[]);
    assert_eq!(out.status.code(), Some(1));
    let err = text(&out.stderr);
    assert!(
        !err.contains('\u{1b}') && !err.contains('\u{202e}'),
        "{err:?}"
    );
    let m = format!("{}/proj\\u{{001B}}[2J\\u{{202E}}x/sky.toml", s.0.display());
    assert!(
        err.starts_with(&format!(
            "{m}:1:1: error: refusing to use {m}: it is a symbolic link\n"
        )),
        "{err:?}"
    );
}
