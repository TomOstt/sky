//! Parser golden tests: every `crates/sky-ast/tests/ui/*.sky` has a
//! `.stderr` next to it holding the exact diagnostics (an empty `.stderr`
//! means the file parses cleanly). The file format and naming follow
//! `tests/ui/README.md`; see `tests/ui/README.md` in this directory for why
//! they live here for now.
//!
//! For each file this runner lexes with `sky_syntax::lex`, parses with
//! `sky_ast::parse_module` if lexing succeeded, and renders the lexer's
//! warnings/error and the parse errors with sky-diag's renderer in the
//! `spec/diagnostics.md` format. It passes the raw source to
//! `SourceFile::new` (no escaping, which could shift spans). Diagnostics are
//! joined by one blank line, and the file name is the path relative to the
//! workspace root, as `sky check` prints it.
//!
//! `ok_*` files must have no error, `err_*` files at least one. Regenerate
//! with `SKY_BLESS=1 cargo test -p sky-ast --test ui_parser` and review the
//! diff.

use std::fs;
use std::path::{Path, PathBuf};

use sky_diag::{Diagnostic, Severity, SourceFile};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/sky-ast is two levels below the workspace root")
        .to_path_buf()
}

/// The rendered diagnostics for one file, and whether any is an error.
fn check(path: &str, src: &str) -> (String, bool) {
    let diagnostics: Vec<Diagnostic> = match sky_syntax::lex(src) {
        Ok(lexed) => {
            let (_, errors) = sky_ast::parse_module(&lexed.tokens, src);
            lexed
                .warnings
                .into_iter()
                .chain(errors.into_iter().map(Diagnostic::from))
                .collect()
        }
        Err(e) => e.diagnostics().cloned().collect(),
    };
    let file = SourceFile::new(path, src);
    let rendered: Vec<String> = diagnostics
        .iter()
        .map(|d| d.render_to_string(&file))
        .collect();
    let has_error = diagnostics.iter().any(|d| d.severity == Severity::Error);
    (rendered.join("\n"), has_error)
}

#[test]
fn ui_parser() {
    let root = workspace_root();
    let dir = root.join("crates/sky-ast/tests/ui");
    let mut files: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .collect();
    files.sort();
    for path in &files {
        if path.extension().is_some_and(|x| x == "stderr") {
            assert!(
                path.with_extension("sky").exists(),
                "{} has no matching .sky file",
                path.display()
            );
        }
    }
    files.retain(|p| p.extension().is_some_and(|x| x == "sky"));
    assert!(
        !files.is_empty(),
        "no .sky files under crates/sky-ast/tests/ui"
    );

    let bless = std::env::var_os("SKY_BLESS").is_some();
    let mut failures = Vec::new();
    for sky in &files {
        let name = sky.file_name().expect("file name").to_string_lossy();
        let rel = format!("crates/sky-ast/tests/ui/{name}");
        let src = fs::read_to_string(sky).expect("read .sky");
        let (actual, has_error) = check(&rel, &src);

        if name.starts_with("ok_") && has_error {
            failures.push(format!("{rel}: ok_ file has an error"));
        }
        if name.starts_with("err_") && !has_error {
            failures.push(format!("{rel}: err_ file has no error"));
        }
        let expected_path = sky.with_extension("stderr");
        if bless {
            fs::write(&expected_path, &actual).expect("write .stderr");
            continue;
        }
        let Ok(expected) = fs::read_to_string(&expected_path) else {
            failures.push(format!("{rel}: missing {}", expected_path.display()));
            continue;
        };
        if actual != expected {
            failures.push(format!(
                "{rel}: stderr mismatch\n--- expected\n{expected}--- actual\n{actual}---"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} parser ui tests failed:\n\n{}",
        failures.len(),
        files.len(),
        failures.join("\n\n")
    );
}
