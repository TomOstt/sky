//! Checker golden tests: every `crates/sky-types/tests/ui/*.sky` has a
//! `.stderr` next to it holding the exact diagnostics (empty means the file
//! checks cleanly). The format and naming follow `tests/ui/README.md` at the
//! workspace root and `crates/sky-ast/tests/ui_parser.rs`; see
//! `tests/ui/README.md` in this directory for the extra files and why they
//! live here for now.
//!
//! For each file this runner lexes with `sky_syntax::lex`, parses with
//! `sky_ast::parse_module`, and, if both succeeded without errors, checks
//! with `sky_types::check_module_with`. Diagnostics are rendered with
//! sky-diag and joined by one blank line.
//!
//! Optional siblings:
//! * `.types`: every binding after the last statement, `name: type`, one per
//!   line, in name order.
//! * `.explain`: the rewrite decisions rendered as notes (what `sky check
//!   --explain` will print). Required whenever the file has a rewrite, so
//!   every rewrite is pinned; an empty `.explain` pins "no rewrite".
//!
//! Declarations now come from real Sky annotations (`A: PositiveDefinite
//! Matrix[N, N]`), so there is no test-only seeding. The cases that need
//! declarations live in `tests/ui/types/` and run end to end through
//! `sky check` (`crates/sky-cli/tests/ui.rs`); this runner also reads their
//! `.types` files, which the CLI harness can't see. Regenerate with
//! `SKY_BLESS=1 cargo test -p sky-types --test ui_check` and review the diff.

use std::fs;
use std::path::{Path, PathBuf};

use sky_diag::{Diagnostic, Severity, SourceFile};
use sky_types::Env;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/sky-types is two levels below the workspace root")
        .to_path_buf()
}

struct Outcome {
    stderr: String,
    has_error: bool,
    types: Option<String>,
    explain: Option<String>,
    rewrites: usize,
}

fn run(path: &str, src: &str) -> Outcome {
    let env = Env::new();
    let file = SourceFile::new(path, src);
    let mut diagnostics: Vec<Diagnostic> = Vec::new();
    let mut checked = None;
    match sky_syntax::lex(src) {
        Ok(lexed) => {
            diagnostics.extend(lexed.warnings);
            let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
            let parse_ok = errors.is_empty();
            diagnostics.extend(errors.into_iter().map(Diagnostic::from));
            if parse_ok {
                let c = sky_types::check_module_with(&module, src, env);
                diagnostics.extend(c.diagnostics.iter().cloned());
                checked = Some(c);
            }
        }
        Err(e) => diagnostics.extend(e.diagnostics().cloned()),
    }
    let render = |ds: &[Diagnostic]| {
        ds.iter()
            .map(|d| d.render_to_string(&file))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let (types, explain, rewrites) = match &checked {
        Some(c) => {
            let mut types = String::new();
            for (n, t) in c.env.bindings() {
                types.push_str(&format!("{n}: {t}\n"));
            }
            let notes: Vec<Diagnostic> = c.rewrites.iter().map(|r| r.to_diagnostic(src)).collect();
            (Some(types), Some(render(&notes)), c.rewrites.len())
        }
        None => (None, None, 0),
    };
    Outcome {
        stderr: render(&diagnostics),
        has_error: diagnostics.iter().any(|d| d.severity == Severity::Error),
        types,
        explain,
        rewrites,
    }
}

/// Compares (or, when blessing, writes) an optional sibling file.
fn sibling(
    failures: &mut Vec<String>,
    rel: &str,
    path: &Path,
    actual: Option<&str>,
    bless: bool,
    what: &str,
) {
    match (actual, bless) {
        (None, _) => failures.push(format!("{rel}: has a .{what} file but did not check")),
        (Some(a), true) => fs::write(path, a).expect("write sibling"),
        (Some(a), false) => {
            let expected = fs::read_to_string(path).expect("read sibling");
            if a != expected {
                failures.push(format!(
                    "{rel}: {what} mismatch\n--- expected\n{expected}\n--- actual\n{a}\n---"
                ));
            }
        }
    }
}

#[test]
fn ui_check() {
    let dir = workspace_root().join("crates/sky-types/tests/ui");
    let mut files: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .collect();
    files.sort();
    for path in &files {
        if path
            .extension()
            .is_some_and(|x| x == "stderr" || x == "types" || x == "explain")
        {
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
        "no .sky files under crates/sky-types/tests/ui"
    );

    let bless = std::env::var_os("SKY_BLESS").is_some();
    let mut failures = Vec::new();
    for sky in &files {
        let name = sky.file_name().expect("file name").to_string_lossy();
        let rel = format!("crates/sky-types/tests/ui/{name}");
        let src = fs::read_to_string(sky).expect("read .sky");
        let out = run(&rel, &src);
        if name.starts_with("ok_") && out.has_error {
            failures.push(format!("{rel}: ok_ file has an error"));
        }
        if name.starts_with("err_") && !out.has_error {
            failures.push(format!("{rel}: err_ file has no error"));
        }
        let types_path = sky.with_extension("types");
        if types_path.exists() {
            sibling(
                &mut failures,
                &rel,
                &types_path,
                out.types.as_deref(),
                bless,
                "types",
            );
        }
        let explain_path = sky.with_extension("explain");
        if explain_path.exists() || (bless && out.rewrites > 0) {
            sibling(
                &mut failures,
                &rel,
                &explain_path,
                out.explain.as_deref(),
                bless,
                "explain",
            );
        } else if out.rewrites > 0 {
            failures.push(format!("{rel}: has a rewrite but no .explain file"));
        }
        let stderr_path = sky.with_extension("stderr");
        if bless {
            fs::write(&stderr_path, &out.stderr).expect("write .stderr");
            continue;
        }
        let Ok(expected) = fs::read_to_string(&stderr_path) else {
            failures.push(format!("{rel}: missing {}", stderr_path.display()));
            continue;
        };
        if out.stderr != expected {
            failures.push(format!(
                "{rel}: stderr mismatch\n--- expected\n{expected}--- actual\n{}---",
                out.stderr
            ));
        }
    }
    // `tests/ui/types/` belongs to the CLI harness (stderr and `--explain`);
    // only its `.types` files are checked here.
    let types_dir = workspace_root().join("tests/ui/types");
    let mut typed: Vec<PathBuf> = fs::read_dir(&types_dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", types_dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "types"))
        .collect();
    typed.sort();
    for types_path in &typed {
        let sky = types_path.with_extension("sky");
        let name = sky.file_name().expect("file name").to_string_lossy();
        let rel = format!("tests/ui/types/{name}");
        let Ok(src) = fs::read_to_string(&sky) else {
            failures.push(format!("{rel}: .types file has no .sky file"));
            continue;
        };
        let out = run(&rel, &src);
        sibling(
            &mut failures,
            &rel,
            types_path,
            out.types.as_deref(),
            bless,
            "types",
        );
    }
    assert!(
        failures.is_empty(),
        "{} of {} checker ui tests failed:\n\n{}",
        failures.len(),
        files.len() + typed.len(),
        failures.join("\n\n")
    );
}
