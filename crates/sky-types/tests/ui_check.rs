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
//! `# env:` comment lines seed names the language can't declare yet (see
//! the README). Regenerate with `SKY_BLESS=1 cargo test -p sky-types --test
//! ui_check` and review the diff.

use std::fs;
use std::path::{Path, PathBuf};

use sky_diag::{Diagnostic, Severity, SourceFile};
use sky_types::{Env, Shape, ShapeDim, Tag, TagSet};
use sky_units::Dimension;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/sky-types is two levels below the workspace root")
        .to_path_buf()
}

/// Parses a `# env: NAME: SHAPE [TAG...] [unit UNIT]` line, e.g.
/// `# env: A: Matrix[N,N] PositiveDefinite` or `# env: v: Vector[3] unit m/s`.
fn seed(env: &mut Env, line: &str) -> Result<(), String> {
    let (name, rest) = line.split_once(':').ok_or("expected `NAME: SHAPE`")?;
    let mut words = rest.split_whitespace();
    let shape_text = words.next().ok_or("missing shape")?;
    let shape = if shape_text == "Scalar" {
        Shape::scalar()
    } else {
        let (kind, dims) = shape_text
            .strip_suffix(']')
            .and_then(|s| s.split_once('['))
            .ok_or_else(|| format!("bad shape `{shape_text}`"))?;
        let dims: Vec<ShapeDim> = dims
            .split(',')
            .map(|d| match d.trim().parse::<u64>() {
                Ok(n) => ShapeDim::Known(n),
                Err(_) => ShapeDim::Var(env.shape_var(d.trim())),
            })
            .collect();
        let want = match kind {
            "Vector" => Some(1),
            "Matrix" => Some(2),
            "Tensor" => None,
            _ => return Err(format!("bad shape kind `{kind}`")),
        };
        if want.is_some_and(|n| n != dims.len()) {
            return Err(format!("wrong rank in `{shape_text}`"));
        }
        Shape::new(dims)
    };
    let mut tags = TagSet::EMPTY;
    let mut dim = Dimension::DIMENSIONLESS;
    while let Some(w) = words.next() {
        if w == "unit" {
            dim = unit(words.next().ok_or("missing unit")?)?;
        } else {
            tags = tags.with(Tag::from_name(w).ok_or_else(|| format!("unknown tag `{w}`"))?);
        }
    }
    env.bind_array(name.trim(), shape, tags, dim)
        .map_err(|e| e.to_string())
}

/// `m/s`, `kg*m^2`: symbols from the unit table joined by `*` and `/`, each
/// with an optional integer power.
fn unit(text: &str) -> Result<Dimension, String> {
    let mut dim = Dimension::DIMENSIONLESS;
    let mut divide = false;
    let mut rest = text;
    loop {
        let end = rest.find(['*', '/']).unwrap_or(rest.len());
        let (factor, tail) = rest.split_at(end);
        let (sym, pow) = match factor.split_once('^') {
            Some((s, p)) => (s, p.parse::<i64>().map_err(|e| e.to_string())?),
            None => (factor, 1),
        };
        let u = sky_units::lookup(sym).ok_or_else(|| format!("unknown unit `{sym}`"))?;
        let d = u.dim.pow_int(pow).map_err(|e| e.to_string())?;
        dim = if divide {
            dim.checked_div(d)
        } else {
            dim.checked_mul(d)
        }
        .map_err(|e| e.to_string())?;
        let Some(op) = tail.chars().next() else {
            return Ok(dim);
        };
        divide = op == '/';
        rest = &tail[1..];
    }
}

struct Outcome {
    stderr: String,
    has_error: bool,
    types: Option<String>,
    explain: Option<String>,
    rewrites: usize,
}

fn run(path: &str, src: &str) -> Result<Outcome, String> {
    let mut env = Env::new();
    for line in src.lines() {
        if let Some(spec) = line.strip_prefix("# env:") {
            seed(&mut env, spec).map_err(|e| format!("{path}: bad `# env:` line {line:?}: {e}"))?;
        }
    }
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
    Ok(Outcome {
        stderr: render(&diagnostics),
        has_error: diagnostics.iter().any(|d| d.severity == Severity::Error),
        types,
        explain,
        rewrites,
    })
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
        let out = match run(&rel, &src) {
            Ok(out) => out,
            Err(e) => {
                failures.push(e);
                continue;
            }
        };
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
    assert!(
        failures.is_empty(),
        "{} of {} checker ui tests failed:\n\n{}",
        failures.len(),
        files.len(),
        failures.join("\n\n")
    );
}
