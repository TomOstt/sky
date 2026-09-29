use std::fs::File;
use std::path::Path;
use std::process::ExitCode;

use sky_diag::{display_safe, SourceFile};
use sky_project::{Project, SystemIdentity};
use sky_syntax::Edition;

mod frontend;
mod repl;
mod run;

const USAGE: &str =
    "usage: sky check [--explain] <file.sky>\n       sky run <file.sky>\n       sky repl";

/// The `sky` command line.
///
/// # Exit status
///
/// The one place the exit codes are defined (the README repeats it):
///
/// * `0`: ok. `sky check` found no error (warnings alone do not fail it),
///   `sky run` finished, or `sky repl` reached end of input.
/// * `1`: diagnostics with at least one error, whatever the phase: a lexer,
///   parser or checker error, a runtime error in `sky run`, or a refused
///   project root (a `sky.toml` that is a link, foreign-owned or
///   world-writable) or an invalid `sky.toml` (unknown key, unsupported
///   edition, ...).
/// * `2`: a usage error (unknown command, wrong arguments), or an input that
///   cannot be read (missing, not a regular file, too large, not UTF-8).
///
/// So `2` means "the check did not run on your source" and `1` means "it ran
/// and rejected it". (A Rust panic would exit 101; that is a bug.)
fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("check") => {
            let (explain, rest): (Vec<&String>, Vec<&String>) =
                args[1..].iter().partition(|a| *a == "--explain");
            let [path] = rest.as_slice() else {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            };
            if explain.len() > 1 || path.starts_with("--") {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            }
            let (file, src) = match read_entry(path) {
                Ok(x) => x,
                Err(code) => return code,
            };
            check(path, &file, &src, !explain.is_empty())
        }
        Some("run") => {
            let [path] = &args[1..] else {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            };
            if path.starts_with("--") {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            }
            let (file, src) = match read_entry(path) {
                Ok(x) => x,
                Err(code) => return code,
            };
            run::run(path, &file, &src)
        }
        Some("repl") => {
            if args.len() != 1 {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            }
            repl::repl()
        }
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// Opens and reads the entry file for `check` and `run`. One open, then
/// fstat and a capped read of that handle: only a regular file of at most
/// sky_syntax::MAX_SOURCE_BYTES is read, and a FIFO can't block the open.
/// Display escapes the path like the renderer does, so an ESC or bidi
/// control can't reach the terminal. On failure the message is printed and
/// the exit code (2) returned.
fn read_entry(path: &str) -> Result<(File, String), ExitCode> {
    let file = sky_project::open_entry_file(Path::new(path)).map_err(|e| {
        eprintln!("error: {e}");
        ExitCode::from(2)
    })?;
    let src = sky_project::read_source(&file, Path::new(path)).map_err(|e| {
        eprintln!("error: {e}");
        ExitCode::from(2)
    })?;
    Ok((file, src))
}

/// `sky check`: finds the project root, then lexes, parses, and type-checks
/// the file and prints its diagnostics to stderr, in source order,
/// separated by blank lines. With `explain`, the checker's rewrite
/// decisions follow as notes. Exits 1 if any diagnostic is an error; prints
/// `ok` only if none is.
fn check(path: &str, file: &File, src: &str, explain: bool) -> ExitCode {
    // Spec v0.1 non-negotiable 2: the root comes only from walking up from
    // the file, never from the cwd or the environment. A refused sky.toml
    // stops the check; its diagnostic goes through the renderer, which
    // escapes the path. Discovery also checks that the path still names
    // the file that was opened and read.
    let discovery = match sky_project::discover_opened(Path::new(path), file, &SystemIdentity) {
        Ok(d) => d,
        Err(e) => {
            eprint!("{}", e.to_diagnostic().display(&e.source_file()));
            return ExitCode::FAILURE;
        }
    };
    // stdout, not stderr: the `--explain` goldens pin stderr, and this is a
    // fact about the run, not a diagnostic. After discovery so an invalid
    // manifest never prints an edition.
    if explain {
        let origin = match &discovery.project {
            Project::Rooted { manifest, .. } => {
                format!(
                    "declared in {}",
                    display_safe(&manifest.display().to_string())
                )
            }
            Project::SingleFile => "no sky.toml: the latest".to_string(),
        };
        println!("edition: {} ({origin})", discovery.edition);
    }
    // The whole front end runs in one call on the known-size stack, so the
    // parser's and checker's depth limits hold (and a panic inside
    // propagates unchanged).
    let (rendered, ok) =
        sky_ast::with_deep_stack(|| check_source(path, src, explain, discovery.edition));
    eprint!("{}", rendered.join("\n"));
    if ok {
        println!("{}: ok", display_safe(path));
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Lexes, parses, and (if parsing succeeded) type-checks `src`, and renders
/// every diagnostic. Returns the rendered diagnostics in source order and
/// whether none is an error. Runs inside `with_deep_stack`.
fn check_source(path: &str, src: &str, explain: bool, edition: Edition) -> (Vec<String>, bool) {
    let front = frontend::analyze(src, sky_types::Env::with_prelude(), explain, edition);
    // Pass the source unchanged: spans are byte offsets into it, and the
    // renderer escapes terminal-unsafe characters itself (spec/diagnostics.md).
    let file = SourceFile::new(path, src);
    (front.render(&file), front.ok)
}
