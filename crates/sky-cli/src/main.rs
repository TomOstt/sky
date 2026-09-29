use std::path::Path;
use std::process::ExitCode;

use sky_ast::ParseError;
use sky_diag::{display_safe, Diagnostic, Severity, SourceFile};
use sky_project::SystemIdentity;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("check") => {
            let Some(path) = args.get(1) else {
                eprintln!("usage: sky check <file.sky>");
                return ExitCode::from(2);
            };
            // One open, then fstat and a capped read of that handle: only a
            // regular file of at most sky_syntax::MAX_SOURCE_BYTES is read, and a FIFO
            // can't block the open.
            let src = match sky_project::read_entry_file(Path::new(path)) {
                Ok(s) => s,
                Err(e) => {
                    // Display escapes the path like the renderer does, so an
                    // ESC or bidi control can't reach the terminal.
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            check(path, &src)
        }
        Some("run") | Some("repl") => {
            eprintln!("not implemented yet");
            ExitCode::FAILURE
        }
        _ => {
            eprintln!("usage: sky <check|run|repl> [file]");
            ExitCode::from(2)
        }
    }
}

/// `sky check`: finds the project root, then lexes and parses the file and
/// prints its diagnostics to stderr, in source order, separated by blank
/// lines. Exits 1 if any is an error; prints `ok` only if none is.
fn check(path: &str, src: &str) -> ExitCode {
    // Spec v0.1 non-negotiable 2: the root comes only from walking up from
    // the file, never from the cwd or the environment. A refused sky.toml
    // stops the check; its diagnostic goes through the renderer, which
    // escapes the path.
    if let Err(e) = sky_project::discover(Path::new(path), &SystemIdentity) {
        eprint!("{}", e.to_diagnostic().display(&e.source_file()));
        return ExitCode::FAILURE;
    }
    // The whole front end runs in one call on the known-size stack, so the
    // parser's depth limit holds (and a panic inside propagates unchanged).
    let (rendered, ok) = sky_ast::with_deep_stack(|| check_source(path, src));
    eprint!("{}", rendered.join("\n"));
    if ok {
        println!("{}: ok", display_safe(path));
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Lexes and parses `src`, and renders every diagnostic. Returns the
/// rendered diagnostics in source order and whether none is an error. Runs
/// inside `with_deep_stack`.
fn check_source(path: &str, src: &str) -> (Vec<String>, bool) {
    let mut diagnostics: Vec<Diagnostic> = match sky_syntax::lex(src) {
        Err(e) => e.diagnostics().cloned().collect(),
        Ok(lexed) => {
            let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
            let mut diagnostics = lexed.warnings;
            diagnostics.extend(errors.into_iter().map(ParseError::into_diagnostic));
            // Type checker seam: when sky-types is wired in, check `module`
            // here (if `errors` was empty) and push its diagnostics.
            let _ = module;
            diagnostics
        }
    };
    // Lexer warnings and parse errors, interleaved by position.
    diagnostics.sort_by_key(|d| d.primary.span.start);
    let ok = !diagnostics.iter().any(|d| d.severity == Severity::Error);
    // Pass the source unchanged: spans are byte offsets into it, and the
    // renderer escapes terminal-unsafe characters itself (spec/diagnostics.md).
    let file = SourceFile::new(path, src);
    let rendered = diagnostics
        .iter()
        .map(|d| d.render_to_string(&file))
        .collect();
    (rendered, ok)
}
