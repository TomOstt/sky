use std::fs::File;
use std::path::Path;
use std::process::ExitCode;

use sky_ast::ParseError;
use sky_diag::{display_safe, Diagnostic, Severity, SourceFile, Span};
use sky_project::SystemIdentity;

const USAGE: &str = "usage: sky check [--explain] <file.sky>\n       sky <run|repl> [file]";

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
            // One open, then fstat and a capped read of that handle: only a
            // regular file of at most sky_syntax::MAX_SOURCE_BYTES is read, and a FIFO
            // can't block the open. Display escapes the path like the
            // renderer does, so an ESC or bidi control can't reach the
            // terminal.
            let file = match sky_project::open_entry_file(Path::new(path.as_str())) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            let src = match sky_project::read_source(&file, Path::new(path.as_str())) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            check(path, &file, &src, !explain.is_empty())
        }
        Some("run") | Some("repl") => {
            eprintln!("not implemented yet");
            ExitCode::FAILURE
        }
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
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
    if let Err(e) = sky_project::discover_opened(Path::new(path), file, &SystemIdentity) {
        eprint!("{}", e.to_diagnostic().display(&e.source_file()));
        return ExitCode::FAILURE;
    }
    // The whole front end runs in one call on the known-size stack, so the
    // parser's and checker's depth limits hold (and a panic inside
    // propagates unchanged).
    let (rendered, ok) = sky_ast::with_deep_stack(|| check_source(path, src, explain));
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
fn check_source(path: &str, src: &str, explain: bool) -> (Vec<String>, bool) {
    let mut diagnostics: Vec<Diagnostic> = match sky_syntax::lex(src) {
        Err(e) => e.diagnostics().cloned().collect(),
        Ok(lexed) => {
            let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
            let mut diagnostics = lexed.warnings;
            if errors.is_empty() {
                // Only a module that parsed cleanly is checked, so the
                // checker never reports on a statement the parser dropped.
                let checked = sky_types::check_module(&module, src);
                diagnostics.extend(checked.diagnostics);
                if explain {
                    diagnostics.extend(checked.rewrites.iter().map(|r| r.to_diagnostic(src)));
                    if checked.rewrites_omitted > 0 {
                        let last = checked.rewrites.last().map_or(Span::point(0), |r| r.span);
                        diagnostics.push(Diagnostic::note(
                            format!(
                                "{} more rewrite{} not shown",
                                checked.rewrites_omitted,
                                if checked.rewrites_omitted == 1 {
                                    ""
                                } else {
                                    "s"
                                }
                            ),
                            last,
                        ));
                    }
                }
            } else {
                diagnostics.extend(errors.into_iter().map(ParseError::into_diagnostic));
            }
            diagnostics
        }
    };
    // Lexer warnings with parse or check errors, interleaved by position.
    // The sort is stable, so diagnostics at one position keep their order.
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
