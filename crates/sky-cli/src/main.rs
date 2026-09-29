use std::path::Path;
use std::process::ExitCode;

use sky_diag::{display_safe, Diagnostic, SourceFile};
use sky_project::SystemIdentity;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("check") => {
            let Some(path) = args.get(1) else {
                eprintln!("usage: sky check <file.sky>");
                return ExitCode::from(2);
            };
            let src = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => {
                    // The path is untrusted text: escape it like the renderer
                    // does, so an ESC or bidi control can't reach the terminal.
                    eprintln!("error: cannot read {}: {e}", display_safe(path));
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

/// `sky check`: finds the project root, lexes the file, and prints its
/// diagnostics to stderr, in source order, separated by blank lines. Exits 1
/// if any is an error.
fn check(path: &str, src: &str) -> ExitCode {
    // Spec v0.1 non-negotiable 2: the root comes only from walking up from
    // the file, never from the cwd or the environment. A refused sky.toml
    // stops the check; its diagnostic goes through the renderer, which
    // escapes the path.
    if let Err(e) = sky_project::discover(Path::new(path), &SystemIdentity) {
        eprint!("{}", e.to_diagnostic().display(&e.source_file()));
        return ExitCode::FAILURE;
    }
    let result = sky_syntax::lex(src);
    let (diagnostics, ok): (Vec<&Diagnostic>, bool) = match &result {
        Ok(lexed) => (lexed.warnings.iter().collect(), true),
        Err(e) => (e.diagnostics().collect(), false),
    };
    // Pass the source unchanged: spans are byte offsets into it, and the
    // renderer escapes terminal-unsafe characters itself (spec/diagnostics.md).
    let file = SourceFile::new(path, src);
    let rendered: Vec<String> = diagnostics
        .iter()
        .map(|d| d.render_to_string(&file))
        .collect();
    eprint!("{}", rendered.join("\n"));
    if ok {
        println!("{}: ok", display_safe(path));
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
