use std::process::ExitCode;

use sky_diag::{Diagnostic, SourceFile};

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
                    eprintln!("error: cannot read {path}: {e}");
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

/// `sky check`: lexes the file and prints its diagnostics to stderr, in
/// source order, separated by blank lines. Exits 1 if any is an error.
fn check(path: &str, src: &str) -> ExitCode {
    let result = sky_syntax::lex(src);
    let (diagnostics, ok): (Vec<&Diagnostic>, bool) = match &result {
        Ok(lexed) => (lexed.warnings.iter().collect(), true),
        Err(e) => (e.diagnostics().collect(), false),
    };
    // Snippets echo source lines; never echo bidi or control characters to
    // the terminal. Offsets and columns are unchanged by the replacement.
    let file = SourceFile::new(path, sky_syntax::source::display_safe(src));
    let rendered: Vec<String> = diagnostics.iter().map(|d| d.render_to_string(&file)).collect();
    eprint!("{}", rendered.join("\n"));
    if ok {
        println!("{path}: ok");
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
