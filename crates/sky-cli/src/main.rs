use std::process::ExitCode;

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
            if let Some(b) = sky_syntax::source::find_bidi_control(&src) {
                eprintln!(
                    "error: {path}: bidirectional control character U+{:04X} at byte {} is not allowed",
                    b.ch as u32, b.byte_offset
                );
                return ExitCode::FAILURE;
            }
            println!("{path}: ok (only the lexer's safety pass exists so far)");
            ExitCode::SUCCESS
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
