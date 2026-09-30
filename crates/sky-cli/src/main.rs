use std::fs::File;
use std::path::Path;
use std::process::ExitCode;

use sky_diag::{display_safe, SourceFile};
use sky_project::{Project, SystemIdentity};
use sky_syntax::Edition;

mod accuracy;
mod cost;
mod frontend;
mod hw;
mod ir;
mod repl;
mod run;
mod tune;

const USAGE: &str = "usage: sky check [--explain] [--emit-ir[=<stage>]] [--threads=N] <file.sky>\n       sky run [--engine=auto|ast|sir] [--threads=N] <file.sky>\n       sky repl\n\
                     sky hw\n\
                     sky tune [--show | --reset] [--budget-ms=<n>] [<kernel>]\n\n\
                     <stage> is `lower`, `final` (the default), or `after:<pass>` for one of\n\
                     fold, egraph, contracts, dce, fuse, cholesky\n\
                     --engine picks how `sky run` executes: `ast` (the tree-walking\n\
                     interpreter, the default), `sir` (lower to SIR, run the optimized IR),\n\
                     or `auto` (SIR, falling back to the tree-walker for programs SIR cannot\n\
                     lower yet)\n\
                     `sky hw` prints what this machine can do and what Sky uses it for;\n\
                     `sky tune` measures kernel parameters here and remembers the winners\n\
                     (`--show` lists them, `--reset` forgets them)\n\
                     --threads sets how many threads independent work may use (default:\n\
                     SKY_THREADS, else the machine's cores); 1 is sequential, and the\n\
                     output is the same for every count. `sky run` uses it on the SIR\n\
                     engine; `sky check --explain` says what would run in parallel";

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
/// * `sky hw` and `sky tune` exit `0` on success; `sky tune` exits `1` when
///   the performance database cannot be written and `2` when no tunable
///   kernel matches its argument.
/// * `2`: a usage error (unknown command, wrong arguments), or an input that
///   cannot be read (missing, not a regular file, too large, not UTF-8).
///
/// So `2` means "the check did not run on your source" and `1` means "it ran
/// and rejected it". (A Rust panic would exit 101; that is a bug.)
fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("check") => {
            let opts = match CheckArgs::parse(&args[1..]) {
                Ok(o) => o,
                Err(msg) => {
                    eprintln!("{USAGE}");
                    if let Some(msg) = msg {
                        eprintln!("\nerror: {msg}");
                    }
                    return ExitCode::from(2);
                }
            };
            let (file, src) = match read_entry(&opts.path) {
                Ok(x) => x,
                Err(code) => return code,
            };
            check(&opts, &file, &src)
        }
        Some("run") => {
            let (path, engine, threads) = match parse_run_args(&args[1..]) {
                Ok(x) => x,
                Err(msg) => {
                    eprintln!("{USAGE}");
                    if let Some(msg) = msg {
                        eprintln!("\nerror: {msg}");
                    }
                    return ExitCode::from(2);
                }
            };
            let (file, src) = match read_entry(&path) {
                Ok(x) => x,
                Err(code) => return code,
            };
            run::run(&path, &file, &src, engine, threads)
        }
        Some("repl") => {
            if args.len() != 1 {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            }
            repl::repl()
        }
        Some("hw") => {
            if args.len() != 1 {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            }
            hw::hw()
        }
        Some("tune") => match tune::parse(&args[1..]) {
            Ok(parsed) => tune::tune(parsed),
            Err(msg) => {
                eprintln!("{USAGE}");
                if let Some(msg) = msg {
                    eprintln!("\nerror: {msg}");
                }
                ExitCode::from(2)
            }
        },
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// Parses `--threads=N`: `N` is a whole number from 1 to
/// `sky_ir::parallel::MAX_THREADS`.
fn parse_threads(text: &str) -> Result<usize, Option<String>> {
    sky_ir::parallel::parse_threads(text).map_err(Some)
}

/// Parses `sky run`'s arguments: one file, at most one
/// `--engine=ast|sir|auto` and at most one `--threads=N` (before or after the
/// file, like `check`'s flags). `Err` is a usage error: the message, if it
/// needs one beyond the usage text.
fn parse_run_args(args: &[String]) -> Result<(String, run::Engine, Option<usize>), Option<String>> {
    let (mut path, mut engine, mut threads) = (None, None, None);
    for arg in args {
        if let Some(text) = arg.strip_prefix("--threads=") {
            if threads.replace(parse_threads(text)?).is_some() {
                return Err(None);
            }
        } else if let Some(text) = arg.strip_prefix("--engine=") {
            let e = run::Engine::parse(text).ok_or_else(|| {
                Some(format!(
                    "unknown engine `{}` (use `auto`, `ast` or `sir`)",
                    display_safe(text)
                ))
            })?;
            if engine.replace(e).is_some() {
                return Err(None);
            }
        } else if arg.starts_with("--") || path.replace(arg.clone()).is_some() {
            return Err(None);
        }
    }
    Ok((path.ok_or(None)?, engine.unwrap_or_default(), threads))
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

/// The arguments of `sky check`.
struct CheckArgs {
    path: String,
    explain: bool,
    /// `--emit-ir[=<stage>]`.
    emit_ir: Option<ir::Stage>,
    /// `--threads=N`: the thread count the parallelism notes speak of.
    threads: Option<usize>,
}

impl CheckArgs {
    /// Parses `sky check`'s arguments (the flags may come before or after
    /// the file). `Err` is a usage error: the message, if it needs one
    /// beyond the usage text.
    fn parse(args: &[String]) -> Result<CheckArgs, Option<String>> {
        let (mut path, mut explain, mut emit_ir, mut threads) = (None, false, None, None);
        for arg in args {
            if arg == "--explain" {
                if std::mem::replace(&mut explain, true) {
                    return Err(None);
                }
            } else if arg == "--emit-ir" || arg.starts_with("--emit-ir=") {
                let stage = match arg.strip_prefix("--emit-ir=") {
                    Some(text) => ir::Stage::parse(text).map_err(|e| Some(e.to_string()))?,
                    None => ir::Stage::default(),
                };
                if emit_ir.replace(stage).is_some() {
                    return Err(None);
                }
            } else if let Some(text) = arg.strip_prefix("--threads=") {
                if threads.replace(parse_threads(text)?).is_some() {
                    return Err(None);
                }
            } else if arg.starts_with("--") || path.replace(arg.clone()).is_some() {
                return Err(None);
            }
        }
        match path {
            Some(path) => Ok(CheckArgs {
                path,
                explain,
                emit_ir,
                threads,
            }),
            None => Err(None),
        }
    }
}

/// `sky check`: finds the project root, then lexes, parses, and type-checks
/// the file and prints its diagnostics to stderr, in source order,
/// separated by blank lines. With `--explain`, the checker's rewrite
/// decisions and the optimization passes' explanations follow as notes.
/// With `--emit-ir`, the SIR text goes to stdout (and only that: `edition:`
/// and `ok` move to stderr, so stdout can be piped into a file and parsed
/// back). Exits 1 if any diagnostic is an error; prints `ok` only if none
/// is. A program that checks but cannot be lowered to SIR yet is still
/// `ok`; `--emit-ir` then prints a note saying why there is no IR.
fn check(opts: &CheckArgs, file: &File, src: &str) -> ExitCode {
    let path = opts.path.as_str();
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
    // Facts about the run, not diagnostics: stdout (the `--explain` goldens
    // pin stderr), unless `--emit-ir` owns stdout. After discovery so an
    // invalid manifest never prints an edition.
    let facts = |line: String| {
        if opts.emit_ir.is_some() {
            eprintln!("{line}");
        } else {
            println!("{line}");
        }
    };
    if opts.explain {
        let origin = match &discovery.project {
            Project::Rooted { manifest, .. } => {
                format!(
                    "declared in {}",
                    display_safe(&manifest.display().to_string())
                )
            }
            Project::SingleFile => "no sky.toml: the latest".to_string(),
        };
        facts(format!("edition: {} ({origin})", discovery.edition));
    }
    // The whole front end runs in one call on the known-size stack, so the
    // parser's and checker's depth limits hold (and a panic inside
    // propagates unchanged).
    let (rendered, ok, ir_text, internal) =
        sky_ast::with_deep_stack(|| check_source(path, src, opts, discovery.edition));
    eprint!("{}", rendered.join("\n"));
    if let Some(text) = ir_text {
        print!("{text}");
    }
    if let Some(message) = internal {
        eprintln!("{message}");
        return ExitCode::FAILURE;
    }
    if ok {
        facts(format!("{}: ok", display_safe(path)));
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Lexes, parses, and (if parsing succeeded) type-checks `src`, and renders
/// every diagnostic. Returns the rendered diagnostics in source order,
/// whether none is an error, the IR text `--emit-ir` asked for (if the
/// program has IR), and the message of an internal error (a pass produced
/// invalid IR: a bug, never the program's fault). Runs inside
/// `with_deep_stack`.
fn check_source(
    path: &str,
    src: &str,
    opts: &CheckArgs,
    edition: Edition,
) -> (Vec<String>, bool, Option<String>, Option<String>) {
    let front = frontend::analyze_ir(
        src,
        sky_types::Env::with_prelude(),
        opts.explain,
        opts.emit_ir.is_some(),
        true,
        opts.threads
            .unwrap_or_else(sky_ir::parallel::Config::default_threads),
        edition,
    );
    // Pass the source unchanged: spans are byte offsets into it, and the
    // renderer escapes terminal-unsafe characters itself (spec/diagnostics.md).
    let file = SourceFile::new(path, src);
    let (mut text, mut internal) = (None, None);
    match (&front.ir, &opts.emit_ir) {
        (Some(ir::Outcome::Ready(ready)), Some(stage)) => {
            text = Some(ready.text(stage).to_string());
        }
        (Some(ir::Outcome::Internal(message)), _) => internal = Some(message.clone()),
        _ => {}
    }
    (front.render(&file), front.ok, text, internal)
}
