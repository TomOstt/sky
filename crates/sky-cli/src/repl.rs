//! `sky repl`: reads Sky from stdin, one input at a time.
//!
//! Each input is lexed, parsed and checked against the environment the
//! earlier inputs built (`check_module_with`), then run against the runtime
//! bindings they built. A top-level expression prints its value, like
//! Python's REPL. An input with any error changes nothing: the checker
//! environment is only replaced after a clean check, and the runtime
//! environment is restored from a snapshot if the run fails.
//!
//! There is no session saving: v0.1 keeps a session only in memory, so no
//! session file or serialized state exists to load (spec non-negotiable 3).

use std::io::{self, BufRead, IsTerminal, Read, Write};
use std::process::ExitCode;

use sky_diag::SourceFile;
use sky_interp::{run_module_with, RuntimeEnv, Value};
use sky_syntax::MAX_SOURCE_BYTES;
use sky_types::Env;

use crate::frontend::analyze;

const PROMPT: &str = "sky> ";
/// Shown while an input is unfinished (no construct needs it yet).
const CONTINUATION_PROMPT: &str = "...  ";
/// What diagnostics call the input.
const NAME: &str = "<repl>";

/// What persists between inputs. No `Default`: a session always starts
/// from the preludes (see `repl_loop`).
struct Session {
    types: Env,
    values: RuntimeEnv,
}

/// Runs the REPL until end of input (Ctrl-D), exit 0. Exit 2 if stdin
/// cannot be read or an input is longer than the source-size limit.
pub fn repl() -> ExitCode {
    let interactive = io::stdin().is_terminal();
    // One deep stack for the whole session: every input is parsed, checked
    // and run on it.
    sky_ast::with_deep_stack(|| repl_loop(interactive))
}

/// Whether `pending` is an unfinished input, so the REPL should read
/// another line before running it. Everything is one line today; the block
/// syntax (`def`, `if`, `while`) will make an open block return true here.
fn needs_continuation(_pending: &str) -> bool {
    false
}

fn repl_loop(interactive: bool) -> ExitCode {
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    // Start where `sky check` and `sky run` start: the physical constants
    // are in scope for the checker and have values at run time.
    let mut session = Session {
        types: Env::with_prelude(),
        values: RuntimeEnv::with_prelude(),
    };
    let mut pending = String::new();
    loop {
        // The prompt goes to stderr and only to a terminal, so piped
        // output stays clean.
        if interactive {
            let p = if pending.is_empty() {
                PROMPT
            } else {
                CONTINUATION_PROMPT
            };
            eprint!("{p}");
            let _ = io::stderr().flush();
        }
        // Bounded read: a line longer than a source file may be is refused
        // instead of buffered without limit.
        let mut buf = Vec::new();
        let limit = (MAX_SOURCE_BYTES as u64).saturating_add(1);
        match (&mut input).take(limit).read_until(b'\n', &mut buf) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                eprintln!("error: cannot read input: {e}");
                return ExitCode::from(2);
            }
        }
        if buf.len() > MAX_SOURCE_BYTES {
            eprintln!("error: input line is longer than {MAX_SOURCE_BYTES} bytes");
            return ExitCode::from(2);
        }
        let Ok(line) = String::from_utf8(buf) else {
            eprintln!("error: input is not valid UTF-8");
            continue;
        };
        pending.push_str(&line);
        if !pending.ends_with('\n') {
            pending.push('\n'); // last line without a newline
        }
        if needs_continuation(&pending) {
            continue;
        }
        let src = std::mem::take(&mut pending);
        eval_input(&mut session, &src, &mut out);
    }
    if interactive {
        eprintln!(); // Ctrl-D leaves the cursor after the prompt
    }
    ExitCode::SUCCESS
}

/// Checks and runs one input, printing diagnostics to stderr and the value
/// of an expression statement to `out`. On any error the session is left
/// as it was.
fn eval_input(session: &mut Session, src: &str, out: &mut impl Write) {
    let file = SourceFile::new(NAME, src);
    let front = analyze(src, session.types.clone(), false);
    eprint!("{}", front.render(&file).join("\n"));
    let module = match front.module {
        Some(m) if front.ok => m,
        _ => return,
    };
    let snapshot = session.values.clone();
    match run_module_with(
        &module,
        src,
        &mut session.values,
        out,
        &front.cholesky_solves,
    ) {
        Ok(value) => {
            session.types = front.env;
            match value {
                Some(Value::None) | None => {}
                Some(v) => {
                    let _ = writeln!(out, "{v}");
                }
            }
        }
        Err(e) => {
            session.values = snapshot;
            eprint!("{}", e.to_diagnostic().render_to_string(&file));
        }
    }
    let _ = out.flush();
}
