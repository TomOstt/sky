//! `sky repl`: reads Sky from stdin, one input at a time.
//!
//! An input is one line, or several: a line that ends in `:` (a block
//! header) or leaves a `(` or `[` open continues on the next line, and a
//! block ends at an empty line (see [`needs_continuation`]). Each input is
//! lexed, parsed and checked against the environment the earlier inputs
//! built (`check_module_with`), then run against the runtime bindings they
//! built. A top-level expression prints its value, like Python's REPL.
//!
//! # A failing input changes nothing
//!
//! The session is only updated after an input checked and ran cleanly: the
//! checker environment is replaced after a clean check, the runtime
//! environment is restored from a snapshot if the run fails part way, and
//! the input's text joins the session history only on success. This covers
//! a multi-line input as a whole: a `def` whose last line fails to check
//! defines nothing, and a loop that fails on its fourth pass leaves the
//! bindings its first three passes made undone.
//!
//! # One coordinate system for the whole session
//!
//! The session keeps the text of every accepted input, joined, as its
//! history. An input is parsed as if it were the tail of that history: the
//! history is replaced by blanks of the same byte length (newlines kept)
//! and the input follows. So every span the front end and the interpreter
//! produce is an offset into the whole session, and two inputs never
//! produce the same span. That matters because the interpreter and the
//! checker identify a `def` by the span of its name, and `f` in the first
//! input and `g` in the second would otherwise both be at offset 4.
//! Diagnostics are translated back before they are shown: the span is made
//! relative to the input it points into, so line numbers count from the
//! first line of that input (`<repl>`, or `<repl:N>` for the N-th accepted
//! input when a runtime error is inside a function defined earlier).
//!
//! # Derivatives
//!
//! `derivative(f, x)(args)` runs on SIR (`sky_derive::SirDerivatives`),
//! which lowers a whole program, and one input is only a fragment. So the
//! REPL gives the hook the whole session as a program: the history followed
//! by the current input, parsed as one module. Its spans agree with the
//! ones in the running input (above), so the `def` the interpreter hands
//! over is found in the lowered program. The program is built lazily, on
//! the first derivative an input asks for, and dropped when the input ends,
//! so a session that never differentiates pays nothing; an input that does
//! pays one parse, one check and one lowering of the session so far, as
//! `sky run` pays for a file. As in `sky run`, what the derivative needs
//! must lower to SIR, and if it does not the derivative is a runtime error
//! that says why. Only accepted inputs are in the history, so a rolled-back
//! input never reaches SIR. A `def` that is redefined later in the session
//! is lowered as the program says, in order, so a derivative asked for
//! after the redefinition sees the new body.
//!
//! There is no session saving: v0.1 keeps a session only in memory, so no
//! session file or serialized state exists to load (spec non-negotiable 3).

use std::cell::OnceCell;
use std::collections::HashSet;
use std::io::{self, BufRead, IsTerminal, Read, Write};
use std::process::ExitCode;

use sky_ast::{Module, ParseError};
use sky_derive::SirDerivatives;
use sky_diag::{Diagnostic, Severity, SourceFile, Span};
use sky_interp::{
    run_module_with_derivatives, DerivativeHook, DerivativeRequest, RuntimeEnv, RuntimeError, Value,
};
use sky_syntax::{LexErrorKind, TokenKind, MAX_SOURCE_BYTES};
use sky_types::Env;

use crate::frontend::FrontEnd;

const PROMPT: &str = "sky> ";
/// Shown while an input is unfinished.
const CONTINUATION_PROMPT: &str = "...  ";
/// What diagnostics call the current input.
const NAME: &str = "<repl>";

const HELP: &str = "\
Type Sky. An expression prints its value.
A line that ends in `:` starts a block (def, if, while, for); it continues
on `...` lines and an empty line ends it. An open ( or [ continues too.
  :help   this text
  :quit   leave (so does Ctrl-D)
";

/// What persists between inputs. No `Default`: a session always starts
/// from the preludes (see `repl_loop`).
struct Session {
    types: Env,
    values: RuntimeEnv,
    /// The text of every accepted input, each ending in a newline.
    history: String,
    /// Where each accepted input starts in `history`.
    starts: Vec<usize>,
}

/// Runs the REPL until end of input (Ctrl-D) or `:quit`, exit 0. Exit 2 if
/// stdin cannot be read or an input is longer than the source-size limit.
pub fn repl() -> ExitCode {
    let interactive = io::stdin().is_terminal();
    // One deep stack for the whole session: every input is parsed, checked
    // and run on it.
    sky_ast::with_deep_stack(|| repl_loop(interactive))
}

/// Whether `pending` (whole lines, each ending in a newline) is an
/// unfinished input, so the REPL should read another line before running it.
///
/// - An empty last line ends the input. It is how a block ends, and it also
///   submits a header that has no body (which then reports the error).
/// - Otherwise the input continues if a bracket is still open, if the last
///   thing on the last logical line is a `:` (a block header wants a body),
///   or if the input holds a block (an indented body), which only an empty
///   line ends.
/// - Any other lexical error does not continue: the input runs, and the
///   error is reported, instead of trapping the user behind `...`.
fn needs_continuation(pending: &str) -> bool {
    let lexed = match sky_syntax::lex(pending) {
        Ok(l) => l,
        Err(e) => return e.kind == LexErrorKind::UnclosedBracket,
    };
    if pending
        .lines()
        .next_back()
        .is_none_or(|l| l.trim().is_empty())
    {
        return false;
    }
    let significant = lexed.tokens.iter().rev().find(|t| {
        !matches!(
            t.kind,
            TokenKind::Newline | TokenKind::Indent | TokenKind::Dedent | TokenKind::Eof
        )
    });
    significant.is_some_and(|t| t.kind == TokenKind::Colon)
        || lexed.tokens.iter().any(|t| t.kind == TokenKind::Indent)
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
        history: String::new(),
        starts: Vec::new(),
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
            Ok(0) => {
                // End of input inside an unfinished input: run it, so its
                // error is reported instead of silently dropped.
                if !pending.is_empty() {
                    let src = std::mem::take(&mut pending);
                    eval_input(&mut session, &src, &mut out);
                }
                break;
            }
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
            // A bad line abandons the input it was part of.
            pending.clear();
            continue;
        };
        if pending.is_empty() && line.trim_start().starts_with(':') {
            match line.trim() {
                ":quit" => break,
                ":help" => eprint!("{HELP}"),
                other => eprintln!("error: unknown command `{other}` (try `:help`)"),
            }
            continue;
        }
        pending.push_str(&line);
        if !pending.ends_with('\n') {
            pending.push('\n'); // last line without a newline
        }
        if pending.len() > MAX_SOURCE_BYTES {
            eprintln!("error: input is longer than {MAX_SOURCE_BYTES} bytes");
            return ExitCode::from(2);
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

/// Lexes, parses and checks `src`, the input that follows `session.history`,
/// against the session's checker environment, and gives what
/// `frontend::analyze` gives. Like `frontend::analyze`,
/// except that the input is lexed alone and its tokens are moved to where
/// they are in the session (`analyze` would lex the history again, and
/// report its warnings again), so every span is a session offset.
///
/// `text` is what a span reads: the history, then the input. The history's
/// newlines are spaces there, so a line number the checker computes counts
/// only the lines of this input, like the line numbers of the diagnostics
/// themselves. (Offsets are unchanged: a newline and a space are one byte.)
fn analyze_tail(session: &Session, src: &str, text: &str) -> FrontEnd {
    let base = u32::try_from(session.history.len()).unwrap_or(u32::MAX);
    let shifted = |s: Span| Span::new(s.start.saturating_add(base), s.end.saturating_add(base));
    let shift_diag = |mut d: Diagnostic| {
        d.primary.span = shifted(d.primary.span);
        for l in &mut d.secondary {
            l.span = shifted(l.span);
        }
        d
    };
    let mut module_out = None;
    let mut env_out = session.types.clone();
    let mut solves = HashSet::new();
    let mut diagnostics: Vec<Diagnostic> = match sky_syntax::lex(src) {
        Err(e) => e.diagnostics().cloned().map(shift_diag).collect(),
        Ok(lexed) => {
            let mut diagnostics: Vec<Diagnostic> =
                lexed.warnings.into_iter().map(shift_diag).collect();
            let mut tokens = lexed.tokens;
            for t in &mut tokens {
                t.span = shifted(t.span);
            }
            let (module, errors) = sky_ast::parse_module(&tokens, text);
            if errors.is_empty() {
                let checked = sky_types::check_module_with(&module, text, session.types.clone());
                diagnostics.extend(checked.diagnostics);
                env_out = checked.env;
                solves = checked.cholesky_solves;
                module_out = Some(module);
            } else {
                diagnostics.extend(errors.into_iter().map(ParseError::into_diagnostic));
            }
            diagnostics
        }
    };
    diagnostics.sort_by_key(|d| d.primary.span.start);
    let ok = !diagnostics.iter().any(|d| d.severity == Severity::Error);
    FrontEnd {
        diagnostics,
        module: module_out,
        env: env_out,
        cholesky_solves: solves,
        // The REPL runs each input on the tree-walker; `sky run`'s SIR
        // engine is the only reader of the checker's result here.
        checked: None,
        ir: None,
        ok,
    }
}

/// `d` as it is shown: its spans made relative to the input they point
/// into, which is also the file it is rendered against. The current input
/// starts at `base` in the session; a span before that is in an earlier one.
fn render(d: &Diagnostic, session: &Session, base: usize, src: &str) -> String {
    let at = d.primary.span.start as usize;
    let (name, start, text) = if at >= base {
        (NAME.to_string(), base, src)
    } else {
        let n = session
            .starts
            .partition_point(|s| *s <= at)
            .saturating_sub(1);
        let start = session.starts[n];
        let end = session
            .starts
            .get(n + 1)
            .copied()
            .unwrap_or(session.history.len());
        (
            format!("<repl:{}>", n + 1),
            start,
            &session.history[start..end],
        )
    };
    let inside = |s: Span| s.start as usize >= start && s.end as usize <= start + text.len();
    let shift = |s: Span| {
        // Offsets are bounded by MAX_SOURCE_BYTES, so they fit in u32.
        let by = u32::try_from(start).unwrap_or(u32::MAX);
        Span::new(s.start.saturating_sub(by), s.end.saturating_sub(by))
    };
    let mut d = d.clone();
    d.primary.span = shift(d.primary.span);
    // A label in another input cannot be drawn on this one: leave it out.
    d.secondary.retain(|l| inside(l.span));
    for l in &mut d.secondary {
        l.span = shift(l.span);
    }
    d.render_to_string(&SourceFile::new(name, text))
}

/// Evaluates derivatives for an input by lowering the whole session; see the
/// module docs.
struct SessionDerivatives<'a> {
    /// The history followed by the input: the session as one source text.
    full: &'a str,
    /// The session as one program, built by the first derivative.
    program: &'a OnceCell<Result<Module, String>>,
    sir: Option<SirDerivatives<'a>>,
}

impl DerivativeHook for SessionDerivatives<'_> {
    fn derivative(
        &mut self,
        request: &DerivativeRequest<'_>,
        out: &mut dyn Write,
    ) -> Result<Value, RuntimeError> {
        let (program, full) = (self.program, self.full);
        let built = program.get_or_init(|| {
            let lexed = sky_syntax::lex(full).map_err(|e| e.diagnostic.message.clone())?;
            let (module, errors) = sky_ast::parse_module(&lexed.tokens, full);
            match errors.into_iter().next() {
                Some(e) => Err(e.into_diagnostic().message),
                None => Ok(module),
            }
        });
        let module = match built {
            Ok(module) => module,
            Err(why) => {
                return Err(RuntimeError::new(
                    format!("internal error: cannot rebuild the session as a program: {why}"),
                    request.span,
                ));
            }
        };
        self.sir
            .get_or_insert_with(|| SirDerivatives::new(module, full))
            .derivative(request, out)
    }
}

/// Checks and runs one input, printing diagnostics to stderr and the value
/// of an expression statement to `out`. On any error the session is left
/// as it was.
fn eval_input(session: &mut Session, src: &str, out: &mut impl Write) {
    let base = session.history.len();
    if base + src.len() > MAX_SOURCE_BYTES {
        eprintln!("error: the session is longer than {MAX_SOURCE_BYTES} bytes; start a new one");
        return;
    }
    // The session as one text: what every span in it reads (see the module
    // docs). The checker gets it with the history's newlines flattened.
    let full = format!("{}{src}", session.history);
    let flat = format!("{}{src}", session.history.replace('\n', " "));
    let front = analyze_tail(session, src, &flat);
    let shown: Vec<String> = front
        .diagnostics
        .iter()
        .map(|d| render(d, session, base, src))
        .collect();
    eprint!("{}", shown.join("\n"));
    let module = match front.module {
        Some(m) if front.ok => m,
        _ => return,
    };
    let snapshot = session.values.clone();
    let program = OnceCell::new();
    let mut derivatives = SessionDerivatives {
        full: &full,
        program: &program,
        sir: None,
    };
    let result = run_module_with_derivatives(
        &module,
        &full,
        &mut session.values,
        out,
        &front.cholesky_solves,
        &mut derivatives,
    );
    match result {
        Ok(value) => {
            session.types = front.env;
            session.starts.push(base);
            session.history.push_str(src);
            match value {
                Some(Value::None) | None => {}
                Some(v) => {
                    let _ = writeln!(out, "{v}");
                }
            }
        }
        Err(e) => {
            session.values = snapshot;
            eprint!("{}", render(&e.to_diagnostic(), session, base, src));
        }
    }
    let _ = out.flush();
}

#[cfg(test)]
mod tests {
    use super::needs_continuation;

    #[test]
    fn continuation_rules() {
        // Complete: statements, blank lines, comments, a finished block.
        for done in [
            "x = 1\n",
            "\n",
            "# note\n",
            "x: m\n",
            "if x:\n    y = 1\n\n",
            "if x:\n\n",
            "(1,\n2)\n",
            "x = 1 # a colon:\n",
            "s = \"a:\"\n",
        ] {
            assert!(!needs_continuation(done), "{done:?}");
        }
        // Unfinished: a header, an open block, an open bracket.
        for open in [
            "def f(x: m) -> m:\n",
            "if x:\n",
            "for i in range(3): # go\n",
            "if x:\n    y = 1\n",
            "if x:\n    y = 1\nelse:\n",
            "(1,\n",
            "v = [1,\n\n",
            "f(1,\n  g(2,\n",
        ] {
            assert!(needs_continuation(open), "{open:?}");
        }
        // A lexical error other than an open bracket does not wait for more.
        assert!(!needs_continuation("x = 1 $\n"));
        assert!(!needs_continuation("x = )\n"));
    }
}
