//! `sky run <file>`: check the file, and only if it has no errors, run it.
//!
//! # Two engines
//!
//! A checked program runs on one of two engines that must behave the same
//! byte for byte (stdout, diagnostics, exit status):
//!
//! * SIR: lower to SIR, run the default optimization pipeline with
//!   verification, and execute on the SIR reference interpreter
//!   (`sky_ir_interp`). `docs/design/ir.md` plans for it to become the
//!   executor; it is not the default yet because, measured with
//!   `sky-bench`, it is slower than the tree-walker today (about 1.6× on a
//!   scalar loop, 2.5× on calls) and gives nothing a user can see in
//!   return. The default flips when SIR is the faster engine (roadmap
//!   Phase 4).
//! * the tree-walking interpreter over the AST (`sky_interp`): the default,
//!   the oracle the differential tests compare SIR against, and the fallback
//!   for the programs SIR cannot express yet.
//!
//! `--engine=auto` picks SIR and falls back to the tree-walker
//! on a `LowerError`, silently: the program is valid, it is only that one
//! engine cannot run it yet. `--engine=ast` and `--engine=sir` force one, for
//! debugging and differential use; a forced SIR run of a program that does
//! not lower is an error, so a fallback can never hide in a differential
//! comparison.

use std::collections::HashSet;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::process::ExitCode;

use sky_derive::SirDerivatives;
use sky_diag::{Diagnostic, SourceFile, Span};
use sky_interp::{run_module_with_derivatives, RuntimeEnv};
use sky_project::SystemIdentity;
use sky_syntax::Edition;

use crate::frontend::analyze;
use crate::ir::{prepare_run, Runnable};

/// Which engine `sky run` executes on (`--engine=`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Engine {
    /// SIR, falling back to the tree-walker when the program does not lower.
    Auto,
    /// The AST tree-walker: the default until SIR is faster (see the module
    /// docs).
    #[default]
    Ast,
    /// SIR only: a program that does not lower is an error.
    Sir,
}

impl Engine {
    /// Parses the text after `--engine=`.
    pub fn parse(text: &str) -> Option<Engine> {
        match text {
            "auto" => Some(Engine::Auto),
            "ast" => Some(Engine::Ast),
            "sir" => Some(Engine::Sir),
            _ => None,
        }
    }
}

/// Refuses to run a file with any error: its diagnostics go to stderr and
/// nothing executes, so a `print` before a type error prints nothing.
/// Warnings are shown and the run continues. Program output goes to stdout,
/// diagnostics (including a runtime error, with the span of the failing
/// expression) to stderr. Exit 0 on success, 1 on any error.
pub fn run(path: &str, file: &File, src: &str, engine: Engine) -> ExitCode {
    // Same discovery as `sky check` (spec non-negotiable 2): the project
    // root comes only from walking up from the opened file.
    let discovery = match sky_project::discover_opened(Path::new(path), file, &SystemIdentity) {
        Ok(d) => d,
        Err(e) => {
            eprint!("{}", e.to_diagnostic().display(&e.source_file()));
            return ExitCode::FAILURE;
        }
    };
    // Front end and interpreter both run on the known-size stack, so the
    // parser's, checker's and evaluator's depth bounds hold.
    sky_ast::with_deep_stack(|| run_source(path, src, discovery.edition, engine))
}

/// Forwards to stdout and remembers whether the reader went away (`sky run
/// x.sky | head`), which is not an error worth a diagnostic.
struct Stdout<'a> {
    inner: io::StdoutLock<'a>,
    closed: bool,
}

impl Write for Stdout<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf).inspect_err(|e| {
            self.closed |= e.kind() == io::ErrorKind::BrokenPipe;
        })
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn run_source(path: &str, src: &str, edition: Edition, engine: Engine) -> ExitCode {
    let file = SourceFile::new(path, src);
    let front = analyze(src, sky_types::Env::with_prelude(), false, edition);
    let rendered = front.render(&file);
    eprint!("{}", rendered.join("\n"));
    let (module, checked) = match (&front.module, &front.checked) {
        (Some(m), Some(c)) if front.ok => (m, c),
        _ => return ExitCode::FAILURE,
    };
    // Lowering and the pipeline run before anything executes, so a program
    // that falls back has printed nothing yet and the tree-walker starts
    // clean. `--engine=ast` never lowers, as before.
    let sir = match engine {
        Engine::Ast => None,
        Engine::Auto | Engine::Sir => match prepare_run(module, src, checked) {
            Runnable::Ready(ir) => Some(ir),
            Runnable::Unsupported(e) if engine == Engine::Sir => {
                let d = Diagnostic::error(format!("cannot run on SIR: {}", e.message), e.span)
                    .with_note("`--engine=sir` runs only programs that lower to SIR".to_string());
                let d = match e.note {
                    Some(n) => d.with_note(n),
                    None => d,
                };
                return fail_with(&d, &rendered, &file);
            }
            Runnable::Unsupported(_) => None,
            Runnable::Internal(message) => {
                eprintln!("{message}");
                return ExitCode::FAILURE;
            }
        },
    };
    let stdout = io::stdout();
    let mut out = Stdout {
        inner: stdout.lock(),
        closed: false,
    };
    let result = match sir {
        // An operation's operands carry the spans of the source they were
        // read from, so SIR's error is worded and labelled exactly as the
        // tree-walker's, with nothing to replay.
        Some(ir) => sky_ir_interp::run(&ir, &mut out).map_err(|e| e.to_diagnostic()),
        None => {
            tree_walk(module, src, &front.cholesky_solves, &mut out).map_err(|e| e.to_diagnostic())
        }
    };
    let _ = out.flush();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(_) if out.closed => ExitCode::FAILURE,
        Err(d) => fail_with(&d, &rendered, &file),
    }
}

/// Runs the checked module on the tree-walking interpreter.
fn tree_walk(
    module: &sky_ast::Module,
    src: &str,
    solves: &HashSet<Span>,
    out: &mut impl Write,
) -> Result<(), sky_interp::RuntimeError> {
    let mut env = RuntimeEnv::with_prelude();
    // `derivative(f, x)(args)` runs on SIR (see `sky_derive`); the program
    // is only lowered if it asks for one.
    let mut derivatives = SirDerivatives::new(module, src);
    run_module_with_derivatives(module, src, &mut env, out, solves, &mut derivatives).map(|_| ())
}

/// Prints a failure's diagnostic to stderr, after the warnings (with a blank
/// line between) if there were any. Exit status 1.
fn fail_with(d: &Diagnostic, warnings: &[String], file: &SourceFile) -> ExitCode {
    if !warnings.is_empty() {
        eprintln!(); // blank line after the warnings
    }
    eprint!("{}", d.render_to_string(file));
    ExitCode::FAILURE
}
