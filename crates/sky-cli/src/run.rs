//! `sky run <file>`: check the file, and only if it has no errors, run it.
//!
//! # Three engines
//!
//! A checked program runs on one of three engines that must behave the same
//! byte for byte (stdout, diagnostics, exit status):
//!
//! * native: the SIR compiled to machine code through Cranelift
//!   (`sky_jit`), whole module, in memory. It runs the optimized SIR of the
//!   `sir` engine and is exact against it. It refuses a program with
//!   something it cannot compile (`CompileError`), which `--engine=auto`
//!   answers by falling back.
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
//! `--threads=N` (or `SKY_THREADS`) sets how many threads the SIR engine may
//! use for work that cannot change the output (`sky_ir::parallel`: big
//! elementwise kernels and independent pure calls); 1 means sequential, and
//! the output is the same for every N. The tree-walker is always sequential.
//!
//! `--engine=auto` picks native code and falls back to SIR when the program
//! does not compile natively, and to the tree-walker on a `LowerError`,
//! silently: the program is valid, it is only that one engine cannot run it
//! yet. `--engine=ast`, `--engine=sir` and `--engine=native` force one, for
//! debugging and differential use; a forced SIR or native run of a program
//! that does not lower (or compile) is an error, so a fallback can never hide
//! in a differential comparison.

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
    /// Native code, falling back to SIR when the program does not compile
    /// natively, and to the tree-walker when it does not lower.
    Auto,
    /// The AST tree-walker: the default until SIR is faster (see the module
    /// docs).
    #[default]
    Ast,
    /// SIR only: a program that does not lower is an error.
    Sir,
    /// Native code only: a program that does not lower or compile is an
    /// error.
    Native,
}

impl Engine {
    /// Parses the text after `--engine=`.
    pub fn parse(text: &str) -> Option<Engine> {
        match text {
            "auto" => Some(Engine::Auto),
            "ast" => Some(Engine::Ast),
            "sir" => Some(Engine::Sir),
            "native" => Some(Engine::Native),
            _ => None,
        }
    }
}

/// Refuses to run a file with any error: its diagnostics go to stderr and
/// nothing executes, so a `print` before a type error prints nothing.
/// Warnings are shown and the run continues. Program output goes to stdout,
/// diagnostics (including a runtime error, with the span of the failing
/// expression) to stderr. Exit 0 on success, 1 on any error.
pub fn run(path: &str, file: &File, src: &str, engine: Engine, threads: Option<usize>) -> ExitCode {
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
    sky_ast::with_deep_stack(|| run_source(path, src, discovery.edition, engine, threads))
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

fn run_source(
    path: &str,
    src: &str,
    edition: Edition,
    engine: Engine,
    threads: Option<usize>,
) -> ExitCode {
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
        Engine::Auto | Engine::Sir | Engine::Native => match prepare_run(module, src, checked) {
            Runnable::Ready(ir) => Some(ir),
            Runnable::Unsupported(e) if engine != Engine::Auto => {
                let (what, flag) = if engine == Engine::Sir {
                    ("on SIR", "sir")
                } else {
                    ("natively", "native")
                };
                let d = Diagnostic::error(format!("cannot run {what}: {}", e.message), e.span)
                    .with_note(format!(
                        "`--engine={flag}` runs only programs that lower to SIR"
                    ));
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
    // Native code is compiled before anything runs, for the same reason: a
    // program it refuses has printed nothing yet.
    let native = match (&sir, engine) {
        (Some(ir), Engine::Native | Engine::Auto) => match sky_jit::compile(ir) {
            Ok(c) => Some(c),
            Err(e) if engine == Engine::Native => {
                let span = e.span.unwrap_or(module.span);
                let d = Diagnostic::error(format!("cannot run natively: {}", e.message), span)
                    .with_note(
                        "`--engine=native` runs only programs that compile to machine code"
                            .to_string(),
                    );
                return fail_with(&d, &rendered, &file);
            }
            Err(_) => None,
        },
        _ => None,
    };
    let stdout = io::stdout();
    let mut out = Stdout {
        inner: stdout.lock(),
        closed: false,
    };
    let result = match (native, sir) {
        // A value the checker could not type that is not what a declaration
        // says stops native code; SIR runs the program instead, from the
        // start, skipping the output already written.
        (Some(code), sir) => {
            let rerun = |w: &mut dyn Write| match &sir {
                Some(ir) => sky_ir_interp::run_with(ir, &mut { w }, &par_config(threads)),
                None => Ok(()),
            };
            code.run_or_rerun(&mut out, rerun)
                .0
                .map_err(|e| e.to_diagnostic())
        }
        // An operation's operands carry the spans of the source they were
        // read from, so SIR's error is worded and labelled exactly as the
        // tree-walker's, with nothing to replay.
        (None, Some(ir)) => {
            let cfg = par_config(threads);
            sky_ir_interp::run_with(&ir, &mut out, &cfg).map_err(|e| e.to_diagnostic())
        }
        (None, None) => tree_walk(module, src, &front.cholesky_solves, checked, &mut out)
            .map_err(|e| e.to_diagnostic()),
    };
    let _ = out.flush();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(_) if out.closed => ExitCode::FAILURE,
        Err(d) => fail_with(&d, &rendered, &file),
    }
}

/// The SIR interpreter's thread configuration for `--threads` (default: the
/// environment's, else the machine's cores).
fn par_config(threads: Option<usize>) -> sky_ir_interp::ParConfig {
    sky_ir_interp::ParConfig::with_threads(
        threads.unwrap_or_else(sky_ir_interp::ParConfig::default_threads),
    )
}

/// Runs the checked module on the tree-walking interpreter.
fn tree_walk(
    module: &sky_ast::Module,
    src: &str,
    solves: &HashSet<Span>,
    checked: &sky_types::Checked,
    out: &mut impl Write,
) -> Result<(), sky_interp::RuntimeError> {
    let mut env = RuntimeEnv::with_prelude();
    // The chains the checker planned (order, solves): the tree-walker runs
    // them as the plan says, as SIR does.
    env.add_chain_plans(&checked.chains);
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
