//! `sky run <file>`: check the file, and only if it has no errors, run it.

use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::process::ExitCode;

use sky_diag::SourceFile;
use sky_interp::{run_module_with, RuntimeEnv};
use sky_project::SystemIdentity;

use crate::frontend::analyze;

/// Refuses to run a file with any error: its diagnostics go to stderr and
/// nothing executes, so a `print` before a type error prints nothing.
/// Warnings are shown and the run continues. Program output goes to stdout,
/// diagnostics (including a runtime error, with the span of the failing
/// expression) to stderr. Exit 0 on success, 1 on any error.
pub fn run(path: &str, file: &File, src: &str) -> ExitCode {
    // Same discovery as `sky check` (spec non-negotiable 2): the project
    // root comes only from walking up from the opened file.
    if let Err(e) = sky_project::discover_opened(Path::new(path), file, &SystemIdentity) {
        eprint!("{}", e.to_diagnostic().display(&e.source_file()));
        return ExitCode::FAILURE;
    }
    // Front end and interpreter both run on the known-size stack, so the
    // parser's, checker's and evaluator's depth bounds hold.
    sky_ast::with_deep_stack(|| run_source(path, src))
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

fn run_source(path: &str, src: &str) -> ExitCode {
    let file = SourceFile::new(path, src);
    let front = analyze(src, sky_types::Env::with_prelude(), false);
    let rendered = front.render(&file);
    eprint!("{}", rendered.join("\n"));
    let module = match front.module {
        Some(m) if front.ok => m,
        _ => return ExitCode::FAILURE,
    };
    let stdout = io::stdout();
    let mut out = Stdout {
        inner: stdout.lock(),
        closed: false,
    };
    let mut env = RuntimeEnv::with_prelude();
    let result = run_module_with(&module, src, &mut env, &mut out, &front.cholesky_solves);
    let _ = out.flush();
    match result {
        Ok(_) => ExitCode::SUCCESS,
        Err(_) if out.closed => ExitCode::FAILURE,
        Err(e) => {
            if !rendered.is_empty() {
                eprintln!(); // blank line after the warnings
            }
            eprint!("{}", e.to_diagnostic().render_to_string(&file));
            ExitCode::FAILURE
        }
    }
}
