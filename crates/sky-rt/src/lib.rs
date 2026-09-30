//! The runtime library of an executable built by `sky build`.
//!
//! An executable is an object file (`sky_jit::aot`) linked with this static
//! library. The object's `main` calls [`sky_rt_start`] with the program's
//! *image* and the address of the compiled `<main>`; the `extern "C"`
//! functions the machine code calls (`sky_rt_*`) are `sky-jit`'s
//! `#[no_mangle]` functions, which this library carries because it depends
//! on that crate.
//!
//! [`start`] does what `sky run --engine=native` does after compiling, and
//! prints what it prints, byte for byte:
//!
//! 1. the front end's warnings, to stderr (they were rendered when the
//!    program was built and are stored in the [`Meta`]);
//! 2. the run, on a thread with the deep stack the interpreters' budgets
//!    assume, `print` output to stdout;
//! 3. if a value of unknown type turns out not to be what a declaration says
//!    (deoptimization), the program is run again from the start on the SIR
//!    interpreter, skipping the output already written;
//! 4. on a run-time error, the diagnostic rendered against the embedded
//!    source text (the same text `sky run` reads), to stderr;
//! 5. the exit status: 0, or 1 after an error.
//!
//! # Unsafe
//!
//! Three operations: reading the image the generated `main` points to,
//! calling the compiled `<main>` (both through the caller's promise), and
//! ignoring `SIGPIPE`. Each is commented where it is made.

#![deny(unsafe_code)]

use std::io::{self, Write};

use sky_diag::{Diagnostic, SourceFile};
use sky_jit::aot::Image;
use sky_jit::{RunFailure, SkipWriter};

/// What `sky build` stores in the image next to the SIR, so that a built
/// program reports exactly what `sky run` reports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Meta {
    /// The path the program was built from, as given to `sky build` (the name
    /// diagnostics show).
    pub path: String,
    /// The program's source text.
    pub source: String,
    /// The front end's rendered warnings, joined as `sky run` prints them.
    pub warnings: String,
    /// Whether there were any (a blank line separates them from a later
    /// error).
    pub has_warnings: bool,
}

impl Meta {
    /// The bytes for [`sky_jit::aot::compile_object`]'s `extra`.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = vec![u8::from(self.has_warnings)];
        for s in [&self.path, &self.source, &self.warnings] {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        out
    }

    /// The inverse of [`Meta::encode`]; `None` for bytes it did not make.
    pub fn decode(bytes: &[u8]) -> Option<Meta> {
        let (&flag, mut rest) = bytes.split_first()?;
        let mut take = || -> Option<String> {
            let (len, tail) = rest.split_at_checked(8)?;
            let len = usize::try_from(u64::from_le_bytes(len.try_into().ok()?)).ok()?;
            let (text, tail) = tail.split_at_checked(len)?;
            rest = tail;
            String::from_utf8(text.to_vec()).ok()
        };
        let path = take()?;
        let source = take()?;
        let warnings = take()?;
        Some(Meta {
            path,
            source,
            warnings,
            has_warnings: flag != 0,
        })
    }
}

/// Forwards to stdout and remembers whether the reader went away (`prog |
/// head`), which is not an error worth a diagnostic (`sky run` does the
/// same).
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

/// Runs a program from its image and returns the process's exit status.
///
/// # Safety
///
/// `entry` must be the address of the compiled `<main>` of the module in
/// `image` (what the generated `main` passes).
#[allow(unsafe_code)]
pub unsafe fn start(image: &[u8], entry: usize) -> i32 {
    let image = match Image::decode(image) {
        Ok(i) => i,
        Err(message) => {
            eprintln!("{message}");
            return 2;
        }
    };
    let Some(meta) = Meta::decode(&image.extra) else {
        eprintln!("the program image is damaged or from another version of Sky");
        return 2;
    };
    // Same as `sky run`: warnings first.
    eprint!("{}", meta.warnings);
    let file = SourceFile::new(meta.path.as_str(), meta.source.as_str());
    // SAFETY: the caller's promise about `entry` is `Image::run`'s.
    let result = sky_diag::with_deep_stack(|| unsafe { run(&image, entry) });
    match result {
        Ok(()) => 0,
        // The reader went away: not an error worth a diagnostic.
        Err(Failure::Closed) => 1,
        Err(Failure::Diagnostic(d)) => {
            if meta.has_warnings {
                eprintln!();
            }
            eprint!("{}", d.render_to_string(&file));
            1
        }
    }
}

enum Failure {
    Closed,
    Diagnostic(Box<Diagnostic>),
}

/// # Safety
///
/// As [`start`].
#[allow(unsafe_code)]
unsafe fn run(image: &Image, entry: usize) -> Result<(), Failure> {
    let stdout = io::stdout();
    let mut out = Stdout {
        inner: stdout.lock(),
        closed: false,
    };
    // SAFETY: the caller's promise about `entry` is `Image::run`'s.
    let native = unsafe { image.run(entry, &mut out) };
    let result = match native {
        Ok(()) => Ok(()),
        Err(RunFailure::Error(e)) => Err(e.to_diagnostic()),
        // A value the checker could not type was not what a declaration says
        // and native code stopped: the SIR interpreter runs the program
        // instead, from the start, skipping the output already written.
        Err(RunFailure::Deopt { written }) => {
            let threads = sky_ir_interp::ParConfig::default_threads();
            let cfg = sky_ir_interp::ParConfig::with_threads(threads);
            let mut skip = SkipWriter::new(&mut out, written);
            sky_ir_interp::run_with(image.module(), &mut skip, &cfg).map_err(|e| e.to_diagnostic())
        }
    };
    let _ = out.flush();
    match result {
        Ok(()) => Ok(()),
        Err(_) if out.closed => Err(Failure::Closed),
        Err(d) => Err(Failure::Diagnostic(Box::new(d))),
    }
}

/// The entry point the generated `main` calls: the process's exit status.
///
/// # Safety
///
/// `image` and `image_len` describe the program's image data, and `entry` is
/// the address of its compiled `<main>` (all three are what `sky build`'s
/// generated `main` passes).
#[allow(unsafe_code)]
#[no_mangle]
pub unsafe extern "C" fn sky_rt_start(image: *const u8, image_len: usize, entry: usize) -> i32 {
    ignore_sigpipe();
    // SAFETY: the caller passes the address and length of the image data.
    let bytes = unsafe { std::slice::from_raw_parts(image, image_len) };
    // SAFETY: the caller passes `<main>`'s address.
    unsafe { start(bytes, entry) }
}

/// Rust's own start-up ignores `SIGPIPE`, so a write to a closed pipe is an
/// error (`EPIPE`) rather than a signal; a C `main` has to do it itself for
/// the executable to end the way `sky run` does.
#[cfg(unix)]
#[allow(unsafe_code)]
fn ignore_sigpipe() {
    extern "C" {
        fn signal(signum: i32, handler: usize) -> usize;
    }
    const SIGPIPE: i32 = 13;
    const SIG_IGN: usize = 1;
    // SAFETY: `signal` is the C library's, called with a valid signal number
    // and the ignore disposition; nothing else in the process handles it.
    unsafe { signal(SIGPIPE, SIG_IGN) };
}

#[cfg(not(unix))]
fn ignore_sigpipe() {}
