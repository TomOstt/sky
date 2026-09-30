//! `sky build <file>`: compile a program ahead of time to a standalone native
//! executable (or, with `--emit=obj`, to an object file).
//!
//! The front end is `sky run`'s: the same discovery, checking, lowering and
//! optimization, the same diagnostics for a program with errors (nothing is
//! written), and the same refusal (`cannot build: ...`) for a program that
//! does not lower or compile to machine code. Then `sky_jit::aot` translates
//! the optimized SIR to an object file, and the system linker (`cc`) links it
//! with the runtime library (`libsky_rt.a`, the `sky-rt` crate) into the
//! executable.
//!
//! The executable behaves like `sky run --engine=native` on the same file:
//! the same stdout, the same diagnostics on stderr, the same exit status. It
//! carries the source text, the front end's warnings and the SIR (see
//! `sky_jit::aot` for why), so it needs nothing from the machine that built
//! it but the C library.
//!
//! # Finding the runtime library
//!
//! In this order: `--runtime=<path>`, the `SKY_RT_LIB` environment variable,
//! and `libsky_rt.a` in the directory of the `sky` executable (where `cargo
//! build --workspace` puts both, and where an installation keeps them
//! together).

use std::fs::{self, File};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use sky_diag::{display_safe, Diagnostic, SourceFile};
use sky_jit::aot::{self, TargetInfo};
use sky_project::SystemIdentity;
use sky_rt::Meta;
use sky_syntax::Edition;

use crate::frontend::analyze;
use crate::ir::{prepare_run, Runnable};

/// What `sky build` writes (`--emit=`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Emit {
    /// A linked executable: the default.
    #[default]
    Exe,
    /// The object file, unlinked.
    Obj,
}

/// The arguments of `sky build`.
#[derive(Debug, PartialEq, Eq)]
pub struct BuildArgs {
    pub path: String,
    pub emit: Emit,
    /// `-o <path>`.
    pub output: Option<String>,
    /// `--target=<triple>`: the machine to build for (default: this one).
    pub target: Option<String>,
    /// `--runtime=<path>`: the runtime library to link.
    pub runtime: Option<String>,
}

impl BuildArgs {
    /// Parses `sky build`'s arguments. `Err` is a usage error: the message,
    /// if it needs one beyond the usage text.
    pub fn parse(args: &[String]) -> Result<BuildArgs, Option<String>> {
        let mut out = BuildArgs {
            path: String::new(),
            emit: Emit::Exe,
            output: None,
            target: None,
            runtime: None,
        };
        let (mut path, mut emit) = (None, None);
        let mut it = args.iter();
        while let Some(arg) = it.next() {
            if arg == "-o" {
                let value = it.next().ok_or_else(|| Some("`-o` needs a path".into()))?;
                if out.output.replace(value.clone()).is_some() {
                    return Err(None);
                }
            } else if let Some(what) = arg.strip_prefix("--emit=") {
                let e = match what {
                    "exe" => Emit::Exe,
                    "obj" => Emit::Obj,
                    _ => {
                        return Err(Some(format!(
                            "unknown --emit target `{}` (use `exe` or `obj`)",
                            display_safe(what)
                        )))
                    }
                };
                if emit.replace(e).is_some() {
                    return Err(None);
                }
            } else if let Some(t) = arg.strip_prefix("--target=") {
                if out.target.replace(t.to_string()).is_some() {
                    return Err(None);
                }
            } else if let Some(p) = arg.strip_prefix("--runtime=") {
                if out.runtime.replace(p.to_string()).is_some() {
                    return Err(None);
                }
            } else if arg.starts_with('-') || path.replace(arg.clone()).is_some() {
                return Err(None);
            }
        }
        out.path = path.ok_or(None)?;
        out.emit = emit.unwrap_or_default();
        Ok(out)
    }

    /// Where the output goes: `-o`, else the file's name without `.sky` (and
    /// `.o` for an object) in the current directory.
    fn output_path(&self) -> PathBuf {
        if let Some(o) = &self.output {
            return PathBuf::from(o);
        }
        let stem = Path::new(&self.path)
            .file_stem()
            .map_or_else(|| "a".into(), |s| s.to_os_string());
        let mut p = PathBuf::from(stem);
        if self.emit == Emit::Obj {
            p.set_extension("o");
        }
        p
    }
}

/// `sky build`: see the module docs. Exit status 0 when the output is
/// written, 1 when the program has errors or cannot be built (nothing is
/// written), like `sky run`.
pub fn build(args: &BuildArgs, file: &File, src: &str) -> ExitCode {
    let path = args.path.as_str();
    let discovery = match sky_project::discover_opened(Path::new(path), file, &SystemIdentity) {
        Ok(d) => d,
        Err(e) => {
            eprint!("{}", e.to_diagnostic().display(&e.source_file()));
            return ExitCode::FAILURE;
        }
    };
    sky_ast::with_deep_stack(|| build_source(args, src, discovery.edition))
}

fn build_source(args: &BuildArgs, src: &str, edition: Edition) -> ExitCode {
    let path = args.path.as_str();
    let file = SourceFile::new(path, src);
    let front = analyze(src, sky_types::Env::with_prelude(), false, edition);
    let rendered = front.render(&file);
    // Warnings are shown now, and again by every run of the executable.
    eprint!("{}", rendered.join("\n"));
    let (module, checked) = match (&front.module, &front.checked) {
        (Some(m), Some(c)) if front.ok => (m, c),
        _ => return ExitCode::FAILURE,
    };
    let refuse = |d: Diagnostic| {
        if !rendered.is_empty() {
            eprintln!();
        }
        eprint!("{}", d.render_to_string(&file));
        ExitCode::FAILURE
    };
    let ir = match prepare_run(module, src, checked) {
        Runnable::Ready(ir) => ir,
        Runnable::Unsupported(e) => {
            let d = Diagnostic::error(format!("cannot build: {}", e.message), e.span)
                .with_note("`sky build` compiles only programs that lower to SIR".to_string());
            return refuse(match e.note {
                Some(n) => d.with_note(n),
                None => d,
            });
        }
        Runnable::Internal(message) => {
            eprintln!("{message}");
            return ExitCode::FAILURE;
        }
    };
    let meta = Meta {
        path: path.to_string(),
        source: src.to_string(),
        warnings: rendered.join("\n"),
        has_warnings: !rendered.is_empty(),
    };
    let object = match aot::compile_object(&ir, args.target.as_deref(), &meta.encode()) {
        Ok(o) => o,
        Err(e) => {
            let span = e.span.unwrap_or(module.span);
            let d = Diagnostic::error(format!("cannot build: {}", e.message), span).with_note(
                "`sky build` compiles only programs that compile to machine code".to_string(),
            );
            return refuse(d);
        }
    };
    let out = args.output_path();
    let result = match args.emit {
        Emit::Obj => write_file(&out, &object.bytes),
        Emit::Exe => link_executable(args, &object, &out),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut f = File::create(path).map_err(|e| {
        format!(
            "cannot write `{}`: {e}",
            display_safe(&path.display().to_string())
        )
    })?;
    f.write_all(bytes).map_err(|e| {
        format!(
            "cannot write `{}`: {e}",
            display_safe(&path.display().to_string())
        )
    })
}

/// The runtime library to link (see the module docs).
fn runtime_library(args: &BuildArgs) -> Result<PathBuf, String> {
    let named = args
        .runtime
        .clone()
        .or_else(|| std::env::var("SKY_RT_LIB").ok().filter(|s| !s.is_empty()));
    if let Some(p) = named {
        let p = PathBuf::from(p);
        return if p.is_file() {
            Ok(p)
        } else {
            Err(format!(
                "the runtime library `{}` does not exist",
                display_safe(&p.display().to_string())
            ))
        };
    }
    let beside = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|d| d.join("libsky_rt.a")));
    match beside {
        Some(p) if p.is_file() => Ok(p),
        _ => Err(
            "the runtime library `libsky_rt.a` was not found next to the `sky` executable; \
                  build it with `cargo build -p sky-rt` or name it with `--runtime=<path>` \
                  (or `SKY_RT_LIB`)"
                .to_string(),
        ),
    }
}

/// Links `object` with the runtime library into the executable `out`.
fn link_executable(args: &BuildArgs, object: &aot::Object, out: &Path) -> Result<(), String> {
    let host = aot::host_target()?;
    let rt = runtime_library(args)?;
    let target = &object.target;
    let flags = linker_flags(target, &host)?;

    // A private directory for the object file the linker reads.
    let dir = std::env::temp_dir().join(format!(
        "sky-build-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));
    make_private_dir(&dir).map_err(|e| format!("cannot create a temporary directory: {e}"))?;
    let obj_path = dir.join("program.o");
    let result = write_file(&obj_path, &object.bytes).and_then(|()| {
        let cc = std::env::var("CC")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "cc".to_string());
        let mut cmd = Command::new(&cc);
        cmd.args(&flags.pre)
            .arg(&obj_path)
            .arg(&rt)
            .arg("-o")
            .arg(out)
            .args(&flags.post);
        let output = cmd.output().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                format!(
                    "no C linker found (looked for `{}`); install one, set `CC`, or use \
                     `--emit=obj` and link the object yourself",
                    display_safe(&cc)
                )
            } else {
                format!("cannot run the linker `{}`: {e}", display_safe(&cc))
            }
        })?;
        if output.status.success() {
            Ok(())
        } else {
            Err(format!(
                "the linker `{}` failed ({})\n{}",
                display_safe(&cc),
                output.status,
                display_safe(&String::from_utf8_lossy(&output.stderr))
            ))
        }
    });
    let _ = fs::remove_dir_all(&dir);
    result
}

#[cfg(unix)]
fn make_private_dir(dir: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    fs::DirBuilder::new().mode(0o700).create(dir)
}

#[cfg(not(unix))]
fn make_private_dir(dir: &Path) -> std::io::Result<()> {
    fs::create_dir(dir)
}

/// The linker arguments a target needs: before the inputs and after them.
struct LinkerFlags {
    pre: Vec<String>,
    post: Vec<String>,
}

fn linker_flags(target: &TargetInfo, host: &TargetInfo) -> Result<LinkerFlags, String> {
    let cross_arch = target.arch != host.arch;
    match (target.os.as_str(), host.os.as_str()) {
        ("darwin" | "macosx", "darwin" | "macosx") => {
            let mut pre = Vec::new();
            if cross_arch {
                let arch = match target.arch.as_str() {
                    "aarch64" => "arm64",
                    other => other,
                };
                pre.extend(["-arch".to_string(), arch.to_string()]);
            }
            // Unused runtime code and local symbols are dropped: a sixth off.
            pre.extend(["-Wl,-dead_strip", "-Wl,-x"].map(String::from));
            Ok(LinkerFlags {
                pre,
                // The runtime's relaxed-numerics kernels call Accelerate
                // (BLAS/LAPACK), which every macOS has.
                post: ["-framework", "Accelerate"].map(String::from).to_vec(),
            })
        }
        ("linux", "linux") if !cross_arch => Ok(LinkerFlags {
            pre: ["-Wl,--gc-sections", "-Wl,-z,noexecstack"]
                .map(String::from)
                .to_vec(),
            // What the Rust standard library in the runtime needs.
            post: ["-lm", "-lpthread", "-ldl"].map(String::from).to_vec(),
        }),
        _ => Err(format!(
            "cannot link for `{}` on `{}`: only this machine's operating system and, on macOS, \
             either architecture can be linked; build the object with `--emit=obj --target=...` \
             and link it there",
            display_safe(&target.triple),
            display_safe(&host.triple)
        )),
    }
}
