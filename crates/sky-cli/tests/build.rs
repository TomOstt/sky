//! `sky build`: a program built to an executable behaves exactly like `sky run
//! --engine=native` on the same file: the same stdout, the same stderr (the
//! diagnostics, rendered from the embedded source), the same exit status.
//!
//! Every test skips (says so, and passes) when the machine has no C linker.
//! The runtime library the executables link with (`libsky_rt.a`) is built
//! on first use with `cargo build -p sky-rt`, next to the `sky` binary under
//! test, because `cargo test` does not build a package's static library.
//!
//! `SKY_TEST_X86_64=1` (macOS on Apple silicon only) also builds for
//! `x86_64-apple-darwin` and runs the executables under Rosetta.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

fn sky() -> Command {
    Command::new(env!("CARGO_BIN_EXE_sky"))
}

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn linker() -> String {
    std::env::var("CC")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "cc".to_string())
}

/// Whether a C linker is installed.
fn have_linker() -> bool {
    static HAVE: OnceLock<bool> = OnceLock::new();
    *HAVE.get_or_init(|| Command::new(linker()).arg("--version").output().is_ok())
}

/// `<target dir>/<debug|release>` of the `sky` under test, for `triple` (`None`:
/// the host) the directory `cargo build --target` writes to.
fn profile_dir(triple: Option<&str>) -> Option<PathBuf> {
    let exe = PathBuf::from(env!("CARGO_BIN_EXE_sky"));
    let dir = exe.parent()?;
    match triple {
        None => Some(dir.to_path_buf()),
        Some(t) => Some(dir.parent()?.join(t).join(dir.file_name()?)),
    }
}

/// The runtime library for `triple`, built if it is not there yet. `None`
/// if it cannot be built here.
fn runtime_lib(triple: Option<&str>) -> Option<PathBuf> {
    static HOST: OnceLock<Option<PathBuf>> = OnceLock::new();
    static X86: OnceLock<Option<PathBuf>> = OnceLock::new();
    let cell = if triple.is_none() { &HOST } else { &X86 };
    cell.get_or_init(|| {
        let dir = profile_dir(triple)?;
        let lib = dir.join("libsky_rt.a");
        if !lib.is_file() {
            let target_dir = PathBuf::from(env!("CARGO_BIN_EXE_sky"))
                .parent()?
                .parent()?
                .to_path_buf();
            let mut cmd = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
            cmd.current_dir(repo())
                .args(["build", "--locked", "-p", "sky-rt", "--target-dir"])
                .arg(target_dir);
            if dir.file_name()? == "release" {
                cmd.arg("--release");
            }
            if let Some(t) = triple {
                cmd.args(["--target", t]);
            }
            let out = cmd.output().ok()?;
            assert!(
                out.status.success(),
                "cargo build -p sky-rt failed:\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        lib.is_file().then_some(lib)
    })
    .clone()
}

/// A private scratch directory for one test.
struct Dir(PathBuf);

impl Dir {
    fn new(name: &str) -> Dir {
        let p = std::env::temp_dir().join(format!("sky-build-test-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        Dir(p)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The three things that must match.
#[derive(Debug, PartialEq)]
struct Outcome {
    stdout: String,
    stderr: String,
    code: Option<i32>,
}

impl From<Output> for Outcome {
    fn from(o: Output) -> Outcome {
        Outcome {
            stdout: String::from_utf8_lossy(&o.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&o.stderr).into_owned(),
            code: o.status.code(),
        }
    }
}

/// How an executable is started: directly, or under Rosetta.
#[derive(Clone, Copy)]
enum Arch {
    Host,
    X86OnArm,
}

impl Arch {
    fn target(self) -> Option<&'static str> {
        match self {
            Arch::Host => None,
            Arch::X86OnArm => Some("x86_64-apple-darwin"),
        }
    }
}

/// `sky build` of `dir/prog.sky` (run in `dir`, so diagnostics name the same
/// path `sky run` is given) to `dir/prog`.
fn build(dir: &Path, arch: Arch, extra: &[&str]) -> Option<Output> {
    let mut cmd = sky();
    cmd.current_dir(dir).arg("build");
    if let Some(t) = arch.target() {
        cmd.arg(format!("--target={t}"));
    }
    let lib = runtime_lib(arch.target())?;
    cmd.arg(format!("--runtime={}", lib.display()));
    cmd.args(extra).args(["prog.sky", "-o", "prog"]);
    Some(cmd.output().expect("run sky build"))
}

fn run_exe(dir: &Path, arch: Arch) -> Outcome {
    let mut cmd = match arch {
        Arch::Host => Command::new(dir.join("prog")),
        Arch::X86OnArm => {
            let mut c = Command::new("arch");
            c.args(["-x86_64"]).arg(dir.join("prog"));
            c
        }
    };
    cmd.current_dir(dir)
        .output()
        .expect("run the executable")
        .into()
}

/// Builds `src`, runs the executable and `sky run --engine=native`, and
/// requires the same stdout, stderr and status. `None` when skipped (no
/// linker). Returns the run's outcome.
fn same_as_native(name: &str, src: &str, arch: Arch) -> Option<Outcome> {
    if !have_linker() {
        eprintln!("skipping {name}: no C linker");
        return None;
    }
    let dir = Dir::new(name);
    fs::write(dir.0.join("prog.sky"), src).unwrap();
    let want: Outcome = sky()
        .current_dir(&dir.0)
        .args(["run", "--engine=native", "prog.sky"])
        .output()
        .expect("run sky")
        .into();
    let built = build(&dir.0, arch, &[])?;
    let checks = sky()
        .current_dir(&dir.0)
        .args(["check", "prog.sky"])
        .output()
        .expect("run sky check")
        .status
        .success();
    if !checks || want.stderr.contains("cannot run natively") {
        // Not a program that runs (a check error or a program native code
        // declines): nothing is built, and the diagnostic is the same.
        assert_eq!(built.status.code(), Some(1), "{name}");
        assert!(!dir.0.join("prog").exists(), "{name}: wrote an executable");
        let got = String::from_utf8_lossy(&built.stderr);
        if want.stderr.contains("cannot run natively") {
            assert!(got.contains("cannot build"), "{name}: {got}");
        } else {
            assert_eq!(got, want.stderr, "{name}");
        }
        return Some(want);
    }
    assert!(
        built.status.success(),
        "{name}: sky build failed:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let got = run_exe(&dir.0, arch);
    assert_eq!(got, want, "{name}: the executable differs from sky run");
    Some(got)
}

fn check(name: &str, src: &str) -> Option<Outcome> {
    same_as_native(name, src, Arch::Host)
}

// ----- the examples and the IR test programs -----

fn sky_files(dir: &str) -> Vec<PathBuf> {
    let mut files: Vec<_> = fs::read_dir(repo().join(dir))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "sky"))
        .collect();
    files.sort();
    files
}

fn each_file(dir: &str, arch: Arch) {
    for path in sky_files(dir) {
        let name = format!("{dir}-{}", path.file_stem().unwrap().to_string_lossy());
        let src = fs::read_to_string(&path).unwrap();
        same_as_native(&name.replace('/', "-"), &src, arch);
    }
}

#[test]
fn every_example_builds_and_behaves_like_sky_run() {
    each_file("examples", Arch::Host);
}

#[test]
fn every_ir_test_program_builds_and_behaves_like_sky_run() {
    each_file("tests/ir", Arch::Host);
}

// ----- programs from the native engine's tests -----

#[test]
fn scalar_loops_and_powers() {
    check(
        "scalar",
        "m = 2 kg\nk = 50 N/m\nx = 1 m\nv = 0 m/s\ni = 0\n\
         while i < 1000:\n    a = -k * x / m\n    v += a * 0.001 s\n    x += v * 0.001 s\n    i += 1\n\
         print(x, v, 0.5 * m * v^2 + 0.5 * k * x^2)\n",
    );
    let mut src = String::new();
    for b in [
        "0.1",
        "1 / 3",
        "2",
        "-2",
        "1e10",
        "-7.5",
        "3.7e80",
        "0",
        "123456.789",
    ] {
        for e in [
            "0", "1", "2", "3", "5", "8", "13", "-1", "-3", "-7", "0.5", "1/3", "-64",
        ] {
            src += &format!("x = {b}\nprint(x^({e}))\n");
        }
    }
    check("powers", &src);
}

#[test]
fn math_functions() {
    let mut src = String::new();
    for x in [
        "0.1", "0.5", "1", "2", "10", "0.001", "1.5", "0.99", "3.14159", "12.7",
    ] {
        src += &format!("x = {x}\nprint(");
        let fs = [
            "sqrt", "cbrt", "abs", "exp", "log", "log10", "log2", "sin", "cos", "tan", "atan",
            "floor", "ceil", "round", "expm1", "log1p", "sinh", "cosh", "tanh", "asinh", "erf",
            "sign",
        ];
        src += &fs.map(|f| format!("{f}(x)")).join(", ");
        src += ")\n";
        src += "print(atan2(x, 0.1), hypot(x, -3), min(x, 7.5), max(x, 7.5))\n";
    }
    check("math", &src);
}

#[test]
fn errors_after_output_have_sky_runs_diagnostics() {
    for (i, src) in [
        "x = 4\nz = 2 - 2\nprint(1)\nprint(x / z)\nprint(2)\n",
        "x = 1e200\nprint(3)\nprint(x * x)\n",
        "print(1)\nprint(sqrt(0 - 1))\n",
        "x = 1e300\nprint(7)\na = x * 1e10\nb = a * a\nprint(a, b)\n",
        "x = 1.0\nn = 0\nwhile n < 5000:\n    print(n)\n    x = x * 3\n    n = n + 1\n",
        "n = 5\nfor i in range(10):\n    print(100 / (n - i))\n",
        "def fall_time(h: m) -> s:\n    requires h >= 0 m\n    return sqrt(2 * h / g_n)\n\
         print(fall_time(20 m))\nprint(fall_time(-1 m))\n",
        "def d(n: m) -> m:\n    if n == 0 m:\n        return 0 m\n    return d(n - 1 m) + 1 m\n\
         print(d(999 m))\nprint(d(1000 m))\n",
        "def f(x):\n    return f(x) + 1\nprint(1)\nprint(f(1))\n",
        "v = [1 m, 2 m, 3 m]\nprint(v)\ni = 5\nprint(v[i])\n",
    ]
    .iter()
    .enumerate()
    {
        if let Some(got) = check(&format!("error{i}"), src) {
            assert_eq!(got.code, Some(1), "{src}");
            assert!(got.stderr.contains("error"), "{src}: {}", got.stderr);
        }
    }
}

#[test]
fn control_flow_calls_arrays_and_strings() {
    check(
        "flow",
        "s = 0\nfor i in range(20):\n    if i / 2 == floor(i / 2):\n        continue\n    for j in range(20):\n        if i * j > 50:\n            break\n        s = s + i * j\nprint(s)\n\
         a = 3\nb = 0 - 2\nprint(a > 0 and b < 0, a > 0 and b > 0, a < 0 or b < 0, not (a < 0 or b > 0))\n",
    );
    check(
        "calls",
        "def shout(n):\n    print(n)\n    return n + 1\n\
         print(shout(1) + shout(10))\nprint(\"done\", 1 < 2, \"a\" == \"a\")\n",
    );
    check(
        "arrays",
        "v = [1 m, 2 m, 3 m]\nA = [[1, 2], [3, 4]]\n\
         print(v + v * 2, v[1], A @ A, Aᵀ, A⁻¹, len(v), zeros(2), identity(2))\n\
         print(v[1:3], A[0], A[1, 0])\n",
    );
}

/// A value of unknown type that is not what a declaration says: native code
/// stops, and the executable finishes the run on the SIR interpreter it
/// carries (deoptimization), skipping the output already written.
#[test]
fn deoptimization_finishes_on_the_embedded_interpreter() {
    let src =
        "def f(x) -> m:\n    return x\nprint(\"before\")\nprint(f(3 s))\nprint(f(3 s) + f(3 s))\n";
    if let Some(got) = check("deopt", src) {
        assert!(got.stdout.starts_with("before\n"), "{got:?}");
    }
    // ... and one that fails after the deoptimization.
    check(
        "deopt-error",
        "def f(x) -> m:\n    return x\nprint(\"before\")\nprint(f(3 s))\nprint(f(3 s) + 1 m)\n",
    );
}

// ----- errors and warnings -----

#[test]
fn a_program_with_a_type_error_is_not_built() {
    check("type-error", "print(1)\nx = 1 m + 2 s\nprint(x)\n");
    check("syntax-error", "print(1\n");
}

/// A differential equation does not lower to SIR yet, so `--engine=native`
/// declines it (and `--engine=auto` runs it on the tree-walker). `sky build`
/// says the same and writes nothing: there is no interpreter to fall back to
/// in an executable, and a built program never runs differently from
/// `sky run --engine=native`.
#[test]
fn a_program_that_does_not_compile_natively_is_not_built() {
    let dir = Dir::new("declined");
    let src = fs::read_to_string(repo().join("examples/orbit.sky")).unwrap();
    fs::write(dir.0.join("prog.sky"), src).unwrap();
    // `--emit=obj`, so no linker or runtime library is involved.
    let out = sky()
        .current_dir(&dir.0)
        .args(["build", "--emit=obj", "prog.sky", "-o", "prog.o"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("error: cannot build: differential equations are not supported by SIR yet"),
        "{err}"
    );
    assert!(!dir.0.join("prog.o").exists());
}

/// The front end's warnings are printed by the build and again, from the
/// text stored in the executable, by every run; and a run-time error after
/// them is separated by a blank line, as `sky run` does.
#[test]
fn warnings_are_shown_by_the_build_and_by_every_run() {
    let warn = "x = 1000000.0\ny = sqrt(x^2 + 1) - x\nprint(y)\n";
    let got = check("warn", warn);
    if let Some(got) = &got {
        assert!(got.stderr.contains("warning:"), "{got:?}");
    }
    let got = check("warn-then-error", &format!("{warn}print(1 / (2 - 2))\n"));
    if let Some(got) = &got {
        assert!(
            got.stderr.contains("warning:") && got.stderr.contains("error:"),
            "{got:?}"
        );
        assert_eq!(got.code, Some(1));
    }
}

/// Every program of the checker's `ok_*` tests that native code takes.
#[test]
fn every_program_the_checker_accepts_builds_and_behaves_like_sky_run() {
    let ok = |p: &PathBuf| {
        p.file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with("ok_"))
    };
    for path in sky_files("tests/ui/check").into_iter().filter(ok) {
        let name = format!("ui-{}", path.file_stem().unwrap().to_string_lossy());
        let src = fs::read_to_string(&path).unwrap();
        same_as_native(&name, &src, Arch::Host);
    }
}

// ----- the command line -----

#[test]
fn usage_errors_exit_2() {
    for args in [
        vec!["build"],
        vec!["build", "--emit=asm", "x.sky"],
        vec!["build", "-o"],
        vec!["build", "a.sky", "b.sky"],
        vec!["build", "--bogus", "a.sky"],
    ] {
        let out = sky().args(&args).output().unwrap();
        assert_eq!(out.status.code(), Some(2), "{args:?}");
    }
}

#[test]
fn a_missing_file_is_exit_2() {
    let out = sky().args(["build", "no-such-file.sky"]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn a_missing_runtime_library_or_linker_is_a_clear_error() {
    let dir = Dir::new("missing");
    fs::write(dir.0.join("prog.sky"), "print(1)\n").unwrap();
    let out = sky()
        .current_dir(&dir.0)
        .args(["build", "--runtime=/no/such/libsky_rt.a", "prog.sky"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("runtime library"));
    let Some(lib) = runtime_lib(None) else { return };
    let out = sky()
        .current_dir(&dir.0)
        .env("CC", "/no/such/cc")
        .arg(format!("--runtime={}", lib.display()))
        .args(["build", "prog.sky"])
        .output()
        .unwrap();
    // The runtime flag before the command is a usage error; put it after.
    assert_eq!(out.status.code(), Some(2));
    let out = sky()
        .current_dir(&dir.0)
        .env("CC", "/no/such/cc")
        .args(["build", &format!("--runtime={}", lib.display()), "prog.sky"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("no C linker found"));
    assert!(!dir.0.join("prog").exists());
}

#[test]
fn emit_obj_writes_an_object_file_for_any_target() {
    let dir = Dir::new("obj");
    fs::write(dir.0.join("prog.sky"), "print(1 + 2)\n").unwrap();
    for (target, magic) in [
        (None, None),
        (Some("x86_64-unknown-linux-gnu"), Some(&b"\x7fELF"[..])),
        (Some("aarch64-unknown-linux-gnu"), Some(&b"\x7fELF"[..])),
        (Some("x86_64-apple-darwin"), Some(&b"\xcf\xfa\xed\xfe"[..])),
        (Some("aarch64-apple-darwin"), Some(&b"\xcf\xfa\xed\xfe"[..])),
    ] {
        let mut cmd = sky();
        cmd.current_dir(&dir.0)
            .args(["build", "--emit=obj", "prog.sky"]);
        if let Some(t) = target {
            cmd.arg(format!("--target={t}"));
        }
        let out = cmd.output().unwrap();
        assert!(
            out.status.success(),
            "{target:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let obj = fs::read(dir.0.join("prog.o")).unwrap();
        if let Some(m) = magic {
            assert!(obj.starts_with(m), "{target:?}");
        }
        assert!(obj.len() > 100);
        fs::remove_file(dir.0.join("prog.o")).unwrap();
    }
    let out = sky()
        .current_dir(&dir.0)
        .args(["build", "--emit=obj", "--target=pdp11-none", "prog.sky"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot build"));
}

#[test]
fn the_default_output_is_named_after_the_file() {
    if !have_linker() {
        return;
    }
    let Some(lib) = runtime_lib(None) else { return };
    let dir = Dir::new("default-name");
    fs::write(dir.0.join("hello.sky"), "print(\"hi\")\n").unwrap();
    let out = sky()
        .current_dir(&dir.0)
        .args([
            "build",
            &format!("--runtime={}", lib.display()),
            "hello.sky",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run: Outcome = Command::new(dir.0.join("hello")).output().unwrap().into();
    assert_eq!(run.stdout, "hi\n");
    assert_eq!(run.code, Some(0));
}

/// The executable is the same file wherever it runs from: the runtime finds
/// its program in the executable, not next to it.
#[test]
fn the_executable_is_standalone() {
    if !have_linker() {
        return;
    }
    let dir = Dir::new("standalone");
    fs::write(dir.0.join("prog.sky"), "print(1)\nprint(1 / (2 - 2))\n").unwrap();
    let Some(out) = build(&dir.0, Arch::Host, &[]) else {
        return;
    };
    assert!(out.status.success());
    let elsewhere = Dir::new("standalone-elsewhere");
    fs::copy(dir.0.join("prog"), elsewhere.0.join("prog")).unwrap();
    fs::remove_dir_all(&dir.0).unwrap();
    let run: Outcome = Command::new(elsewhere.0.join("prog"))
        .current_dir(&elsewhere.0)
        .output()
        .unwrap()
        .into();
    assert_eq!(run.stdout, "1\n");
    assert_eq!(run.code, Some(1));
    // The diagnostic quotes the source the program was built from.
    assert!(run.stderr.contains("prog.sky"), "{}", run.stderr);
    assert!(run.stderr.contains("1 / (2 - 2)"), "{}", run.stderr);
}

#[cfg(unix)]
#[test]
fn a_closed_pipe_ends_the_run_quietly_like_sky_run() {
    if !have_linker() {
        return;
    }
    let dir = Dir::new("pipe");
    let src = "for i in range(200000):\n    print(i)\n";
    fs::write(dir.0.join("prog.sky"), src).unwrap();
    let Some(out) = build(&dir.0, Arch::Host, &[]) else {
        return;
    };
    assert!(out.status.success());
    let shell = |cmd: &str| {
        let o = Command::new("sh")
            .current_dir(&dir.0)
            // `head` closes the pipe after one line; the status is the
            // writer's (through `PIPESTATUS`-free trick: run it in a
            // subshell that reports it).
            .args([
                "-c",
                &format!("({cmd}; echo \"status=$?\" >&2) | head -1 >/dev/null"),
            ])
            .output()
            .unwrap();
        String::from_utf8_lossy(&o.stderr).into_owned()
    };
    let run = shell(&format!(
        "{} run --engine=native prog.sky",
        env!("CARGO_BIN_EXE_sky")
    ));
    let built = shell("./prog");
    assert_eq!(built, run);
}

// ----- x86-64 under Rosetta (macOS on Apple silicon; opt in) -----

#[test]
fn x86_64_executables_run_under_rosetta() {
    let wanted = std::env::var("SKY_TEST_X86_64").is_ok_and(|v| v == "1");
    if !wanted || !cfg!(all(target_os = "macos", target_arch = "aarch64")) || !have_linker() {
        eprintln!("skipping the x86-64 build (set SKY_TEST_X86_64=1 on Apple silicon)");
        return;
    }
    for path in sky_files("examples") {
        let name = format!("x86-{}", path.file_stem().unwrap().to_string_lossy());
        let src = fs::read_to_string(&path).unwrap();
        same_as_native(&name, &src, Arch::X86OnArm);
    }
    for (i, src) in [
        "x = 1.0\nn = 0\nwhile n < 5000:\n    print(n)\n    x = x * 3\n    n = n + 1\n",
        "def f(x) -> m:\n    return x\nprint(\"before\")\nprint(f(3 s))\n",
        "v = [1 m, 2 m, 3 m]\nA = [[1, 2], [3, 4]]\nprint(v + v * 2, A @ A, A⁻¹)\n",
        "print(floor(2.5), ceil(2.5), round(2.5), sqrt(2), sin(1), 2^0.5)\n",
    ]
    .iter()
    .enumerate()
    {
        same_as_native(&format!("x86-{i}"), src, Arch::X86OnArm);
    }
}
