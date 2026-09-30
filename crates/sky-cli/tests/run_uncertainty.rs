//! `sky run` and `sky repl` with uncertain values (spec "Uncertainty"): the
//! pendulum period end to end, the engines (SIR does not carry an
//! uncertainty, so `--engine=auto` falls back to the tree-walker and
//! `--engine=sir` refuses), and the checker's warning reaching stderr.
//!
//! Scripts are written to a scratch directory that has a `.git`, so project
//! discovery stops there and never looks at the real checkout.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("unc-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(d.join(".git")).unwrap();
    d
}

fn run_engine(name: &str, src: &str, engine: Option<&str>) -> Output {
    let dir = scratch(name);
    let file = dir.join("prog.sky");
    fs::write(&file, src).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_sky"));
    cmd.arg("run");
    if let Some(engine) = engine {
        cmd.arg(format!("--engine={engine}"));
    }
    let out = cmd.arg(&file).current_dir(&dir).output().expect("run sky");
    fs::remove_dir_all(&dir).unwrap();
    out
}

fn out(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn err(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

const PENDULUM: &str = "\
# T = 2 pi sqrt(L/g) with a measured length and gravity.
L = 1.000 ± 0.005 m
g = 9.81 ± 0.05 m/s^2
T = 2 * π * sqrt(L / g)
print(\"T =\", T)
print(\"T =\", T as ms)
print(\"relative uncertainty:\", uncertainty(T) / nominal(T))
";

#[test]
fn the_pendulum_period() {
    let o = run_engine("pendulum", PENDULUM, None);
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    assert_eq!(
        out(&o),
        "T = 2.006 ± 0.007 s\n\
         T = 2006 ± 7 ms\n\
         relative uncertainty: 0.003569936188853925\n"
    );
    assert_eq!(err(&o), "");
}

#[test]
fn ascii_spelling_and_the_default_engine_agree() {
    let ascii = PENDULUM.replace('±', "+/-");
    let a = run_engine("pendulum_ascii", &ascii, Some("ast"));
    let b = run_engine("pendulum_default", PENDULUM, None);
    assert_eq!(out(&a), out(&b));
    assert_eq!(a.status.code(), Some(0));
}

#[test]
fn auto_falls_back_and_sir_refuses() {
    let ast = run_engine("engines_ast", PENDULUM, Some("ast"));
    let auto = run_engine("engines_auto", PENDULUM, Some("auto"));
    assert_eq!(auto.status.code(), Some(0), "{}", err(&auto));
    assert_eq!(out(&auto), out(&ast));
    assert_eq!(err(&auto), err(&ast));

    let sir = run_engine("engines_sir", PENDULUM, Some("sir"));
    assert_eq!(sir.status.code(), Some(1));
    assert_eq!(out(&sir), "");
    let e = err(&sir);
    assert!(
        e.contains("error: cannot run on SIR: uncertainty (`±`) is not supported by SIR yet"),
        "{e}"
    );
    assert!(e.contains("--engine=auto"), "{e}");
}

#[test]
fn nominal_alone_also_falls_back() {
    // No `±` in the program, but `nominal` is not a SIR operation either.
    let src = "print(nominal(5 m), uncertainty(5 m))\n";
    let ast = run_engine("nominal_ast", src, Some("ast"));
    let auto = run_engine("nominal_auto", src, Some("auto"));
    assert_eq!(out(&ast), "5 m 0 m\n");
    assert_eq!(out(&auto), out(&ast));
    let sir = run_engine("nominal_sir", src, Some("sir"));
    assert!(
        err(&sir).contains("`nominal` is not supported by SIR yet"),
        "{}",
        err(&sir)
    );
}

#[test]
fn correlations_end_to_end() {
    let src = "\
x = 5.31 ± 0.02 kg
print(x - x)
print(x * x, x ^ 2)
print(x + x)
y = 5.31 ± 0.02 kg
print(x - y)
";
    let o = run_engine("correlations", src, None);
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    let want = [
        "0 ± 0 kg",
        // x * x and x^2 are the same (2 x σ = 0.212), in kg².
        "28.20 ± 0.21 kg^2 28.20 ± 0.21 kg^2",
        "10.62 ± 0.04 kg",
        // Independent: sqrt(2) · 0.02 = 0.028.
        "0.000 ± 0.028 kg",
    ];
    assert_eq!(out(&o), want.join("\n") + "\n");
}

#[test]
fn the_comparison_warning_is_on_stderr_and_the_run_goes_on() {
    let src = "\
a = 9.81 ± 0.05 m/s^2
b = 9.79 ± 0.03 m/s^2
print(a > b)
";
    let o = run_engine("warn", src, None);
    assert_eq!(o.status.code(), Some(0));
    assert_eq!(out(&o), "true\n");
    assert!(
        err(&o).contains("warning: comparing an uncertain value compares only its nominal value"),
        "{}",
        err(&o)
    );
}

#[test]
fn a_runtime_error_points_at_the_uncertain_expression() {
    let src = "x = 0 ± 0.1\nprint(sqrt(x))\n";
    let o = run_engine("slope", src, None);
    assert_eq!(o.status.code(), Some(1));
    let e = err(&o);
    assert!(
        e.contains("error: cannot propagate an uncertainty through `sqrt` at 0"),
        "{e}"
    );
    assert!(e.contains("the derivative is infinite here"), "{e}");
}

#[test]
fn the_repl_echoes_uncertain_values() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_sky"))
        .arg("repl")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run sky");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"m = 5.31 +/- 0.02 kg\nm\nm * 2\nm as g\nm - m\n")
        .unwrap();
    let o = child.wait_with_output().unwrap();
    let text = out(&o);
    for want in [
        "5.31 ± 0.02 kg",
        "10.62 ± 0.04 kg",
        "5310 ± 20 g",
        "0 ± 0 kg",
    ] {
        assert!(text.contains(want), "no {want:?} in {text:?}");
    }
}
