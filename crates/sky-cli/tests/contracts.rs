//! Contracts end to end (spec "Contracts"): `sky run` with `assert`,
//! `requires` and `ensures`, on both engines, and `sky check` for what it
//! says about them.
//!
//! Scripts run in a scratch directory with a `.git`, so project discovery
//! stops there.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("contracts-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(d.join(".git")).unwrap();
    d
}

/// `sky <args> prog.sky` on `src`, in a scratch directory.
fn sky(name: &str, args: &[&str], src: &str) -> Output {
    let dir = scratch(name);
    fs::write(dir.join("prog.sky"), src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_sky"))
        .args(args)
        .arg("prog.sky")
        .current_dir(&dir)
        .output()
        .expect("run sky");
    fs::remove_dir_all(&dir).unwrap();
    out
}

fn out(o: &Output) -> String {
    String::from_utf8(o.stdout.clone()).expect("utf-8")
}

fn err(o: &Output) -> String {
    String::from_utf8(o.stderr.clone()).expect("utf-8")
}

/// `sky run` under both engines, which must agree on stdout, stderr and
/// exit status byte for byte. Returns the tree-walker's run.
fn run_both(name: &str, src: &str) -> Output {
    let ast = sky(name, &["run", "--engine=ast"], src);
    let sir = sky(name, &["run", "--engine=sir"], src);
    assert!(
        !err(&sir).contains("cannot run on SIR"),
        "SIR refused:\n{}",
        err(&sir)
    );
    assert_eq!(sir.status.code(), ast.status.code(), "status of:\n{src}");
    assert_eq!(out(&sir), out(&ast), "stdout of:\n{src}");
    assert_eq!(err(&sir), err(&ast), "stderr of:\n{src}");
    ast
}

const PROJECTILE: &str = "\
def fall_time(h: m) -> s:
    requires h >= 0 m
    ensures result > 0 s
    return sqrt(2 * h / g_n)

def range_of(v: m/s, angle) -> m:
    requires v > 0 m/s, \"launch speed\"
    requires angle > 0 and angle < 1.5
    ensures result >= 0 m
    return v^2 * sin(2 * angle) / g_n

print(fall_time(20 m))
print(range_of(30 m/s, 0.7))
";

#[test]
fn a_projectile_with_contracts_runs_on_both_engines() {
    let o = run_both("projectile", PROJECTILE);
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    assert_eq!(
        out(&o),
        format!("2.0196199771025523 s\n{} m\n", range_text())
    );
    assert_eq!(err(&o), "");
}

/// `30^2 * sin(1.4) / g_n`, computed the way the interpreter does.
fn range_text() -> String {
    let x = 30f64 * 30f64 * (2.0f64 * 0.7).sin() / 9.80665;
    format!("{x}")
}

#[test]
fn a_failed_precondition_points_at_the_contract_and_the_call() {
    let src = format!("{PROJECTILE}print(fall_time(-1 m))\nprint(\"unreachable\")\n");
    let o = run_both("pre", &src);
    assert_eq!(o.status.code(), Some(1));
    // Output before the failure stays.
    assert!(out(&o).starts_with("2.0196199771025523 s\n"), "{}", out(&o));
    assert!(!out(&o).contains("unreachable"));
    let e = err(&o);
    assert!(
        e.contains(
            "prog.sky:2:14: error: precondition of `fall_time` failed: `h >= 0 m` (h = -1 m)"
        ),
        "{e}"
    );
    assert!(e.contains("requires h >= 0 m"), "{e}");
    assert!(e.contains("this is false"), "{e}");
    assert!(e.contains("print(fall_time(-1 m))"), "{e}");
    assert!(e.contains("called here"), "{e}");
}

#[test]
fn a_failed_postcondition_names_the_result() {
    let src = "def fall_time(h: m) -> s:\n    requires h >= 0 m\n    ensures result > 0 s\n    return sqrt(2 * h / g_n)\nprint(fall_time(0 m))\n";
    let o = run_both("post", src);
    assert_eq!(o.status.code(), Some(1));
    let e = err(&o);
    assert!(
        e.contains("prog.sky:3:13: error: postcondition of `fall_time` failed: `result > 0 s` (result = 0 s)"),
        "{e}"
    );
    assert!(e.contains("ensures result > 0 s"), "{e}");
}

#[test]
fn a_failed_assertion_shows_the_values_of_its_names() {
    let src = "E = 3 J\nassert E > 0 J\nprint(\"fine\")\nE = -3.2 J\nassert E > 0 J\nprint(\"unreachable\")\n";
    let o = run_both("assert", src);
    assert_eq!(o.status.code(), Some(1));
    assert_eq!(out(&o), "fine\n");
    assert_eq!(
        err(&o),
        "prog.sky:5:8: error: assertion failed: `E > 0 J` (E = -3.2 J)\n  |\n5 | assert E > 0 J\n  |        ^^^^^^^ this is false\n"
    );
    let src = "E = -3.2 J\nassert E > 0 J, \"energy must be positive\"\n";
    let o = run_both("assert-msg", src);
    assert!(
        err(&o).starts_with(
            "prog.sky:2:8: error: assertion failed: energy must be positive (`E > 0 J`, E = -3.2 J)\n"
        ),
        "{}",
        err(&o)
    );
}

#[test]
fn a_contract_that_never_holds_is_a_warning_at_check_time() {
    let src = "def f(x: m):\n    requires x > 0 m\n    return x\nassert 2 m < 1 m\nf(-3 m)\n";
    let o = sky("warn", &["check"], src);
    assert_eq!(
        o.status.code(),
        Some(0),
        "warnings alone do not fail a check"
    );
    let e = err(&o);
    assert!(
        e.contains("warning: this `assert` always fails: `2 m < 1 m` is false"),
        "{e}"
    );
    assert!(
        e.contains("warning: this call breaks the precondition of `f`"),
        "{e}"
    );
    assert!(e.contains("with x = -3 m"), "{e}");
    // `sky run` still runs (a warning does not stop it) and fails at the assert.
    let o = sky("warn-run", &["run"], src);
    assert_eq!(o.status.code(), Some(1));
    assert!(
        err(&o).contains("error: assertion failed: `2 m < 1 m`"),
        "{}",
        err(&o)
    );
}

#[test]
fn a_contract_that_always_holds_is_explained_and_its_check_is_removed() {
    let src = "def fall_time(h: m) -> s:\n    requires h >= 0 m\n    return sqrt(2 * h / g_n)\nprint(fall_time(20 m))\n";
    let o = sky("explain", &["check", "--explain"], src);
    assert_eq!(o.status.code(), Some(0));
    let e = err(&o);
    assert!(
        e.contains("note: [CONTRACT-002] this `requires` always holds, so its check is removed: `h >= 0 m`"),
        "{e}"
    );
    assert!(e.contains("applied by the `contracts` pass"), "{e}");
    // The constant call gets its own function (`specialize`), where `h` is 20.
    assert!(
        e.contains("`fall_time[h=20]` is `fall_time` specialized for h = 20"),
        "{e}"
    );
    assert!(
        e.contains("note: [SPEC-001] specialized `fall_time` for h = 20 at this call"),
        "{e}"
    );
    // The IR after the passes has no assert; before, it has one.
    let final_ir = out(&sky("ir-final", &["check", "--emit-ir"], src));
    assert!(!final_ir.contains("assert["), "{final_ir}");
    let lowered = out(&sky("ir-lower", &["check", "--emit-ir=lower"], src));
    assert!(
        lowered.contains("assert[requires] \"h >= 0 m\" fn \"fall_time\" names(\"h\")"),
        "{lowered}"
    );
}

#[test]
fn errors_in_contracts_are_check_errors() {
    let src = "def f(x: m):\n    requires x > 0\n    return x\n";
    let o = sky("err", &["check"], src);
    assert_eq!(o.status.code(), Some(1));
    assert!(
        err(&o).contains("error: cannot compare length and dimensionless"),
        "{}",
        err(&o)
    );
    // `sky run` stops before running anything.
    let src = "print(1)\nassert 3 m\n";
    let o = sky("err-run", &["run"], src);
    assert_eq!(o.status.code(), Some(1));
    assert_eq!(out(&o), "");
    assert!(
        err(&o).contains("`assert` needs a Bool condition"),
        "{}",
        err(&o)
    );
}

#[test]
fn the_repl_checks_contracts_across_inputs() {
    let dir = scratch("repl");
    let mut child = Command::new(env!("CARGO_BIN_EXE_sky"))
        .arg("repl")
        .current_dir(&dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(
            b"def f(x: m) -> m:\n    requires x > 0 m\n    return x * 2\n\nf(3 m)\nf(-1 m)\nf(4 m)\n",
        )
        .unwrap();
    let o = child.wait_with_output().expect("wait");
    fs::remove_dir_all(&dir).unwrap();
    assert_eq!(out(&o), "6 m\n8 m\n");
    assert!(
        err(&o).contains("error: precondition of `f` failed: `x > 0 m` (x = -1 m)"),
        "{}",
        err(&o)
    );
}
