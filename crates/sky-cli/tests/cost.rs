//! Performance contracts end to end (spec "Performance contracts"): a `cost`
//! line does nothing when a program runs, `sky check` proves or refutes it
//! from the static cost model, and `--explain` reports what each function
//! costs. The goldens in `tests/ui/check/` pin the wording; these pin the
//! behaviour around it.
//!
//! Scripts run in a scratch directory with a `.git`, so project discovery
//! stops there.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("cost-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(d.join(".git")).unwrap();
    d
}

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

const STEP: &str = "\
def step(v: Vector[N], w: Vector[N]) -> Vector[N]:
    requires len(v) > 0
    cost flops == 2 * N
    ensures len(result) > 0
    return v + 2 * w

print(step([1, 2], [3, 4]))
";

#[test]
fn a_run_ignores_cost_lines_on_both_engines() {
    let ast = sky("run-ast", &["run", "--engine=ast"], STEP);
    let sir = sky("run-sir", &["run", "--engine=sir"], STEP);
    let native = sky("run-native", &["run", "--engine=native"], STEP);
    assert_eq!(ast.status.code(), Some(0), "{}", err(&ast));
    assert_eq!(out(&ast), "[7, 10]\n");
    assert_eq!(out(&sir), out(&ast));
    assert_eq!(err(&sir), err(&ast));
    assert_eq!(out(&native), out(&ast));
    assert_eq!(err(&native), err(&ast));
    // A refuted contract does not stop a run, only a check.
    let wrong = STEP.replace("2 * N", "N");
    let o = sky("run-wrong", &["run"], &wrong);
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    assert_eq!(out(&o), "[7, 10]\n");
    let o = sky("check-wrong", &["check"], &wrong);
    assert_eq!(o.status.code(), Some(1));
}

#[test]
fn proven_is_silent_and_explained_under_explain() {
    let o = sky("proven", &["check"], STEP);
    assert_eq!(o.status.code(), Some(0));
    assert_eq!(err(&o), "");
    let o = sky("proven-explain", &["check", "--explain"], STEP);
    assert_eq!(o.status.code(), Some(0));
    let e = err(&o);
    assert!(
        e.contains("note: `step` costs exactly 2·N flops, as `cost flops == 2 * N` states"),
        "{e}"
    );
    // The report needs no contract: it comes with the summary.
    assert!(
        e.contains("note: cost of `step`: 2·N flops, 1 allocation of N floats"),
        "{e}"
    );
}

#[test]
fn a_refuted_contract_names_the_operation_and_the_term() {
    let src = "def mm(A: Matrix[N, K], B: Matrix[K, M]) -> Matrix[N, M]:\n    cost flops <= 2 * N^3\n    return A @ B\n";
    let o = sky("refuted", &["check"], src);
    assert_eq!(o.status.code(), Some(1));
    let e = err(&o);
    assert!(
        e.contains("error: `mm` costs 2·K·M·N flops, which is more than `2 * N^3`"),
        "{e}"
    );
    assert!(e.contains("`A @ B` costs 2·K·M·N flops"), "{e}");
    assert!(
        e.contains("the dominant term of the excess is 2·K·M·N"),
        "{e}"
    );
}

#[test]
fn an_unbounded_loop_is_a_warning_that_says_where() {
    let src = "def f(v: Vector[N]) -> Vector[N]:\n    cost flops <= 5 * N\n    while v[0] > 1:\n        v = v * 2\n    return v\n";
    let o = sky("unbounded", &["check"], src);
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    let e = err(&o);
    assert!(e.contains("warning: cannot prove that `f` meets `cost flops <= 5 * N`"));
    assert!(
        e.contains("the cost has no upper bound: loop at line 3"),
        "{e}"
    );
    let o = sky("unbounded-explain", &["check", "--explain"], src);
    assert!(
        err(&o).contains("note: cost of `f`: unbounded (loop at line 3)"),
        "{}",
        err(&o)
    );
}

#[test]
fn a_program_that_does_not_lower_is_warned_not_ignored() {
    // Iterating over a parameter without an annotation does not lower.
    let src = "def total(v: Vector[N], w) -> m/m:\n    cost flops <= N\n    s = 0\n    for x in w:\n        s += x\n    return s\n";
    let o = sky("nolower", &["check"], src);
    assert_eq!(o.status.code(), Some(0));
    assert!(
        err(&o).contains("warning: the cost contracts were not checked"),
        "{}",
        err(&o)
    );
    // With no contract there is nothing to say.
    let o = sky(
        "nolower-quiet",
        &["check"],
        &src.replace("    cost flops <= N\n", ""),
    );
    assert_eq!(err(&o), "");
}

#[test]
fn a_program_without_cost_lines_checks_as_before() {
    let src = "def f(x: m) -> m:\n    return x * 2\nprint(f(2 m))\n";
    let o = sky("none", &["check"], src);
    assert_eq!(err(&o), "");
    assert_eq!(out(&o), format!("{}: ok\n", "prog.sky"));
}

#[test]
fn the_report_is_in_source_order_with_the_other_notes() {
    let src = "def a(v: Vector[N]) -> Vector[N]:\n    return v * 2\n\ndef b(v: Vector[N]) -> Vector[N]:\n    return v + v\n";
    let o = sky("order", &["check", "--explain"], src);
    let e = err(&o);
    let a = e.find("cost of `a`").expect("a");
    let b = e.find("cost of `b`").expect("b");
    assert!(a < b, "{e}");
}

/// `norm` and `dot` became SIR operations in parallel with the cost model:
/// they cost their entries (2·n, and 2·n + 1 with the `sqrt`), never zero,
/// so a contract that leaves them out is refuted, not proven.
#[test]
fn norm_and_dot_are_counted() {
    let src = "def energy(v: Vector[N] m/s, p: Vector[N] m) -> m^2/s^2:\n    cost flops == 4 * N + 5\n    return 0.5 * norm(v)^2 + dot(p, p) / (1 s^2)\n";
    let o = sky("norm-dot", &["check", "--explain"], src);
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    assert!(err(&o).contains("`energy` costs exactly"), "{}", err(&o));
    let o = sky("norm-dot-free", &["check"], &src.replace("4 * N + 5", "5"));
    assert_eq!(o.status.code(), Some(1), "{}", err(&o));
    assert!(
        err(&o).contains("`norm(v)` costs") || err(&o).contains("`dot(p, p)` costs"),
        "{}",
        err(&o)
    );
}

/// The algorithm rewrites (built in parallel) turn `inv(A) @ b` into an LU
/// solve for an untagged `A`, which costs 2·N³/3 to factor, twice the
/// Cholesky solve of a PositiveDefinite one; the model prices each method.
#[test]
fn solves_are_priced_by_method() {
    let lu = "def f(A: Matrix[N, N], b: Vector[N]) -> Vector[N]:\n    cost flops <= N^3 / 3 + 2 * N^2\n    return inv(A) @ b\n";
    let o = sky("lu", &["check"], lu);
    assert_eq!(
        o.status.code(),
        Some(1),
        "an LU solve is not a Cholesky solve:\n{}",
        err(&o)
    );
    let o = sky("lu-ok", &["check"], &lu.replace("N^3 / 3", "2 * N^3 / 3"));
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    let pd = "def f(A: PositiveDefinite Matrix[N, N], b: Vector[N]) -> Vector[N]:\n    cost flops <= N^3 / 3 + 2 * N^2\n    return inv(A) @ b\n";
    let o = sky("pd", &["check"], pd);
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
}
