//! Tags of names a loop body reassigns: the body must be checked with the
//! tags the next iteration starts with, not only those from before the
//! loop. `Q = 2 * Q` leaves `Q` diagonal but not orthogonal, so a
//! `Q⁻¹ @ b` at the top of the body must not become `Qᵀ @ b`, on any
//! engine. And an argument must have the tags its parameter declares.
//!
//! Scripts run in a scratch directory with a `.git`, so project discovery
//! stops there.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("loop-tags-{}-{name}", std::process::id()));
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

/// Runs `src` on every engine and checks each prints `expected`.
fn runs_everywhere(name: &str, src: &str, expected: &str) {
    for engine in ["ast", "sir", "native", "auto"] {
        let flag = format!("--engine={engine}");
        let o = sky(&format!("{name}-{engine}"), &["run", &flag], src);
        assert!(o.status.success(), "{engine}: {}", err(&o));
        assert_eq!(out(&o), expected, "{engine}");
    }
}

const REASSIGNED: &str = "\
Q = identity(2)
b = [1, 1]
for i in range(3):
    x = Q⁻¹ @ b
    print(x)
    Q = 2 * Q
";

#[test]
fn a_tag_the_body_breaks_is_not_trusted_on_the_next_iteration() {
    runs_everywhere(
        "reassigned",
        REASSIGNED,
        "[1, 1]\n[0.5, 0.5]\n[0.25, 0.25]\n",
    );
}

#[test]
fn explain_solves_instead_of_transposing() {
    let o = sky("reassigned-explain", &["check", "--explain"], REASSIGNED);
    assert!(o.status.success(), "{}", err(&o));
    let e = err(&o);
    assert!(
        e.contains("[LA-CHAIN-001] `Q⁻¹ @ b` is rewritten to `solve[diagonal](Q, b)`"),
        "{e}"
    );
    assert!(!e.contains("Qᵀ"), "{e}");
    assert!(!e.contains("[LA-001]"), "{e}");
}

#[test]
fn a_tag_the_body_keeps_is_still_used() {
    let src = REASSIGNED.replace("Q = 2 * Q", "Q = Q");
    runs_everywhere("kept", &src, "[1, 1]\n[1, 1]\n[1, 1]\n");
    let o = sky("kept-explain", &["check", "--explain"], &src);
    let e = err(&o);
    assert!(
        e.contains("[LA-CHAIN-001] `Q⁻¹ @ b` is rewritten to `Qᵀ @ b`"),
        "{e}"
    );
    assert!(e.contains("[LA-001] `Qᵀ` is rewritten to `Q`"), "{e}");
}

#[test]
fn conditions_continues_nested_loops_and_unknown_results() {
    let src = "\
def grow(A):
    return A + A

b = [1, 1]
Q = identity(2)
k = 0
while (Q⁻¹ @ b)[0] > 0.2:
    print(Q⁻¹ @ b)
    Q = 2 * Q
    k = k + 1
print(k)
P = identity(2)
for i in range(2):
    for j in range(2):
        print(P⁻¹ @ b)
    P = grow(P)
R = identity(2)
for i in range(3):
    print(R⁻¹ @ b)
    if i == 0:
        R = 3 * R
        continue
    R = R
S = identity(2)
for i in range(3):
    print(Sᵀ @ b)
    S = [[1, 2], [3, 4]]
";
    let third = "[0.3333333333333333, 0.3333333333333333]";
    let expected = format!(
        "[1, 1]\n[0.5, 0.5]\n[0.25, 0.25]\n3\n\
         [1, 1]\n[1, 1]\n[0.5, 0.5]\n[0.5, 0.5]\n\
         [1, 1]\n{third}\n{third}\n\
         [1, 1]\n[4, 6]\n[4, 6]\n"
    );
    runs_everywhere("paths", src, &expected);
}

#[test]
fn deeply_nested_loops_check_quickly() {
    // Each level would double the work if every loop re-checked its body
    // from scratch; a nested loop settles in one extra pass instead.
    let depth = 24;
    let mut src = String::from("Q = identity(2)\nA = identity(2)\nb = [1, 1]\n");
    for k in 0..depth {
        src.push_str(&format!("{}for i{k} in range(1):\n", "    ".repeat(k)));
    }
    let ind = "    ".repeat(depth);
    src.push_str(&format!(
        "{ind}print(Q⁻¹ @ b)\n{ind}A = Q\n{ind}Q = 2 * Q\n"
    ));
    let start = std::time::Instant::now();
    let o = sky("deep", &["check"], &src);
    assert!(o.status.success(), "{}", err(&o));
    assert!(start.elapsed().as_secs() < 20, "took {:?}", start.elapsed());
}

#[test]
fn an_argument_must_have_the_tags_of_its_parameter() {
    let src = "\
def f(Q: Orthogonal Matrix[2, 2], b: Vector[2]) -> Vector[2]:
    return Q⁻¹ @ b

A = [[2, 0], [0, 2]]
print(f(A, [1, 1]))
";
    let o = sky("argument", &["run"], src);
    assert!(!o.status.success());
    assert_eq!(out(&o), "");
    assert!(
        err(&o).contains("`Q` of `f` must be Orthogonal, but `A` is not known to be Orthogonal"),
        "{}",
        err(&o)
    );
    let ok = src.replace("f(A,", "f(identity(2),");
    runs_everywhere("argument-ok", &ok, "[1, 1]\n");
}
