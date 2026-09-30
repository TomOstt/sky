//! Planned `@` chains end to end: `sky run` on the tree-walker and on SIR
//! agree byte for byte (stdout, stderr, exit status), and `sky check
//! --explain` says what it did.
//!
//! Scripts run in a scratch directory with a `.git`, so project discovery
//! stops there.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("chains-{}-{name}", std::process::id()));
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

const PROGRAM: &str = "\
A = [[1, 2], [3, 4]]
B = [[0.5, 1], [1, 0.25]]
C = [[2, 0.1], [0, 2]]
v = [1, 2]
print(A @ B @ C @ v)
print(inv(A) @ inv(B) @ v)
print(v @ inv(A), inv(A) @ B)
D: Diagonal Matrix[2, 2] = [[2, 0], [0, 8]]
print(inv(D) @ v)
Q = identity(2)
print(inv(Q) @ v)
";

#[test]
fn both_engines_print_the_same_bytes() {
    let ast = sky("ast", &["run", "--engine=ast"], PROGRAM);
    let sir = sky("sir", &["run", "--engine=sir"], PROGRAM);
    let native = sky("native", &["run", "--engine=native"], PROGRAM);
    let auto = sky("auto", &["run"], PROGRAM);
    assert!(ast.status.success(), "{}", err(&ast));
    assert_eq!(out(&ast).lines().count(), 5);
    assert_eq!(out(&sir), out(&ast));
    assert_eq!(out(&native), out(&ast));
    assert_eq!(out(&auto), out(&ast));
    assert_eq!(err(&sir), err(&ast));
    assert_eq!(err(&native), err(&ast));
}

#[test]
fn both_engines_fail_with_the_same_diagnostic() {
    let src = "A = [[1, 2], [2, 4]]\nprint(1)\nprint(inv(A) @ [1, 2])\n";
    let ast = sky("fail-ast", &["run", "--engine=ast"], src);
    let sir = sky("fail-sir", &["run", "--engine=sir"], src);
    let native = sky("fail-native", &["run", "--engine=native"], src);
    assert!(!ast.status.success());
    assert_eq!(out(&ast), "1\n");
    assert_eq!(out(&sir), out(&ast));
    assert_eq!(err(&sir), err(&ast));
    assert_eq!(out(&native), out(&ast));
    assert_eq!(err(&native), err(&ast));
    assert!(
        err(&ast).contains("the matrix is singular"),
        "{}",
        err(&ast)
    );
}

#[test]
fn explain_names_the_chain_and_the_saving() {
    let src = "\
A: Matrix[500, 500]
B: Matrix[500, 500]
C: Matrix[500, 500]
v: Vector[500]
x = A @ B @ C @ v
y = inv(A) @ v
";
    let o = sky("explain", &["check", "--explain"], src);
    let e = err(&o);
    assert!(
        e.contains("`A @ B @ C @ v` is rewritten to `A @ (B @ (C @ v))`"),
        "{e}"
    );
    assert!(
        e.contains(
            "estimated cost: 500,500,000 flops as written, 1,500,000 flops after the rewrite"
        ),
        "{e}"
    );
    assert!(
        e.contains("`inv(A) @ v` is rewritten to `solve[lu](A, v)`"),
        "{e}"
    );
    assert!(e.contains("A⁻¹ is never formed"), "{e}");
    assert!(e.contains("not bit-identical to the written form"), "{e}");
    // Without --explain the rewrite is silent.
    let quiet = err(&sky("quiet", &["check"], src));
    assert!(!quiet.contains("rewritten"), "{quiet}");
}

#[test]
fn parentheses_force_the_written_order() {
    let src = "\
A: Matrix[500, 500]
B: Matrix[500, 500]
C: Matrix[500, 500]
v: Vector[500]
x = ((A @ B) @ C) @ v
";
    let e = err(&sky("forced", &["check", "--explain"], src));
    assert!(!e.contains("rewritten"), "{e}");
}
