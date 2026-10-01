//! A parameter's tags are promises the caller makes, and the rewrites that
//! rely on them do not all check them (`Kᵀ` of a Symmetric `K` is `K`). So
//! every engine checks the promise at the call, at the argument, with the
//! same message: a broken one is never a wrong answer, and never a
//! different answer on each engine. One differential test per tag, and per
//! way a tag is relied on.
//!
//! Scripts run in a scratch directory with a `.git`, so project discovery
//! stops there.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

const ENGINES: [&str; 4] = ["ast", "sir", "native", "auto"];

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("promises-{}-{name}", std::process::id()));
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

fn text(b: &[u8]) -> String {
    String::from_utf8(b.to_vec()).expect("utf-8")
}

/// Runs `src` on every engine: each prints `expected` and succeeds.
fn runs_everywhere(name: &str, src: &str, expected: &str) {
    for engine in ENGINES {
        let flag = format!("--engine={engine}");
        let o = sky(&format!("{name}-{engine}"), &["run", &flag], src);
        assert!(o.status.success(), "{engine}: {}", text(&o.stderr));
        assert_eq!(text(&o.stdout), expected, "{engine}");
    }
}

/// Runs `src` on every engine: each prints `stdout`, then fails with one
/// and the same diagnostic, which contains every line of `want`.
fn fails_everywhere(name: &str, src: &str, stdout: &str, want: &[&str]) {
    let mut first: Option<String> = None;
    for engine in ENGINES {
        let flag = format!("--engine={engine}");
        let o = sky(&format!("{name}-{engine}"), &["run", &flag], src);
        let e = text(&o.stderr);
        assert!(
            !o.status.success(),
            "{engine} succeeded: {}",
            text(&o.stdout)
        );
        assert_eq!(text(&o.stdout), stdout, "{engine}: {e}");
        for w in want {
            assert!(e.contains(w), "{engine}: no `{w}` in\n{e}");
        }
        match &first {
            None => first = Some(e),
            Some(f) => assert_eq!(&e, f, "{engine} differs from ast"),
        }
    }
}

/// A computed matrix the checker cannot prove anything of.
const UPPER: &str = "[[1, 2], [0, 1]] + 0 * identity(2)";
const DENSE: &str = "[[2, 1], [1, 2]] + 0 * identity(2)";

fn not_symmetric(declared: &str) -> String {
    format!("{declared}, but entries [0, 1] and [1, 0] differ")
}

#[test]
fn symmetric_transpose_is_not_trusted() {
    // `Kᵀ` of a Symmetric `K` is `K` on SIR: an unchecked promise would
    // print `[1, 0]` there and `[1, 2]` on the tree-walker.
    let src = format!(
        "def t(K: Symmetric Matrix[2, 2], f: Vector[2]) -> Vector[2]:\n    return Kᵀ @ f\nprint(1)\nprint(t({UPPER}, [1, 0]))\n"
    );
    fails_everywhere(
        "sym-transpose",
        &src,
        "1\n",
        &[
            "error: the matrix is not symmetric",
            "4:9",
            &not_symmetric("declared Symmetric"),
            "a tag on a parameter is checked when the function is called",
        ],
    );
}

#[test]
fn symmetric_gram_product_is_not_trusted() {
    let src = format!(
        "def t(K: Symmetric Matrix[2, 2]) -> Matrix[2, 2]:\n    return K @ Kᵀ\nprint(t({UPPER}))\n"
    );
    fails_everywhere(
        "sym-gram",
        &src,
        "",
        &["error: the matrix is not symmetric", "3:9"],
    );
}

#[test]
fn positive_definite_implies_a_checked_symmetry() {
    let src = format!(
        "def t(K: PositiveDefinite Matrix[2, 2], f: Vector[2]) -> Vector[2]:\n    return Kᵀ @ f\nprint(t({UPPER}, [1, 0]))\n"
    );
    fails_everywhere(
        "pd-transpose",
        &src,
        "",
        &[&not_symmetric(
            "declared PositiveDefinite, which implies symmetric",
        )],
    );
    // Symmetric but not positive definite: the Cholesky solve that relies
    // on it says so, in the body, as before.
    let src = "def t(K: PositiveDefinite Matrix[2, 2], f: Vector[2]) -> Vector[2]:\n    return K⁻¹ @ f\nprint(t([[1, 2], [2, 1]] + 0 * identity(2), [1, 1]))\n";
    fails_everywhere(
        "pd-indefinite",
        src,
        "",
        &["error: the matrix is not positive definite", "2:12"],
    );
}

#[test]
fn diagonal_products_are_not_trusted() {
    // `D @ x` of a Diagonal `D` is a diagonal product on SIR; on the
    // tree-walker it is a full one. Each use, and none at all.
    for (i, body) in ["D @ f", "Dᵀ @ f", "(D @ D) @ f", "D⁻¹ @ f", "f"]
        .into_iter()
        .enumerate()
    {
        let src = format!(
            "def t(D: Diagonal Matrix[2, 2], f: Vector[2]) -> Vector[2]:\n    return {body}\nprint(t({DENSE}, [1, 1]))\n"
        );
        fails_everywhere(
            &format!("diag-{i}"),
            &src,
            "",
            &[
                "error: the matrix is not diagonal",
                "3:9",
                "declared Diagonal, but entry [0, 1] is not zero",
            ],
        );
    }
}

#[test]
fn an_argument_the_checker_cannot_type_is_checked_too() {
    let src = "def g(x):\n    return [[1, x], [0, 1]]\ndef t(K: Symmetric Matrix[2, 2], f: Vector[2]) -> Vector[2]:\n    return Kᵀ @ f\nprint(t(g(2), [1, 0]))\n";
    fails_everywhere(
        "unknown-arg",
        src,
        "",
        &["error: the matrix is not symmetric", "5:9"],
    );
}

#[test]
fn a_derivative_checks_the_promise_like_a_call() {
    let src = format!(
        "def t(K: Symmetric Matrix[2, 2], x: Scalar) -> Scalar:\n    v = Kᵀ @ [x, 1]\n    return v[0] * x\nprint(derivative(t, x)({UPPER}, 3))\n"
    );
    fails_everywhere(
        "derivative",
        &src,
        "",
        &["error: the matrix is not symmetric", "4:24"],
    );
}

#[test]
fn promises_that_hold_run_as_before() {
    let src = "\
def t(K: Symmetric Matrix[2, 2], f: Vector[2]) -> Vector[2]:
    return Kᵀ @ f
def d(D: Diagonal Matrix[2, 2], f: Vector[2]) -> Vector[2]:
    return D @ f
def p(K: PositiveDefinite Matrix[2, 2], f: Vector[2]) -> Vector[2]:
    return K⁻¹ @ f
A = [[1, 2], [3, 4]]
print(t(A + Aᵀ, [1, 0]))
print(t([[1, 2], [2, 1]] + 0 * identity(2), [1, 0]))
print(t(A @ Aᵀ, [1, 0]))
print(d(2 * identity(2), [1, 1]))
print(d([[3, 0], [0, 4]] + 0 * identity(2), [1, 1]))
B = [[2, 1], [0, 3]]
print(p(B @ Bᵀ, [1, 1]))
";
    runs_everywhere(
        "hold",
        src,
        "[2, 5]\n[1, 2]\n[5, 11]\n[2, 2]\n[3, 4]\n[0.16666666666666666, 0.05555555555555556]\n",
    );
}

#[test]
fn a_promise_is_checked_on_every_call() {
    // The first two calls keep the promise, the third breaks it.
    let src = "\
def t(K: Symmetric Matrix[2, 2], f: Vector[2]) -> Vector[2]:
    return Kᵀ @ f
A = [[1, 2], [3, 4]]
for i in range(3):
    print(t(A + Aᵀ + i * (A - Aᵀ) * (i - 1), [1, 0]))
";
    fails_everywhere(
        "every-call",
        src,
        "[2, 5]\n[2, 5]\n",
        &["error: the matrix is not symmetric", "5:13"],
    );
}

#[test]
fn sums_of_symmetric_matrices_are_symmetric() {
    // `+` and `-` keep Symmetric and Diagonal, so these bind as declared
    // and a symmetric sum is passed on without a check.
    let src = "\
S: Symmetric Matrix[2, 2] = [[1, 2], [2, 5]]
D: Diagonal Matrix[2, 2] = 3 * identity(2)
T: Symmetric Matrix[2, 2] = S + S - D
E: Diagonal Matrix[2, 2] = D - identity(2)
print(T, E)
";
    runs_everywhere("sums", src, "[[-1, 4], [4, 7]] [[2, 0], [0, 2]]\n");
    let o = sky("sums-pd", &["check"], "P: PositiveDefinite Matrix[2, 2] = [[2, 1], [1, 2]]\nQ: PositiveDefinite Matrix[2, 2] = P - P\n");
    assert!(!o.status.success());
    let e = text(&o.stderr);
    assert!(
        e.contains("`P - P` is not known to be PositiveDefinite"),
        "{e}"
    );
    assert!(
        e.contains("pass it to a function whose parameter is declared PositiveDefinite"),
        "{e}"
    );
}
