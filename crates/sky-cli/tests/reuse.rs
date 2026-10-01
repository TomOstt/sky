//! Buffer reuse, end to end: a value that is still needed must keep its
//! contents when an array operation reuses the buffer of an operand that
//! dies. Every program runs under the tree-walker (`--engine=ast`) and under
//! SIR (`--engine=sir`, which runs the optimized module and moves each
//! operand at its last use), and both must print exactly the expected text.
//!
//! The aliasing cases are the ones a wrong "last use" would break: `b = a`
//! before `a = a + 1`, a value used after a branch, a value read by the next
//! iteration of a loop, a function that returns its argument, the same array
//! twice in one operation, and a global that a called function reads while
//! the caller reassigns it. `--explain` is checked at the end: the note that
//! says a loop reuses a buffer.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("reuse-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(d.join(".git")).unwrap();
    d
}

fn sky(name: &str, src: &str, args: &[&str]) -> Output {
    let dir = scratch(name);
    let file = dir.join("prog.sky");
    fs::write(&file, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_sky"))
        .args(args)
        .arg(&file)
        .current_dir(&dir)
        .output()
        .expect("run sky");
    fs::remove_dir_all(&dir).unwrap();
    out
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).unwrap()
}

/// Runs `src` on both engines and asserts they print `expected` and succeed.
fn prints(name: &str, src: &str, expected: &str) {
    for engine in ["--engine=ast", "--engine=sir"] {
        let o = sky(name, src, &["run", engine]);
        assert!(
            o.status.success(),
            "{engine} failed on:\n{src}\n{}",
            text(&o.stderr)
        );
        assert_eq!(text(&o.stdout), expected, "{engine} on:\n{src}");
    }
}

#[test]
fn an_alias_keeps_the_old_value() {
    prints(
        "alias",
        "a = [1 m, 2 m, 3 m]\nb = a\na = a + [1 m, 1 m, 1 m]\nprint(b)\nprint(a)\n",
        "[1 m, 2 m, 3 m]\n[2 m, 3 m, 4 m]\n",
    );
}

#[test]
fn an_alias_of_a_scaled_array() {
    prints(
        "alias-scale",
        "a = [1 m, 2 m]\nb = a\na = a * 3\nc = a\na = a / 3\nprint(a, b, c)\n",
        "[1 m, 2 m] [1 m, 2 m] [3 m, 6 m]\n",
    );
}

#[test]
fn the_same_array_twice_in_one_operation() {
    prints(
        "twice",
        "a = [1 m, 2 m, 3 m]\nb = a + a\nc = b - a\nprint(a, b, c)\nd = a + a + a\nprint(a, d)\n",
        "[1 m, 2 m, 3 m] [2 m, 4 m, 6 m] [1 m, 2 m, 3 m]\n[1 m, 2 m, 3 m] [3 m, 6 m, 9 m]\n",
    );
}

#[test]
fn a_value_used_after_a_branch() {
    for cond in ["true", "false"] {
        let src = format!(
            "a = [1 m, 2 m]\nb = [10 m, 20 m]\nif {cond}:\n    c = a + b\nelse:\n    c = b - a\nprint(a, b, c)\n"
        );
        let expected = if cond == "true" {
            "[1 m, 2 m] [10 m, 20 m] [11 m, 22 m]\n"
        } else {
            "[1 m, 2 m] [10 m, 20 m] [9 m, 18 m]\n"
        };
        prints("branch", &src, expected);
    }
}

#[test]
fn a_value_reassigned_in_one_branch_only() {
    for cond in ["true", "false"] {
        let src = format!(
            "a = [1 m, 2 m]\nsnap = a\nif {cond}:\n    a = a + a\nprint(a, snap)\nprint(a + snap)\n"
        );
        let expected = if cond == "true" {
            "[2 m, 4 m] [1 m, 2 m]\n[3 m, 6 m]\n"
        } else {
            "[1 m, 2 m] [1 m, 2 m]\n[2 m, 4 m]\n"
        };
        prints("one-branch", &src, expected);
    }
}

#[test]
fn a_loop_updates_an_array_and_the_snapshot_keeps_the_start() {
    prints(
        "loop",
        "dt = 0.5 s\nv = [1 m/s, 2 m/s]\nx = [0 m, 0 m]\nstart = x\nfor i in range(3):\n    x = x + dt * v\n    v = v + dt * [2 m/s^2, 4 m/s^2]\nprint(start)\nprint(x)\nprint(v)\n",
        "[0 m, 0 m]\n[3 m, 6 m]\n[4 m/s, 8 m/s]\n",
    );
}

#[test]
fn a_value_read_by_the_next_iteration_of_a_loop() {
    // `base` is defined outside the loop and read in every iteration: it must
    // not be consumed by the first one.
    prints(
        "loop-invariant",
        "base = [1 m, 2 m]\nacc = [0 m, 0 m]\nfor i in range(4):\n    acc = acc + base\nprint(base, acc)\n",
        "[1 m, 2 m] [4 m, 8 m]\n",
    );
    prints(
        "loop-invariant-while",
        "base = [1 m, 2 m]\nacc = base\nn = 0\nwhile n < 3:\n    acc = acc + base\n    n = n + 1\nprint(base, acc)\n",
        "[1 m, 2 m] [4 m, 8 m]\n",
    );
}

#[test]
fn break_and_continue_with_arrays() {
    prints(
        "break",
        "a = [1 m, 1 m]\nb = a\nfor i in range(10):\n    if i == 2:\n        continue\n    a = a + a\n    if i == 4:\n        break\nprint(a, b)\n",
        "[16 m, 16 m] [1 m, 1 m]\n",
    );
}

#[test]
fn a_function_that_returns_its_argument() {
    prints(
        "identity",
        "def same(v: Vector[2] m) -> Vector[2] m:\n    return v\na = [1 m, 2 m]\nb = same(a)\na = a + a\nc = same(a)\nprint(a, b, c)\nb = b + b\nprint(a, b, c)\n",
        "[2 m, 4 m] [1 m, 2 m] [2 m, 4 m]\n[2 m, 4 m] [2 m, 4 m] [2 m, 4 m]\n",
    );
}

#[test]
fn a_function_that_updates_and_returns_its_argument() {
    prints(
        "callee-updates",
        "def bump(v: Vector[2] m) -> Vector[2] m:\n    v = v + v\n    return v\na = [1 m, 2 m]\nb = bump(a)\nprint(a, b)\nc = bump(bump(a))\nprint(a, c)\n",
        "[1 m, 2 m] [2 m, 4 m]\n[1 m, 2 m] [4 m, 8 m]\n",
    );
}

#[test]
fn the_same_array_passed_twice() {
    prints(
        "pass-twice",
        "def add2(u: Vector[2] m, w: Vector[2] m) -> Vector[2] m:\n    return u + w\na = [1 m, 2 m]\nb = add2(a, a)\nprint(a, b)\nc = add2(b, a)\nprint(a, b, c)\n",
        "[1 m, 2 m] [2 m, 4 m]\n[1 m, 2 m] [2 m, 4 m] [3 m, 6 m]\n",
    );
}

#[test]
fn a_function_reads_the_global_the_caller_reassigns() {
    // The right side calls a function that reads `v` as a module-level name:
    // the caller must not have moved `v` out first.
    prints(
        "global",
        "v = [1, 2, 3]\ndef first():\n    return v[0] + v[2]\nv = v * first()\nprint(v)\nv = v + v * first()\nprint(v)\n",
        "[4, 8, 12]\n[68, 136, 204]\n",
    );
}

#[test]
fn a_local_shadows_a_module_array() {
    // Inside the function `w = w + w` reads the parameter `w`, which holds
    // the module's array, and rebinds the local: the module's must survive.
    // (Reading the module's `w` itself before binding a local `w` is a
    // check error: the local hides it in the whole body.)
    prints(
        "shadow",
        "w = [1, 2]\ndef f(w: Vector[2]) -> Vector[2]:\n    w = w + w\n    return w\nprint(f(w))\nprint(w)\n",
        "[2, 4]\n[1, 2]\n",
    );
}

#[test]
fn compound_assignment_and_indexing() {
    prints(
        "compound",
        "a = [1 m, 2 m, 3 m]\nb = a\na += a\na -= b\nprint(a, b)\nc = a[1]\na *= 2\nprint(a, c)\n",
        "[1 m, 2 m, 3 m] [1 m, 2 m, 3 m]\n[2 m, 4 m, 6 m] 2 m\n",
    );
}

#[test]
fn a_matrix_and_its_transpose() {
    prints(
        "matrix",
        "M = [[1, 2], [3, 4]]\nN = M\nM = M + M\nprint(M, N)\nprint(M @ Nᵀ)\nM = Mᵀ\nprint(M, N)\n",
        "[[2, 4], [6, 8]] [[1, 2], [3, 4]]\n[[10, 22], [22, 50]]\n[[2, 6], [4, 8]] [[1, 2], [3, 4]]\n",
    );
}

#[test]
fn errors_are_the_same_when_a_buffer_was_being_reused() {
    let src = "a = [1 m, 2 m]\nb = [1 m, 0 m]\nprint(a)\na = a / (b[1] / 1 m)\nprint(a)\n";
    // The same file name for both: the diagnostic prints it.
    let ast = sky("err", src, &["run", "--engine=ast"]);
    let sir = sky("err", src, &["run", "--engine=sir"]);
    assert!(!ast.status.success());
    assert_eq!(sir.status.code(), ast.status.code());
    assert_eq!(text(&sir.stdout), text(&ast.stdout));
    assert_eq!(text(&sir.stderr), text(&ast.stderr));
    assert_eq!(text(&ast.stdout), "[1 m, 2 m]\n");
    assert!(text(&ast.stderr).contains("division by zero"));
}

#[test]
fn a_time_stepping_loop_prints_the_same_numbers() {
    let src = "dt = 0.001 s\nk = 2 / (1 s)^2\nx = [1 m, 2 m, 3 m, 4 m]\nv = [0 m/s, 0 m/s, 0 m/s, 0 m/s]\nn = 0\nwhile n < 1000:\n    a = -k * x\n    v = v + dt * a\n    x = x + dt * v\n    n = n + 1\nprint(x, v)\n";
    let ast = sky("step", src, &["run", "--engine=ast"]);
    let sir = sky("step", src, &["run", "--engine=sir"]);
    assert!(ast.status.success() && sir.status.success());
    assert_eq!(text(&ast.stdout), text(&sir.stdout));
}

#[test]
fn explain_says_which_loop_reuses_a_buffer() {
    let src = "dt = 0.001 s\nk = 2 / (1 s)^2\nx = [1 m, 2 m]\nv = [0 m/s, 0 m/s]\nn = 0\nwhile n < 1000:\n    a = -k * x\n    v = v + dt * a\n    x = x + dt * v\n    n = n + 1\nprint(x, v)\n";
    let o = sky("explain", src, &["check", "--explain"]);
    assert!(o.status.success());
    let out = text(&o.stdout) + &text(&o.stderr);
    assert!(
        out.contains("[MEM-001] `v + dt * a` reuses `v`'s buffer: no allocation in this loop"),
        "{out}"
    );
    assert!(
        out.contains("[MEM-001] `x + dt * v` reuses `x`'s buffer: no allocation in this loop"),
        "{out}"
    );
    // The note is about `v` and `x`, the arrays the loop carries, not about
    // `a`, which is not reused (it is a fresh array inside the kernel).
    assert!(!out.contains("reuses `a`"), "{out}");
}

#[test]
fn explain_is_quiet_when_nothing_is_reused() {
    // Both operands are still needed afterwards.
    let src = "a = [1 m, 2 m]\nb = [3 m, 4 m]\nc = a + b\nprint(a, b, c)\n";
    let o = sky("explain-none", src, &["check", "--explain"]);
    let out = text(&o.stdout) + &text(&o.stderr);
    assert!(!out.contains("MEM-001"), "{out}");
}
