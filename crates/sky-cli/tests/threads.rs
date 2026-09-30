//! `--threads=N` and `SKY_THREADS` (spec "Threads"): the SIR engine may run
//! independent work on several threads, and nothing a program prints, no
//! error and no exit status depends on N. `sky check --explain` says what
//! would run in parallel.
//!
//! Scripts run in a scratch directory with a `.git`, so project discovery
//! stops there.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("threads-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(d.join(".git")).unwrap();
    d
}

/// `sky <args> prog.sky` on `src` with `env` set, in a scratch directory.
fn sky(name: &str, args: &[&str], env: &[(&str, &str)], src: &str) -> Output {
    let dir = scratch(name);
    fs::write(dir.join("prog.sky"), src).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_sky"));
    cmd.args(args)
        .arg("prog.sky")
        .env_remove("SKY_THREADS")
        .current_dir(&dir);
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("run sky");
    fs::remove_dir_all(&dir).unwrap();
    out
}

fn out(o: &Output) -> String {
    String::from_utf8(o.stdout.clone()).expect("utf-8")
}

fn err(o: &Output) -> String {
    String::from_utf8(o.stderr.clone()).expect("utf-8")
}

/// Two independent expensive calls, then a print, then a failure in a
/// third: output before an error, and the error, must be the same on every
/// thread count.
const PROGRAM: &str = "\
def work(n):
    total = 0
    i = 0
    while i < n:
        total = total + sqrt(i + 1) / (i + 1)
        i = i + 1
    return total

def bad(x):
    return 1 / x

a = work(3000)
b = work(4000)
print(a + b)
c = bad(0)
print(c)
";

#[test]
fn the_output_and_the_error_do_not_depend_on_the_thread_count() {
    let baseline = sky(
        "base",
        &["run", "--engine=sir", "--threads=1"],
        &[],
        PROGRAM,
    );
    assert!(out(&baseline).starts_with("233."), "{}", out(&baseline));
    assert!(err(&baseline).contains("division by zero"));
    assert_eq!(baseline.status.code(), Some(1));
    for n in ["2", "4", "8", "64"] {
        let flag = format!("--threads={n}");
        let o = sky("n", &["run", "--engine=sir", &flag], &[], PROGRAM);
        assert_eq!(out(&o), out(&baseline), "stdout with {flag}");
        assert_eq!(err(&o), err(&baseline), "stderr with {flag}");
        assert_eq!(
            o.status.code(),
            baseline.status.code(),
            "status with {flag}"
        );
    }
    // The environment variable is the default, the flag wins over it.
    let env = sky(
        "env",
        &["run", "--engine=sir"],
        &[("SKY_THREADS", "3")],
        PROGRAM,
    );
    assert_eq!(out(&env), out(&baseline));
    assert_eq!(err(&env), err(&baseline));
    let flag = sky(
        "flag",
        &["run", "--engine=sir", "--threads=2"],
        &[("SKY_THREADS", "nonsense")],
        PROGRAM,
    );
    assert_eq!(out(&flag), out(&baseline));
    // And the tree-walker, which is sequential, agrees with all of them.
    let ast = sky("ast", &["run", "--engine=ast", "--threads=4"], &[], PROGRAM);
    assert_eq!(out(&ast), out(&baseline));
    assert_eq!(err(&ast), err(&baseline));
}

#[test]
fn bad_thread_counts_are_usage_errors() {
    for flag in [
        "--threads=0",
        "--threads=x",
        "--threads=",
        "--threads=-2",
        "--threads=99999",
    ] {
        for cmd in ["run", "check"] {
            let o = sky("bad", &[cmd, flag], &[], "print(1)\n");
            assert_eq!(o.status.code(), Some(2), "{cmd} {flag}");
            assert!(
                err(&o).contains("thread count"),
                "{cmd} {flag}: {}",
                err(&o)
            );
        }
    }
    let o = sky(
        "twice",
        &["run", "--threads=2", "--threads=3"],
        &[],
        "print(1)\n",
    );
    assert_eq!(o.status.code(), Some(2));
}

#[test]
fn explain_says_what_runs_in_parallel_and_what_does_not() {
    let o = sky(
        "explain",
        &["check", "--explain", "--threads=4"],
        &[],
        PROGRAM,
    );
    let text = err(&o);
    assert!(
        text.contains(
            "[PAR-002] these 2 calls are independent (pure): run concurrently on up to 4 threads"
        ),
        "{text}"
    );
    // Sequential: no group.
    let o = sky(
        "explain1",
        &["check", "--explain", "--threads=1"],
        &[],
        PROGRAM,
    );
    assert!(!err(&o).contains("[PAR-002]"), "{}", err(&o));
    // The same on the thread count from the environment.
    let o = sky(
        "explain-env",
        &["check", "--explain"],
        &[("SKY_THREADS", "8")],
        PROGRAM,
    );
    assert!(
        err(&o).contains("run concurrently on up to 8 threads"),
        "{}",
        err(&o)
    );
}

#[test]
fn explain_says_why_a_small_kernel_is_not_split_and_why_a_call_that_prints_is_not() {
    let src = "\
a = [1, 2, 3]
b = [4, 5, 6]
x = (a + b) * 2 - a
print(x)
";
    let o = sky("small", &["check", "--explain", "--threads=8"], &[], src);
    let text = err(&o);
    assert!(
        text.contains("[PAR-003] this elementwise kernel over 3 elements is not parallelized: it may fail and is small"),
        "{text}"
    );
    let src = "\
def noisy(x):
    print(x)
    return x * 2

def work(n):
    total = 0
    i = 0
    while i < n:
        total = total + i
        i = i + 1
    return total

a = noisy(1)
b = work(10)
c = work(20)
print(a + b + c)
";
    let o = sky("prints", &["check", "--explain", "--threads=8"], &[], src);
    let text = err(&o);
    assert!(
        text.contains("`noisy` prints, and output order is behaviour"),
        "{text}"
    );
}
