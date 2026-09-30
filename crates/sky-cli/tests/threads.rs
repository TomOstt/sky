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

/// A `sky tune` database whose thresholds make every kernel of two or more
/// elements split (`fused_work` 0 stays 0 for native code, which scales it),
/// so that small test programs take the parallel paths on every engine.
fn eager_db(name: &str) -> String {
    let dir = scratch(name);
    let db = dir.join("tune.db");
    let fp = sky_hw::current_fingerprint();
    let row = |kernel: &str, params: &str, default: &str| {
        format!(
            "result fp={fp} kernel={kernel} size=all params={params} default={default} best_ns=1 default_ns=2 when=1790000000\n"
        )
    };
    let mut text = String::from("sky-tune-db 1\n");
    text += &row("par_fused_work", "work:0", "work:4096");
    text += &row("par_array_elements", "elements:2", "elements:262144");
    fs::write(&db, text).unwrap();
    db.to_str().unwrap().to_string()
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
        text.contains("`noisy[x:Scalar]` prints, and output order is behaviour"),
        "{text}"
    );
}

/// A vector literal of `n` entries, `special` at the indexes in `at`.
fn vector(n: usize, base: &str, special: &str, at: &[usize]) -> String {
    let entries: Vec<&str> = (0..n)
        .map(|i| if at.contains(&i) { special } else { base })
        .collect();
    format!("[{}]", entries.join(", "))
}

/// A kernel big enough to be split on every thread count (with the default
/// thresholds: 2000 elements of a body of three operations, in four chunks),
/// that overflows in three different chunks, and whose other operations fail
/// too: the error is the first operation's, at every `N`.
fn big_kernel_program() -> String {
    format!(
        "x = {}\nk = 1e300\nz = 0\na = x * k\nb = a + x\nc = b / z\nprint(c[0])\n",
        vector(2_000, "2", "3e10", &[100, 900, 1_999])
    )
}

#[test]
fn native_output_and_error_do_not_depend_on_the_thread_count() {
    let src = big_kernel_program();
    let baseline = sky(
        "native-base",
        &["run", "--engine=native", "--threads=1"],
        &[],
        &src,
    );
    assert!(
        err(&baseline).contains("arithmetic overflow"),
        "{}",
        err(&baseline)
    );
    assert_eq!(baseline.status.code(), Some(1));
    // The SIR interpreter, sequential, is the oracle.
    let sir = sky(
        "native-sir",
        &["run", "--engine=sir", "--threads=1"],
        &[],
        &src,
    );
    assert_eq!(err(&baseline), err(&sir), "native differs from SIR");
    for n in ["2", "4", "8", "64"] {
        let flag = format!("--threads={n}");
        let o = sky("native-n", &["run", "--engine=native", &flag], &[], &src);
        assert_eq!(out(&o), out(&baseline), "stdout with {flag}");
        assert_eq!(err(&o), err(&baseline), "stderr with {flag}");
        assert_eq!(
            o.status.code(),
            baseline.status.code(),
            "status with {flag}"
        );
    }
    // With every threshold at its minimum the kernel is split in every chunk
    // (`sky tune`'s database, in the child's environment only).
    let db = eager_db("native-db");
    for n in ["2", "4", "8"] {
        let flag = format!("--threads={n}");
        let o = sky(
            "native-eager",
            &["run", "--engine=native", &flag],
            &[("SKY_TUNE_DB", &db)],
            &src,
        );
        assert_eq!(out(&o), out(&baseline), "stdout with {flag}, eager");
        assert_eq!(err(&o), err(&baseline), "stderr with {flag}, eager");
        assert_eq!(o.status.code(), baseline.status.code(), "{flag}, eager");
    }
    // The environment variable is the default; the flag wins; `auto` is native.
    for (args, env) in [
        (vec!["run", "--engine=native"], vec![("SKY_THREADS", "3")]),
        (vec!["run"], vec![("SKY_THREADS", "4")]),
        (
            vec!["run", "--engine=native", "--threads=2"],
            vec![("SKY_THREADS", "nonsense")],
        ),
    ] {
        let o = sky("native-env", &args, &env, &src);
        assert_eq!(out(&o), out(&baseline), "{args:?} {env:?}");
        assert_eq!(err(&o), err(&baseline), "{args:?} {env:?}");
    }
}

#[test]
fn native_successful_output_does_not_depend_on_the_thread_count() {
    let entries: Vec<String> = (0..2_500)
        .map(|i| format!("{}", (i % 89) as f64 * 0.37 - 11.0))
        .collect();
    let src = format!(
        "x = [{}]\ny = x * 3 - x * 0.5\nz = (x + y) * 0.25 + (x - y) * 0.125\nprint(z[0], z[2499], z[1234])\n",
        entries.join(", ")
    );
    let baseline = sky(
        "native-ok-base",
        &["run", "--engine=native", "--threads=1"],
        &[],
        &src,
    );
    assert_eq!(baseline.status.code(), Some(0), "{}", err(&baseline));
    let db = eager_db("native-ok-db");
    for n in ["2", "4", "8"] {
        let flag = format!("--threads={n}");
        for env in [vec![], vec![("SKY_TUNE_DB", db.as_str())]] {
            let o = sky("native-ok", &["run", "--engine=native", &flag], &env, &src);
            assert_eq!(out(&o), out(&baseline), "stdout with {flag} {env:?}");
            assert_eq!(err(&o), err(&baseline), "stderr with {flag} {env:?}");
        }
    }
    // And the tree-walker agrees.
    let ast = sky("native-ok-ast", &["run", "--engine=ast"], &[], &src);
    assert_eq!(out(&ast), out(&baseline));
}

#[test]
fn explain_says_native_code_runs_a_kernel_on_the_threads_too() {
    let entries: Vec<String> = (0..3000).map(|i| format!("{}", i % 7)).collect();
    let src = format!(
        "x = [{}]\ny = x * 3 - x * 2\nz = (x + y) * 0.25 + (x - y) * 0.125\nprint(z[0])\n",
        entries.join(", ")
    );
    let o = sky(
        "explain-native",
        &["check", "--explain", "--threads=4"],
        &[],
        &src,
    );
    let text = err(&o);
    assert!(
        text.contains("this elementwise kernel over 3e3 elements runs on 4 threads"),
        "{text}"
    );
    // Native code computes an element so fast that 3000 are not worth splitting.
    assert!(
        text.contains("native code runs it in one piece: a compiled kernel computes an element so fast that splitting pays only from"),
        "{text}"
    );
    // With the threshold tuned down, native code splits it too.
    let db = eager_db("explain-native-db");
    let o = sky(
        "explain-native-eager",
        &["check", "--explain", "--threads=4"],
        &[("SKY_TUNE_DB", &db)],
        &src,
    );
    assert!(
        err(&o).contains("native code runs it on 4 threads too, in the same chunks"),
        "{}",
        err(&o)
    );
    // Sequential: no such claim.
    let o = sky(
        "explain-native1",
        &["check", "--explain", "--threads=1"],
        &[],
        &src,
    );
    assert!(!err(&o).contains("native code runs it"), "{}", err(&o));
}

#[test]
fn explain_says_which_calls_native_code_starts() {
    // Two small independent calls: the interpreter starts them, native code
    // (which runs an operation 64 times faster) does not.
    let o = sky(
        "explain-calls",
        &["check", "--explain", "--threads=4"],
        &[],
        PROGRAM,
    );
    let text = err(&o);
    assert!(text.contains("native code runs them in place"), "{text}");
}
