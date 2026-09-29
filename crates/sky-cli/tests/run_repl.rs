//! `sky run` and `sky repl`: stdout, stderr and exit status end to end.
//!
//! Scripts are written to a scratch directory that has a `.git`, so project
//! discovery stops there and never looks at the real checkout.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("run-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(d.join(".git")).unwrap();
    d
}

/// Writes `src` to a fresh script and runs `sky run` on it.
fn run_script(name: &str, src: &str) -> Output {
    let dir = scratch(name);
    let file = dir.join("prog.sky");
    fs::write(&file, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_sky"))
        .arg("run")
        .arg(&file)
        .current_dir(&dir)
        .output()
        .expect("run sky");
    fs::remove_dir_all(&dir).unwrap();
    out
}

/// Runs `sky repl` with `input` on stdin.
fn repl(input: &str) -> Output {
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
        .write_all(input.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

fn out(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn err(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

#[test]
fn run_prints_a_physics_script_in_si() {
    let o = run_script(
        "physics",
        "g = 9.81 m/s^2\n\
         t = 2 s\n\
         print(\"fall:\", 0.5 * g * t^2)\n\
         v = 36 km/h\n\
         print(v, v * 10 s)\n\
         m = 2 kg\n\
         print(\"F =\", m * g)\n\
         print(\"E =\", 0.5 * m * v^2)\n\
         print(10 ms, 1_000, 3 * (100 ms) == 0.3 s)\n\
         print()\n\
         m * g == 19.62 N\n",
    );
    assert_eq!(err(&o), "");
    assert_eq!(
        out(&o),
        "fall: 19.62 m\n\
         10 m/s 100 m\n\
         F = 19.62 N\n\
         E = 100 J\n\
         0.01 s 1000 false\n\
         \n"
    );
    assert_eq!(o.status.code(), Some(0));
}

#[test]
fn top_level_expressions_are_not_printed_by_run() {
    let o = run_script("silent", "1 + 1\n5 m\n");
    assert_eq!((out(&o).as_str(), o.status.code()), ("", Some(0)));
}

#[test]
fn a_check_error_runs_nothing() {
    // The `print` before the error must not run: the whole file is checked
    // first.
    let o = run_script("check-error", "print(\"MARKER_RAN\")\ny = 3 m + 2 s\n");
    assert_eq!(o.status.code(), Some(1));
    assert_eq!(out(&o), "");
    let e = err(&o);
    assert!(e.contains("2:9: error: cannot add length and time"), "{e}");
    assert!(!e.contains("MARKER_RAN"), "{e}");
}

#[test]
fn a_parse_error_runs_nothing() {
    let o = run_script("parse-error", "print(1)\nx = = 2\n");
    assert_eq!(o.status.code(), Some(1));
    assert_eq!(out(&o), "");
    assert!(err(&o).contains("error:"));
}

#[test]
fn a_lex_error_runs_nothing() {
    let o = run_script("lex-error", "print(1)\nx = \"open\n");
    assert_eq!(o.status.code(), Some(1));
    assert_eq!(out(&o), "");
    assert!(err(&o).contains("unterminated string"));
}

#[test]
fn a_runtime_error_is_a_rendered_diagnostic() {
    let o = run_script(
        "runtime-error",
        "print(\"before\")\nt = 2 s\nprint(1 m / (t - 2 s))\nprint(\"after\")\n",
    );
    assert_eq!(o.status.code(), Some(1));
    // Output up to the failure stays, nothing after it runs.
    assert_eq!(out(&o), "before\n");
    let e = err(&o);
    assert!(e.contains("prog.sky:3:7: error: division by zero"), "{e}");
    assert!(e.contains("3 | print(1 m / (t - 2 s))"), "{e}");
    assert!(e.contains("this is zero"), "{e}");
}

#[test]
fn overflow_and_arrays_are_runtime_errors() {
    let o = run_script("overflow", "print(1e308 * 10)\n");
    assert_eq!(o.status.code(), Some(1));
    assert!(err(&o).contains("error: arithmetic overflow"));

    let o = run_script("arrays", "x = 1\nprint(x[0])\n");
    assert_eq!(o.status.code(), Some(1));
    assert!(err(&o).contains("arrays are not supported by the interpreter yet"));
}

#[test]
fn warnings_print_and_the_run_continues() {
    // A Cyrillic `а` in an identifier is a confusable-identifier warning.
    let o = run_script("warning", "аx = 1\nprint(аx)\n");
    assert_eq!(o.status.code(), Some(0));
    assert_eq!(out(&o), "1\n");
    assert!(err(&o).contains("warning:"));
}

#[test]
fn a_user_print_shadows_the_builtin() {
    let o = run_script("shadow", "print = 3\nprint(1)\n");
    // The checker lets the call through (its type is unknown); running it
    // fails because a number is not a function.
    assert_eq!(o.status.code(), Some(1));
    assert!(err(&o).contains("cannot call a number"));
}

#[test]
fn deep_nesting_runs() {
    let src = format!("print({}1{})\n", "(".repeat(200), ")".repeat(200));
    let o = run_script("deep", &src);
    assert_eq!((out(&o).as_str(), o.status.code()), ("1\n", Some(0)));
}

#[test]
fn check_still_never_runs_code() {
    let dir = scratch("check-no-exec");
    let file = dir.join("prog.sky");
    fs::write(&file, "print(\"side effect\")\n").unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_sky"))
        .arg("check")
        .arg(&file)
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(0));
    let stdout = out(&o);
    assert!(stdout.ends_with("prog.sky: ok\n"), "{stdout}");
    assert!(!stdout.contains("side effect"));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn run_usage_and_input_errors_exit_2() {
    let sky = env!("CARGO_BIN_EXE_sky");
    for args in [
        &["run"][..],
        &["run", "a", "b"],
        &["run", "--x"],
        &["repl", "x"],
    ] {
        let o = Command::new(sky).args(args).output().unwrap();
        assert_eq!(o.status.code(), Some(2), "{args:?}");
        assert!(err(&o).starts_with("usage:"), "{args:?}");
    }
    let o = Command::new(sky)
        .args(["run", "/nonexistent/prog.sky"])
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2));
}

#[test]
fn repl_evaluates_lines_and_echoes_expressions() {
    let o = repl("x = 10 km\nx / 2 s\nx > 5 m\nprint(\"hi\", x)\n\"a\\nb\"\n");
    assert_eq!(o.status.code(), Some(0));
    assert_eq!(out(&o), "5000 m/s\ntrue\nhi 10000 m\n\"a\\nb\"\n");
    // Piped stdin: no prompt.
    assert_eq!(err(&o), "");
}

#[test]
fn repl_output_can_be_pasted_back() {
    let o = repl("9.81 m / 1 s^2\n5 kg * 2 m/s^2\n2 J / 1 s\n");
    assert_eq!(out(&o), "9.81 m/s^2\n10 N\n2 W\n");
    let o = repl("9.81 m/s^2\n10 N\n2 W\n");
    assert_eq!(out(&o), "9.81 m/s^2\n10 N\n2 W\n");
}

#[test]
fn repl_recovers_from_errors_without_poisoning_the_session() {
    let o = repl(
        "x = 1 m\n\
         y = x + 1 s\n\
         y\n\
         z = 1 / 0\n\
         z\n\
         w = = 3\n\
         x = 5 m\n\
         x\n",
    );
    assert_eq!(o.status.code(), Some(0));
    // Only the last line echoes: the failed lines bound nothing, and `x`
    // (rebound after them) kept working.
    assert_eq!(out(&o), "5 m\n");
    let e = err(&o);
    assert!(
        e.contains("<repl>:1:7: error: cannot add length and time"),
        "{e}"
    );
    // `y` and `z` do not exist: the checker refuses, so it is a check error.
    assert_eq!(e.matches("cannot find `y` in this scope").count(), 1, "{e}");
    assert!(e.contains("division by zero"), "{e}");
    assert_eq!(e.matches("cannot find `z` in this scope").count(), 1, "{e}");
}

#[test]
fn repl_runtime_error_rolls_back_that_line_only() {
    // The failing line binds `a` before it fails on `1 / 0`: `a` must not survive.
    let o = repl("a = 1\nb = a / 0\nb\nprint(a)\nc: Vector[2]\nprint(a)\n");
    assert_eq!(out(&o), "1\n1\n");
    let e = err(&o);
    assert!(e.contains("division by zero"), "{e}");
    assert!(e.contains("arrays are not supported"), "{e}");
}

#[test]
fn repl_keeps_names_across_lines_and_compound_assignment() {
    let o = repl("d = 3 m\nd += 4 m\nd *= 2\nd\nprint(d)\n");
    assert_eq!(out(&o), "14 m\n14 m\n");
}

#[test]
fn repl_handles_blank_lines_comments_and_a_missing_final_newline() {
    let o = repl("\n# just a comment\n1 + 1\n\n2 + 2");
    assert_eq!(out(&o), "2\n4\n");
    assert_eq!(o.status.code(), Some(0));
}

#[test]
fn repl_handles_empty_input_and_invalid_utf8() {
    let o = repl("");
    assert_eq!((out(&o).as_str(), o.status.code()), ("", Some(0)));
    let mut child = Command::new(env!("CARGO_BIN_EXE_sky"))
        .arg("repl")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"\xff\xfe\n1 + 1\n")
        .unwrap();
    let o = child.wait_with_output().unwrap();
    assert_eq!(out(&o), "2\n");
    assert!(err(&o).contains("not valid UTF-8"));
    assert_eq!(o.status.code(), Some(0));
}

#[test]
fn repl_warnings_do_not_stop_a_line() {
    let o = repl("аx = 4\nаx\n");
    assert_eq!(out(&o), "4\n");
    assert!(err(&o).contains("warning:"));
}
