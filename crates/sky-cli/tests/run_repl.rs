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

/// Writes `src` to a fresh script and runs `sky run [--engine=<engine>]`
/// on it.
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

/// The marker of a forced SIR run refusing a program that does not lower.
const NOT_ON_SIR: &str = "error: cannot run on SIR:";

/// The two limits that stop unbounded recursion. Which one trips first
/// depends on how much Rust stack each engine spends per call (a debug
/// tree-walker runs out of its stack budget before 1000 calls), not on the
/// program, so runs that end in either are compared without the message.
fn hit_a_limit(o: &Output) -> bool {
    let e = err(o);
    e.contains("recursion is too deep") || e.contains("nests too deeply")
}

/// Runs the program under `--engine=ast` and `--engine=sir` and asserts that
/// they agree on stdout, stderr and exit status, byte for byte. A program
/// SIR does not lower is refused by `--engine=sir` (that is the only thing
/// it may do differently); it is then reported, so the caller can tell a
/// comparison from a fallback. Returns whether SIR ran the program.
fn engines_agree(name: &str, src: &str) -> bool {
    let ast = run_engine(name, src, Some("ast"));
    let sir = run_engine(name, src, Some("sir"));
    if err(&sir).contains(NOT_ON_SIR) {
        assert_eq!(sir.status.code(), Some(1), "{}", err(&sir));
        assert_eq!(out(&sir), "", "a refused program prints nothing");
        eprintln!("REFUSED {name}: {}", err(&sir));
        return false;
    }
    assert_eq!(
        sir.status.code(),
        ast.status.code(),
        "exit status of:\n{src}"
    );
    assert_eq!(out(&sir), out(&ast), "stdout of:\n{src}");
    if !(hit_a_limit(&ast) && hit_a_limit(&sir)) {
        assert_eq!(err(&sir), err(&ast), "stderr of:\n{src}");
    }
    true
}

/// Writes `src` to a fresh script and runs `sky run` on it, under the
/// default engine (`auto`). Every program the tests here run is also run
/// under `--engine=ast` and `--engine=sir` and the three must agree
/// (see [`engines_agree`]), so each of these tests is also a differential
/// test of the SIR engine against the tree-walker.
fn run_script(name: &str, src: &str) -> Output {
    let auto = run_engine(name, src, None);
    let ast = run_engine(name, src, Some("ast"));
    assert_eq!(
        auto.status.code(),
        ast.status.code(),
        "auto exit status of:\n{src}"
    );
    assert_eq!(out(&auto), out(&ast), "auto stdout of:\n{src}");
    if !(hit_a_limit(&ast) && hit_a_limit(&auto)) {
        assert_eq!(err(&auto), err(&ast), "auto stderr of:\n{src}");
    }
    engines_agree(name, src);
    auto
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

    // A variable index past the end is a runtime error, on the index.
    let o = run_script("arrays", "v = [1, 2, 3]\ni = 3\nprint(v[i])\n");
    assert_eq!(o.status.code(), Some(1));
    assert!(
        err(&o).contains("index 3 is out of bounds for axis 0 with length 3"),
        "{}",
        err(&o)
    );
    let o = run_script("singular", "print([[1, 2], [2, 4]]⁻¹)\n");
    assert_eq!(o.status.code(), Some(1));
    assert!(
        err(&o).contains("error: the matrix is singular"),
        "{}",
        err(&o)
    );
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
    // The user's `print` is a number, so the call is a check error: nothing
    // runs.
    assert_eq!(o.status.code(), Some(1));
    assert_eq!(out(&o), "");
    assert!(err(&o).contains("cannot call a dimensionless number"));
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
fn repl_echoes_values_shown_in_a_unit() {
    let o = repl(
        "v = 90 km/h\n\
         v\n\
         v as km/h\n\
         (v as m/s)\n\
         1 AU as km\n\
         511 keV as MeV\n\
         u = [0.1 m, 0.2 m]\n\
         u as mm\n\
         print(v as km/h, v)\n\
         v * 2 as km/h\n",
    );
    assert_eq!(o.status.code(), Some(0));
    assert_eq!(
        out(&o),
        "25 m/s\n90 km/h\n25 m/s\n149597870.7 km\n0.511 MeV\n[100 mm, 200 mm]\n90 km/h 25 m/s\n180 km/h\n"
    );
    assert_eq!(err(&o), "");
}

#[test]
fn repl_shown_values_are_display_only_and_errors_do_not_poison_the_session() {
    let o = repl(
        "v = 90 km/h\n\
         v as kg\n\
         w = v as km/h\n\
         (v as km/h) + 1\n\
         v as km/h as m\n\
         v as parsecs\n\
         v as km/h\n",
    );
    assert_eq!(o.status.code(), Some(0));
    // Only the last line echoes; `w` was never bound.
    assert_eq!(out(&o), "90 km/h\n");
    let e = err(&o);
    assert!(
        e.contains("<repl>:1:1: error: cannot show a velocity in kg"),
        "{e}"
    );
    assert!(
        e.contains("error: a value shown in km/h can only be printed"),
        "{e}"
    );
    assert!(
        e.contains("error: a value can be shown in only one unit"),
        "{e}"
    );
    assert!(e.contains("error: unknown unit `parsecs`"), "{e}");
    assert!(e.contains("= help: did you mean `pc`?"), "{e}");
}

#[test]
fn run_prints_values_in_the_units_people_think_in() {
    let o = run_script(
        "as-units",
        "v = 90 km/h\n\
         print(v)\n\
         print(v as km/h)\n\
         print(\"1 AU is\", 1 AU as km)\n\
         print(\"E =\", 511 keV as eV, 511 keV as MeV)\n\
         u = [100 mm, 200 mm]\n\
         print(u as mm)\n\
         v as km/h\n",
    );
    assert_eq!(err(&o), "");
    assert_eq!(
        out(&o),
        "25 m/s\n90 km/h\n1 AU is 149597870.7 km\nE = 511000 eV 0.511 MeV\n[100 mm, 200 mm]\n"
    );
}

#[test]
fn run_rejects_a_wrong_dimension_and_arithmetic_on_shown_values() {
    let o = run_script("as-wrong", "print(\"MARKER\")\nprint(90 km/h as kg)\n");
    assert_eq!(o.status.code(), Some(1));
    assert_eq!(out(&o), "");
    assert!(
        err(&o).contains("prog.sky:2:7: error: cannot show a velocity in kg"),
        "{}",
        err(&o)
    );
    let o = run_script("as-arith", "print((1 km as m) + 1)\n");
    assert_eq!(o.status.code(), Some(1));
    assert!(
        err(&o).contains("error: a value shown in m can only be printed"),
        "{}",
        err(&o)
    );
}

#[test]
fn a_display_overflow_is_a_runtime_error() {
    let o = run_script("as-overflow", "print(1)\nprint(1e300 m as nm)\n");
    assert_eq!(o.status.code(), Some(1));
    assert_eq!(out(&o), "1\n");
    assert!(
        err(&o).contains("error: arithmetic overflow"),
        "{}",
        err(&o)
    );
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
    assert!(e.contains("`c` is declared but has no value"), "{e}");
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

#[test]
fn physical_constants_have_values_in_run_and_repl() {
    // The checker's prelude and the interpreter's prelude must agree: a
    // constant that checks must also have a value at run time.
    let o = run_script(
        "constants.sky",
        "m = 2 kg\nE = m * c^2\nprint(E)\nprint(hbar == ħ)\n",
    );
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    assert_eq!(out(&o), "1.7975103574736352e17 J\ntrue\n");

    let o = repl("k_B * 300 K\nq_e\n");
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    assert_eq!(out(&o), "4.141947e-21 J\n1.602176634e-19 C\n");
}

#[test]
fn functions_loops_and_conditionals_run() {
    let src = "\
def kinetic(m: kg, v: m/s) -> J:
    return 0.5 * m * v^2

print(kinetic(2 kg, 3 m/s))

def fact(n: Scalar) -> Scalar:
    if n <= 1:
        return 1
    return n * fact(n - 1)

print(fact(10))

total = 0 m
i = 0
while true:
    i += 1
    if i > 5:
        break
    if i == 3:
        continue
    total += 1 m
print(total, i)

speed = 400 m/s
print(\"supersonic\" if speed > 343 m/s else \"subsonic\")
";
    let o = run_script("blocks.sky", src);
    assert_eq!(o.status.code(), Some(0), "{}", err(&o));
    assert_eq!(out(&o), "9 J\n3628800\n4 m 6\nsupersonic\n");
}

#[test]
fn function_locals_do_not_leak_and_globals_are_read() {
    let src = "\
g = 2 m
def f(x: m) -> m:
    y = x + g
    return y
print(f(1 m))
print(y)
";
    // `y` is local to `f`: using it at module level is a check error.
    let o = run_script("locals.sky", src);
    assert_eq!(o.status.code(), Some(1));
    assert!(err(&o).contains("cannot find `y`"), "{}", err(&o));
    let o = run_script("locals_ok.sky", &src.replace("print(y)\n", ""));
    assert_eq!(out(&o), "3 m\n");
}

#[test]
fn runaway_recursion_is_an_error_not_a_crash() {
    // Deep nesting inside each recursive call multiplies the stack each
    // call needs; the interpreter must stop cleanly (exit 1, a diagnostic),
    // never abort on a stack overflow, in debug and release builds alike.
    for nesting in [0, 5, 250] {
        let body = format!("{}loop(n + 1){}", "(".repeat(nesting), ")".repeat(nesting));
        let src = format!("def loop(n: Scalar) -> Scalar:\n    return {body}\n\nloop(0)\n");
        let o = run_script("runaway.sky", &src);
        assert_eq!(o.status.code(), Some(1), "nesting {nesting}: {}", err(&o));
        let e = err(&o);
        assert!(
            e.contains("recursion is too deep") || e.contains("nests too deeply"),
            "nesting {nesting}: {e}"
        );
    }
}

#[test]
fn run_a_damped_oscillator_with_math_builtins() {
    // x = A exp(-t/tau) cos(omega t): `t / tau` and `2 pi omega t` are
    // dimensionless (Hz is 1/s), so the checker accepts them and the result
    // keeps A's dimension.
    let o = run_script(
        "oscillator",
        "A = 2 m\n\
         tau = 1 s\n\
         omega = 0.25 Hz\n\
         t = 2 s\n\
         x = A * exp(-t / tau) * cos(2 * pi * omega * t)\n\
         print(x)\n\
         print(sqrt(9 m^2), hypot(3 m, 4 m), max(1 s, 2 s), round(2.5))\n",
    );
    assert_eq!(err(&o), "");
    // cos(pi) = -1, so x = -2 exp(-2) m.
    let o_text = out(&o);
    let (first, rest) = o_text.split_once('\n').expect("two lines");
    let x: f64 = first.trim_end_matches(" m").parse().expect("a number");
    assert!((x + 2.0 * (-2.0f64).exp()).abs() < 1e-12, "{first}");
    assert_eq!(rest, "3 m 5 m 2 s 3\n");
    assert_eq!(o.status.code(), Some(0));
}

#[test]
fn math_domain_errors_are_rendered_and_units_are_checked_first() {
    let o = run_script("log-zero", "print(\"a\")\nprint(log(0))\n");
    assert_eq!(o.status.code(), Some(1));
    assert_eq!(out(&o), "a\n");
    assert!(
        err(&o).contains("error: `log` of zero has no real value"),
        "{}",
        err(&o)
    );

    // A unit error stops the run before anything prints.
    let o = run_script("exp-length", "print(\"x\")\nprint(exp(3 m))\n");
    assert_eq!(o.status.code(), Some(1));
    assert_eq!(out(&o), "");
    assert!(err(&o).contains("the argument of `exp` must be dimensionless, found length"));
}

#[test]
fn arrays_run_and_print_as_sky() {
    let o = run_script(
        "arrays_basic",
        "v = [1 m, 2 m, 3 m]\nA = [[1, 2], [3, 4]]\nprint(v)\nprint(A @ [1, 1])\n\
         print(v[1:] * 2, len(v), identity(2))\nprint(zeros(2))\n",
    );
    assert_eq!(err(&o), "");
    assert_eq!(
        out(&o),
        "[1 m, 2 m, 3 m]\n[3, 7]\n[4 m, 6 m] 3 [[1, 0], [0, 1]]\n[0, 0]\n"
    );
}

/// The numbers in printed output such as `[0.1 m, 0.2 m]`.
fn numbers(text: &str) -> Vec<f64> {
    text.split(|c: char| !(c.is_ascii_digit() || "+-.e".contains(c)))
        .filter_map(|t| t.parse().ok())
        .collect()
}

#[test]
fn spring_system_displacements_in_metres() {
    // Two springs in series, fixed at the left: K u = f. K is declared
    // Symmetric (proven from the entries), so u = K⁻¹ f is the general
    // inverse; the displacements come out in metres.
    let src = "\
K: Symmetric Matrix[2, 2] N/m = [[2000 N/m, -1000 N/m], [-1000 N/m, 1000 N/m]]
f: Vector[2] N = [0 N, 100 N]
u = K⁻¹ @ f
print(u)
print(u[1] / 1 mm)
";
    let o = run_script("spring", src);
    assert_eq!(err(&o), "");
    assert_eq!(out(&o), "[0.1 m, 0.2 m]\n200\n");
}

#[test]
fn positive_definite_stiffness_is_solved_by_cholesky() {
    // Declaring PositiveDefinite (proven from the entries) makes `K⁻¹ @ f`
    // a Cholesky solve: `sky check --explain` says so, and `sky run`
    // honours it.
    let src = "\
K: PositiveDefinite Matrix[2, 2] N/m = [[2000 N/m, -1000 N/m], [-1000 N/m, 1000 N/m]]
f: Vector[2] N = [0 N, 100 N]
u = K⁻¹ @ f
print(u)
";
    let dir = scratch("spring_pd");
    let file = dir.join("prog.sky");
    fs::write(&file, src).unwrap();
    let explain = Command::new(env!("CARGO_BIN_EXE_sky"))
        .args(["check", "--explain"])
        .arg(&file)
        .current_dir(&dir)
        .output()
        .expect("run sky");
    assert!(
        err(&explain).contains("rewritten to a Cholesky solve"),
        "{}",
        err(&explain)
    );
    let o = Command::new(env!("CARGO_BIN_EXE_sky"))
        .arg("run")
        .arg(&file)
        .current_dir(&dir)
        .output()
        .expect("run sky");
    fs::remove_dir_all(&dir).unwrap();
    assert_eq!(err(&o), "");
    let u = numbers(&out(&o));
    assert_eq!(u.len(), 2, "{}", out(&o));
    assert!(
        (u[0] - 0.1).abs() < 1e-12 && (u[1] - 0.2).abs() < 1e-12,
        "{u:?}"
    );
    assert!(out(&o).trim_end().ends_with(" m]"), "{}", out(&o));
}

#[test]
fn a_dimension_error_inside_a_literal_stops_the_run() {
    let o = run_script("literal_dim", "print(1)\nv = [1 m, 2 s]\n");
    assert_eq!(o.status.code(), Some(1));
    assert_eq!(out(&o), "", "nothing runs when the check fails");
    assert!(err(&o).contains("array elements must have the same dimension"));
}

#[test]
fn repl_echoes_arrays() {
    let o = repl("v = [1 m, 2 m]\nv\nv[0]\n[[1, 2], [3, 4]]⁻¹\n");
    assert_eq!(err(&o), "");
    let printed = out(&o);
    let mut lines = printed.lines();
    assert_eq!(lines.next(), Some("[1 m, 2 m]"));
    assert_eq!(lines.next(), Some("1 m"));
    let inv = numbers(lines.next().unwrap());
    let want = [-2.0, 1.0, 1.5, -0.5];
    assert_eq!(inv.len(), 4);
    assert!(
        inv.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-14),
        "{inv:?}"
    );
}

#[test]
fn repl_solves_with_a_positive_definite_input_it_was_given_a_value() {
    let o = repl("K: PositiveDefinite Matrix[2, 2] = [[4, 1], [1, 3]]\nK⁻¹ @ [1, 2]\n");
    assert_eq!(err(&o), "");
    let x = numbers(&out(&o));
    assert!(
        (x[0] - 1.0 / 11.0).abs() < 1e-15 && (x[1] - 7.0 / 11.0).abs() < 1e-15,
        "{x:?}"
    );
}

// ----- for loops -----

#[test]
fn run_sums_a_vector_of_forces() {
    let o = run_script(
        "for-forces",
        "forces = [10 N, 20 N, 5.5 N]\n\
         total = 0 N\n\
         for f in forces:\n    total += f\n\
         print(\"total =\", total)\n\
         print(\"last =\", f)\n",
    );
    assert_eq!(err(&o), "");
    assert_eq!(out(&o), "total = 35.5 N\nlast = 5.5 N\n");
    assert_eq!(o.status.code(), Some(0));
}

#[test]
fn run_integrates_with_a_riemann_sum() {
    // The midpoint rule for the integral of x^2 on [0, 3 m]: 9 m^3.
    let o = run_script(
        "for-riemann",
        "n = 3000\n\
         a = 0 m\n\
         b = 3 m\n\
         dx = (b - a) / n\n\
         area = 0 m^3\n\
         for i in range(n):\n    x = a + (i + 0.5) * dx\n    area += x^2 * dx\n\
         print(area)\n",
    );
    assert_eq!(err(&o), "");
    let text = out(&o);
    let value: f64 = text
        .trim()
        .strip_suffix(" m^3")
        .expect("a volume")
        .parse()
        .expect("a number");
    assert!((value - 9.0).abs() < 1e-5, "{text}");
    assert_eq!(o.status.code(), Some(0));
}

#[test]
fn run_loops_over_matrix_rows_and_uses_break_and_continue() {
    let o = run_script(
        "for-rows",
        "A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\n\
         for row in A:\n    if row[0] == 4:\n        continue\n    print(row)\n\
         for i in range(10, 0, -3):\n    if i < 3:\n        break\n    print(i)\n",
    );
    assert_eq!(err(&o), "");
    assert_eq!(out(&o), "[1, 2, 3]\n[7, 8, 9]\n10\n7\n4\n");
    assert_eq!(o.status.code(), Some(0));
}

#[test]
fn run_rejects_iterating_a_scalar_before_running_anything() {
    let o = run_script(
        "for-scalar",
        "print(\"MARKER_RAN\")\nn = 3\nfor i in n:\n    print(i)\n",
    );
    assert_eq!(o.status.code(), Some(1));
    assert_eq!(out(&o), "");
    let e = err(&o);
    assert!(
        e.contains("3:10: error: cannot iterate over a dimensionless number"),
        "{e}"
    );
    assert!(e.contains("for i in range(n)"), "{e}");
}

#[test]
fn run_reports_a_bad_range_bound_at_run_time() {
    let o = run_script(
        "for-bad-bound",
        "print(\"before\")\nn = 5 / 2\nfor i in range(n):\n    print(i)\n",
    );
    assert_eq!(o.status.code(), Some(1));
    assert_eq!(out(&o), "before\n");
    assert!(
        err(&o).contains("3:16: error: `range` needs an integer, found 2.5"),
        "{}",
        err(&o)
    );
}

#[test]
fn run_reports_iterating_a_string_at_run_time() {
    let o = run_script("for-string", "s = \"abc\"\nfor c in s:\n    print(c)\n");
    assert_eq!(o.status.code(), Some(1));
    assert!(
        err(&o).contains("2:10: error: cannot iterate over a string"),
        "{}",
        err(&o)
    );
}

#[test]
fn run_a_loop_variable_after_a_loop_that_may_not_run_is_a_check_error() {
    let o = run_script(
        "for-maybe",
        "n = 0\nfor i in range(n):\n    pass\nprint(i)\n",
    );
    assert_eq!(o.status.code(), Some(1));
    assert!(err(&o).contains("4:7: error: `i` may be undefined here"));
}

#[test]
fn run_evaluates_a_derivative_with_its_unit() {
    // x(t) = 5 m/s^2 t^2: x'(2 s) = 20 m/s, x''(2 s) = 10 m/s^2.
    let o = run_script(
        "derivative",
        "def x(t: s) -> m:\n    return 5 m/s^2 * t^2\n\
         def v(t: s) -> m/s:\n    return derivative(x, t)(t)\n\
         print(x(2 s))\n\
         print(derivative(x, t)(2 s))\n\
         print(derivative(v, t)(2 s))\n",
    );
    assert_eq!(err(&o), "");
    assert_eq!(out(&o), "20 m\n20 m/s\n10 m/s^2\n");
    assert!(o.status.success());
}

#[test]
fn a_derivative_unit_error_is_a_check_error_and_nothing_runs() {
    let o = run_script(
        "derivative_unit",
        "def x(t: s) -> m:\n    return 5 m/s^2 * t^2\n\
         print(\"before\")\n\
         print(x(2 s) + derivative(x, t)(2 s))\n",
    );
    assert_eq!(out(&o), "");
    assert!(
        err(&o).contains("cannot add length and velocity"),
        "{}",
        err(&o)
    );
    assert_eq!(o.status.code(), Some(1));
}

#[test]
fn a_derivative_that_fails_is_a_runtime_error_on_the_call() {
    // The slope of sqrt at 0 is infinite: division by zero, reported where
    // the derivative was asked for, after the output before it.
    let o = run_script(
        "derivative_domain",
        "def root(a: m^2) -> m:\n    return sqrt(a)\n\
         print(derivative(root, a)(4 m^2))\n\
         print(derivative(root, a)(0 m^2))\n",
    );
    assert_eq!(out(&o), "0.25 m^-1\n");
    assert!(err(&o).contains("division by zero"), "{}", err(&o));
    assert_eq!(o.status.code(), Some(1));
}

// ---- multi-line input in the REPL ----

#[test]
fn repl_defines_a_function_across_lines_and_calls_it_later() {
    let o = repl(
        "def area(w: m, h: m) -> m^2:\n\
         \x20   a = w * h\n\
         \x20   return a\n\
         \n\
         area(3 m, 4 m)\n\
         print(area(1 m, 2 m))\n\
         # a comment after the block\n\
         def twice(x: m^2) -> m^2:\n\
         \x20   return 2 * x\n\
         \n\
         twice(area(1 m, 1 m))\n",
    );
    assert_eq!(err(&o), "");
    assert_eq!(out(&o), "12 m^2\n2 m^2\n2 m^2\n");
    assert_eq!(o.status.code(), Some(0));
}

#[test]
fn repl_runs_for_while_and_if_blocks() {
    let o = repl(
        "total = 0\n\
         for i in range(4):\n\
         \x20   total += i\n\
         \n\
         total\n\
         n = 0 m\n\
         while n < 3 m:\n\
         \x20   n += 1 m\n\
         \x20   if n == 2 m:\n\
         \x20       print(\"two\")\n\
         \x20   else:\n\
         \x20       print(n)\n\
         \n\
         n\n",
    );
    assert_eq!(err(&o), "");
    assert_eq!(out(&o), "6\n1 m\ntwo\n3 m\n3 m\n");
}

#[test]
fn repl_error_in_a_block_is_reported_against_the_whole_input() {
    let o = repl(
        "x = 1 m\n\
         if x > 0 m:\n\
         \x20   y = 2 m\n\
         \x20   z = y + 1 s\n\
         \n\
         x\n",
    );
    assert_eq!(out(&o), "1 m\n");
    let e = err(&o);
    // The bad line is the third of its input, not the fourth line typed.
    assert!(
        e.contains("<repl>:3:11: error: cannot add length and time"),
        "{e}"
    );
    assert!(e.contains("3 |     z = y + 1 s"), "{e}");
    assert!(e.contains("(line 2)"), "{e}");
}

#[test]
fn repl_a_syntax_error_in_a_block_leaves_the_session_usable() {
    let o = repl("if 1 > 0:\n    x = = 2\n\n1 + 1\n");
    assert_eq!(out(&o), "2\n");
    assert!(err(&o).contains("<repl>:2:9: error"), "{}", err(&o));
}

#[test]
fn repl_continues_inside_open_brackets() {
    let o = repl(
        "v = [1 m,\n\
         \x20    2 m,\n\
         \x20    3 m]\n\
         v[1]\n\
         s = (1 +\n\
         \n\
         \x20    2)\n\
         s\n\
         print(3,\n\
         4\n\
         )\n",
    );
    // The empty line inside the parentheses does not end the input.
    assert_eq!(err(&o), "");
    assert_eq!(out(&o), "2 m\n3\n3 4\n");
    // Errors count lines from the start of the input.
    let o = repl("x = (1 m +\n     2 s)\n");
    assert!(
        err(&o).contains("<repl>:1:10: error: cannot add length and time"),
        "{}",
        err(&o)
}

// ---- the two engines of `sky run` ----

/// Programs for the engine comparison: what each does is not the point, that
/// SIR and the tree-walker do it the same way is. `true` when the program
/// lowers, so `--engine=sir` runs it (the comparison is then real, not a
/// refusal).
const ENGINE_PROGRAMS: &[(&str, bool, &str)] = &[
    (
        "eng-arith",
        true,
        "x = 3 m\nv = 36 km/h\nprint(x * 2, v, v * 2 s, 1 / 4, 2 ^ 10)\nprint(v as km/h)\n",
    ),
    (
        "eng-while",
        true,
        "i = 0\ntotal = 0\nwhile i < 100:\n    i += 1\n    if i % 2 == 0:\n        continue\n    if i > 50:\n        break\n    total += i\nprint(i, total)\n",
    ),
    (
        "eng-funcs",
        true,
        "def fact(n):\n    if n <= 1:\n        return 1\n    return n * fact(n - 1)\nprint(fact(10))\n",
    ),
    (
        "eng-div-in-func",
        true,
        "def ratio(a: m, b: m):\n    return a / b\nprint(\"first\")\nprint(ratio(1 m, 2 m))\nprint(ratio(1 m, 0 m))\nprint(\"never\")\n",
    ),
    (
        "eng-div-nested",
        true,
        "def f(x: s):\n    return 1 m / (x - 1 s)\ndef g(y: s):\n    return f(y) * 2\nprint(g(3 s))\nprint(g(1 s))\n",
    ),
    (
        "eng-overflow",
        true,
        "x = 1e308\nprint(x)\nprint(x * 10)\n",
    ),
    (
        "eng-sqrt-neg",
        true,
        "def r(a: m^2):\n    return sqrt(a)\nprint(r(4 m^2))\nprint(r(-1 m^2))\n",
    ),
    (
        "eng-index",
        true,
        "v = [1, 2, 3]\ni = 3\nprint(v[0], v[2])\nprint(v[i])\n",
    ),
    (
        "eng-slice",
        true,
        "v = [1, 2, 3, 4]\nprint(v[1:3])\nn = 9\nprint(v[0:n])\n",
    ),
    (
        "eng-matrix",
        true,
        "A = [[2, 0], [0, 4]]\nb = [2, 4]\nprint(A @ b, A⁻¹, A⁻¹ @ b)\n",
    ),
    (
        "eng-singular",
        true,
        "A = [[1, 2], [2, 4]]\nprint(\"before\")\nprint(A⁻¹)\n",
    ),
    (
        "eng-cholesky",
        true,
        "K: PositiveDefinite Matrix[2, 2] = [[4, 1], [1, 3]]\nf = [1, 2]\nprint(K⁻¹ @ f)\n",
    ),
    (
        "eng-const-range",
        true,
        "s = 0\nfor i in range(1, 11):\n    s += i\nprint(s, i)\nfor j in range(10, 0, -3):\n    print(j)\n",
    ),
    (
        "eng-warning-then-error",
        true,
        "аx = 1\nprint(аx / 0)\n",
    ),
    (
        "eng-derivative",
        true,
        "def x(t: s) -> m:\n    return 5 m/s^2 * t^2\nprint(derivative(x, t)(2 s))\nprint(x(1 s))\n",
    ),
    (
        "eng-derivative-fails",
        true,
        "def root(a: m^2) -> m:\n    return sqrt(a)\nprint(derivative(root, a)(0 m^2))\n",
    ),
    (
        "eng-deep-recursion",
        true,
        "def f(n):\n    return f(n + 1)\nprint(\"start\")\nf(0)\n",
    ),
    // Programs SIR cannot run yet: `--engine=sir` refuses them, and `auto`
    // must give the tree-walker's answer (`run_script` checks that).
    (
        "eng-computed-range",
        false,
        "n = 2 + 1\nfor i in range(n):\n    print(i)\n",
    ),
    (
        "eng-array-loop",
        false,
        "v = [1 m, 2 m]\nfor x in v:\n    print(x)\n",
    ),
    (
        "eng-input-after-output",
        false,
        "print(\"before\")\nA: Matrix[2, 2]\nprint(\"after\")\n",
    ),
    // The checker leaves `"a" + "b"` to run time; lowering builds an `add`
    // of strings that does not verify, which is not the pass pipeline's fault.
    (
        "eng-string-add",
        false,
        "print(\"before\")\ns = \"a\" + \"b\"\n",
    ),
    (
        "eng-input-in-func",
        false,
        "def f():\n    v: Vector[3]\n    return 1\nprint(2)\n",
    ),
];

#[test]
fn the_engines_agree_on_every_program() {
    for &(name, on_sir, src) in ENGINE_PROGRAMS {
        // `run_script` also compares `auto` with `--engine=ast`.
        let auto = run_script(name, src);
        assert_eq!(engines_agree(name, src), on_sir, "{name}");
        // A refused program still runs, on the fallback.
        assert_ne!(auto.status.code(), Some(2), "{name}: {}", err(&auto));
    }
}

#[test]
fn auto_falls_back_to_the_tree_walker_for_a_program_sir_cannot_lower() {
    // An input declaration inside a function is a `LowerError` (see
    // sky-lower's tests/errors.rs), and the function never runs.
    let src = "def f():\n    v: Vector[3]\n    return v\nprint(2 m)\n";
    let sir = run_engine("fallback", src, Some("sir"));
    assert_eq!(sir.status.code(), Some(1));
    assert_eq!(out(&sir), "");
    let e = err(&sir);
    assert!(e.contains(NOT_ON_SIR), "{e}");
    assert!(e.contains("top level of the file"), "{e}");
    assert!(e.contains("v: Vector[3]"), "{e}");
    // `auto` says nothing about it and runs the program.
    let auto = run_engine("fallback", src, None);
    assert_eq!((out(&auto).as_str(), err(&auto).as_str()), ("2 m\n", ""));
    assert_eq!(auto.status.code(), Some(0));
    let ast = run_engine("fallback", src, Some("ast"));
    assert_eq!((out(&ast), err(&ast)), (out(&auto), err(&auto)));
}

#[test]
fn a_fallback_program_prints_what_the_tree_walker_prints_before_it_fails() {
    // The input has no value to run with. Lowering accepts it, but SIR would
    // refuse the whole run up front, so the output before it would be lost.
    let src = "print(\"before\")\nA: Matrix[2, 2]\nprint(\"after\")\n";
    let auto = run_engine("fallback-input", src, None);
    assert_eq!(out(&auto), "before\n");
    assert_eq!(auto.status.code(), Some(1));
    assert!(
        err(&auto).contains("`A` is declared but has no value"),
        "{}",
        err(&auto)
    );
}

#[test]
fn repl_an_unclosed_bracket_at_end_of_input_is_reported() {
    let o = repl("1 + 1\nx = (1,\n");
    assert_eq!(out(&o), "2\n");
    assert!(
        err(&o).contains("<repl>:1:5: error: unclosed `(`"),
        "{}",
        err(&o)
    );
    assert_eq!(o.status.code(), Some(0));
}

#[test]
fn repl_an_unfinished_block_at_end_of_input_still_runs() {
    let o = repl("for i in range(2):\n    print(i)");
    assert_eq!(out(&o), "0\n1\n");
}

#[test]
fn repl_a_header_without_a_body_is_an_error_not_a_trap() {
    let o = repl("if 1 > 0:\n\nprint(3)\n");
    assert_eq!(out(&o), "3\n");
    assert!(
        err(&o).contains("expected an indented block after `if`"),
        "{}",
        err(&o)
    );
}

#[test]
fn repl_a_failed_multi_line_input_changes_nothing() {
    // Runtime failure on the second pass: `n` was rebound by the first
    // pass, `m` was bound, and neither must survive.
    let o = repl(
        "n = 1\n\
         while n < 5:\n\
         \x20   n += 1\n\
         \x20   m = 1 / (n - 3)\n\
         \n\
         n\n\
         m\n",
    );
    assert_eq!(out(&o), "1\n");
    let e = err(&o);
    assert!(e.contains("<repl>:3:9: error: division by zero"), "{e}");
    assert!(e.contains("cannot find `m` in this scope"), "{e}");

    // A def that fails to check defines nothing, and does not disturb `f`.
    let o = repl(
        "def f(x: m) -> m:\n\
         \x20   return x\n\
         \n\
         def g(x: m) -> m:\n\
         \x20   return f(x) + 1 s\n\
         \n\
         f(2 m)\n\
         g(1 m)\n",
    );
    assert_eq!(out(&o), "2 m\n");
    let e = err(&o);
    assert!(
        e.contains("<repl>:2:17: error: cannot add length and time"),
        "{e}"
    );
    assert!(e.contains("cannot find `g` in this scope"), "{e}");
}

#[test]
fn repl_a_runtime_error_inside_an_earlier_function_names_that_input() {
    let o = repl(
        "def inv1(x: m) -> m^-1:\n\
         \x20   return 1 / x\n\
         \n\
         inv1(0 m)\n\
         inv1(2 m)\n",
    );
    assert_eq!(out(&o), "0.5 m^-1\n");
    let e = err(&o);
    assert!(e.contains("<repl:1>:2:12: error: division by zero"), "{e}");
    assert!(e.contains("2 |     return 1 / x"), "{e}");
}

#[test]
fn repl_differentiates_functions_defined_in_it() {
    // x(t) = 5 m/s^2 t^2. The function, its derivative, and a function that
    // differentiates it are defined by different inputs, and `y` starts at
    // the same offset in its input as `x` does in its own.
    let o = repl(
        "def x(t: s) -> m:\n\
         \x20   return 5 m/s^2 * t^2\n\
         \n\
         derivative(x, t)(2 s)\n\
         def v(t: s) -> m/s:\n\
         \x20   return derivative(x, t)(t)\n\
         \n\
         v(3 s)\n\
         derivative(v, t)(2 s)\n\
         def y(t: s) -> m:\n\
         \x20   return 2 m/s * t\n\
         \n\
         derivative(y, t)(1 s)\n\
         derivative(x, t)(1 s)\n",
    );
    assert_eq!(err(&o), "");
    assert_eq!(out(&o), "20 m/s\n30 m/s\n10 m/s^2\n2 m/s\n10 m/s\n");
}

#[test]
fn repl_differentiates_a_redefined_function_with_its_new_body() {
    let o = repl(
        "def f(x: m) -> m:\n\
         \x20   return x * 2\n\
         \n\
         derivative(f, x)(1 m)\n\
         def f(x: m) -> m:\n\
         \x20   return x * 3\n\
         \n\
         derivative(f, x)(1 m)\n",
    );
    assert_eq!(err(&o), "");
    assert_eq!(out(&o), "2\n3\n");
}

#[test]
fn repl_a_failed_input_is_not_part_of_the_program_derivatives_see() {
    let o = repl(
        "def f(x: m) -> m:\n\
         \x20   return x * 2\n\
         \n\
         def f(x: m) -> m:\n\
         \x20   return x + 1 s\n\
         \n\
         derivative(f, x)(1 m)\n",
    );
    assert_eq!(out(&o), "2\n");
    assert!(
        err(&o).contains("cannot add length and time"),
        "{}",
        err(&o)
    );
}

#[test]
fn repl_commands_help_and_quit() {
    let o = repl(":help\n1 + 1\n:nope\n:quit\n3 + 3\n");
    assert_eq!(out(&o), "2\n");
    let e = err(&o);
    assert!(e.contains(":quit"), "{e}");
    assert!(e.contains("unknown command `:nope`"), "{e}");
    assert_eq!(o.status.code(), Some(0));
    // Not a command inside a block: it is part of the input.
    let o = repl("if 1 > 0:\n    :quit\n\n4\n");
    assert_eq!(out(&o), "4\n");
}

fn a_computed_range_bound_is_still_checked_at_run_time() {
    // SIR cannot stop a run with the message `range` gives for 2.5, so the
    // loop stays on the tree-walker under `auto`, which fails as before.
    let src = "n = 5 / 2\nfor i in range(n):\n    print(i)\n";
    let auto = run_engine("range-bound", src, None);
    assert_eq!(auto.status.code(), Some(1));
    assert!(
        err(&auto).contains("`range` needs an integer, found 2.5"),
        "{}",
        err(&auto)
    );
    let sir = run_engine("range-bound", src, Some("sir"));
    assert!(err(&sir).contains(NOT_ON_SIR), "{}", err(&sir));
}

#[test]
fn a_runtime_error_on_sir_points_where_the_tree_walker_points() {
    // SIR knows the failing operation; the operands' spans in the labels
    // are the tree-walker's, and every engine must print the same text.
    let src = "v = [1, 2, 3]\ni = 3\nprint(v[i])\n";
    for engine in ["sir", "ast", "auto"] {
        let o = run_engine("span", src, Some(engine));
        let e = err(&o);
        assert!(
            e.contains("3:9: error: index 3 is out of bounds"),
            "{engine}: {e}"
        );
        assert!(e.contains("this is Vector[3]"), "{engine}: {e}");
    }
}

#[test]
fn engine_flag_usage_errors_exit_2() {
    let dir = scratch("engine-usage");
    let file = dir.join("prog.sky");
    fs::write(&file, "print(1)\n").unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_sky"))
            .arg("run")
            .args(args)
            .arg(&file)
            .output()
            .unwrap()
    };
    let o = run(&["--engine=jit"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(err(&o).contains("unknown engine `jit`"), "{}", err(&o));
    assert_eq!(
        run(&["--engine=ast", "--engine=sir"]).status.code(),
        Some(2)
    );
    assert_eq!(run(&["--engine"]).status.code(), Some(2));
    // The flag may follow the file, as `check`'s flags may.
    let after = Command::new(env!("CARGO_BIN_EXE_sky"))
        .arg("run")
        .arg(&file)
        .arg("--engine=sir")
        .output()
        .unwrap();
    assert_eq!(
        (out(&after).as_str(), after.status.code()),
        ("1\n", Some(0))
    );
    // No file.
    let o = Command::new(env!("CARGO_BIN_EXE_sky"))
        .args(["run", "--engine=ast"])
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_closed_stdout_is_not_a_diagnostic_on_any_engine() {
    // `sky run x.sky | head -0`: the reader is gone before the output is
    // written. No engine prints a diagnostic; all exit 1.
    let src = "i = 0\nwhile i < 20000:\n    print(i)\n    i += 1\n";
    for engine in ["ast", "sir", "auto"] {
        let dir = scratch(&format!("pipe-{engine}"));
        let file = dir.join("prog.sky");
        fs::write(&file, src).unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_sky"))
            .arg("run")
            .arg(format!("--engine={engine}"))
            .arg(&file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        drop(child.stdout.take());
        let o = child.wait_with_output().unwrap();
        assert_eq!(o.status.code(), Some(1), "{engine}");
        assert_eq!(err(&o), "", "{engine}");
        fs::remove_dir_all(&dir).unwrap();
    }
}
