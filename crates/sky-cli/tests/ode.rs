//! Differential equations end to end: `sky run` and `sky check` on scripts
//! with `equation` and `solve` (ROADMAP Phase 3: "an orbit script declares
//! its tolerance and conserved quantity, and Sky picks and explains the
//! integrator").
//!
//! Accuracy is asserted against the tolerance the script asked for (see
//! [`bound`]).

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("ode-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(d.join(".git")).unwrap();
    d
}

/// Runs `sky <args> prog.sky` on `src`, with the engine flag if any.
fn sky(name: &str, command: &str, flags: &[&str], src: &str) -> Output {
    let dir = scratch(name);
    let file = dir.join("prog.sky");
    fs::write(&file, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_sky"))
        .arg(command)
        .args(flags)
        .arg(&file)
        .current_dir(&dir)
        .output()
        .expect("run sky");
    fs::remove_dir_all(&dir).unwrap();
    out
}

fn out(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn err(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// Runs a script under the default engine and `--engine=auto`, which must
/// agree (SIR cannot lower equations, so `auto` falls back), and checks that
/// a forced `--engine=sir` refuses the program with the reason.
fn run(name: &str, src: &str) -> Output {
    let default = sky(name, "run", &[], src);
    let auto = sky(name, "run", &["--engine=auto"], src);
    assert_eq!(out(&auto), out(&default), "auto stdout of:\n{src}");
    assert_eq!(err(&auto), err(&default), "auto stderr of:\n{src}");
    assert_eq!(auto.status.code(), default.status.code());
    let sir = sky(name, "run", &["--engine=sir"], src);
    assert_eq!(sir.status.code(), Some(1));
    assert!(
        err(&sir)
            .contains("cannot run on SIR: differential equations are not supported by SIR yet"),
        "{}",
        err(&sir)
    );
    assert_eq!(out(&sir), "");
    // ... and so does `--engine=native`, which runs the same lowered SIR.
    let native = sky(name, "run", &["--engine=native"], src);
    assert_eq!(native.status.code(), Some(1));
    assert!(
        err(&native)
            .contains("cannot run natively: differential equations are not supported by SIR yet"),
        "{}",
        err(&native)
    );
    assert_eq!(out(&native), "");
    default
}

/// The numbers in `text`, whatever surrounds them (brackets, commas, units).
fn numbers(text: &str) -> Vec<f64> {
    text.split(|c: char| !(c.is_ascii_digit() || "+-.e".contains(c)))
        .filter_map(|t| t.parse().ok())
        .collect()
}

/// The error a test accepts, `tight` (usually a multiple of the tolerance the
/// program asked for). `loose` was the bound for the v0 fixed-step solver;
/// the real solvers are in, so an explanation saying v0 ran is a failure.
fn bound(explanation: &str, _loose: f64, tight: f64) -> f64 {
    assert!(
        !explanation.contains("not available yet"),
        "the v0 solver ran:
{explanation}"
    );
    tight
}

/// The lines of `text` after the first `n`, the explanation.
fn explanation_after(text: &str, n: usize) -> String {
    text.lines().skip(n).collect::<Vec<_>>().join("\n")
}

#[test]
fn exponential_decay() {
    let src = "\
k = 0.5 Hz
equation decay(t: s):
    n: Scalar
    n' = -k * n

sol = solve decay from 0 s to 4 s with n = 1:
    tolerance 1e-9
print(sol.n)
print(explain(sol))
";
    let o = run("decay", src);
    assert_eq!(err(&o), "");
    let text = out(&o);
    let n = numbers(text.lines().next().unwrap())[0];
    let want = (-2.0f64).exp();
    assert!(
        (n - want).abs() <= bound(&explanation_after(&text, 1), 1e-8, 100.0 * 1e-9),
        "{n} vs {want}"
    );
    assert!(explanation_after(&text, 1).len() > 10, "explains itself");
}

#[test]
fn harmonic_oscillator_against_cosine() {
    let src = "\
omega = 3 Hz
equation oscillator(t: s):
    x: m
    v: m/s
    x' = v
    v' = -omega^2 * x

sol = solve oscillator from 0 s to 2 s with x = 0.5 m, v = 0 m/s:
    tolerance 1e-9
    conserve energy = 0.5 * v^2 + 0.5 * omega^2 * x^2
exact = 0.5 * cos(omega * sol.t)
print((sol.x / 1 m) - exact)
e0 = 0.5 * omega^2 * (0.5 m)^2
print((sol.energy - e0) / e0)
print(explain(sol))
";
    let o = run("oscillator", src);
    assert_eq!(err(&o), "");
    let text = out(&o);
    let v: Vec<f64> = text.lines().take(2).flat_map(numbers).collect();
    let tol = bound(&explanation_after(&text, 2), 1e-7, 100.0 * 1e-9);
    assert!(v[0].abs() < tol, "x error {}", v[0]);
    assert!(v[1].abs() < tol, "energy drift {}", v[1]);
}

/// Earth's orbit: `GM` of the Sun, one AU, the speed of a circular orbit
/// (29.8 km/s is close to it).
const ORBIT: &str = "\
GM = 1.327e20 m^3/s^2
equation orbit(t: s):
    r: Vector[2] m
    v: Vector[2] m/s
    r' = v
    v' = -GM * r / norm(r)^3

r0 = [1 AU, 0 m]
v0 = [0 m/s, 29.8 km/s]
e0 = 0.5 * norm(v0)^2 - GM / norm(r0)
";

#[test]
fn one_year_brings_earth_back_near_its_start() {
    let src = format!(
        "{ORBIT}
sol = solve orbit from 0 s to 1 yr with r = r0, v = v0:
    tolerance 1e-9
    conserve energy = 0.5 * norm(v)^2 - GM / norm(r)
print(norm(sol.r - r0) / (1 AU))
print(abs(sol.energy - e0) / abs(e0))
print(explain(sol))
"
    );
    let o = run("orbit-year", &src);
    assert_eq!(err(&o), "");
    let text = out(&o);
    let v: Vec<f64> = text.lines().take(2).flat_map(numbers).collect();
    // Near its start: the initial speed is a little off circular, so a
    // year is a little off a period. This is physics, not the solver.
    assert!(v[0] < 0.03, "{} AU from the start after a year", v[0]);
    let drift = bound(&explanation_after(&text, 2), 1e-7, 100.0 * 1e-9);
    assert!(v[1] < drift, "energy drift {}", v[1]);
    let explanation = explanation_after(&text, 2);
    assert!(!explanation.is_empty());
}

#[test]
fn the_orbit_closes_after_exactly_one_period() {
    // The period from the initial state (vis-viva): after it the planet is
    // back where it started, to the accuracy of the solver.
    let src = format!(
        "{ORBIT}
a = 1 / (2 / norm(r0) - dot(v0, v0) / GM)
period = 2 * pi * sqrt(a^3 / GM)
sol = solve orbit from 0 s to period with r = r0, v = v0:
    tolerance 1e-9
    conserve energy = 0.5 * norm(v)^2 - GM / norm(r)
print(norm(sol.r - r0) / (1 AU))
print(explain(sol))
"
    );
    let o = run("orbit-period", &src);
    assert_eq!(err(&o), "");
    let text = out(&o);
    let miss = numbers(text.lines().next().unwrap())[0];
    let allowed = bound(&explanation_after(&text, 1), 1e-4, 1e3 * 1e-9);
    assert!(miss < allowed, "{miss} AU from the start after a period");
}

#[test]
fn the_explanation_is_printed_by_the_program_and_named_in_check() {
    let src = format!(
        "{ORBIT}
sol = solve orbit from 0 s to 1 yr with r = r0, v = v0:
    tolerance 1e-9
    conserve energy = 0.5 * norm(v)^2 - GM / norm(r)
print(explain(sol))
"
    );
    let o = run("explain", &src);
    let text = out(&o);
    assert!(text.lines().count() >= 2, "{text}");
    // `sky check --explain` says what the checker found, without running.
    let c = sky("explain-check", "check", &["--explain"], &src);
    let notes = err(&c);
    assert!(
        notes.contains("`orbit` is a second-order system, so a symplectic integrator applies"),
        "{notes}"
    );
    assert!(notes.contains("to be conserved: `energy`"), "{notes}");
    assert!(notes.contains("tolerance: 1e-9"), "{notes}");
    assert!(out(&c).ends_with("prog.sky: ok\n"), "{}", out(&c));
    // Without `--explain` the notes stay out of the way.
    let quiet = sky("explain-quiet", "check", &[], &src);
    assert_eq!(err(&quiet), "");
}

#[test]
fn a_wrong_dimension_in_an_equation_runs_nothing() {
    let src = "\
equation fall(t: s):
    h: m
    v: m/s
    h' = v
    v' = v
print(1)
";
    let o = sky("wrong-dim", "run", &[], src);
    assert_eq!(o.status.code(), Some(1));
    assert_eq!(out(&o), "", "nothing runs after a check error");
    let e = err(&o);
    assert!(
        e.contains(
            "error: `v'` is a quantity of dimension L·T⁻², but the right side is a velocity"
        ),
        "{e}"
    );
}

#[test]
fn errors_at_run_time_are_diagnostics() {
    // A right-hand side that divides by zero: reported on the division,
    // with the solve as a secondary location.
    let src = "\
equation bad(t: s):
    x: Scalar
    x' = 1 Hz / x
sol = solve bad from 0 s to 1 s with x = 0
";
    let o = run("rhs-error", src);
    assert_eq!(o.status.code(), Some(1));
    let e = err(&o);
    assert!(e.contains("error: division by zero"), "{e}");
    assert!(e.contains("3 |     x' = 1 Hz / x"), "{e}");
    assert!(e.contains("while solving `bad` (at t = 0 s)"), "{e}");

    // A drift bound the solve cannot meet: an error on the solve.
    let src = "\
w = 1 Hz
equation osc(t: s):
    x: m
    v: m/s
    x' = v
    v' = -w^2 * x
sol = solve osc from 0 s to 10 s with x = 1 m, v = 0 m/s:
    conserve energy = 0.5 * v^2 + 0.5 * w^2 * x^2 within 1e-30
";
    let o = run("drift", src);
    assert_eq!(o.status.code(), Some(1));
    let e = err(&o);
    assert!(
        e.contains("error: cannot solve `osc`: energy drifted by"),
        "{e}"
    );
    assert!(
        e.contains("4 | sol = solve") || e.contains("sol = solve osc"),
        "{e}"
    );

    // A span that runs backwards.
    let src = "\
equation e(t: s):
    x: m
    x' = 1 m/s
sol = solve e from 2 s to 1 s with x = 0 m
";
    let o = run("backwards", src);
    assert_eq!(o.status.code(), Some(1));
    assert!(
        err(&o).contains("the time span runs backwards: it ends before it starts"),
        "{}",
        err(&o)
    );

    // A tolerance outside (0, 1).
    let src = "\
equation e(t: s):
    x: m
    x' = 1 m/s
sol = solve e from 0 s to 1 s with x = 0 m:
    tolerance 5
";
    let o = run("tolerance", src);
    assert_eq!(o.status.code(), Some(1));
    assert!(
        err(&o).contains("the tolerance must be greater than 0 and less than 1"),
        "{}",
        err(&o)
    );
}

#[test]
fn a_solution_that_blows_up_is_an_error_not_infinity() {
    let src = "\
equation blowup(t: s):
    x: Scalar
    x' = x^2 * 1 Hz
sol = solve blowup from 0 s to 2 s with x = 1
";
    let o = run("blowup", src);
    assert_eq!(o.status.code(), Some(1));
    let e = err(&o);
    assert!(
        ["overflow", "not finite", "gave up", "step size"]
            .iter()
            .any(|w| e.contains(w)),
        "{e}"
    );
    assert!(!out(&o).contains("inf") && !out(&o).contains("NaN"));
}

#[test]
fn the_repl_defines_an_equation_and_solves_it_on_the_next_line() {
    use std::io::Write;
    use std::process::Stdio;
    let mut child = Command::new(env!("CARGO_BIN_EXE_sky"))
        .arg("repl")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run sky");
    let input = "\
k = 1 Hz
equation decay(t: s):
    n: Scalar
    n' = -k * n

s = solve decay from 0 s to 1 s with n = 1
print(s.n)
";
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let o = child.wait_with_output().unwrap();
    assert_eq!(err(&o), "");
    let n = numbers(&out(&o))[0];
    assert!((n - (-1.0f64).exp()).abs() < 1e-6, "{}", out(&o));
}

#[test]
fn an_uncertain_value_in_solve_says_how_to_proceed() {
    let o = sky(
        "uncertain",
        "run",
        &[],
        "k = 2 ± 0.1 Hz\nequation decay(t: s):\n    x: m\n    x' = -k * x\nsol = solve decay from 0 s to 1 s with x = 1 m:\n    tolerance 1e-8\nprint(sol.x)\n",
    );
    assert_eq!(o.status.code(), Some(1));
    let e = err(&o);
    // Found by the check, before anything runs.
    assert!(
        e.contains("the equations of `decay` read `k`, which may be an uncertain number"),
        "{e}"
    );
    assert!(e.contains("`k` is read here"), "{e}");
    assert!(
        e.contains("`solve` does not propagate uncertainties"),
        "{e}"
    );
    assert!(e.contains("nominal(v)"), "{e}");
}

const FALL: &str = "equation fall(t: s):
    y: m
    v: m/s
    y' = v
    v' = -9.81 m/s^2

def drop(h: m) -> m:
    sol = solve fall from 0 s to 1 s with y = h, v = 0 m/s
    return sol.y

def twice(h: m) -> m:
    return 2 * drop(h)

def sq(x: m) -> m^2:
    return x * x
";

/// A `solve` inside one `def` does not stop the derivative of another that
/// never reaches it (before, every `def` had to lower to SIR).
#[test]
fn a_solve_in_one_def_does_not_stop_the_derivative_of_another() {
    for engine in ["--engine=ast", "--engine=auto"] {
        let o = sky(
            "solve_def_derivative",
            "run",
            &[engine],
            &format!("{FALL}print(derivative(sq, x)(3 m))\nprint(drop(10 m))\n"),
        );
        assert_eq!(err(&o), "", "{engine}");
        assert_eq!(out(&o), "6 m\n5.094999999999999 m\n", "{engine}");
    }
}

/// A derivative of a function that does reach a `solve` names the calls
/// that lead there and points at the `solve`.
#[test]
fn a_derivative_through_a_solve_names_the_calls_and_the_solve() {
    let o = sky(
        "solve_def_derivative_err",
        "run",
        &[],
        &format!("{FALL}print(derivative(twice, h)(10 m))\n"),
    );
    assert_eq!(o.status.code(), Some(1));
    let e = err(&o);
    assert!(
        e.contains("cannot evaluate a derivative: `twice` calls `drop`, which cannot run on SIR: differential equations are not supported by SIR yet"),
        "{e}"
    );
    assert!(
        e.contains("sol = solve fall from 0 s to 1 s with y = h, v = 0 m/s")
            && e.contains("SIR stops here"),
        "{e}"
    );
    let o = sky(
        "solve_def_derivative_err2",
        "run",
        &[],
        &format!("{FALL}print(derivative(drop, h)(10 m))\n"),
    );
    assert!(
        err(&o).contains("cannot evaluate a derivative: `drop` cannot run on SIR"),
        "{}",
        err(&o)
    );
}

/// The force as the gradient of a potential, `v' = -grad(U)(r)`: the
/// derivative runs on SIR while the equation runs on the tree walker, and
/// the orbit it gives is the one the explicit force gives.
#[test]
fn a_force_from_grad_of_a_potential_drives_the_orbit() {
    let explicit = format!(
        "{ORBIT}
sol = solve orbit from 0 s to 1 yr with r = r0, v = v0:
    tolerance 1e-9
    conserve energy = 0.5 * norm(v)^2 - GM / norm(r)
print(sol.r / (1 AU))
"
    );
    let gradient = "\
GM = 1.327e20 m^3/s^2
def U(p: Vector[2] m) -> J/kg:
    return -GM / sqrt(p @ p)

equation orbit(t: s):
    r: Vector[2] m
    v: Vector[2] m/s
    r' = v
    v' = -grad(U)(r)

sol = solve orbit from 0 s to 1 yr with r = [1 AU, 0 m], v = [0 m/s, 29.8 km/s]:
    tolerance 1e-9
    conserve energy = 0.5 * norm(v)^2 + U(r)
print(sol.r / (1 AU))
print(explain(sol))
";
    let a = run("orbit-explicit", &explicit);
    let g = run("orbit-gradient", gradient);
    assert_eq!(err(&g), "", "{}", err(&g));
    let (text_a, text_g) = (out(&a), out(&g));
    let ra = numbers(text_a.lines().next().unwrap());
    let rg = numbers(text_g.lines().next().unwrap());
    assert_eq!(ra.len(), 2);
    for (x, y) in ra.iter().zip(&rg) {
        assert!((x - y).abs() < 1e-6, "explicit {ra:?} vs gradient {rg:?}");
    }
    assert!(
        explanation_after(&text_g, 1).contains("symplectic"),
        "{text_g}"
    );
}

/// A derivative only needs the `def`s to lower to SIR, so `±` at the top
/// level of the file does not stop it.
#[test]
fn a_derivative_works_beside_uncertain_values() {
    let o = sky(
        "derivative-uncertain",
        "run",
        &[],
        "L = 1.000 ± 0.005 m\ndef x(t: s) -> m:\n    return 3 m/s^2 * t^2\nprint(derivative(x, t)(2 s))\nprint(L)\n",
    );
    assert_eq!(err(&o), "");
    assert_eq!(out(&o), "12 m/s\n1.000 ± 0.005 m\n");
}
