//! `equation` and `solve` on the interpreter, and the vector builtins
//! `norm` and `dot`. The programs run without the checker (like the other
//! interpreter tests), so what is checked is what the interpreter itself
//! does: flattening the state, evaluating right-hand sides, the layout of a
//! second-order system, and the errors.
//!
//! The accuracy asserted is loose while `Method::Auto` is v0's fixed-step
//! RK4 (its explanation says so) and tightens to what the tolerance asked
//! for when the real solvers land (see [`bound`]).

use sky_ast::parse_module;
use sky_syntax::lex;

use crate::{run_module, RuntimeEnv, RuntimeError};

fn run(src: &str) -> Result<String, RuntimeError> {
    let src = format!("{src}\n");
    let lexed = lex(&src).expect("lexes");
    let (module, errors) = parse_module(&lexed.tokens, &src);
    assert!(errors.is_empty(), "{errors:?} in {src:?}");
    let mut out = Vec::new();
    let mut env = RuntimeEnv::with_prelude();
    run_module(&module, &src, &mut env, &mut out)?;
    Ok(String::from_utf8(out).expect("utf-8"))
}

fn ok(src: &str) -> String {
    run(src).unwrap_or_else(|e| panic!("{}: {src}", e.message))
}

fn err(src: &str) -> RuntimeError {
    run(src).expect_err("fails")
}

/// The numbers in `text`, one per whitespace- or comma-separated token
/// (units and brackets stripped).
fn numbers(text: &str) -> Vec<f64> {
    text.split(|c: char| c.is_whitespace() || c == ',' || c == '[' || c == ']')
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

#[test]
fn norm_and_dot() {
    assert_eq!(ok("print(norm([3 m, 4 m]))"), "5 m\n");
    assert_eq!(ok("print(norm([[1, 2], [2, 4]]))"), "5\n");
    assert_eq!(ok("print(norm(-2 s))"), "2 s\n");
    assert_eq!(ok("print(norm([0 m, 0 m]))"), "0 m\n");
    assert_eq!(ok("print(dot([1 N, 2 N, 2 N], [1 m, 0 m, 3 m]))"), "7 J\n");
    // Scaled: neither overflows nor underflows on the way.
    let big = numbers(&ok("print(norm([3e200, 4e200]))"))[0];
    assert!((big / 5e200 - 1.0).abs() < 1e-15, "{big}");
    let small = numbers(&ok("print(norm([3e-200, 4e-200]))"))[0];
    assert!((small / 5e-200 - 1.0).abs() < 1e-15, "{small}");
    // A user's own `norm` wins.
    assert_eq!(ok("def norm(x):\n    return 7\nprint(norm(1))"), "7\n");
}

#[test]
fn norm_and_dot_fail_cleanly() {
    assert_eq!(
        err("print(dot([1, 2], [1, 2, 3]))").message,
        "`dot` needs vectors of one length, found Vector[2] and Vector[3]"
    );
    assert_eq!(
        err("print(dot([[1, 2]], [[1, 2]]))").message,
        "`dot` needs two vectors"
    );
    assert_eq!(
        err("print(dot([1e200, 1e200], [1e200, 1e200]))").message,
        "arithmetic overflow"
    );
    assert_eq!(
        err("print(norm(true))").message,
        "`norm` needs a vector, found a Bool"
    );
    assert_eq!(
        err("print(norm([1], [2]))").message,
        "`norm` takes 1 argument, found 2"
    );
}

const DECAY: &str = "\
equation decay(t: s):
    n: Scalar
    k = 0.5 Hz
    n' = -k * n
";

#[test]
fn exponential_decay() {
    let text = ok(&format!(
        "{DECAY}
s = solve decay from 0 s to 4 s with n = 1:
    tolerance 1e-9
print(s.n, s.t)
print(explain(s))
"
    ));
    let (numbers_line, explanation) = text.split_once('\n').expect("two parts");
    let v = numbers(numbers_line);
    let err = (v[0] - (-2.0f64).exp()).abs();
    assert!(err < bound(explanation, 1e-8, 1e-7), "{err} in {text}");
    assert_eq!(v[1], 4.0, "the final time is the end of the span");
    assert!(!explanation.trim().is_empty(), "the solver explains itself");
}

#[test]
fn a_solution_prints_as_itself_and_has_no_other_values() {
    let text = ok(&format!(
        "{DECAY}s = solve decay from 0 s to 1 s with n = 1\nprint(s)\nprint(decay)\n"
    ));
    assert_eq!(text, "<solution of decay>\n<equation decay>\n");
}

const OSCILLATOR: &str = "\
omega = 2 Hz
equation oscillator(t: s):
    x: m
    v: m/s
    x' = v
    v' = -omega^2 * x
";

#[test]
fn harmonic_oscillator_follows_cosine_and_conserves_energy() {
    let text = ok(&format!(
        "{OSCILLATOR}
sol = solve oscillator from 0 s to 3 s with x = 1 m, v = 0 m/s:
    tolerance 1e-9
    conserve energy = 0.5 * v^2 + 0.5 * omega^2 * x^2
print(sol.x / 1 m, cos(omega * sol.t))
print(sol.energy / (1 m^2/s^2))
print(explain(sol))
"
    ));
    let lines: Vec<&str> = text.lines().collect();
    let explanation = lines[2..].join("\n");
    let v = numbers(lines[0]);
    let tol = bound(&explanation, 1e-6, 1e-7);
    assert!((v[0] - v[1]).abs() < tol, "x = {} vs cos = {}", v[0], v[1]);
    // E = 0.5 * 4 * 1 = 2 (in m^2/s^2), the initial energy.
    let e = numbers(lines[1])[0];
    assert!((e - 2.0).abs() / 2.0 < tol, "energy {e}");
}

#[test]
fn the_state_layout_follows_the_equation_not_the_order_of_the_declarations() {
    // Velocities declared before positions, and two pairs interleaved: the
    // flattened layout for a second-order system is `[q..., v...]`, and must
    // still map every variable to its own values.
    let text = ok("\
w = 1 Hz
equation plane(t: s):
    vy: m/s
    vx: m/s
    y: m
    x: m
    x' = vx
    y' = vy
    vx' = -w^2 * x
    vy' = -w^2 * y
sol = solve plane from 0 s to 1.5 s with x = 1 m, y = 0 m, vx = 0 m/s, vy = 2 m/s
print(sol.x / 1 m, sol.y / 1 m, sol.vx / 1 m/s, sol.vy / 1 m/s)
print(explain(sol))
");
    let v = numbers(text.lines().next().expect("line"));
    let t = 1.5f64;
    let want = [t.cos(), 2.0 * t.sin(), -t.sin(), 2.0 * t.cos()];
    for (got, want) in v.iter().zip(want) {
        assert!((got - want).abs() < 1e-6, "{v:?} vs {want:?}");
    }
}

#[test]
fn a_vector_state_that_is_not_second_order() {
    // Rotation: u' = w * (-u1, u0). A quarter turn takes (1, 0) to (0, 1).
    let text = ok("\
w = 2 Hz
equation spin(t: s):
    u: Vector[2] m
    u' = w * [-u[1], u[0]]
q = 3.141592653589793 / 4
sol = solve spin from 0 s to q * 1 s with u = [1 m, 0 m]
print(sol.u)
print(explain(sol))
");
    let v = numbers(text.lines().next().expect("line"));
    assert!(v[0].abs() < 1e-6 && (v[1] - 1.0).abs() < 1e-6, "{v:?}");
}

#[test]
fn a_matrix_state_and_named_intermediate_values() {
    let text = ok("\
equation cool(t: s):
    M: Matrix[2, 2] K
    rate = 1 Hz
    M' = -rate * M
sol = solve cool from 0 s to 2 s with M = [[1 K, 2 K], [3 K, 4 K]]
print(sol.M)
");
    let v = numbers(&text);
    let f = (-2.0f64).exp();
    for (got, want) in v.iter().zip([1.0, 2.0, 3.0, 4.0]) {
        assert!((got - want * f).abs() < 1e-6, "{v:?}");
    }
}

#[test]
fn equations_call_functions_and_read_names_from_around_them() {
    let text = ok("\
def drag(v):
    return 0.1 Hz * v
g = 9.8 m/s^2
equation fall(t: s):
    h: m
    v: m/s
    h' = v
    v' = -g - drag(v)
sol = solve fall from 0 s to 2 s with h = 100 m, v = 0 m/s
print(sol.v)
");
    // Terminal velocity is g / 0.1 = 98 m/s downwards; after 2 s it is
    // (98)(1 - e^-0.2) = 17.76 m/s.
    let v = numbers(&text)[0];
    assert!((v + 98.0 * (1.0 - (-0.2f64).exp())).abs() < 1e-5, "{v}");
}

#[test]
fn an_equation_defined_and_solved_inside_a_function_sees_its_locals() {
    let text = ok("\
def decayed(k, t_end):
    equation e(t: s):
        n: Scalar
        n' = -k * n
    s = solve e from 0 s to t_end with n = 1
    return s.n
print(decayed(0.5 Hz, 4 s))
");
    let v = numbers(&text)[0];
    assert!((v - (-2.0f64).exp()).abs() < 1e-7, "{v}");
}

#[test]
fn the_solution_can_be_read_by_name() {
    let text = ok(&format!(
        "{DECAY}s = solve decay from 1 s to 3 s with n = 2\nprint(s.t, s.n)\n"
    ));
    let v = numbers(&text);
    assert_eq!(v[0], 3.0);
    assert!((v[1] - 2.0 * (-1.0f64).exp()).abs() < 1e-6, "{v:?}");
}

// ----- errors -----

#[test]
fn a_failing_right_hand_side_is_reported_where_it_fails() {
    let src = "\
equation bad(t: s):
    x: Scalar
    x' = 1 Hz / x
s = solve bad from 0 s to 1 s with x = 0";
    let e = err(src);
    assert_eq!(e.message, "division by zero");
    // The primary span is the division; the solve is a secondary label.
    assert_eq!(&src[e.span.range()], "1 Hz / x");
    assert!(
        e.secondary
            .iter()
            .any(|(span, label)| label.starts_with("while solving `bad`")
                && src[span.range()].starts_with("s = solve bad")),
        "{:?}",
        e.secondary
    );
}

#[test]
fn a_time_span_that_runs_backwards_is_an_error() {
    let e = err(&format!(
        "{DECAY}s = solve decay from 2 s to 1 s with n = 1"
    ));
    assert_eq!(
        e.message,
        "the time span runs backwards: it ends before it starts"
    );
}

#[test]
fn tolerances_are_between_zero_and_one() {
    for tol in ["0", "-1e-3", "1", "2"] {
        let e = err(&format!(
            "{DECAY}s = solve decay from 0 s to 1 s with n = 1:\n    tolerance {tol}"
        ));
        assert_eq!(
            e.message, "the tolerance must be greater than 0 and less than 1",
            "{tol}"
        );
    }
}

#[test]
fn a_conserved_quantity_that_drifts_more_than_allowed_stops_the_solve() {
    let e = err(&format!(
        "{OSCILLATOR}
sol = solve oscillator from 0 s to 3 s with x = 1 m, v = 0 m/s:
    conserve energy = 0.5 * v^2 + 0.5 * omega^2 * x^2 within 1e-30"
    ));
    assert!(e.message.contains("energy drifted by"), "{}", e.message);
    assert!(e.message.starts_with("cannot solve `oscillator`"));
}

#[test]
fn a_solve_that_blows_up_is_an_error_not_infinity() {
    // x' = x^2 blows up at t = 1 for x(0) = 1.
    let e = err("\
equation blowup(t: s):
    x: Scalar
    x' = x^2 * 1 Hz
s = solve blowup from 0 s to 2 s with x = 1");
    assert!(
        ["overflow", "not finite", "gave up", "step size"]
            .iter()
            .any(|w| e.message.contains(w)),
        "{}",
        e.message
    );
}

#[test]
fn solving_something_that_is_not_an_equation() {
    assert_eq!(
        err("x = 3\ns = solve x from 0 s to 1 s").message,
        "`x` is a number, not an equation"
    );
    assert_eq!(
        err("s = solve nothing from 0 s to 1 s").message,
        "cannot find equation `nothing` in this scope"
    );
}

#[test]
fn reading_what_a_solution_does_not_have() {
    let e = err(&format!(
        "{DECAY}s = solve decay from 0 s to 1 s with n = 1\nprint(s.q)"
    ));
    assert_eq!(e.message, "the solution of `decay` has no `q`");
    let e = err(&format!("{DECAY}print(decay.n)"));
    assert_eq!(e.message, "an equation has no attribute `n`");
    let e = err("x = 3 m\nprint(x.y)");
    assert!(e.message.contains("has no attribute `"), "{}", e.message);
}

#[test]
fn explain_needs_a_solution() {
    assert_eq!(
        err("print(explain(3))").message,
        "`explain` needs a solution, found a number"
    );
}
