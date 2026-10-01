//! Differential equations in the checker: what `equation` and `solve` bind,
//! the structure read from an equation, scoping, and the way an equation
//! carries over between REPL lines. The wording of the errors is pinned by
//! the goldens (`tests/ui/check/err_equation_units.sky`).

use sky_ast::{with_deep_stack, StmtKind};
use sky_types::{analyze_equation, check_module, check_module_with, Checked, OdeTy, Ty};
use sky_units::Dimension;

fn check(src: &str) -> Checked {
    with_deep_stack(|| {
        let lexed = sky_syntax::lex(src).expect("lexes");
        let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
        assert!(errors.is_empty(), "{errors:#?}");
        check_module(&module, src)
    })
}

fn messages(c: &Checked) -> Vec<&str> {
    c.diagnostics.iter().map(|d| d.message.as_str()).collect()
}

fn ty(c: &Checked, name: &str) -> String {
    c.env
        .get(name)
        .map_or_else(|| "<unbound>".to_string(), |t| t.to_string())
}

/// The second-order pairs of the first equation in `src`, by name.
fn structure(src: &str) -> Option<Vec<(String, String)>> {
    with_deep_stack(|| {
        let lexed = sky_syntax::lex(src).expect("lexes");
        let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
        assert!(errors.is_empty(), "{errors:#?}");
        let StmtKind::Equation(eq) = &module.stmts[0].kind else {
            panic!("not an equation");
        };
        let info = analyze_equation(src, eq);
        info.second_order.as_ref()?;
        Some(
            info.pair_names()
                .into_iter()
                .map(|(q, v)| (q.to_string(), v.to_string()))
                .collect(),
        )
    })
}

fn pair(q: &str, v: &str) -> (String, String) {
    (q.to_string(), v.to_string())
}

#[test]
fn a_second_order_system_is_recognised() {
    let orbit = "\
equation orbit(t: s):
    r: Vector[2] m
    v: Vector[2] m/s
    r' = v
    v' = -GM * r / norm(r)^3
";
    assert_eq!(structure(orbit), Some(vec![pair("r", "v")]));
}

#[test]
fn declaration_order_does_not_matter_for_the_structure() {
    let src = "\
equation e(t: s):
    v: m/s
    w: m/s
    x: m
    y: m
    y' = w
    x' = v
    v' = -x
    w' = -y - x
";
    // Pairs come in the order of the positions' declaration.
    assert_eq!(structure(src), Some(vec![pair("x", "v"), pair("y", "w")]));
}

#[test]
fn acceleration_through_a_local_name_still_counts() {
    let src = "\
equation e(t: s):
    x: m
    v: m/s
    a = -k * x
    x' = v
    v' = a
";
    assert_eq!(structure(src), Some(vec![pair("x", "v")]));
}

#[test]
fn a_velocity_dependent_acceleration_is_not_second_order() {
    // Drag: v' depends on v.
    assert_eq!(
        structure("equation e(t: s):\n    x: m\n    v: m/s\n    x' = v\n    v' = -k * v\n"),
        None
    );
    // ... also through a local name.
    assert_eq!(
        structure(
            "equation e(t: s):\n    x: m\n    v: m/s\n    drag = c * v\n    x' = v\n    v' = -drag\n"
        ),
        None
    );
}

#[test]
fn other_systems_are_general() {
    // One state, an odd number of states, a state that is not paired, and
    // pairs that share a velocity.
    assert_eq!(
        structure("equation e(t: s):\n    n: Scalar\n    n' = -n / 1 s\n"),
        None
    );
    assert_eq!(
        structure("equation e(t: s):\n    x: m\n    v: m/s\n    T: K\n    x' = v\n    v' = 0 m/s^2\n    T' = 0 K/s\n"),
        None
    );
    assert_eq!(
        structure("equation e(t: s):\n    a: m\n    b: m\n    v: m/s\n    a' = v\n    b' = v\n    v' = 0 m/s^2\n"),
        None
    );
    // `x' = v` and `v' = x` chain both ways: not a pair.
    assert_eq!(
        structure("equation e(t: s):\n    x: m\n    v: m\n    x' = v\n    v' = x\n"),
        None
    );
}

const ORBIT: &str = "\
GM = 1.327e20 m^3/s^2
equation orbit(t: s):
    r: Vector[2] m
    v: Vector[2] m/s
    r' = v
    v' = -GM * r / norm(r)^3
";

#[test]
fn an_equation_and_its_solution_have_types() {
    let c = check(&format!(
        "{ORBIT}
sol = solve orbit from 0 s to 1 yr with r = [1 AU, 0 m], v = [0 m/s, 29.8 km/s]:
    conserve energy = 0.5 * norm(v)^2 - GM / norm(r)
"
    ));
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    assert_eq!(ty(&c, "orbit"), "equation `orbit`");
    assert_eq!(ty(&c, "sol"), "solution of `orbit`");
    let Some(Ty::Ode(o)) = c.env.get("sol") else {
        panic!("not a solution");
    };
    let OdeTy::Solution(sol) = &*o else {
        panic!("not a solution");
    };
    let fields: Vec<(&str, String)> = sol
        .fields
        .iter()
        .map(|(n, v)| (n.as_str(), v.to_string()))
        .collect();
    assert_eq!(
        fields,
        [
            ("t", "Scalar [T]".to_string()),
            ("r", "Vector[2] [L]".to_string()),
            ("v", "Vector[2] [L·T⁻¹]".to_string()),
            ("energy", "Scalar [L²·T⁻²]".to_string()),
        ]
    );
    // Reading them gives their types.
    let c = check(&format!(
        "{ORBIT}
sol = solve orbit from 0 s to 1 yr with r = [1 AU, 0 m], v = [0 m/s, 29.8 km/s]
r = sol.r
speed = norm(sol.v)
"
    ));
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    assert_eq!(ty(&c, "r"), "Vector[2] [L]");
    assert_eq!(ty(&c, "speed"), "Scalar [L·T⁻¹]");
}

#[test]
fn states_locals_and_time_do_not_leak_out_of_an_equation() {
    let c = check(
        "\
x = 1 m
equation e(t: s):
    x: kg
    a = 2 kg
    x' = a / 1 s
y = x
z = t
w = a
",
    );
    // `x` outside is still the length, and the equation's names are gone.
    assert_eq!(ty(&c, "y"), "Scalar [L]");
    assert_eq!(
        messages(&c),
        [
            "cannot find `t` in this scope",
            "cannot find `a` in this scope"
        ]
    );
}

#[test]
fn equations_read_names_from_around_them_but_only_defined_ones() {
    let c = check("equation e(t: s):\n    x: m\n    x' = later / 1 s\nlater = 1 m\n");
    assert_eq!(messages(&c), ["cannot find `later` in this scope"]);
}

#[test]
fn every_state_needs_exactly_one_derivative() {
    let c = check("equation e(t: s):\n    x: m\n    y: m\n    x' = 1 m/s\n    x' = 2 m/s\n");
    assert_eq!(
        messages(&c),
        ["`x'` has more than one equation", "no equation for `y'`"]
    );
}

#[test]
fn state_and_time_names_must_be_distinct() {
    let c = check("equation e(t: s):\n    t: m\n    x: m\n    x: m\n    x' = 1 m/s\n");
    assert_eq!(
        messages(&c),
        [
            "`t` is the independent variable, so it cannot be declared again",
            "`x` is already a state variable, so it cannot be declared again"
        ]
    );
}

#[test]
fn state_sizes_must_be_fixed() {
    let c = check("equation e(t: s):\n    x: Vector[N] m\n    x' = x / 1 s\n");
    assert_eq!(
        messages(&c),
        ["the size of state variable `x` must be a number"]
    );
}

#[test]
fn a_derivative_has_the_dimension_of_its_variable_per_time() {
    let c = check("equation e(t: s):\n    x: m\n    x' = 1 m\n");
    assert_eq!(
        messages(&c),
        ["`x'` is a velocity, but the right side is a length"]
    );
    // Dimensionless variable: the derivative is a frequency.
    let c = check("equation e(t: s):\n    n: Scalar\n    n' = 3 Hz\n");
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    // A vector variable wants a vector derivative.
    let c = check("equation e(t: s):\n    r: Vector[2] m\n    r' = 1 m/s\n");
    assert_eq!(
        messages(&c),
        ["declared as a vector, but the value is a scalar"]
    );
}

#[test]
fn the_independent_variable_need_not_be_time() {
    let c = check(
        "\
equation ray(x: m):
    y: m
    y' = 0.5
s = solve ray from 0 m to 10 m with y = 1 m
height = s.y
",
    );
    assert!(c.diagnostics.is_empty(), "{:#?}", c.diagnostics);
    assert_eq!(ty(&c, "height"), "Scalar [L]");
    // The bounds must be lengths here.
    let c = check(
        "equation ray(x: m):\n    y: m\n    y' = 0.5\ns = solve ray from 0 s to 10 s with y = 1 m\n",
    );
    assert_eq!(
        messages(&c),
        [
            "the start of the time span must be a length (the dimension of `x`), found a time",
            "the end of the time span must be a length (the dimension of `x`), found a time"
        ]
    );
}

#[test]
fn solve_needs_an_equation_and_initial_values() {
    let c = check("x = 3\ns = solve x from 0 s to 1 s\n");
    assert_eq!(
        messages(&c),
        ["`x` is not an equation, it is Scalar [dimensionless]"]
    );
    let c = check("equation e(t: s):\n    x: m\n    x' = 1 m/s\ns = solve e from 0 s to 1 s\n");
    assert_eq!(
        messages(&c),
        ["no initial value for `x` in the `solve` of `e`"]
    );
    let c = check(
        "equation e(t: s):\n    x: m\n    x' = 1 m/s\ns = solve e from 0 s to 1 s with x = 1 m, x = 2 m, y = 3 m\n",
    );
    assert_eq!(
        messages(&c),
        [
            "`x` is given an initial value twice",
            "`e` has no state variable `y`"
        ]
    );
}

#[test]
fn options_are_typed() {
    let c = check(
        "\
equation e(t: s):
    x: m
    v: m/s
    x' = v
    v' = -x / 1 s^2
s = solve e from 0 s to 1 s with x = 1 m, v = 0 m/s:
    tolerance 1e-6
    tolerance 1e-3
    conserve energy = 0.5 * v^2 + 0.5 * x^2 / 1 s^2
    conserve bad = v * x within 2
    conserve pair = [x, x]
e = s.energy
b = s.bad
",
    );
    assert_eq!(
        messages(&c),
        [
            "`tolerance` is given twice",
            "a conserved quantity must be a single number, found vector"
        ]
    );
    assert_eq!(ty(&c, "e"), "Scalar [L²·T⁻²]");
    assert_eq!(ty(&c, "b"), "Scalar [L²·T⁻¹]");
}

#[test]
fn an_equation_carries_over_to_the_next_repl_line() {
    // The REPL checks each line against the environment the last one left.
    let first = "equation e(t: s):\n    x: m\n    x' = 1 m/s\n";
    let c1 = check(first);
    assert!(c1.diagnostics.is_empty());
    let src = "s = solve e from 0 s to 2 s with x = 0 m\nlength = s.x\n";
    let c2 = with_deep_stack(|| {
        let lexed = sky_syntax::lex(src).expect("lexes");
        let (module, errors) = sky_ast::parse_module(&lexed.tokens, src);
        assert!(errors.is_empty());
        check_module_with(&module, src, c1.env.clone())
    });
    assert!(c2.diagnostics.is_empty(), "{:#?}", c2.diagnostics);
    assert_eq!(ty(&c2, "length"), "Scalar [L]");
}

#[test]
fn an_equation_is_not_a_number() {
    // Like a function: the operators leave it untyped, and the runtime says
    // what is wrong; indexing and annotating it are checker errors.
    let c = check(&format!("{ORBIT}y = orbit[0]\nz: m = orbit\n"));
    assert_eq!(
        messages(&c),
        [
            "cannot index equation `orbit`",
            "expected a number, found equation `orbit`"
        ]
    );
}

#[test]
fn a_state_named_like_a_constant_warns_once() {
    let c = check("equation e(t: s):\n    c: m\n    c' = 1 m/s\n");
    assert_eq!(
        c.diagnostics
            .iter()
            .filter(|d| d.message.contains("shadows"))
            .count(),
        1,
        "{:#?}",
        messages(&c)
    );
    // The constant is back afterwards.
    let c = check("equation e(t: s):\n    c: m\n    c' = 1 m/s\nx = c\n");
    assert_eq!(ty(&c, "x"), "Scalar [L·T⁻¹]");
}

#[test]
fn the_solution_time_has_the_dimension_of_the_independent_variable() {
    let c = check(
        "equation e(t: s):\n    x: m\n    x' = 1 m/s\ns = solve e from 0 s to 2 s with x = 0 m\nt_end = s.t\n",
    );
    assert_eq!(c.env.get("t_end"), Some(Ty::scalar(Dimension::TIME)));
}

#[test]
fn explain_notes_record_the_structure_the_checker_found() {
    let c = check(&format!(
        "{ORBIT}
sol = solve orbit from 0 s to 1 yr with r = [1 AU, 0 m], v = [0 m/s, 29.8 km/s]:
    tolerance 1e-9
    conserve energy = 0.5 * norm(v)^2 - GM / norm(r)
"
    ));
    assert_eq!(c.ode_notes.len(), 1);
    let note = &c.ode_notes[0];
    assert_eq!(
        note.message,
        "`orbit` is a second-order system, so a symplectic integrator applies"
    );
    let notes = note.notes.join("\n");
    assert!(notes.contains("`r' = v`"), "{notes}");
    assert!(notes.contains("tolerance: 1e-9"), "{notes}");
    assert!(notes.contains("to be conserved: `energy`"), "{notes}");
}
