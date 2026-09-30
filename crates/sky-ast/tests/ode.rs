//! `equation` and `solve`: trees, spans, and errors (spec "Differential
//! equations").

mod common;

use common::{lex, module_errors, module_sexpr};
use sky_ast::{parse_module, with_deep_stack, SolveOption, StmtKind, MAX_DEPTH};

fn messages(src: &str) -> Vec<String> {
    module_errors(src).into_iter().map(|e| e.message).collect()
}

const ORBIT: &str = "\
equation orbit(t: s):
    r: Vector[2] m
    v: Vector[2] m/s
    r' = v
    v' = -GM * r / norm(r)^3
";

#[test]
fn an_equation_is_a_header_and_typed_lines() {
    assert_eq!(
        module_sexpr(ORBIT),
        "(equation orbit (time t (type (unit s))) \
         (: r (type (Vector 2) (unit m))) \
         (: v (type (Vector 2) (unit (/ m s)))) \
         (d r v) \
         (d v (/ (* (neg GM) r) (^ (call norm r) 3))))"
    );
}

#[test]
fn locals_and_blank_lines_and_comments_in_the_body() {
    let src = "\
equation decay(t: s):
    n: Scalar

    # rate
    k = 0.5 Hz
    n' = -k * n
y = 1
";
    assert_eq!(
        module_sexpr(src),
        "(equation decay (time t (type (unit s))) (: n (type (Scalar))) (= k (unit 0.5 Hz)) (d n (* (neg k) n)))\n(= y 1)"
    );
}

#[test]
fn solve_with_initial_values_and_options() {
    let src = "\
sol = solve orbit from 0 s to 1 yr with r = [1 AU, 0 m], v = [0 m/s, 29.8 km/s]:
    tolerance 1e-9
    conserve energy = 0.5 * norm(v)^2 - GM / norm(r) within 1e-6
";
    assert_eq!(
        module_sexpr(src),
        "(solve sol orbit (unit 0 s) (unit 1 yr) \
         (with (r (array (unit 1 AU) (unit 0 m))) (v (array (unit 0 (/ m s)) (unit 29.8 (/ km s))))) \
         (tolerance 1e-9) \
         (conserve energy (- (* 0.5 (^ (call norm v) 2)) (/ GM (call norm r))) (within 1e-6)))"
    );
}

#[test]
fn solve_without_a_block() {
    assert_eq!(
        module_sexpr("sol = solve decay from 0 s to 10 s with n = 1\nprint(sol.n)\n"),
        "(solve sol decay (unit 0 s) (unit 10 s) (with (n 1)))\n(call print (attr sol n))"
    );
}

#[test]
fn a_number_before_to_or_with_is_not_a_unit() {
    // The parse succeeds (the checker reports the missing unit): `to` and
    // `with` end the expression rather than naming a unit.
    assert_eq!(
        module_sexpr("sol = solve e from 0 to 1 s with x = 2\n"),
        "(solve sol e 0 (unit 1 s) (with (x 2)))"
    );
}

#[test]
fn solve_words_are_ordinary_names_elsewhere() {
    assert_eq!(
        module_sexpr("to = 3\nwith = 4\nfrom = to + with\ntolerance = 1\n"),
        "(= to 3)\n(= with 4)\n(= from (+ to with))\n(= tolerance 1)"
    );
}

#[test]
fn statement_spans() {
    let src = format!("{ORBIT}x = 1\n");
    let tokens = lex(&src);
    let (module, errors) = parse_module(&tokens, &src);
    assert!(errors.is_empty(), "{errors:?}");
    let eq = &module.stmts[0];
    assert!(matches!(eq.kind, StmtKind::Equation(_)));
    assert_eq!(
        &src[eq.span.range()],
        ORBIT.trim_end_matches('\n'),
        "an equation spans its header through its last line"
    );
    let src = "sol = solve e from 0 s to 1 s with x = 2:\n    tolerance 1e-6\nz = 1\n";
    let tokens = lex(src);
    let (module, errors) = parse_module(&tokens, src);
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(
        &src[module.stmts[0].span.range()],
        "sol = solve e from 0 s to 1 s with x = 2:\n    tolerance 1e-6"
    );
    let StmtKind::Solve(solve) = &module.stmts[0].kind else {
        panic!("not a solve");
    };
    assert_eq!(&src[solve.keyword.range()], "solve");
    assert!(matches!(solve.options[0], SolveOption::Tolerance { .. }));
}

#[test]
fn equations_nest_in_blocks() {
    let src = "def f():\n    equation e(t: s):\n        x: m\n        x' = 1 m/s\n    return 1\n";
    assert_eq!(
        module_sexpr(src),
        "(def f (params) (block (equation e (time t (type (unit s))) (: x (type (unit m))) (d x (unit 1 (/ m s)))) (return 1)))"
    );
}

// ----- errors -----

#[test]
fn equation_header_errors() {
    assert_eq!(
        messages("equation 3(t: s):\n    x: m\n"),
        ["expected an equation name after `equation`, found integer literal `3`"]
    );
    assert_eq!(
        messages("equation e:\n    x: m\n"),
        ["expected `(` after the equation name, found `:`"]
    );
    assert_eq!(
        messages("equation e():\n    x: m\n"),
        ["expected the name of the independent variable, found `)`"]
    );
    assert_eq!(
        messages("equation e(t):\n    x: m\n"),
        ["expected `:` and a unit after `t`, found `)`"]
    );
    assert_eq!(
        messages("equation e(t: s, u: s):\n    x: m\n"),
        ["expected `)` after the independent variable, found `,`"]
    );
    assert_eq!(
        messages("equation e(t: s)\n    x: m\n"),
        ["expected `:` after the `equation` header, found end of line"]
    );
}

#[test]
fn equation_body_errors() {
    assert_eq!(
        messages("equation e(t: s):\nx = 1\n"),
        ["expected an indented block after `equation`"]
    );
    assert_eq!(
        messages("equation e(t: s): x: m\n"),
        ["the body of an `equation` must be on its own indented line"]
    );
    assert_eq!(
        messages("equation e(t: s):\n    x: m\n    x'' = 1\n"),
        ["second derivatives are not supported"]
    );
    assert_eq!(
        messages("equation e(t: s):\n    x: m = 3 m\n"),
        ["a state variable is declared without a value"]
    );
    assert_eq!(
        messages("equation e(t: s):\n    x' 1\n"),
        ["expected `=` after `x'`, found integer literal `1`"]
    );
    assert_eq!(
        messages("equation e(t: s):\n    x 1\n"),
        ["expected `:`, `'` or `=` after `x`, found integer literal `1`"]
    );
    assert_eq!(
        messages("equation e(t: s):\n    3 = x\n"),
        ["expected a state variable (`x: m`), an equation (`x' = ...`) or a name (`a = ...`), found integer literal `3`"]
    );
}

#[test]
fn one_bad_line_does_not_hide_the_next() {
    let errors = messages("equation e(t: s):\n    x 1\n    y = = 2\n    z: m\n");
    assert_eq!(errors.len(), 2, "{errors:?}");
}

#[test]
fn solve_errors() {
    assert_eq!(
        messages("solve e from 0 s to 1 s\n"),
        ["`solve` gives a solution, which needs a name"]
    );
    assert_eq!(
        messages("s = solve 3 from 0 s to 1 s\n"),
        ["expected an expression, found keyword `solve`"]
    );
    assert_eq!(
        messages("s = solve e 0 s to 1 s\n"),
        ["expected `from`, found integer literal `0`"]
    );
    assert_eq!(
        messages("s = solve e from 0 s 1 s\n"),
        ["expected `to`, found integer literal `1`"]
    );
    assert_eq!(
        messages("s = solve e from 0 s to 1 s with x 1\n"),
        ["expected `=` after `x`, found integer literal `1`"]
    );
    assert_eq!(
        messages("s = solve e from 0 s to 1 s with = 1\n"),
        ["expected a state variable name, found `=`"]
    );
    assert_eq!(
        messages("s = solve e from 0 s to 1 s:\n    tol 1\n"),
        ["unknown `solve` option `tol`"]
    );
    assert_eq!(
        messages("s = solve e from 0 s to 1 s:\n    conserve = 1\n"),
        ["expected a name for the conserved quantity, found `=`"]
    );
    assert_eq!(
        messages("s = solve e from 0 s to 1 s: tolerance 1\n"),
        ["the options of a `solve` must be on their own indented lines"]
    );
    // The options need the `:`; without it the indented lines are strays.
    assert_eq!(
        messages("s = solve e from 0 s to 1 s\n    tolerance 1\n")[0],
        "unexpected indentation"
    );
}

#[test]
fn a_bare_prime_is_still_an_error_in_expressions() {
    let errors = messages("y = x' + 1\n");
    assert_eq!(errors, ["expected end of line, found `'`"]);
}

#[test]
fn deeply_nested_equations_are_bounded() {
    with_deep_stack(|| {
        // An equation body counts as one level, like any block.
        let mut src = String::new();
        for i in 0..MAX_DEPTH as usize + 2 {
            src.push_str(&format!("{}if a:\n", "    ".repeat(i)));
        }
        let pad = "    ".repeat(MAX_DEPTH as usize + 2);
        src.push_str(&format!("{pad}equation e(t: s):\n{pad}    x: m\n"));
        let tokens = lex(&src);
        let (_module, errors) = parse_module(&tokens, &src);
        assert!(!errors.is_empty());
    });
}
