//! Uncertain numbers: `±`, propagation against hand computations,
//! correlations, the math functions, printing. Like the other interpreter
//! tests these run without the checker, so they also cover the dynamic
//! checks a checked program never reaches.

use sky_ast::parse_module;
use sky_syntax::lex;

use crate::uncertain::Lin;
use crate::{run_module, RuntimeEnv, RuntimeError, Value};

fn run(src: &str) -> Result<(Option<Value>, String), RuntimeError> {
    let src = format!("{src}\n");
    let lexed = lex(&src).expect("lexes");
    let (module, errors) = parse_module(&lexed.tokens, &src);
    assert!(errors.is_empty(), "parse errors in {src:?}: {errors:?}");
    let mut out = Vec::new();
    let v = run_module(&module, &src, &mut RuntimeEnv::new(), &mut out)?;
    Ok((v, String::from_utf8(out).expect("utf-8")))
}

fn eval(src: &str) -> Value {
    run(src).expect("runs").0.expect("a value")
}

/// The printed form of the last expression.
fn show(src: &str) -> String {
    eval(src).to_string()
}

fn printed(src: &str) -> String {
    run(src).expect("runs").1
}

fn err(src: &str) -> RuntimeError {
    run(src).expect_err("fails at run time")
}

/// Nominal value and standard uncertainty of the last expression, in SI.
fn parts(src: &str) -> (f64, f64) {
    match eval(src) {
        Value::Uncertain(u) => (u.value, u.std_dev()),
        other => panic!("{src:?} is not uncertain: {other:?}"),
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1e-300)
}

#[track_caller]
fn assert_parts(src: &str, value: f64, sigma: f64) {
    let (v, s) = parts(src);
    assert!(
        close(v, value) && close(s, sigma),
        "{src:?}: got {v} ± {s}, expected {value} ± {sigma}"
    );
}

// ----- the value -----

#[test]
fn plus_minus_makes_an_uncertain_number() {
    let Value::Uncertain(u) = eval("5.31 ± 0.02 kg") else {
        panic!("not uncertain");
    };
    assert_eq!(u.value, 5.31);
    assert_eq!(u.std_dev(), 0.02);
    assert_eq!(u.dim.name(), Some("mass"));
    assert_eq!(u.lin.terms().len(), 1);
}

#[test]
fn ascii_spelling_and_the_kg_g_form() {
    assert_eq!(show("5.31 +/- 0.02 kg"), "5.31 ± 0.02 kg");
    // Each side has its own unit; both are converted to SI at the literal.
    assert_eq!(show("5.31 kg ± 20 g"), "5.31 ± 0.02 kg");
    assert_eq!(show("5310 g ± 0.02 kg"), "5.31 ± 0.02 kg");
    assert_eq!(show("2 ± 0.1"), "2.0 ± 0.1");
    assert_eq!(show("-2 ± 0.1 m"), "-2.0 ± 0.1 m");
}

#[test]
fn any_expression_may_be_the_value_and_the_uncertainty() {
    assert_parts("x = 3 m\ndx = 2 cm\nx ± dx", 3.0, 0.02);
    assert_parts("(1 m + 2 m) ± 0.5 * 2 cm", 3.0, 0.01);
    assert_parts("sqrt(9) ± 0.1", 3.0, 0.1);
}

#[test]
fn zero_uncertainty_is_exact_but_still_written_as_uncertain() {
    let Value::Uncertain(u) = eval("5 kg ± 0 kg") else {
        panic!("not uncertain");
    };
    assert!(u.lin.is_exact());
    assert_eq!(show("5 kg ± 0 kg"), "5 ± 0 kg");
}

#[test]
fn a_second_plus_minus_adds_an_independent_uncertainty() {
    assert_parts("(1 ± 0.3) ± 0.4", 1.0, 0.5);
    assert_parts("1 ± 0.3 ± 0.4", 1.0, 0.5);
}

#[test]
fn errors_of_plus_minus() {
    let e = err("5 ± -0.1");
    assert_eq!(e.message, "an uncertainty cannot be negative");
    let e = err("5 kg ± 1 s");
    assert_eq!(e.message, "cannot use a time as the uncertainty of a mass");
    let e = err("5 ± (1 ± 0.1)");
    assert_eq!(
        e.message,
        "the uncertainty after `±` must be an exact number"
    );
    let e = err("true ± 1");
    assert_eq!(e.message, "`±` needs a number on its left, found a Bool");
    let e = err("1 ± true");
    assert_eq!(e.message, "`±` needs a number on its right, found a Bool");
}

// ----- propagation, against hand computations -----

#[test]
fn sum_and_difference_add_in_quadrature() {
    // sqrt(0.3² + 0.4²) = 0.5.
    assert_parts("(3 ± 0.3) + (4 ± 0.4)", 7.0, 0.5);
    assert_parts("(3 ± 0.3) - (4 ± 0.4)", -1.0, 0.5);
    assert_eq!(show("(3 ± 0.3) + (4 ± 0.4)"), "7.0 ± 0.5");
    // An exact operand adds nothing.
    assert_parts("(3 ± 0.3) + 4", 7.0, 0.3);
    assert_parts("10 - (3 ± 0.3)", 7.0, 0.3);
}

#[test]
fn product_and_quotient_add_relative_uncertainties() {
    // x = 10 ± 1, y = 20 ± 2.
    // x*y = 200, σ² = (y σx)² + (x σy)² = 20² + 20² = 800.
    assert_parts("(10 ± 1) * (20 ± 2)", 200.0, 800f64.sqrt());
    // x/y = 0.5, σ² = (σx/y)² + (x σy/y²)² = 0.05² + 0.05² = 0.005.
    assert_parts("(10 ± 1) / (20 ± 2)", 0.5, 0.005f64.sqrt());
    // Scaling by an exact number scales the uncertainty.
    assert_parts("3 * (10 ± 1)", 30.0, 3.0);
    assert_parts("(10 ± 1) / 4", 2.5, 0.25);
    assert_parts("4 / (10 ± 1)", 0.4, 0.04);
}

#[test]
fn units_carry_through() {
    let Value::Uncertain(u) = eval("(10 ± 1 m) / (2 ± 0.1 s)") else {
        panic!("not uncertain");
    };
    assert_eq!(u.dim.name(), Some("velocity"));
    // σ² = (1/2)² + (10·0.1/4)² = 0.25 + 0.0625.
    assert!(close(u.std_dev(), 0.3125f64.sqrt()));
    assert_eq!(show("(10 ± 1 m) / (2 ± 0.1 s)"), "5.0 ± 0.6 m/s");
}

#[test]
fn powers_with_an_exact_exponent() {
    // d(x^n) = n x^(n-1) dx: 3 · 4 · 0.1 = 1.2 for x = 2.
    assert_parts("(2 ± 0.1) ^ 3", 8.0, 3.0 * 4.0 * 0.1);
    assert_parts("(4 ± 0.4) ^ 0.5", 2.0, 0.5 / 2.0 * 0.4);
    assert_parts("(2 ± 0.1) ^ -1", 0.5, 0.025);
    assert_parts("(2 ± 0.1) ^ 0", 1.0, 0.0);
    // With a unit: the dimension comes from the constant exponent.
    let Value::Uncertain(u) = eval("(3 ± 0.1 m) ^ 2") else {
        panic!("not uncertain");
    };
    assert!(close(u.std_dev(), 2.0 * 3.0 * 0.1));
    assert_eq!(show("(3 ± 0.1 m) ^ 2"), "9.0 ± 0.6 m^2");
    // 0 ^ 2 has slope 0, 0 ^ 1 has slope 1.
    assert_parts("(0 ± 0.1) ^ 2", 0.0, 0.0);
    assert_parts("(0 ± 0.1) ^ 1", 0.0, 0.1);
}

#[test]
fn powers_with_an_uncertain_exponent() {
    // d(2^x)/dx = 2^x ln 2: at x = 3, 8 ln 2 · 0.1.
    assert_parts("2 ^ (3 ± 0.1)", 8.0, 8.0 * std::f64::consts::LN_2 * 0.1);
    // Both uncertain: a = 2 ± 0.1, x = 3 ± 0.2.
    let (v, s) = parts("(2 ± 0.1) ^ (3 ± 0.2)");
    let da = 3.0 * 4.0 * 0.1;
    let dx = 8.0 * std::f64::consts::LN_2 * 0.2;
    assert!(close(v, 8.0));
    assert!(close(s, (da * da + dx * dx).sqrt()));
    // A negative base has no logarithm.
    let e = err("(-2) ^ (3 ± 0.1)");
    assert_eq!(
        e.message,
        "the exponent's uncertainty needs a positive base"
    );
}

#[test]
fn negation_and_reciprocal() {
    assert_parts("-(3 ± 0.3)", -3.0, 0.3);
    assert_parts("+(3 ± 0.3)", 3.0, 0.3);
    // d(1/x) = -dx/x²: 0.3/9.
    assert_parts("inv(3 ± 0.3)", 1.0 / 3.0, 0.3 / 9.0);
    assert_parts("(2 ± 0.1)⁻¹", 0.5, 0.025);
}

// ----- correlations -----

#[test]
fn a_value_minus_itself_is_exactly_zero() {
    let Value::Uncertain(u) = eval("x = 5.31 ± 0.02\nx - x") else {
        panic!("not uncertain");
    };
    assert_eq!(u.value, 0.0);
    assert_eq!(u.std_dev(), 0.0);
    assert!(u.lin.is_exact());
    assert_eq!(show("x = 5.31 ± 0.02 kg\nx - x"), "0 ± 0 kg");
    // The spec's example: it equals `0 ± 0` (comparison is of nominal
    // values, and the difference is exact).
    assert_eq!(
        eval("x = 5.31 ± 0.02 kg\nx - x == 0 ± 0 kg"),
        Value::Bool(true)
    );
    assert_eq!(
        eval("x = 5.31 ± 0.02 kg\nuncertainty(x - x) == 0 kg"),
        Value::Bool(true)
    );
    // Two independent values with the same numbers do not cancel.
    assert_parts("(5 ± 0.5) - (5 ± 0.5)", 0.0, 0.5 * 2f64.sqrt());
}

#[test]
fn a_value_plus_itself_doubles_and_does_not_add_in_quadrature() {
    assert_parts("x = 3 ± 0.1\nx + x", 6.0, 0.2);
    // Independent: sqrt(2) · 0.1.
    assert_parts("(3 ± 0.1) + (3 ± 0.1)", 6.0, 0.1 * 2f64.sqrt());
}

#[test]
fn x_times_x_agrees_with_x_squared() {
    // Both are 2 x σ, not sqrt(2) x σ (which independent factors give).
    let a = parts("x = 3 ± 0.1\nx * x");
    let b = parts("x = 3 ± 0.1\nx ^ 2");
    assert_eq!(a, b);
    assert!(close(a.0, 9.0) && close(a.1, 0.6));
    let independent = parts("(3 ± 0.1) * (3 ± 0.1)");
    assert!(close(independent.1, 3.0 * 0.1 * 2f64.sqrt()));
}

#[test]
fn x_over_x_is_exactly_one() {
    assert_parts("x = 3 ± 0.1\nx / x", 1.0, 0.0);
}

#[test]
fn shared_sources_cancel_through_intermediate_values() {
    // (x + y) - y is x; (x + y) - x - y is exactly zero.
    assert_parts("x = 3 ± 0.1\ny = 4 ± 0.2\n(x + y) - y", 3.0, 0.1);
    assert_parts("x = 3 ± 0.1\ny = 4 ± 0.2\n(x + y) - x - y", 0.0, 0.0);
    // z = x + y and w = x - y are correlated through x and y: z + w = 2x.
    assert_parts("x = 3 ± 0.1\ny = 4 ± 0.2\n(x + y) + (x - y)", 6.0, 0.2);
}

#[test]
fn the_covariance_of_shared_values() {
    let Value::Uncertain(a) = eval("x = 3 ± 0.1\ny = 4 ± 0.2\nx + y") else {
        panic!()
    };
    let Value::Uncertain(b) = eval("x = 3 ± 0.1\ny = 4 ± 0.2\nx - y") else {
        panic!()
    };
    // Different sources (each run makes its own), so nothing is shared.
    assert_eq!(a.lin.covariance(&b.lin), 0.0);
    // Within one run: cov(x + y, x - y) = σx² - σy² = 0.01 - 0.04.
    let (Value::Uncertain(z), Value::Uncertain(w)) = (
        eval("x = 3 ± 0.1\ny = 4 ± 0.2\nz = x + y\nz"),
        eval("x = 3 ± 0.1\ny = 4 ± 0.2\nz = x + y\nz - 2 * y"),
    ) else {
        panic!()
    };
    assert!(close(z.lin.covariance(&z.lin), z.std_dev().powi(2)));
    assert!(w.std_dev() > 0.0);
}

#[test]
fn a_repeated_measurement_averages_down() {
    // 100 independent readings of 10 ± 1: the mean has σ = 1/sqrt(100).
    let src = "s = 0 ± 0\nfor i in range(100):\n    s += 10 ± 1\ns / 100";
    assert_parts(src, 10.0, 0.1);
    // The same reading counted 100 times does not average down.
    let src = "x = 10 ± 1\ns = 0 ± 0\nfor i in range(100):\n    s += x\ns / 100";
    assert_parts(src, 10.0, 1.0);
}

// ----- math functions -----

#[test]
fn math_functions_use_the_derivative() {
    // sqrt: 0.4 / (2·2).
    assert_parts("sqrt(4 ± 0.4)", 2.0, 0.1);
    // exp: e · 0.1.
    assert_parts("exp(1 ± 0.1)", 1f64.exp(), 1f64.exp() * 0.1);
    // ln: 0.1 / 2.
    assert_parts("log(2 ± 0.1)", 2f64.ln(), 0.05);
    // sin: cos(0) · 0.1.
    assert_parts("sin(0 ± 0.1)", 0.0, 0.1);
    assert_parts("cos(0 ± 0.1)", 1.0, 0.0);
    assert_parts("tan(0 ± 0.1)", 0.0, 0.1);
    // atan2 and hypot: the 3-4-5 triangle. hypot: (3/5)² 0.1² + (4/5)² 0.2².
    let s = (0.36 * 0.01 + 0.64 * 0.04f64).sqrt();
    assert_parts("hypot(3 ± 0.1, 4 ± 0.2)", 5.0, s);
    // atan2(y, x): ∂y = x/25, ∂x = -y/25.
    let s = ((4.0 / 25.0 * 0.1f64).powi(2) + (3.0 / 25.0 * 0.2f64).powi(2)).sqrt();
    assert_parts("atan2(3 ± 0.1, 4 ± 0.2)", 3f64.atan2(4.0), s);
    // With units.
    let Value::Uncertain(u) = eval("sqrt(4 ± 0.4 m^2)") else {
        panic!()
    };
    assert_eq!(u.dim.name(), Some("length"));
    assert!(close(u.std_dev(), 0.1));
}

#[test]
fn math_functions_of_correlated_values() {
    // sqrt(x * x) is |x| with the uncertainty of x, not of x²'s root of a
    // independent product.
    assert_parts("x = 3 ± 0.1\nsqrt(x * x)", 3.0, 0.1);
    // sin² + cos² = 1 exactly, with no uncertainty: the slopes cancel.
    let (v, s) = parts("x = 0.7 ± 0.1\nsin(x) ^ 2 + cos(x) ^ 2");
    assert!(close(v, 1.0));
    assert!(s < 1e-15, "sin²+cos² has σ {s}");
}

#[test]
fn steps_have_no_uncertainty() {
    assert_parts("floor(2.5 ± 0.1)", 2.0, 0.0);
    assert_parts("ceil(2.5 ± 0.1)", 3.0, 0.0);
    assert_parts("round(2.5 ± 0.1)", 3.0, 0.0);
    assert_parts("sign(-2.5 ± 0.1)", -1.0, 0.0);
}

#[test]
fn abs_min_and_max_take_the_arm_that_runs() {
    assert_parts("abs(-2 ± 0.1)", 2.0, 0.1);
    assert_parts("abs(2 ± 0.1)", 2.0, 0.1);
    // At zero the right-hand slope, +1, as for derivatives.
    assert_parts("abs(0 ± 0.1)", 0.0, 0.1);
    assert_parts("min(2 ± 0.1, 3 ± 0.2)", 2.0, 0.1);
    assert_parts("max(2 ± 0.1, 3 ± 0.2)", 3.0, 0.2);
    // Equal: the first operand.
    assert_parts("min(2 ± 0.1, 2 ± 0.2)", 2.0, 0.1);
    assert_parts("max(2 ± 0.1, 2 ± 0.2)", 2.0, 0.1);
    // An exact operand that is returned is exact.
    assert_eq!(eval("min(1, 2 ± 0.1)"), Value::scalar(1.0));
}

#[test]
fn an_infinite_slope_is_an_error_and_an_exact_argument_is_not() {
    let e = err("sqrt(0 ± 0.1)");
    assert_eq!(
        e.message,
        "cannot propagate an uncertainty through `sqrt` at 0"
    );
    let e = err("asin(1 ± 0.01)");
    assert!(e.message.contains("`asin` at 1"), "{}", e.message);
    // The domain errors are the exact functions'.
    let e = err("sqrt(-1 ± 0.1)");
    assert_eq!(e.message, "`sqrt` of a negative number has no real value");
    // An exact argument never needs a slope.
    assert_eq!(show("sqrt(0)"), "0");
    assert_parts("sqrt(0 ± 0)", 0.0, 0.0);
    assert_eq!(
        err("(0 ± 0.1) ^ 0.5").message,
        "cannot propagate an uncertainty through `^` at a zero base"
    );
}

#[test]
fn dimension_rules_are_the_exact_ones() {
    let e = err("sin(1 ± 0.1 m)");
    assert!(e.message.contains("must be dimensionless"), "{}", e.message);
    let e = err("(1 ± 0.1 m) + (1 ± 0.1 s)");
    assert!(e.message.contains("cannot add"), "{}", e.message);
    let e = err("(1 ± 0.1 m) ^ (2 ± 0.1)");
    assert!(
        e.message.contains("constant number"),
        "an exponent with a unit base must be constant: {}",
        e.message
    );
}

#[test]
fn overflow_is_an_error_not_infinity() {
    let e = err("(1e200 ± 1e200) * (1e200 ± 1e200)");
    assert_eq!(e.message, "arithmetic overflow");
    let e = err("(1e200 ± 1e200) * 1e200");
    assert_eq!(e.message, "arithmetic overflow");
}

// ----- comparisons and parts -----

#[test]
fn comparisons_compare_nominal_values() {
    assert_eq!(eval("(3 ± 5) < (4 ± 5)"), Value::Bool(true));
    assert_eq!(eval("(3 ± 5) >= (4 ± 0)"), Value::Bool(false));
    assert_eq!(eval("(3 ± 5) == 3"), Value::Bool(true));
    assert_eq!(eval("(3 ± 5) != (3 ± 1)"), Value::Bool(false));
    assert_eq!(eval("2 kg < 3 ± 1 kg"), Value::Bool(true));
    let e = err("(3 ± 5) < 4 kg");
    assert!(e.message.starts_with("cannot compare"), "{}", e.message);
}

#[test]
fn nominal_and_uncertainty() {
    assert_eq!(eval("nominal(5.31 ± 0.02 kg)"), eval("5.31 kg"));
    assert_eq!(eval("uncertainty(5.31 ± 0.02 kg)"), eval("0.02 kg"));
    // Exact numbers: themselves, and zero.
    assert_eq!(eval("nominal(5 m)"), eval("5 m"));
    assert_eq!(eval("uncertainty(5 m)"), eval("0 m"));
    assert_eq!(eval("uncertainty(3)"), Value::scalar(0.0));
    // A consistency test written with them.
    assert_eq!(
        eval("a = 9.81 ± 0.05\nb = 9.79 ± 0.03\nabs(nominal(a - b)) < 2 * uncertainty(a - b)"),
        Value::Bool(true)
    );
    let e = err("nominal(true)");
    assert_eq!(e.message, "`nominal` needs a number, found a Bool");
    let e = err("uncertainty(1, 2)");
    assert_eq!(e.message, "`uncertainty` takes 1 argument, found 2");
    // A user's own name wins, as for the other builtins.
    assert_eq!(
        eval("def nominal(x):\n    return 7\nnominal(1 ± 0.1)"),
        Value::scalar(7.0)
    );
}

#[test]
fn control_flow_and_functions_carry_uncertainty() {
    let src = "def area(r):\n    return 3 * r ^ 2\narea(2 ± 0.1)";
    assert_parts(src, 12.0, 3.0 * 2.0 * 2.0 * 0.1);
    assert_parts("x = 2 ± 0.1\nx = x * 2\nx", 4.0, 0.2);
    assert_parts("x = 2 ± 0.1\nx *= 2\nx", 4.0, 0.2);
    assert_parts("x = 1 ± 0.1 if true else 2 ± 0.2\nx", 1.0, 0.1);
}

// ----- printing -----

#[test]
fn printing() {
    assert_eq!(printed("print(5.31 ± 0.02 kg)"), "5.31 ± 0.02 kg\n");
    assert_eq!(
        printed("print(9.8123 ± 0.0234 m/s^2)"),
        "9.812 ± 0.023 m/s^2\n"
    );
    assert_eq!(printed("print(1 ± 0.1, 2)"), "1.0 ± 0.1 2\n");
    assert_eq!(
        printed("m = 5.31 ± 0.02 kg\nprint(\"m =\", m)"),
        "m = 5.31 ± 0.02 kg\n"
    );
}

#[test]
fn as_shows_the_value_and_the_uncertainty_in_a_unit() {
    assert_eq!(
        printed("v = 5.31 ± 0.02 kg\nprint(v as g)"),
        "5310 ± 20 g\n"
    );
    assert_eq!(printed("print(1500 ± 30 m as km)"), "1.50 ± 0.03 km\n");
    assert_eq!(
        printed("print((10 ± 1 m) / (2 ± 0.1 s) as km/h)"),
        "18 ± 2 km/h\n"
    );
    assert_eq!(show("5.31 ± 0.02 kg as g"), "5310 ± 20 g");
    let e = err("(5.31 ± 0.02 kg) as m");
    assert!(e.message.contains("cannot show"), "{}", e.message);
}

#[test]
fn a_printed_value_reads_back() {
    // `x ± dx unit` is valid Sky and gives the same value and uncertainty
    // (to the printed digits).
    let text = show("(10 ± 1 m) / (2 ± 0.1 s)");
    assert_eq!(text, "5.0 ± 0.6 m/s");
    assert_parts(&text, 5.0, 0.6);
}

// ----- the pendulum -----

#[test]
fn pendulum_period() {
    // T = 2π sqrt(L/g) with L = 1.000 ± 0.005 m and g = 9.81 ± 0.05 m/s².
    let src =
        "L = 1.000 ± 0.005 m\ng = 9.81 ± 0.05 m/s^2\nT = 2 * 3.141592653589793 * sqrt(L / g)\nT";
    let Value::Uncertain(t) = eval(src) else {
        panic!()
    };
    let t0 = 2.0 * std::f64::consts::PI * (1.0f64 / 9.81).sqrt();
    // dT/T = ½ sqrt((dL/L)² + (dg/g)²).
    let rel = 0.5 * ((0.005f64 / 1.0).powi(2) + (0.05f64 / 9.81).powi(2)).sqrt();
    assert!(close(t.value, t0));
    assert!(close(t.std_dev(), t0 * rel), "σ = {}", t.std_dev());
    assert_eq!(t.dim.name(), Some("time"));
    assert_eq!(show(src), "2.006 ± 0.007 s");
}

// ----- Lin -----

#[test]
fn combine_cancels_and_sorts() {
    let a = Lin::source(0.1);
    let b = Lin::source(0.2);
    let sp = sky_diag::Span::point(0);
    // 1·a + 1·b, then minus a.
    let ab = Lin::combine(&[(a.terms(), 1.0), (b.terms(), 1.0)], sp).expect("fits");
    assert_eq!(ab.terms().len(), 2);
    assert!(ab.terms().windows(2).all(|w| w[0].source < w[1].source));
    let back = Lin::combine(&[(ab.terms(), 1.0), (a.terms(), -1.0)], sp).expect("fits");
    assert_eq!(back, b);
    let none = Lin::combine(&[(a.terms(), 1.0), (a.terms(), -1.0)], sp).expect("fits");
    assert!(none.is_exact());
    assert_eq!(none.std_dev(), 0.0);
    // A zero weight contributes nothing.
    let zero = Lin::combine(&[(a.terms(), 0.0)], sp).expect("fits");
    assert!(zero.is_exact());
    // Overflow is an error.
    let big = Lin::source(1e300);
    assert!(Lin::combine(&[(big.terms(), 1e300)], sp).is_err());
}

#[test]
fn std_dev_survives_huge_and_tiny_coefficients() {
    let sp = sky_diag::Span::point(0);
    let a = Lin::source(1e200);
    let b = Lin::source(1e200);
    let ab = Lin::combine(&[(a.terms(), 1.0), (b.terms(), 1.0)], sp).expect("fits");
    assert!(close(ab.std_dev(), 1e200 * 2f64.sqrt()));
    let a = Lin::source(1e-200);
    let b = Lin::source(1e-200);
    let ab = Lin::combine(&[(a.terms(), 1.0), (b.terms(), 1.0)], sp).expect("fits");
    assert!(close(ab.std_dev(), 1e-200 * 2f64.sqrt()));
}

#[test]
fn a_run_is_deterministic() {
    let src = "x = 3 ± 0.1\ny = 4 ± 0.2\nz = sqrt(x * y + x) / y\nz";
    let (a, b) = (parts(src), parts(src));
    assert_eq!(a.0.to_bits(), b.0.to_bits());
    assert_eq!(a.1.to_bits(), b.1.to_bits());
}

#[test]
fn arrays_do_not_carry_uncertainty_yet() {
    let e = err("[1 ± 0.1, 2]");
    assert_eq!(
        e.message,
        "an array element must be a number, found an uncertain number"
    );
}
