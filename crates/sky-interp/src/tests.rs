//! Interpreter unit tests: source in, printed value or error out. These run
//! without the checker, so they also cover the dynamic checks a checked
//! program never reaches.

use sky_ast::parse_module;
use sky_syntax::lex;

use crate::{format_number, run_module, RuntimeEnv, RuntimeError, Value};

/// Runs `src` in `env`, returning the last expression's value and the
/// `print` output.
fn run_in(env: &mut RuntimeEnv, src: &str) -> Result<(Option<Value>, String), RuntimeError> {
    let lexed = lex(src).expect("lexes");
    let (module, errors) = parse_module(&lexed.tokens, src);
    assert!(
        errors.is_empty(),
        "parse errors in {:?}",
        &src[..src.len().min(80)]
    );
    let mut out = Vec::new();
    let v = run_module(&module, src, env, &mut out)?;
    Ok((v, String::from_utf8(out).expect("utf-8 output")))
}

fn eval(src: &str) -> Value {
    let (v, _) = run_in(&mut RuntimeEnv::new(), &format!("{src}\n")).expect("runs");
    v.expect("an expression statement")
}

/// The printed form of the value of `src`.
fn show(src: &str) -> String {
    eval(src).to_string()
}

fn err(src: &str) -> RuntimeError {
    run_in(&mut RuntimeEnv::new(), &format!("{src}\n")).expect_err("fails at run time")
}

#[test]
fn number_literals() {
    assert_eq!(show("42"), "42");
    assert_eq!(show("1_000"), "1000");
    assert_eq!(show("2.5"), "2.5");
    assert_eq!(show("1_0.5e1_0"), "105000000000");
    assert_eq!(show("2e-3"), "0.002");
    // Too large for f64: an error on the literal, not inf.
    let e = err("1e999");
    assert_eq!(e.message, "number too large");
}

#[test]
fn unit_literals_convert_to_si_once() {
    assert_eq!(show("10 km"), "10000 m");
    assert_eq!(show("10 ms"), "0.01 s");
    assert_eq!(show("2 min"), "120 s");
    assert_eq!(show("1 h"), "3600 s");
    assert_eq!(show("9.81 m/s^2"), "9.81 m/s^2");
    assert_eq!(show("1 km + 1 m"), "1001 m");
    // Factors apply one at a time, so this is exact.
    assert_eq!(eval("36 km/h"), eval("10 m/s"));
    assert_eq!(show("2 kg*m^2/s^2"), "2 J");
    assert_eq!(show("4 m^(1/2)"), "4 m^(1/2)");
    assert_eq!(show("3 s^-2"), "3 s^-2");
    assert_eq!(show("1 eV"), "1.602176634e-19 J");
}

#[test]
fn spec_float_case() {
    // spec "Unit values": 100 ms is 0.1 and 3 * 0.1 is 0.30000000000000004.
    assert_eq!(show("3 * (100 ms)"), "0.30000000000000004 s");
    assert_eq!(eval("3 * (100 ms) == 0.3 s"), Value::Bool(false));
    assert_eq!(eval("100 ms == 0.1 s"), Value::Bool(true));
}

#[test]
fn arithmetic_operators() {
    assert_eq!(show("2 + 3"), "5");
    assert_eq!(show("2 - 3"), "-1");
    assert_eq!(show("2 * 3"), "6");
    assert_eq!(show("7 / 2"), "3.5");
    assert_eq!(show("2 ^ 10"), "1024");
    assert_eq!(show("2 ^ -1"), "0.5");
    assert_eq!(show("2 ^ 0.5 == 1.4142135623730951"), "true");
    // `-x^2` is `-(x^2)`.
    assert_eq!(show("-3 ^ 2"), "-9");
    assert_eq!(show("+3"), "3");
    assert_eq!(show("(1 + 2) * 3"), "9");
    assert_eq!(show("2 ^ 3 ^ 2"), "512");
}

#[test]
fn dimensions_follow_the_operators() {
    assert_eq!(show("10 m / 2 s"), "5 m/s");
    assert_eq!(show("2 m * 3 m"), "6 m^2");
    assert_eq!(show("2 kg * 3 m/s^2"), "6 N");
    assert_eq!(show("3 m / 3 m"), "1");
    assert_eq!(show("(3 m)^2"), "9 m^2");
    assert_eq!(show("(9 m^2)^(1/2)"), "3 m");
    assert_eq!(show("(8 m^3)^(1/3)"), "2 m");
    assert_eq!(show("1 m - 3 m"), "-2 m");
}

#[test]
fn inverse_and_inv() {
    assert_eq!(show("4⁻¹"), "0.25");
    assert_eq!(show("(2 s)⁻¹"), "0.5 Hz");
    assert_eq!(show("inv(4 m)"), "0.25 m^-1");
    assert_eq!(show("inv(inv(8))"), "8");
    assert_eq!(err("inv(0)").message, "division by zero");
    assert_eq!(err("(0 m)⁻¹").message, "division by zero");
}

#[test]
fn comparisons_use_si_values() {
    assert_eq!(eval("1 km > 999 m"), Value::Bool(true));
    assert_eq!(eval("1 km < 999 m"), Value::Bool(false));
    assert_eq!(eval("1 km >= 1000 m"), Value::Bool(true));
    assert_eq!(eval("1 km <= 999 m"), Value::Bool(false));
    assert_eq!(eval("60 s == 1 min"), Value::Bool(true));
    assert_eq!(eval("60 s != 1 min"), Value::Bool(false));
    assert_eq!(eval("true == true"), Value::Bool(true));
    assert_eq!(eval("true != false"), Value::Bool(true));
    assert_eq!(eval(r#""a" == "a""#), Value::Bool(true));
    assert_eq!(eval(r#""a" != "b""#), Value::Bool(true));
}

#[test]
fn boolean_operators() {
    assert_eq!(eval("not true"), Value::Bool(false));
    assert_eq!(eval("true and false"), Value::Bool(false));
    assert_eq!(eval("true and true and true"), Value::Bool(true));
    assert_eq!(eval("false or false or true"), Value::Bool(true));
    assert_eq!(eval("false or false"), Value::Bool(false));
    assert_eq!(eval("true and false or true"), Value::Bool(true));
    assert_eq!(eval("not 1 < 2"), Value::Bool(false));
}

#[test]
fn and_or_short_circuit() {
    // The right sides would each be a runtime error if evaluated.
    assert_eq!(eval("false and (1/0 > 0)"), Value::Bool(false));
    assert_eq!(eval("true or (1/0 > 0)"), Value::Bool(true));
    assert_eq!(eval("true and false and (1/0 > 0)"), Value::Bool(false));
    // And they are evaluated when the left side does not decide.
    assert_eq!(err("true and (1/0 > 0)").message, "division by zero");
    assert_eq!(err("false or (1/0 > 0)").message, "division by zero");
    // A side effect on the right must not happen either.
    let (v, out) = run_in(&mut RuntimeEnv::new(), "false and (print(1) == 1)\n").unwrap();
    assert_eq!((v, out.as_str()), (Some(Value::Bool(false)), ""));
}

#[test]
fn no_truthiness_at_run_time() {
    let e = err("1 and true");
    assert_eq!(e.message, "`and` needs a Bool operand, found a number");
    assert!(err("not 0").message.contains("needs a Bool"));
}

#[test]
fn division_by_zero_is_an_error_with_a_span() {
    let src = "x = 1 / 0\n";
    let e = run_in(&mut RuntimeEnv::new(), src).expect_err("fails");
    assert_eq!(e.message, "division by zero");
    // The primary span is the whole division; the zero is a secondary label.
    assert_eq!(&src[e.span.range()], "1 / 0");
    assert_eq!(&src[e.secondary[0].0.range()], "0");
    assert!(err("1 m / 0 s").message == "division by zero");
    assert!(err("0 ^ -1").message == "division by zero");
}

#[test]
fn overflow_is_an_error() {
    assert_eq!(err("1e308 * 10").message, "arithmetic overflow");
    assert_eq!(err("1e308 + 1e308").message, "arithmetic overflow");
    assert_eq!(err("10 ^ 400").message, "arithmetic overflow");
    assert_eq!(err("1e-320⁻¹").message, "arithmetic overflow");
    assert_eq!(err("1e308 m * 1e10").message, "arithmetic overflow");
    // Underflow is not an error.
    assert_eq!(show("1e-200 * 1e-200"), "0");
}

#[test]
fn no_real_result_is_an_error() {
    let e = err("(-8) ^ (1/3)");
    assert_eq!(e.message, "the result is not a real number");
    assert_eq!(show("(-8) ^ 3"), "-512");
}

#[test]
fn exponent_rules() {
    assert!(err("(2 m) ^ (2 m)").message.contains("dimensionless"));
    let mut env = RuntimeEnv::new();
    run_in(&mut env, "n = 2\n").unwrap();
    let e = run_in(&mut env, "(3 m) ^ n\n").unwrap_err();
    assert!(e.message.contains("constant"));
    // A dimensionless base may have a computed exponent.
    assert_eq!(
        run_in(&mut env, "3 ^ n\n").unwrap().0,
        Some(Value::scalar(9.0))
    );
}

#[test]
fn dynamic_type_errors() {
    assert_eq!(
        err("1 m + 1 s").message,
        "cannot add length and time",
        "the checker rejects this; the interpreter still does not add them"
    );
    assert_eq!(err("1 m < 1 s").message, "cannot compare length and time");
    assert_eq!(err("true + 1").message, "cannot add a Bool and a number");
    assert_eq!(
        err("1 - true").message,
        "cannot subtract a Bool from a number"
    );
    assert_eq!(
        err(r#""a" * 2"#).message,
        "cannot multiply a string and a number"
    );
    assert_eq!(
        err(r#"true < false"#).message,
        "cannot order a Bool and a Bool with `<`"
    );
    assert_eq!(err("-true").message, "`-` needs a number, found a Bool");
    assert_eq!(
        err("1 == true").message,
        "cannot compare a number and a Bool"
    );
}

#[test]
fn names_and_assignment() {
    let mut env = RuntimeEnv::new();
    run_in(&mut env, "x = 10 km\n").unwrap();
    assert_eq!(
        env.get("x"),
        Some(&Value::number(10000.0, sky_units::Dimension::LENGTH))
    );
    let (v, _) = run_in(&mut env, "x / 2 s\n").unwrap();
    assert_eq!(v.unwrap().to_string(), "5000 m/s");
    // Rebinding.
    run_in(&mut env, "x = 1\n").unwrap();
    assert_eq!(env.get("x"), Some(&Value::scalar(1.0)));
    // An assignment is not an expression statement: no value to echo.
    assert_eq!(run_in(&mut env, "y = 2\n").unwrap().0, None);
    let e = run_in(&mut env, "z\n").unwrap_err();
    assert_eq!(e.message, "cannot find `z` in this scope");
}

#[test]
fn compound_assignment() {
    let mut env = RuntimeEnv::new();
    run_in(&mut env, "x = 1 m\n").unwrap();
    for (line, want) in [
        ("x += 2 m\n", "3 m"),
        ("x -= 1 m\n", "2 m"),
        ("x *= 3\n", "6 m"),
        ("x /= 2\n", "3 m"),
        ("x ^= 2\n", "9 m^2"),
    ] {
        run_in(&mut env, line).unwrap();
        assert_eq!(env.get("x").unwrap().to_string(), want, "{line}");
    }
    assert_eq!(
        run_in(&mut env, "q += 1\n").unwrap_err().message,
        "cannot find `q` in this scope"
    );
}

#[test]
fn annotated_binding_binds_its_value() {
    let mut env = RuntimeEnv::new();
    run_in(&mut env, "a: Vector[3] = 1\n").unwrap();
    assert_eq!(env.get("a"), Some(&Value::scalar(1.0)));
    // A declaration without a value would be an array.
    let e = run_in(&mut env, "A: Matrix[3, 3]\n").unwrap_err();
    assert_eq!(e.message, "arrays are not supported by the interpreter yet");
}

#[test]
fn arrays_are_a_runtime_error_not_a_panic() {
    let want = "arrays are not supported by the interpreter yet";
    assert_eq!(err("x = 1\nx[0]").message, want);
    let mut env = RuntimeEnv::new();
    run_in(&mut env, "A = 2\n").unwrap();
    assert_eq!(run_in(&mut env, "A[0, :]\n").unwrap_err().message, want);
    assert_eq!(run_in(&mut env, "A @ A\n").unwrap_err().message, want);
    assert_eq!(
        run_in(&mut env, "Matrix[3, 3]\n").unwrap_err().message,
        want
    );
}

#[test]
fn unsupported_operators_are_errors() {
    assert!(err("∂1").message.contains("not supported"));
    assert!(err("∇1").message.contains("not supported"));
    let mut env = RuntimeEnv::new();
    run_in(&mut env, "x = 1\n").unwrap();
    assert!(run_in(&mut env, "x.y\n")
        .unwrap_err()
        .message
        .contains("attribute `y`"));
    // The checker rejects writes through targets; if one gets here anyway
    // it is an error, not a panic.
    assert!(run_in(&mut env, "x[0] = 1\n")
        .unwrap_err()
        .message
        .contains("cannot assign"));
    assert!(run_in(&mut env, "x.y = 1\n")
        .unwrap_err()
        .message
        .contains("cannot assign"));
    assert!(run_in(&mut env, "x[0] += 1\n")
        .unwrap_err()
        .message
        .contains("cannot assign"));
}

#[test]
fn print_builtin() {
    let mut env = RuntimeEnv::new();
    let (v, out) = run_in(
        &mut env,
        "print(10 ms, 3, true)\nprint()\nprint(\"a b\", \"c\\td\")\n",
    )
    .unwrap();
    assert_eq!(out, "0.01 s 3 true\n\na b c\td\n");
    assert_eq!(v, Some(Value::None));
    // No partial line when an argument fails.
    let src = "print(1, 1/0)\n";
    let lexed = lex(src).unwrap();
    let (module, _) = parse_module(&lexed.tokens, src);
    let mut out = Vec::new();
    let e = run_module(&module, src, &mut env, &mut out).unwrap_err();
    assert_eq!(e.message, "division by zero");
    assert!(out.is_empty());
}

#[test]
fn user_bindings_shadow_builtins() {
    let mut env = RuntimeEnv::new();
    run_in(&mut env, "print = 3\n").unwrap();
    let e = run_in(&mut env, "print(1)\n").unwrap_err();
    assert_eq!(e.message, "cannot call a number");
    run_in(&mut env, "inv = 2\n").unwrap();
    assert_eq!(
        run_in(&mut env, "inv(4)\n").unwrap_err().message,
        "cannot call a number"
    );
    assert_eq!(err("nope(1)").message, "cannot find `nope` in this scope");
    assert_eq!(err("inv(1, 2)").message, "`inv` takes 1 argument, found 2");
    assert_eq!(
        err("inv(true)").message,
        "`inv` needs a number, found a Bool"
    );
}

#[test]
fn print_write_failure_is_an_error() {
    struct Broken;
    impl std::io::Write for Broken {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "closed",
            ))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let src = "print(1)\n";
    let lexed = lex(src).unwrap();
    let (module, _) = parse_module(&lexed.tokens, src);
    let e = run_module(&module, src, &mut RuntimeEnv::new(), &mut Broken).unwrap_err();
    assert!(e.message.starts_with("cannot write output"));
}

#[test]
fn strings_decode_escapes() {
    assert_eq!(eval(r#""a\nb\"c\\""#), Value::Str("a\nb\"c\\".to_string()));
    // The echo form is valid source: quoted and escaped.
    assert_eq!(show(r#""a\nb\"c\\""#), r#""a\nb\"c\\""#);
}

#[test]
fn printing_format() {
    // Dimensionless: bare.
    assert_eq!(show("0.5"), "0.5");
    // Coherent derived units.
    assert_eq!(show("1 N"), "1 N");
    assert_eq!(show("1 J / 1 s"), "1 W");
    assert_eq!(show("1 N / 1 m^2"), "1 Pa");
    assert_eq!(show("1 / 1 s"), "1 Hz");
    // Composed base units, valid Sky source.
    assert_eq!(show("9.81 m / 1 s^2"), "9.81 m/s^2");
    assert_eq!(show("1 kg * 1 m^2"), "1 kg*m^2");
    assert_eq!(show("1 kg / 1 m^3"), "1 kg/m^3");
    assert_eq!(show("1 / 1 m"), "1 m^-1");
    assert_eq!(show("1 / (1 m * 1 s)"), "1 m^-1*s^-1");
    assert_eq!(show("1 m / 1 s / 1 K"), "1 m/s/K");
    assert_eq!(show("1 J / 1 K"), "1 kg*m^2/s^2/K");
    assert_eq!(show("(1 m)^(1/2)"), "1 m^(1/2)");
    assert_eq!(show("(1 m)^(-3/2)"), "1 m^(-3/2)");
    assert_eq!(show("(1 kg)^(1/2) / (1 s)^(1/2)"), "1 kg^(1/2)/s^(1/2)");
    // Negative numbers and units.
    assert_eq!(show("-3 m"), "-3 m");
}

#[test]
fn printed_values_read_back_as_the_same_value() {
    // The REPL echo is pasteable: printing then re-evaluating is identity.
    for src in [
        "10 ms",
        "9.81 m / 1 s^2",
        "2 kg * 3 m^2 / (1 s)^2",
        "(2 m)^(1/2)",
        "1 / (3 m * 7 s)",
        "6.674e-11 m^3 / (1 kg * 1 s^2)",
        "0.1 + 0.2",
        "123456789.123456789 km",
        "1e300 * 1e10",
    ] {
        let Ok((Some(v), _)) = run_in(&mut RuntimeEnv::new(), &format!("{src}\n")) else {
            continue; // `1e300 * 1e10` overflows: not a value to read back
        };
        let text = v.to_string();
        assert_eq!(eval(&text), v, "{src} printed as {text}");
    }
}

#[test]
fn number_formatting() {
    assert_eq!(format_number(0.0), "0");
    assert_eq!(format_number(10000.0), "10000");
    assert_eq!(format_number(0.01), "0.01");
    assert_eq!(format_number(0.0001), "0.0001");
    assert_eq!(format_number(0.00001), "1e-5");
    assert_eq!(format_number(9.109e-31), "9.109e-31");
    assert_eq!(format_number(1e15), "1000000000000000");
    assert_eq!(format_number(1e16), "1e16");
    assert_eq!(format_number(-1.5e20), "-1.5e20");
    assert_eq!(format_number(0.1 + 0.2), "0.30000000000000004");
    for x in [0.1, 1e-7, 123456.789, 5e-324, f64::MAX, 1e22, 0.3] {
        assert_eq!(format_number(x).parse::<f64>().unwrap(), x);
    }
}

#[test]
fn deep_expressions_do_not_overflow_the_stack() {
    // Nesting deeper than the parser's limit is a parse error, so the
    // evaluator's recursion is bounded by MAX_DEPTH; run at that depth on
    // the same known-size stack the CLI uses.
    let depth = 200;
    let src = format!("{}1{}\n", "(".repeat(depth), ")".repeat(depth));
    let v = sky_ast::with_deep_stack(|| eval(&src));
    assert_eq!(v, Value::scalar(1.0));
    let chain = format!("{}\n", vec!["1"; 250].join(" + "));
    let v = sky_ast::with_deep_stack(|| eval(&chain));
    assert_eq!(v, Value::scalar(250.0));
}

#[test]
fn decimal_exponents_match_the_checker() {
    // The checker and the interpreter share `sky_types::const_exponent`, so
    // a decimal exponent the checker accepts is also one the interpreter
    // can evaluate, with the same dimension.
    assert_eq!(show("(4 m)^0.5"), "2 m^(1/2)");
    assert_eq!(show("(8 s)^(1/3)"), "2 s^(1/3)");
    assert_eq!(show("(9 m^2)^0.5"), "3 m");
}

// Math builtins (`crate::math`). These run without the checker, so the
// dynamic dimension re-checks are covered too.

#[test]
fn math_dimension_rules() {
    assert_eq!(show("sqrt(9 m^2)"), "3 m");
    assert_eq!(show("sqrt(16)"), "4");
    assert_eq!(show("cbrt(-8 m^3)"), "-2 m");
    assert_eq!(show("abs(-3 m/s)"), "3 m/s");
    assert_eq!(show("hypot(3 m, 4 m)"), "5 m");
    assert_eq!(show("min(2 s, 1 s)"), "1 s");
    assert_eq!(show("max(1 kg, 2 kg)"), "2 kg");
    assert_eq!(
        show("atan2(1 m, 1 m)"),
        format_number(std::f64::consts::FRAC_PI_4)
    );
    // Hz times s is dimensionless, so it may feed a trig function.
    assert_eq!(show("cos(0 Hz * 2 s)"), "1");
    assert_eq!(show("exp(0)"), "1");
    assert_eq!(show("log2(8) + log10(1000)"), "6");
    assert_eq!(show("sin(0 rad)"), "0");
}

#[test]
fn math_rounding_is_half_away_from_zero_and_never_negative_zero() {
    assert_eq!(show("round(2.5)"), "3");
    assert_eq!(show("round(-2.5)"), "-3");
    assert_eq!(show("round(-0.4)"), "0");
    assert_eq!(show("floor(-0.5)"), "-1");
    assert_eq!(show("ceil(-0.5)"), "0");
}

#[test]
fn math_domain_errors_never_produce_nan_or_inf() {
    assert_eq!(err("log(0)").message, "`log` of zero has no real value");
    assert_eq!(
        err("log10(-1)").message,
        "`log10` of a negative number has no real value"
    );
    assert_eq!(
        err("sqrt(-1 m^2)").message,
        "`sqrt` of a negative number has no real value"
    );
    assert_eq!(
        err("asin(2)").message,
        "`asin` of a number outside [-1, 1] has no real value"
    );
    assert_eq!(
        err("atan2(0 m, 0 m)").message,
        "`atan2` of the zero vector has no real value"
    );
    assert_eq!(err("exp(1000)").message, "arithmetic overflow");
    assert_eq!(
        err("hypot(1.5e308 m, 1.5e308 m)").message,
        "arithmetic overflow"
    );
    // Underflow is a correct zero, not an error.
    assert_eq!(show("exp(-1000)"), "0");
}

#[test]
fn math_dimension_and_type_errors_at_run_time() {
    assert_eq!(
        err("exp(3 m)").message,
        "the argument of `exp` must be dimensionless, found length"
    );
    assert_eq!(
        err("max(1 m, 1 s)").message,
        "the arguments of `max` must have the same dimension, found length and time"
    );
    assert_eq!(
        err("round(1.5 km)").message,
        "the argument of `round` must be dimensionless, found length"
    );
    assert!(err("sin(true)").message.starts_with("`sin` needs a number"));
    assert_eq!(
        err("sqrt(1, 2)").message,
        "`sqrt` takes 1 argument, found 2"
    );
    assert_eq!(
        err("hypot(1)").message,
        "`hypot` takes 2 arguments, found 1"
    );
}

#[test]
fn a_user_binding_or_def_shadows_a_math_builtin() {
    let (_, out) = run_in(
        &mut RuntimeEnv::new(),
        "def sqrt(x):\n    return x + 1\nprint(sqrt(9))\nabs = 5\nprint(abs)\n",
    )
    .expect("runs");
    assert_eq!(out, "10\n5\n");
}

#[test]
fn pi_is_a_dimensionless_prelude_constant() {
    let mut env = RuntimeEnv::with_prelude();
    let (v, _) = run_in(&mut env, "2 * pi == 2 * π\n").expect("runs");
    assert_eq!(v.map(|v| v.to_string()), Some("true".to_string()));
    let (v, _) = run_in(&mut env, "pi\n").expect("runs");
    assert_eq!(
        v.map(|v| v.to_string()),
        Some(format_number(std::f64::consts::PI))
    );
}
