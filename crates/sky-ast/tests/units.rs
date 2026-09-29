//! Unit literals: grammar in spec/v0.1.md "Unit literals" (signed off by Eng Lead).

mod common;
use common::{expr_errors, module_sexpr, sexpr};

fn check(cases: &[(&str, &str)]) {
    for (src, want) in cases {
        assert_eq!(sexpr(src), *want, "source: {src:?}");
    }
}

#[test]
fn simple_unit_literals() {
    check(&[
        ("10 km", "(unit 10 km)"),
        ("9.8 m", "(unit 9.8 m)"),
        ("1e3 m", "(unit 1e3 m)"),
        ("3 d", "(unit 3 d)"),
        ("1 eV", "(unit 1 eV)"),
    ]);
}

#[test]
fn number_glued_to_a_name_is_still_a_unit_for_the_parser() {
    // The lexer rejects `10km` ("put a space before the unit"), but the
    // parser only sees tokens: Int directly followed by Ident is a unit.
    assert!(sky_syntax::lex("10km").is_err());
    let src = "10km";
    let tokens = [
        sky_ast::Token {
            kind: sky_ast::TokenKind::Int,
            span: sky_ast::Span::new(0, 2),
        },
        sky_ast::Token {
            kind: sky_ast::TokenKind::Ident,
            span: sky_ast::Span::new(2, 4),
        },
    ];
    let (expr, errors) = sky_ast::parse_expr(&tokens, src);
    assert!(errors.is_empty());
    assert_eq!(expr.unwrap().to_sexpr(src), "(unit 10 km)");
}

#[test]
fn unclosed_rational_exponent() {
    // The lexer rejects `1 m^(1/2` itself; drop the `)` to reach the parser.
    let src = "1 m^(1/2)";
    let errors = common::tokens_errors(&common::lex_prefix(src, 7), src);
    assert_eq!(errors.len(), 1);
    assert!(
        errors[0].message.contains("malformed rational"),
        "{}",
        errors[0].message
    );
}

#[test]
fn compound_units_need_adjacent_operators() {
    check(&[
        ("9.8 m/s^2", "(unit 9.8 (/ m (^ s 2)))"),
        ("1 kg*m/s^2", "(unit 1 (/ (* kg m) (^ s 2)))"),
        ("2 kg*m^2/s^2", "(unit 2 (/ (* kg (^ m 2)) (^ s 2)))"),
        ("3 s^-2", "(unit 3 (^ s -2))"),
        ("9.8 m/s^-2", "(unit 9.8 (/ m (^ s -2)))"),
        ("1 m/s/s", "(unit 1 (/ (/ m s) s))"),
        ("5 J*s", "(unit 5 (* J s))"),
        ("5 kg*m", "(unit 5 (* kg m))"),
    ]);
}

#[test]
fn rational_exponents() {
    check(&[
        ("1 m^(1/2)", "(unit 1 (^ m 1/2))"),
        ("1 m^(-3/2)", "(unit 1 (^ m -3/2))"),
        ("1 Hz^(1/2)*m", "(unit 1 (* (^ Hz 1/2) m))"),
        // Stored as written, not reduced.
        ("1 m^(2/4)", "(unit 1 (^ m 2/4))"),
        ("1 m^(4/2)", "(unit 1 (^ m 4/2))"),
    ]);
}

#[test]
fn spaced_operators_are_arithmetic_on_the_quantity() {
    check(&[
        ("10 km / 30 min", "(/ (unit 10 km) (unit 30 min))"),
        ("10 m / t", "(/ (unit 10 m) t)"),
        ("10 m/ s", "(/ (unit 10 m) s)"),
        ("10 m /s", "(/ (unit 10 m) s)"),
        ("10 m * 2", "(* (unit 10 m) 2)"),
        ("10 m ^ 2", "(^ (unit 10 m) 2)"),
        ("10 m^ 2", "(^ (unit 10 m) 2)"),
        ("10 m ^2", "(^ (unit 10 m) 2)"),
        ("-10 km", "(neg (unit 10 km))"),
        ("(10 km)⁻¹", "(inverse (paren (unit 10 km)))"),
        ("f(10 km, 2 s)", "(call f (unit 10 km) (unit 2 s))"),
    ]);
}

#[test]
fn a_name_is_a_unit_only_directly_after_a_number() {
    check(&[("2 * km", "(* 2 km)"), ("x*km", "(* x km)"), ("km", "km")]);
}

#[test]
fn mismatched_units_parse_fine() {
    // A type error later, not a parse error.
    assert_eq!(
        module_sexpr("x = 5 meters + 20 kelvin\n"),
        "(= x (+ (unit 5 meters) (unit 20 kelvin)))"
    );
}

#[test]
fn unit_values_carry_exponents() {
    let src = "1 m^(-3/2)";
    let tokens = common::lex(src);
    let (expr, errors) = sky_ast::parse_expr(&tokens, src);
    assert!(errors.is_empty());
    let Some(sky_ast::Expr {
        kind: sky_ast::ExprKind::UnitLiteral { unit, .. },
        ..
    }) = expr
    else {
        panic!("not a unit literal");
    };
    let sky_ast::UnitExpr::Pow {
        num,
        den,
        exponent_span,
        ..
    } = unit
    else {
        panic!("not a power");
    };
    assert_eq!((num, den), (-3, 2));
    assert_eq!(common::at(src, exponent_span), "(-3/2)");
}

fn one_error(src: &str) -> sky_ast::ParseError {
    let errors = expr_errors(src);
    assert_eq!(errors.len(), 1, "{src:?}: {errors:#?}");
    errors.into_iter().next().unwrap()
}

#[test]
fn juxtaposed_units_are_an_error_suggesting_star() {
    let err = one_error("5 kg m");
    assert!(
        err.message.contains("after the unit `kg`"),
        "{}",
        err.message
    );
    assert_eq!(
        err.help.as_deref(),
        Some("compound units are written with `*` and no spaces: `kg*m`")
    );
    assert_eq!(common::at("5 kg m", err.span), "m");

    let err = one_error("9.8 m/s s");
    assert!(err.help.unwrap().contains("`m/s*s`"));
}

#[test]
fn malformed_unit_parts_are_errors() {
    for (src, needle) in [
        ("10 m/2", "expected a unit name after `/`"),
        ("10 m*(s)", "expected a unit name after `*`"),
        ("3 s^x", "must be an integer or a parenthesized rational"),
        ("3 s^2.5", "must be an integer or a parenthesized rational"),
        ("3 s^-x", "must be an integer or a parenthesized rational"),
        ("3 s^2^3", "cannot be chained"),
        ("1 m^(1)", "malformed rational"),
        ("1 m^(1/)", "malformed rational"),
        ("1 m^( 1/2)", "malformed rational"),
        ("1 m^(1 /2)", "malformed rational"),
        ("1 m^(1/x)", "malformed rational"),
        ("1 m^(1/-2)", "malformed rational"),
    ] {
        let err = one_error(src);
        assert!(err.message.contains(needle), "{src:?}: {}", err.message);
    }
}

#[test]
fn zero_denominator_is_an_error() {
    let err = one_error("1 m^(1/0)");
    assert!(err.message.contains("denominator"), "{}", err.message);
    assert_eq!(common::at("1 m^(1/0)", err.span), "0");
}

#[test]
fn huge_exponents_are_errors_not_panics() {
    for src in [
        "1 m^(99999999999999999999/2)",
        "1 m^(1/99999999999999999999)",
        "1 m^99999999999999999999",
        "1 m^-99999999999999999999",
        // Fits in u64 but not i64: fine as a denominator, not as a numerator.
        "1 m^18446744073709551615",
        "1 m^(9223372036854775808/1)",
    ] {
        let err = one_error(src);
        assert!(
            err.message.contains("not a valid 64-bit integer"),
            "{src:?}: {}",
            err.message
        );
    }
    check(&[
        (
            "1 m^9223372036854775807",
            "(unit 1 (^ m 9223372036854775807))",
        ),
        (
            "1 m^-9223372036854775807",
            "(unit 1 (^ m -9223372036854775807))",
        ),
        (
            "1 m^(1/18446744073709551615)",
            "(unit 1 (^ m 1/18446744073709551615))",
        ),
    ]);
}

#[test]
fn inverse_directly_after_a_unit_is_an_error() {
    let src = "10 km⁻¹";
    let err = one_error(src);
    assert_eq!(err.message, "`⁻¹` cannot follow a unit");
    assert_eq!(common::at(src, err.span), "⁻¹");
    assert_eq!(common::at(src, err.secondary[0].span), "km");
    assert_eq!(
        err.help.as_deref(),
        Some("write `10 km^-1` for per-km, or `(10 km)⁻¹`")
    );

    for (src, help) in [
        (
            "9.8 m/s⁻¹",
            "write `9.8 m/s^-1` to invert `s`, or `(9.8 m/s)⁻¹`",
        ),
        (
            "1 kg*m⁻¹",
            "write `1 kg*m^-1` to invert `m`, or `(1 kg*m)⁻¹`",
        ),
        (
            "10 m^2⁻¹",
            "put a single exponent after `^` (e.g. `s^-2`), or write `(10 m^2)⁻¹`",
        ),
        ("10 km ⁻¹", "write `10 km^-1` for per-km, or `(10 km)⁻¹`"),
        ("f(10 km⁻¹)", "write `10 km^-1` for per-km, or `(10 km)⁻¹`"),
    ] {
        let err = one_error(src);
        assert_eq!(err.message, "`⁻¹` cannot follow a unit", "{src:?}");
        assert_eq!(common::at(src, err.span), "⁻¹", "{src:?}");
        assert_eq!(err.help.as_deref(), Some(help), "{src:?}");
    }
}

#[test]
fn inverse_of_a_quantity_or_name_stays_valid() {
    check(&[
        ("(10 km)⁻¹", "(inverse (paren (unit 10 km)))"),
        ("10 km^-1", "(unit 10 (^ km -1))"),
        ("A⁻¹", "(inverse A)"),
        ("A⁻¹ @ b", "(@ (inverse A) b)"),
        ("10 km * A⁻¹", "(* (unit 10 km) (inverse A))"),
    ]);
}

#[test]
fn superscript_exponents_never_produce_a_quantity() {
    // The lexer rejects superscript digits and a lone `⁻`, so `10 m²` and
    // `10 m⁻²` never reach the parser as a unit literal.
    for src in ["10 m²", "10 m⁻²", "x = 10 m²\n", "a = 3 s⁻²\n"] {
        let err = sky_syntax::lex(src).expect_err(src);
        assert_eq!(
            err.kind,
            sky_syntax::LexErrorKind::UnexpectedChar,
            "{src:?}"
        );
    }
}

#[test]
fn i64_min_exponent_is_accepted() {
    check(&[
        (
            "1 m^-9223372036854775808",
            "(unit 1 (^ m -9223372036854775808))",
        ),
        (
            "1 m^(-9223372036854775808/3)",
            "(unit 1 (^ m -9223372036854775808/3))",
        ),
    ]);
    for src in ["1 m^-9223372036854775809", "1 m^(-9223372036854775809/3)"] {
        let err = one_error(src);
        assert!(
            err.message.contains("not a valid 64-bit integer"),
            "{src:?}: {}",
            err.message
        );
    }
}

#[test]
fn double_star_suggests_caret() {
    for (src, span) in [
        ("x**2", "**"),
        ("x ** 2", "**"),
        ("10 m**2", "**"),
        ("2 kg*m**2", "**"),
    ] {
        let err = one_error(src);
        assert_eq!(err.message, "`**` is not an operator", "{src:?}");
        assert_eq!(
            err.help.as_deref(),
            Some("Sky writes powers as `^`"),
            "{src:?}"
        );
        assert_eq!(common::at(src, err.span), span, "{src:?}");
    }
    // Separated stars are just a missing operand.
    let err = one_error("x * *2");
    assert_eq!(err.message, "expected an expression, found `*`");
}
