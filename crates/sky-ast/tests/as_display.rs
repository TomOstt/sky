//! `v as unit`: precedence, the unit sub-grammar, spans, and errors.

mod common;
use common::{at, expr_errors, module_errors, module_sexpr, sexpr};
use sky_ast::{parse_expr, ExprKind, MAX_DEPTH};

fn check(cases: &[(&str, &str)]) {
    for (src, want) in cases {
        assert_eq!(sexpr(src), *want, "source: {src:?}");
    }
}

#[test]
fn as_is_the_loosest_construct() {
    check(&[
        ("v as km", "(as v km)"),
        // Looser than arithmetic, comparison, and the boolean operators.
        ("a + b as km", "(as (+ a b) km)"),
        ("a * b - c as J", "(as (- (* a b) c) J)"),
        ("-a^2 as m", "(as (neg (^ a 2)) m)"),
        ("A @ b as N", "(as (@ A b) N)"),
        ("a < b as km", "(as (< a b) km)"),
        // Looser than the conditional expression too.
        ("x if c else y as km", "(as (ifelse c x y) km)"),
        // Postfix forms bind tighter.
        ("f(x) as km", "(as (call f x) km)"),
        ("A[0] as km", "(as (index A 0) km)"),
        // A unit literal keeps its own unit.
        ("2 km as m", "(as (unit 2 km) m)"),
        // Parentheses and call arguments start a fresh expression.
        ("(a + b as km) + 1", "(+ (paren (as (+ a b) km)) 1)"),
        ("print(a + b as km, c)", "(call print (as (+ a b) km) c)"),
        ("print(u as mm, v)", "(call print (as u mm) v)"),
        ("[a as m, b]", "(array (as a m) b)"),
    ]);
}

#[test]
fn the_unit_is_the_unit_literal_grammar() {
    check(&[
        ("v as km/h", "(as v (/ km h))"),
        ("E as MeV", "(as E MeV)"),
        ("E as kg*m^2/s^2", "(as E (/ (* kg (^ m 2)) (^ s 2)))"),
        ("a as m/s^2", "(as a (/ m (^ s 2)))"),
        ("a as s^-2", "(as a (^ s -2))"),
        ("a as m^(1/2)", "(as a (^ m 1/2))"),
        ("a as ms", "(as a ms)"),
        // `Ω` is an ordinary identifier character.
        ("R as kΩ", "(as R kΩ)"),
        // The parser does not know unit names: the checker does.
        ("v as nonsense", "(as v nonsense)"),
    ]);
}

#[test]
fn statements() {
    assert_eq!(module_sexpr("v as km/h\n"), "(as v (/ km h))");
    assert_eq!(
        module_sexpr("print(v as km/h)\n"),
        "(call print (as v (/ km h)))"
    );
    assert_eq!(module_sexpr("x = v as km\n"), "(= x (as v km))");
}

#[test]
fn spans() {
    let src = "print(a + b as km/h)";
    let tokens = common::lex(src);
    let (expr, errors) = parse_expr(&tokens, src);
    assert!(errors.is_empty(), "{errors:#?}");
    let expr = expr.expect("expression");
    let ExprKind::Call { args, .. } = &expr.kind else {
        panic!("not a call: {expr:?}");
    };
    let arg = &args[0];
    assert_eq!(at(src, arg.span), "a + b as km/h");
    let ExprKind::As {
        value,
        unit,
        as_span,
    } = &arg.kind
    else {
        panic!("not `as`: {arg:?}");
    };
    assert_eq!(at(src, value.span), "a + b");
    assert_eq!(at(src, unit.span()), "km/h");
    assert_eq!(at(src, *as_span), "as");
}

#[test]
fn as_is_a_keyword_but_still_an_attribute_name() {
    check(&[("x.as", "(attr x as)")]);
    let errors = module_errors("as = 3\n");
    assert_eq!(errors.len(), 1, "{errors:#?}");
    assert_eq!(
        errors[0].message,
        "expected an expression, found keyword `as`"
    );
    // `as` is not read as a unit after a number.
    check(&[("10 as m", "(as 10 m)")]);
}

#[test]
fn errors() {
    for (src, message, help) in [
        (
            "v as",
            "expected a unit after `as`, found end of line",
            "write the unit to show the value in, e.g. `v as km/h`",
        ),
        (
            "v as 5",
            "expected a unit after `as`, found integer literal `5`",
            "write the unit to show the value in, e.g. `v as km/h`",
        ),
        (
            "v as km as m",
            "a value can be shown in only one unit",
            "remove the second `as`",
        ),
        (
            "v as m / s",
            "a unit's `/` cannot have spaces around it",
            "write the unit without spaces: `m/s`",
        ),
    ] {
        let errors = expr_errors(src);
        assert_eq!(errors.len(), 1, "{src:?}: {errors:#?}");
        assert_eq!(errors[0].message, message, "{src:?}");
        assert_eq!(errors[0].help.as_deref(), Some(help), "{src:?}");
    }
    // Something after the unit that would use the shown value.
    for src in [
        "v as km + 1",
        "v as km and b",
        "v as km * 2",
        "v as km == w",
    ] {
        let errors = expr_errors(src);
        assert_eq!(errors.len(), 1, "{src:?}: {errors:#?}");
        assert!(
            errors[0].message.ends_with("cannot follow `as`"),
            "{src:?}: {}",
            errors[0].message
        );
        assert!(errors[0]
            .help
            .as_deref()
            .is_some_and(|h| h.contains("arithmetic before")));
    }
    let errors = expr_errors("v as km if c else w");
    assert_eq!(errors[0].message, "`as` must come last in an expression");
}

#[test]
fn nesting_counts_toward_the_depth_limit() {
    let n = MAX_DEPTH as usize;
    let src = format!("{}v as m{}", "(".repeat(n - 2), ")".repeat(n - 2));
    assert!(expr_errors(&src).is_empty());
    let src = format!("{}v as m{}", "(".repeat(n), ")".repeat(n));
    assert!(!expr_errors(&src).is_empty());
}

#[test]
fn unit_text_is_exactly_one_unit_literal_unit() {
    use sky_ast::parse_unit_text;
    for text in ["m", "km/h", "kg*m^2/s^2", "MeV", "m^(1/2)", "s^-2", "μm"] {
        let u = parse_unit_text(text).unwrap_or_else(|| panic!("{text:?} is a unit"));
        assert_eq!(at(&u.src, u.unit.span()), text);
    }
    for text in [
        "", " km", "km ", "km h", "km / h", "km/h + 1", "5", "km/", "km^", "(km)", "km as m", "\n",
        "km\nm",
    ] {
        assert!(parse_unit_text(text).is_none(), "{text:?}");
    }
}
