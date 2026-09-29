//! `and`, `or`, `not`: precedence, the flat `BoolOp` node, spans, errors,
//! and depth.

mod common;
use common::{at, expr_errors, lex, sexpr};
use sky_ast::{parse_expr, BoolOp, ExprKind, Literal, UnaryOp, MAX_DEPTH};

fn check(cases: &[(&str, &str)]) {
    for (src, want) in cases {
        assert_eq!(sexpr(src), *want, "source: {src:?}");
    }
}

#[test]
fn precedence_python_style() {
    check(&[
        // `not` takes a whole comparison.
        ("not a < b", "(not (< a b))"),
        ("not a + b == c", "(not (== (+ a b) c))"),
        // Comparisons bind tighter than `and`/`or`.
        ("a < b and b < c", "(and (< a b) (< b c))"),
        ("a == 1 or b != 2", "(or (== a 1) (!= b 2))"),
        // `and` binds tighter than `or`.
        ("a and b or c", "(or (and a b) c)"),
        ("a or b and c", "(or a (and b c))"),
        ("a or b and c or d", "(or a (and b c) d)"),
        // `not` binds tighter than `and`.
        ("not a and b", "(and (not a) b)"),
        ("a and not b", "(and a (not b))"),
        ("not a or not b", "(or (not a) (not b))"),
        ("not not x", "(not (not x))"),
        // Parentheses and call arguments reset the level.
        ("not (a and b)", "(not (paren (and a b)))"),
        ("(a or b) and c", "(and (paren (or a b)) c)"),
        ("f(not a, b or c)", "(call f (not a) (or b c))"),
        ("x[a and b]", "(index x (and a b))"),
        ("-(not x)", "(neg (paren (not x)))"),
        // Tighter operators inside operands.
        ("not -x^2", "(not (neg (^ x 2)))"),
        ("not A⁻¹ @ b == c", "(not (== (@ (inverse A) b) c))"),
        ("10 km < d and ok", "(and (< (unit 10 km) d) ok)"),
    ]);
}

#[test]
fn runs_are_flat_and_left_to_right() {
    check(&[
        ("a and b", "(and a b)"),
        ("a and b and c", "(and a b c)"),
        ("a or b or c or d", "(or a b c d)"),
        // A parenthesized run stays its own node.
        ("(a and b) and c", "(and (paren (and a b)) c)"),
        ("a and (b and c)", "(and a (paren (and b c)))"),
        // Mixed runs nest by precedence.
        ("a and b or c and d", "(or (and a b) (and c d))"),
    ]);
}

#[test]
fn keywords_are_attribute_names() {
    check(&[
        ("x.and", "(attr x and)"),
        ("x.or.not", "(attr (attr x or) not)"),
        (
            "np.logical.not(a) and b",
            "(and (call (attr (attr np logical) not) a) b)",
        ),
    ]);
}

#[test]
fn bool_op_node_and_spans() {
    let src = "a < b and c or not d";
    let (expr, errors) = parse_expr(&lex(src), src);
    assert!(errors.is_empty(), "{errors:#?}");
    let expr = expr.expect("expression");
    assert_eq!(at(src, expr.span), src);
    let ExprKind::BoolOp {
        op: BoolOp::Or,
        operands,
        op_spans,
    } = &expr.kind
    else {
        panic!("not an `or`: {expr:#?}");
    };
    assert_eq!(operands.len(), 2);
    assert_eq!(op_spans.len(), 1);
    assert_eq!(at(src, op_spans[0]), "or");
    assert_eq!(at(src, operands[0].span), "a < b and c");
    assert_eq!(at(src, operands[1].span), "not d");
    let ExprKind::BoolOp {
        op: BoolOp::And,
        operands: inner,
        op_spans: inner_ops,
    } = &operands[0].kind
    else {
        panic!("not an `and`: {:#?}", operands[0]);
    };
    assert_eq!(at(src, inner[0].span), "a < b");
    assert_eq!(at(src, inner[1].span), "c");
    assert_eq!(at(src, inner_ops[0]), "and");
    let ExprKind::Unary {
        op: UnaryOp::Not,
        op_span,
        operand,
    } = &operands[1].kind
    else {
        panic!("not a `not`: {:#?}", operands[1]);
    };
    assert_eq!(at(src, *op_span), "not");
    assert_eq!(at(src, operand.span), "d");

    // Every keyword of a long run is kept, in order.
    let src = "a and  b and\tc and d";
    let (expr, _) = parse_expr(&lex(src), src);
    let expr = expr.expect("expression");
    let ExprKind::BoolOp {
        operands, op_spans, ..
    } = &expr.kind
    else {
        panic!("{expr:#?}");
    };
    assert_eq!(operands.len(), 4);
    let ops: Vec<_> = op_spans.iter().map(|s| s.start).collect();
    assert_eq!(ops, [2, 9, 15]);
}

#[test]
fn not_under_a_tighter_operator_is_an_error() {
    for (src, before, operand, help) in [
        ("-not x", "-", "not", "write `(not x)`"),
        ("+not x", "+", "not", "write `(not x)`"),
        ("a < not b", "<", "not", "write `(not b)`"),
        ("a + not b", "+", "not", "write `(not b)`"),
        ("2^not x", "^", "not", "write `(not x)`"),
        ("∂not x", "∂", "not", "write `(not x)`"),
        ("a == -not b", "-", "not", "write `(not b)`"),
    ] {
        let errors = expr_errors(src);
        assert_eq!(errors.len(), 1, "{src:?}: {errors:#?}");
        let err = &errors[0];
        assert_eq!(
            err.message,
            format!("`not` binds more loosely than `{before}`, so it needs parentheses here"),
            "{src:?}"
        );
        assert_eq!(at(src, err.span), operand, "{src:?}");
        assert_eq!(at(src, err.secondary[0].span), before, "{src:?}");
        assert_eq!(err.help.as_deref(), Some(help), "{src:?}");
    }
}

#[test]
fn misplaced_bool_keywords_are_errors() {
    for (src, message) in [
        ("and", "expected an expression, found keyword `and`"),
        ("a or or b", "expected an expression, found keyword `or`"),
        ("a and", "expected an expression, found end of line"),
        ("not", "expected an expression, found end of line"),
        ("a not b", "expected end of input, found keyword `not`"),
    ] {
        let errors = expr_errors(src);
        assert_eq!(errors.len(), 1, "{src:?}: {errors:#?}");
        assert_eq!(errors[0].message, message, "{src:?}");
    }
}

#[test]
fn comparisons_inside_runs_stay_non_associative() {
    let errors = expr_errors("a < b < c and d");
    assert_eq!(errors.len(), 1, "{errors:#?}");
    assert_eq!(
        errors[0].message,
        "comparison operators cannot be chained in Sky v0.1"
    );
    // One comparison per operand is fine, however many operands.
    assert!(expr_errors("a < b and b < c and c < d").is_empty());
    assert!(expr_errors("not a < b or c == d").is_empty());
}

fn nots(n: usize) -> String {
    format!("{}x", "not ".repeat(n))
}

#[test]
fn not_chain_depth() {
    let src = nots(MAX_DEPTH as usize);
    let (expr, errors) = parse_expr(&lex(&src), &src);
    assert!(errors.is_empty(), "{errors:#?}");
    let mut e = &expr.expect("expression");
    let mut count = 0;
    while let ExprKind::Unary {
        op: UnaryOp::Not,
        operand,
        ..
    } = &e.kind
    {
        count += 1;
        e = operand;
    }
    assert_eq!(count, MAX_DEPTH);

    let src = nots(MAX_DEPTH as usize + 1);
    let errors = expr_errors(&src);
    assert_eq!(errors.len(), 1, "{errors:#?}");
    assert_eq!(
        errors[0].message,
        "expression is nested too deeply (the limit is 256 levels)"
    );
}

/// Parses, prints, and drops `src` on a 256 KiB thread: far too small for
/// anything that recurses per operand.
fn parse_on_small_stack(src: String) -> (usize, usize) {
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(move || {
            let tokens = lex(&src);
            let (expr, errors) = parse_expr(&tokens, &src);
            let expr = expr.expect("expression");
            let printed = expr.to_sexpr(&src);
            let operands = match &expr.kind {
                ExprKind::BoolOp { operands, .. } => operands.len(),
                _ => 1,
            };
            assert!(printed.len() > src.len() / 4);
            (operands, errors.len())
        })
        .expect("spawn")
        .join()
        .expect("no stack overflow")
}

#[test]
fn long_runs_do_not_recurse_per_operand() {
    let n = 10_000;
    let and_run = format!("a{}", " and a".repeat(n));
    assert_eq!(parse_on_small_stack(and_run), (n + 1, 0));

    let or_run = format!("a{}", " or a".repeat(n));
    assert_eq!(parse_on_small_stack(or_run), (n + 1, 0));

    // Alternating: an `or` run of 5001 `and` runs; still height 2.
    let mixed = format!("a{}", " and a or a".repeat(n / 2));
    assert_eq!(parse_on_small_stack(mixed), (n / 2 + 1, 0));

    // Operands with their own comparisons and `not`s.
    let rich = format!("not x < 1{}", " and not x < 1".repeat(n));
    assert_eq!(parse_on_small_stack(rich), (n + 1, 0));
}

#[test]
fn bool_literals() {
    check(&[
        ("true", "true"),
        ("false", "false"),
        ("not false", "(not false)"),
        (
            "true or (3 m + 2 s)",
            "(or true (paren (+ (unit 3 m) (unit 2 s))))",
        ),
        ("a == true and false", "(and (== a true) false)"),
        ("f(true, false)", "(call f true false)"),
        // Keywords stay attribute names.
        ("x.true or y.false", "(or (attr x true) (attr y false))"),
        // Python spellings are plain names (the checker explains them).
        ("True and False", "(and True False)"),
    ]);
    for (src, value) in [("true", true), ("false", false)] {
        let (expr, errors) = parse_expr(&lex(src), src);
        assert!(errors.is_empty(), "{errors:#?}");
        let expr = expr.expect("expression");
        let ExprKind::Literal(Literal::Bool { value: v, span }) = expr.kind else {
            panic!("not a Bool literal: {expr:#?}");
        };
        assert_eq!(v, value);
        assert_eq!(at(src, span), src);
        assert_eq!(expr.span, span);
    }
}

#[test]
fn a_unit_cannot_follow_a_bool() {
    for (src, name) in [("true m", "m"), ("false kg", "kg"), ("x = true km", "km")] {
        let errors = common::module_errors(src);
        assert_eq!(errors.len(), 1, "{src:?}: {errors:#?}");
        assert_eq!(
            errors[0].message,
            format!(
                "unexpected name `{name}` after an expression; a unit may only follow a numeric literal"
            ),
            "{src:?}"
        );
        assert_eq!(at(src, errors[0].span), name);
    }
    // Nor can a bool be a unit, or be assigned to.
    assert!(!common::module_errors("x = 10 true").is_empty());
    assert!(!common::module_errors("true = 1").is_empty());
}

#[test]
fn bool_literal_spans_inside_expressions() {
    let src = "a or false";
    let (expr, _) = parse_expr(&lex(src), src);
    let expr = expr.expect("expression");
    let ExprKind::BoolOp { operands, .. } = &expr.kind else {
        panic!("{expr:#?}");
    };
    let ExprKind::Literal(Literal::Bool { value, span }) = operands[1].kind else {
        panic!("{:#?}", operands[1]);
    };
    assert!(!value);
    assert_eq!((span.start, span.end), (5, 10));
}
