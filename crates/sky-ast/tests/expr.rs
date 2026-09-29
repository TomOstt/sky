//! Expression parsing: precedence, associativity, atoms, and postfix forms.

mod common;
use common::sexpr;

fn check(cases: &[(&str, &str)]) {
    for (src, want) in cases {
        assert_eq!(sexpr(src), *want, "source: {src:?}");
    }
}

#[test]
fn atoms() {
    check(&[
        ("42", "42"),
        ("3.14", "3.14"),
        ("1e-3", "1e-3"),
        ("\"hello\"", "\"hello\""),
        ("x", "x"),
        ("ħ", "ħ"),
        ("(x)", "(paren x)"),
        ("((x))", "(paren (paren x))"),
    ]);
}

#[test]
fn every_precedence_level_pair() {
    check(&[
        // comparison < additive
        ("a + b < c - d", "(< (+ a b) (- c d))"),
        ("a == b - c", "(== a (- b c))"),
        // comparison < multiplicative
        ("a * b >= c @ d", "(>= (* a b) (@ c d))"),
        // comparison < prefix
        ("-a != -b", "(!= (neg a) (neg b))"),
        // comparison < power
        ("a^2 <= b^2", "(<= (^ a 2) (^ b 2))"),
        // comparison < postfix
        ("A⁻¹ > f(x)", "(> (inverse A) (call f x))"),
        // additive < multiplicative
        ("a + b * c", "(+ a (* b c))"),
        ("a * b + c", "(+ (* a b) c)"),
        ("a - b @ c", "(- a (@ b c))"),
        // additive < prefix
        ("-a + b", "(+ (neg a) b)"),
        ("a - -b", "(- a (neg b))"),
        // additive < power
        ("a + b^c", "(+ a (^ b c))"),
        // additive < postfix
        ("a + b.c", "(+ a (attr b c))"),
        // multiplicative < prefix
        ("-a * b", "(* (neg a) b)"),
        ("a / -b", "(/ a (neg b))"),
        ("∂f / ∂x", "(/ (partial f) (partial x))"),
        // multiplicative < power
        ("a * b^c", "(* a (^ b c))"),
        ("a^b @ c", "(@ (^ a b) c)"),
        // multiplicative < postfix
        ("A⁻¹ @ b", "(@ (inverse A) b)"),
        ("inv(A) @ b", "(@ (call inv A) b)"),
        ("a * M[0]", "(* a (index M 0))"),
        // prefix < power
        ("-x^2", "(neg (^ x 2))"),
        ("∇f^2", "(nabla (^ f 2))"),
        // prefix < postfix
        ("-A⁻¹", "(neg (inverse A))"),
        ("-f(x)", "(neg (call f x))"),
        ("+x.y", "(pos (attr x y))"),
        // power < postfix
        ("A⁻¹^2", "(^ (inverse A) 2)"),
        ("x^f(y)", "(^ x (call f y))"),
    ]);
}

#[test]
fn associativity() {
    check(&[
        ("a + b - c", "(- (+ a b) c)"),
        ("a - b - c", "(- (- a b) c)"),
        ("a / b / c", "(/ (/ a b) c)"),
        ("a * b / c", "(/ (* a b) c)"),
        ("a @ b @ c", "(@ (@ a b) c)"),
        ("a * b @ c", "(@ (* a b) c)"),
        // `^` is right-associative.
        ("a^b^c", "(^ a (^ b c))"),
        ("a ^ b ^ c ^ d", "(^ a (^ b (^ c d)))"),
    ]);
}

#[test]
fn power_right_operand_may_be_prefix() {
    check(&[
        ("2^-1", "(^ 2 (neg 1))"),
        ("2^-1 * 3", "(* (^ 2 (neg 1)) 3)"),
        ("x^-y^2", "(^ x (neg (^ y 2)))"),
        ("x^+2", "(^ x (pos 2))"),
    ]);
}

#[test]
fn prefix_operators() {
    check(&[
        ("-x", "(neg x)"),
        ("+x", "(pos x)"),
        ("--x", "(neg (neg x))"),
        ("∂f", "(partial f)"),
        ("∇f", "(nabla f)"),
        ("∂f/∂x", "(/ (partial f) (partial x))"),
    ]);
}

#[test]
fn all_comparisons() {
    check(&[
        ("a == b", "(== a b)"),
        ("a != b", "(!= a b)"),
        ("a < b", "(< a b)"),
        ("a <= b", "(<= a b)"),
        ("a > b", "(> a b)"),
        ("a >= b", "(>= a b)"),
        // Parenthesized comparisons are not chains.
        ("(a < b) == c", "(== (paren (< a b)) c)"),
        ("f(a < b, c < d)", "(call f (< a b) (< c d))"),
    ]);
}

#[test]
fn inverse_is_its_own_node() {
    check(&[
        ("A⁻¹", "(inverse A)"),
        ("A⁻¹ @ b", "(@ (inverse A) b)"),
        ("A⁻¹⁻¹", "(inverse (inverse A))"),
        ("(A @ B)⁻¹", "(inverse (paren (@ A B)))"),
        ("M[0]⁻¹", "(inverse (index M 0))"),
    ]);
}

#[test]
fn calls_indexing_and_attributes() {
    check(&[
        ("f()", "(call f)"),
        ("f(a)", "(call f a)"),
        ("f(a, b)", "(call f a b)"),
        ("f(a, b,)", "(call f a b)"),
        ("f(a)(b)", "(call (call f a) b)"),
        ("inv(A)", "(call inv A)"),
        ("diff(f, x)", "(call diff f x)"),
        ("grad(f)", "(call grad f)"),
        ("d(x)", "(call d x)"),
        ("Matrix[3, 4]", "(index Matrix 3 4)"),
        ("Matrix[M, N]", "(index Matrix M N)"),
        ("Tensor[2, 3, 4,]", "(index Tensor 2 3 4)"),
        (
            "Matrix[3, 4] @ Matrix[4, 8]",
            "(@ (index Matrix 3 4) (index Matrix 4 8))",
        ),
        ("a.b", "(attr a b)"),
        ("a.b.c", "(attr (attr a b) c)"),
        ("a.b(c)[d]⁻¹", "(inverse (index (call (attr a b) c) d))"),
        (
            "np.linalg.lstsq(A, b)",
            "(call (attr (attr np linalg) lstsq) A b)",
        ),
        ("f(g(x), -y^2)", "(call f (call g x) (neg (^ y 2)))"),
        ("(a + b) * c", "(* (paren (+ a b)) c)"),
    ]);
}

#[test]
fn plus_minus_ascii_is_plus_then_negation() {
    // PR #4 has no `±` token and lexes `+-` as `Plus`, `Minus`.
    check(&[
        ("x +- dx", "(+ x (neg dx))"),
        ("10 m +- 2 cm", "(+ (unit 10 m) (neg (unit 2 cm)))"),
    ]);
}

#[test]
fn unicode_comparison_glyphs() {
    check(&[
        ("a ≠ b", "(!= a b)"),
        ("a ≤ b", "(<= a b)"),
        ("a ≥ b", "(>= a b)"),
    ]);
}

#[test]
fn leading_and_trailing_newlines_are_skipped_by_parse_expr() {
    check(&[("\n\na + b\n\n", "(+ a b)")]);
}

#[test]
fn inverse_is_never_lowered_to_a_power() {
    use sky_ast::{parse_expr, BinaryOp, ExprKind, Literal, UnaryOp};

    // `A⁻¹ @ b`: a MatMul whose left operand is an Inverse node, which the
    // checker needs to rewrite it to a Cholesky solve.
    let src = "A⁻¹ @ b";
    let (expr, errors) = parse_expr(&common::lex(src), src);
    assert!(errors.is_empty(), "{errors:#?}");
    let expr = expr.expect("expression");
    let ExprKind::Binary {
        op: BinaryOp::MatMul,
        lhs,
        ..
    } = &expr.kind
    else {
        panic!("not a MatMul: {expr:#?}");
    };
    let ExprKind::Inverse { operand, op_span } = &lhs.kind else {
        panic!("left operand is not an Inverse: {lhs:#?}");
    };
    assert_eq!(common::at(src, operand.span), "A");
    assert_eq!(common::at(src, *op_span), "⁻¹");

    // `A^-1`: a plain Pow whose exponent is the expression `-1`.
    let src = "A^-1";
    let (expr, errors) = parse_expr(&common::lex(src), src);
    assert!(errors.is_empty(), "{errors:#?}");
    let expr = expr.expect("expression");
    let ExprKind::Binary {
        op: BinaryOp::Pow,
        lhs,
        rhs,
        ..
    } = &expr.kind
    else {
        panic!("not a Pow: {expr:#?}");
    };
    assert!(matches!(lhs.kind, ExprKind::Name(_)), "{lhs:#?}");
    let ExprKind::Unary {
        op: UnaryOp::Neg,
        operand,
        ..
    } = &rhs.kind
    else {
        panic!("exponent is not `-1`: {rhs:#?}");
    };
    assert!(matches!(operand.kind, ExprKind::Literal(Literal::Int(_))));
    assert_eq!(common::at(src, rhs.span), "-1");
}

#[test]
fn array_literals() {
    check(&[
        ("[1, 2, 3]", "(array 1 2 3)"),
        ("[1 m, 2 m,]", "(array (unit 1 m) (unit 2 m))"),
        ("[[1, 2], [3, 4]]", "(array (array 1 2) (array 3 4))"),
        ("[a + b, -c]", "(array (+ a b) (neg c))"),
        ("[1, 2][0]", "(index (array 1 2) 0)"),
        ("A @ [1, 2]", "(@ A (array 1 2))"),
        // Newlines inside brackets are not statement ends.
        ("[[1, 2],\n [3, 4]]", "(array (array 1 2) (array 3 4))"),
    ]);
}

#[test]
fn array_literal_errors() {
    let errs = common::expr_errors("[]");
    assert_eq!(errs.len(), 1);
    assert_eq!(
        errs[0].message,
        "an array literal needs at least one element"
    );
    assert!(errs[0]
        .help
        .as_deref()
        .is_some_and(|h| h.contains("zeros(n)")));
    assert_eq!(common::expr_errors("[1 2]").len(), 1);
}
