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
