//! Top-level simple statements: assignment, annotated bindings, expressions.

mod common;
use common::{module_errors, module_sexpr};
use sky_ast::{parse_module, Span, Token, TokenKind};

#[test]
fn assignments_and_bindings() {
    let src = "\
x = 1
d = 3
inv = 2
A: Matrix[3, 4]
B: Matrix[3, 4] = zeros(3, 4)
velocity = 10 km / 30 min
df = ∂f/∂x
x = A⁻¹ @ b
E = ħ * ω
f(x)
";
    let want = "\
(= x 1)
(= d 3)
(= inv 2)
(: A (index Matrix 3 4))
(: B (index Matrix 3 4) (call zeros 3 4))
(= velocity (/ (unit 10 km) (unit 30 min)))
(= df (/ (partial f) (partial x)))
(= x (@ (inverse A) b))
(= E (* ħ ω))
(call f x)";
    assert_eq!(module_sexpr(src), want);
}

#[test]
fn spec_examples_parse() {
    assert_eq!(module_sexpr("10 km"), "(unit 10 km)");
    assert_eq!(
        module_sexpr("5 meters + 20 kelvin"),
        "(+ (unit 5 meters) (unit 20 kelvin))"
    );
    assert_eq!(
        module_sexpr("Matrix[3,4] @ Matrix[4,8]"),
        "(@ (index Matrix 3 4) (index Matrix 4 8))"
    );
    assert_eq!(module_sexpr("A⁻¹ @ b"), "(@ (inverse A) b)");
}

#[test]
fn blank_lines_comments_and_missing_final_newline() {
    let src = "\n# a comment\n\nx = 1   # trailing\n\n\ny = 2";
    assert_eq!(module_sexpr(src), "(= x 1)\n(= y 2)");
}

#[test]
fn empty_module_is_fine() {
    for src in ["", "\n", "\n\n# only a comment\n"] {
        assert_eq!(module_sexpr(src), "");
    }
    let (module, errors) = parse_module(&[], "");
    assert!(module.stmts.is_empty() && errors.is_empty());
    assert_eq!(module.span, Span::new(0, 0));
}

#[test]
fn recovery_skips_to_the_next_line() {
    let src = "x = (1 +\ny = 2\nz = 3 3\nw = 4\n";
    let tokens = common::lex(src);
    let (module, errors) = parse_module(&tokens, src);
    assert_eq!(module.to_sexpr(src), "(= y 2)\n(= w 4)");
    assert_eq!(errors.len(), 2, "{errors:#?}");
    assert!(errors[0].message.contains("expected an expression, found end of line"));
    assert!(errors[1].message.contains("unexpected integer literal `3`"));
}

#[test]
fn invalid_targets_and_chained_assignment() {
    for (src, needle) in [
        ("A[0] = 1", "invalid assignment target"),
        ("1 = x", "invalid assignment target"),
        ("f(x): Real", "invalid annotation target"),
        ("a = b = c", "chained assignment"),
        ("a: T = b = c", "chained assignment"),
    ] {
        let errors = module_errors(src);
        assert_eq!(errors.len(), 1, "{src:?}: {errors:#?}");
        assert!(errors[0].message.contains(needle), "{src:?}: {}", errors[0].message);
    }
}

#[test]
fn keywords_are_not_supported_yet() {
    let errors = module_errors("def f(x):\n    return x\nx = 1\n");
    assert!(errors[0].message.contains("keyword `def`"), "{}", errors[0].message);
    assert!(errors[0].message.contains("not supported"));
}

fn tok(kind: TokenKind, start: u32, end: u32) -> Token {
    Token {
        kind,
        span: Span::new(start, end),
    }
}

#[test]
fn indent_is_reported_once_and_the_block_still_parses() {
    // "x = 1\n    y = 2\n" as the real lexer would tokenize it.
    use TokenKind::*;
    let src = "x = 1\n    y = 2\n";
    let tokens = [
        tok(Ident, 0, 1),
        tok(Assign, 2, 3),
        tok(Int, 4, 5),
        tok(Newline, 5, 6),
        tok(Indent, 10, 10),
        tok(Ident, 10, 11),
        tok(Assign, 12, 13),
        tok(Int, 14, 15),
        tok(Newline, 15, 16),
        tok(Dedent, 16, 16),
        tok(Eof, 16, 16),
    ];
    let (module, errors) = parse_module(&tokens, src);
    assert_eq!(module.to_sexpr(src), "(= x 1)\n(= y 2)");
    assert_eq!(errors.len(), 1);
    assert!(errors[0].message.contains("indent"));
    assert_eq!(errors[0].span, Span::new(10, 10));
}

#[test]
fn missing_eof_and_tokens_after_eof() {
    let src = "x = 1";
    let mut tokens = common::lex(src);
    tokens.pop(); // no Eof
    let (module, errors) = parse_module(&tokens, src);
    assert!(errors.is_empty());
    assert_eq!(module.to_sexpr(src), "(= x 1)");

    let src = "x = 1 y";
    let mut tokens = common::lex("x = 1");
    tokens.push(tok(TokenKind::Ident, 6, 7)); // after Eof: ignored
    let (module, errors) = parse_module(&tokens, src);
    assert!(errors.is_empty());
    assert_eq!(module.to_sexpr(src), "(= x 1)");
}
