//! Helpers shared by the parser tests. Token streams come from the real
//! lexer (`sky_syntax::lex`, #4).

#![allow(dead_code)]

use sky_ast::{Span, Token};

/// Lexes `src` with the real lexer, panicking if it rejects the source.
pub fn lex(src: &str) -> Vec<Token> {
    match sky_syntax::lex(src) {
        Ok(lexed) => lexed.tokens,
        Err(e) => panic!("lexer rejected {src:?}: {}", e.diagnostic.message),
    }
}

/// The first `keep` tokens of `src` (no `Eof`), for inputs the lexer would
/// reject but the parser must still handle, e.g. an unclosed `(`.
pub fn lex_prefix(src: &str, keep: usize) -> Vec<Token> {
    let mut tokens = lex(src);
    tokens.truncate(keep);
    tokens
}

/// Parses `src` as one expression, asserts there are no errors, and returns
/// the S-expression.
pub fn sexpr(src: &str) -> String {
    let tokens = lex(src);
    let (expr, errors) = sky_ast::parse_expr(&tokens, src);
    assert!(
        errors.is_empty(),
        "unexpected errors for {src:?}: {errors:#?}"
    );
    expr.expect("no expression").to_sexpr(src)
}

/// Parses `src` as a module, asserts there are no errors, and returns the
/// S-expression (one statement per line).
pub fn module_sexpr(src: &str) -> String {
    let tokens = lex(src);
    let (module, errors) = sky_ast::parse_module(&tokens, src);
    assert!(
        errors.is_empty(),
        "unexpected errors for {src:?}: {errors:#?}"
    );
    module.to_sexpr(src)
}

pub fn expr_errors(src: &str) -> Vec<sky_ast::ParseError> {
    tokens_errors(&lex(src), src)
}

pub fn tokens_errors(tokens: &[Token], src: &str) -> Vec<sky_ast::ParseError> {
    let (expr, errors) = sky_ast::parse_expr(tokens, src);
    assert!(
        expr.is_some() || !errors.is_empty(),
        "None without an error for {src:?}"
    );
    errors
}

pub fn module_errors(src: &str) -> Vec<sky_ast::ParseError> {
    let tokens = lex(src);
    sky_ast::parse_module(&tokens, src).1
}

/// `src[span]`, panicking on a bad span (tests only).
pub fn at(src: &str, span: Span) -> &str {
    &src[span.range()]
}
