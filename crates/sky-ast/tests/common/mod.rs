//! TEST SCAFFOLDING ONLY: a tiny tokenizer so parser tests can be written as
//! source strings. It is NOT the Sky lexer (that is Eng Lead's, in
//! `sky-syntax`): no indentation tracking (never emits Indent/Dedent), no
//! string escapes, no bidi or confusable checks, and it panics on characters
//! it does not know (including `±`, which PR #4's lexer rejects). Like #4,
//! `+-` lexes as `Plus`, `Minus`, and `≠ ≤ ≥` as `Ne`, `Le`, `Ge`. It does produce exact byte spans following the token
//! contract: non-overlapping, `Newline` covers `\n`, `Eof` is zero-width at
//! the end. Replace with the real lexer once it lands.

#![allow(dead_code)]

use sky_ast::{Kw, Span, Token, TokenKind};

fn span(start: usize, end: usize) -> Span {
    Span::new(start as u32, end as u32)
}

pub fn lex(src: &str) -> Vec<Token> {
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(c) = src[i..].chars().next() {
        let rest = &src[i..];
        let start = i;
        let mut push = |kind, len: usize| {
            out.push(Token {
                kind,
                span: span(start, start + len),
            });
            start + len
        };
        if c == ' ' || c == '\t' || c == '\r' {
            i += 1;
        } else if c == '#' {
            i += rest.find('\n').unwrap_or(rest.len());
        } else if c == '\n' {
            i = push(TokenKind::Newline, 1);
        } else if c.is_ascii_digit() {
            let (len, is_float) = number_len(rest);
            i = push(if is_float { TokenKind::Float } else { TokenKind::Int }, len);
        } else if c == '_' || c.is_alphabetic() {
            let len = rest
                .char_indices()
                .find(|&(_, c)| !(c == '_' || c.is_alphabetic() || c.is_ascii_digit()))
                .map_or(rest.len(), |(n, _)| n);
            let kind = keyword(&rest[..len]).map_or(TokenKind::Ident, TokenKind::Keyword);
            i = push(kind, len);
        } else if c == '"' {
            let close = rest[1..].find('"').expect("test lexer: unterminated string");
            i = push(TokenKind::Str, close + 2);
        } else if let Some((kind, len)) = operator(rest) {
            i = push(kind, len);
        } else {
            panic!("test lexer: unexpected character {c:?} at byte {i}");
        }
    }
    out.push(Token {
        kind: TokenKind::Eof,
        span: span(src.len(), src.len()),
    });
    out
}

fn number_len(s: &str) -> (usize, bool) {
    let b = s.as_bytes();
    let digits = |from: usize| {
        let mut j = from;
        while j < b.len() && (b[j].is_ascii_digit() || b[j] == b'_') {
            j += 1;
        }
        j
    };
    let mut j = digits(0);
    let mut is_float = false;
    if j + 1 < b.len() && b[j] == b'.' && b[j + 1].is_ascii_digit() {
        j = digits(j + 1);
        is_float = true;
    }
    if j < b.len() && (b[j] == b'e' || b[j] == b'E') {
        let k = if j + 1 < b.len() && (b[j + 1] == b'+' || b[j + 1] == b'-') { j + 2 } else { j + 1 };
        if k < b.len() && b[k].is_ascii_digit() {
            j = digits(k);
            is_float = true;
        }
    }
    (j, is_float)
}

fn keyword(word: &str) -> Option<Kw> {
    Some(match word {
        "def" => Kw::Def,
        "return" => Kw::Return,
        "if" => Kw::If,
        "else" => Kw::Else,
        "for" => Kw::For,
        "in" => Kw::In,
        "import" => Kw::Import,
        "equation" => Kw::Equation,
        "solve" => Kw::Solve,
        "field" => Kw::Field,
        _ => return None,
    })
}

fn operator(s: &str) -> Option<(TokenKind, usize)> {
    const OPS: &[(&str, TokenKind)] = &[
        ("⁻¹", TokenKind::Inverse),
        ("==", TokenKind::Eq),
        ("!=", TokenKind::Ne),
        ("<=", TokenKind::Le),
        (">=", TokenKind::Ge),
        ("->", TokenKind::Arrow),
        ("(", TokenKind::LParen),
        (")", TokenKind::RParen),
        ("[", TokenKind::LBracket),
        ("]", TokenKind::RBracket),
        (",", TokenKind::Comma),
        (":", TokenKind::Colon),
        ("=", TokenKind::Assign),
        ("+", TokenKind::Plus),
        ("-", TokenKind::Minus),
        ("*", TokenKind::Star),
        ("/", TokenKind::Slash),
        ("^", TokenKind::Caret),
        ("@", TokenKind::At),
        ("≠", TokenKind::Ne),
        ("≤", TokenKind::Le),
        ("≥", TokenKind::Ge),
        ("∂", TokenKind::Partial),
        ("∇", TokenKind::Nabla),
        ("<", TokenKind::Lt),
        (">", TokenKind::Gt),
        (".", TokenKind::Dot),
    ];
    OPS.iter()
        .find(|(text, _)| s.starts_with(text))
        .map(|&(text, kind)| (kind, text.len()))
}

// ----- helpers shared by the test files -----

/// Parses `src` as one expression, asserts there are no errors, and returns
/// the S-expression.
pub fn sexpr(src: &str) -> String {
    let tokens = lex(src);
    let (expr, errors) = sky_ast::parse_expr(&tokens, src);
    assert!(errors.is_empty(), "unexpected errors for {src:?}: {errors:#?}");
    expr.expect("no expression").to_sexpr(src)
}

/// Parses `src` as a module, asserts there are no errors, and returns the
/// S-expression (one statement per line).
pub fn module_sexpr(src: &str) -> String {
    let tokens = lex(src);
    let (module, errors) = sky_ast::parse_module(&tokens, src);
    assert!(errors.is_empty(), "unexpected errors for {src:?}: {errors:#?}");
    module.to_sexpr(src)
}

pub fn expr_errors(src: &str) -> Vec<sky_ast::ParseError> {
    let tokens = lex(src);
    let (expr, errors) = sky_ast::parse_expr(&tokens, src);
    assert!(expr.is_some() || !errors.is_empty(), "None without an error for {src:?}");
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
