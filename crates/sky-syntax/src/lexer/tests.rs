use super::*;
use crate::token::Kw;
use TokenKind::*;

/// Checks the span rules from the module docs, and that the gaps between
/// tokens hold only whitespace, line terminators, comments, or a leading BOM.
fn check_invariants(src: &str, tokens: &[Token]) {
    let last = tokens.last().expect("at least Eof");
    assert_eq!(last.kind, Eof);
    assert_eq!(last.span, Span::point(src.len() as u32));
    assert_eq!(tokens.iter().filter(|t| t.kind == Eof).count(), 1);
    let mut prev_end = 0usize;
    for t in tokens {
        let r = t.span.range();
        assert!(r.start >= prev_end, "overlap at {t:?} in {src:?}");
        assert!(src.is_char_boundary(r.start) && src.is_char_boundary(r.end));
        let gap = &src[prev_end..r.start];
        assert_gap_is_trivia(gap, prev_end == 0, src);
        match t.kind {
            Indent | Dedent | Eof => assert!(t.span.is_empty(), "{t:?} not zero-width"),
            Newline => assert!(
                matches!(&src[r.clone()], "\n" | "\r\n")
                    || (t.span.is_empty() && r.start == src.len()),
                "bad Newline {t:?} in {src:?}"
            ),
            _ => assert!(!t.span.is_empty(), "{t:?} is empty"),
        }
        prev_end = r.end;
    }
    // Indent/Dedent sit at the start of the next non-layout token.
    for (i, t) in tokens.iter().enumerate() {
        if matches!(t.kind, Indent | Dedent) {
            let next = tokens[i + 1..]
                .iter()
                .find(|n| !matches!(n.kind, Indent | Dedent))
                .unwrap();
            assert_eq!(t.span.start, next.span.start, "{t:?} not at next token");
        }
    }
}

fn assert_gap_is_trivia(gap: &str, at_file_start: bool, src: &str) {
    let gap = if at_file_start {
        gap.strip_prefix('\u{FEFF}').unwrap_or(gap)
    } else {
        gap
    };
    let mut chars = gap.chars();
    while let Some(c) = chars.next() {
        match c {
            ' ' | '\t' | '\n' | '\r' => {}
            '#' => {
                for c in chars.by_ref() {
                    assert!(c != '\r', "comment swallowed CR in {src:?}");
                    if c == '\n' {
                        break;
                    }
                }
            }
            _ => panic!("non-trivia {c:?} in gap {gap:?} of {src:?}"),
        }
    }
}

fn lexed(src: &str) -> Lexed {
    let l = lex(src).unwrap_or_else(|e| panic!("lex({src:?}) failed: {:?}", e.diagnostic));
    check_invariants(src, &l.tokens);
    l
}

fn kinds(src: &str) -> Vec<TokenKind> {
    lexed(src).tokens.iter().map(|t| t.kind).collect()
}

/// Kinds of the first line's tokens, without the trailing Newline/Eof.
fn line(src: &str) -> Vec<TokenKind> {
    let mut k = kinds(src);
    assert_eq!(k.pop(), Some(Eof));
    assert_eq!(k.pop(), Some(Newline));
    k
}

fn texts(src: &str) -> Vec<(TokenKind, &str)> {
    lexed(src)
        .tokens
        .iter()
        .map(|t| (t.kind, &src[t.span.range()]))
        .collect()
}

fn lex_err(src: &str) -> LexError {
    match lex(src) {
        Ok(l) => panic!("lex({src:?}) succeeded: {:?}", l.tokens),
        Err(e) => e,
    }
}

/// Asserts the error kind and that its primary span covers `text`.
fn assert_err(src: &str, kind: LexErrorKind, span_text: &str) -> LexError {
    let e = lex_err(src);
    assert_eq!(e.kind, kind, "{src:?}: {:?}", e.diagnostic);
    assert_eq!(e.diagnostic.severity, sky_diag::Severity::Error);
    assert_eq!(
        &src[e.diagnostic.primary.span.range()],
        span_text,
        "{src:?}"
    );
    e
}

fn sp(start: u32, end: u32) -> Span {
    Span::new(start, end)
}

// ---- basics ---------------------------------------------------------------

#[test]
fn empty_and_trivia_only_sources() {
    for src in [
        "",
        "\n",
        "\n\n  \n",
        "# only a comment",
        "# c\n   # indented c\n",
        "\u{FEFF}",
    ] {
        assert_eq!(kinds(src), [Eof], "{src:?}");
    }
}

#[test]
fn simple_binding_spans() {
    let l = lexed("x = 1\n");
    let spans: Vec<_> = l.tokens.iter().map(|t| (t.kind, t.span)).collect();
    assert_eq!(
        spans,
        [
            (Ident, sp(0, 1)),
            (Assign, sp(2, 3)),
            (Int, sp(4, 5)),
            (Newline, sp(5, 6)),
            (Eof, sp(6, 6)),
        ]
    );
    assert!(l.warnings.is_empty());
}

#[test]
fn missing_final_newline_gets_zero_width_newline() {
    let l = lexed("x = 1");
    assert_eq!(
        l.tokens[3],
        Token {
            kind: Newline,
            span: sp(5, 5)
        }
    );
    assert_eq!(
        l.tokens[4],
        Token {
            kind: Eof,
            span: sp(5, 5)
        }
    );
    // Also after a trailing comment.
    let l = lexed("x = 1  # c");
    assert_eq!(
        l.tokens[3],
        Token {
            kind: Newline,
            span: sp(10, 10)
        }
    );
}

#[test]
fn comments_produce_no_tokens() {
    assert_eq!(kinds("x = 1 # set x\n"), kinds("x = 1\n"));
    assert_eq!(line("a # b = c\n"), [Ident]);
    assert_eq!(kinds("x = 1\n# c\n"), kinds("x = 1\n"));
}

#[test]
fn crlf_newline_covers_both_bytes() {
    let l = lexed("x = 1\r\ny = 2\r\n");
    assert_eq!(
        l.tokens[3],
        Token {
            kind: Newline,
            span: sp(5, 7)
        }
    );
    assert_eq!(
        l.tokens[7],
        Token {
            kind: Newline,
            span: sp(12, 14)
        }
    );
}

#[test]
fn bare_carriage_return_is_an_error() {
    assert_err("x = 1\ry = 2\n", LexErrorKind::BareCarriageReturn, "\r");
    assert_err("x = 1 # c\r", LexErrorKind::BareCarriageReturn, "\r");
}

#[test]
fn leading_bom_is_skipped() {
    let src = "\u{FEFF}x = 1\n";
    assert_eq!(texts(src)[0], (Ident, "x"));
    assert_eq!(kinds(src), [Ident, Assign, Int, Newline, Eof]);
    // A BOM anywhere else is an invisible character.
    assert_err("x = \u{FEFF}1\n", LexErrorKind::UnexpectedChar, "\u{FEFF}");
}

// ---- indentation ----------------------------------------------------------

#[test]
fn indent_and_dedent() {
    let src = "def f(x):\n    return x\ny = 1\n";
    assert_eq!(
        kinds(src),
        [
            Keyword(Kw::Def),
            Ident,
            LParen,
            Ident,
            RParen,
            Colon,
            Newline,
            Indent,
            Keyword(Kw::Return),
            Ident,
            Newline,
            Dedent,
            Ident,
            Assign,
            Int,
            Newline,
            Eof,
        ]
    );
    let l = lexed(src);
    let indent = l.tokens.iter().find(|t| t.kind == Indent).unwrap();
    assert_eq!(indent.span, Span::point(src.find("return").unwrap() as u32));
    let dedent = l.tokens.iter().find(|t| t.kind == Dedent).unwrap();
    assert_eq!(dedent.span, Span::point(src.find("y =").unwrap() as u32));
}

#[test]
fn multiple_dedents_at_once_and_at_eof() {
    let src = "if a:\n  if b:\n    x\n  y\nz\nif c:\n  if d:\n    w";
    assert_eq!(
        kinds(src),
        [
            Keyword(Kw::If),
            Ident,
            Colon,
            Newline,
            Indent,
            Keyword(Kw::If),
            Ident,
            Colon,
            Newline,
            Indent,
            Ident,
            Newline,
            Dedent,
            Ident,
            Newline,
            Dedent,
            Ident,
            Newline,
            Keyword(Kw::If),
            Ident,
            Colon,
            Newline,
            Indent,
            Keyword(Kw::If),
            Ident,
            Colon,
            Newline,
            Indent,
            Ident,
            Newline,
            Dedent,
            Dedent,
            Eof,
        ]
    );
    let l = lexed(src);
    let end = Span::point(src.len() as u32);
    let tail: Vec<_> = l.tokens[l.tokens.len() - 4..]
        .iter()
        .map(|t| t.span)
        .collect();
    assert_eq!(tail, [end, end, end, end]);
}

#[test]
fn trailing_newline_then_dedents_before_eof() {
    let k = kinds("def f():\n    return 1\n");
    assert_eq!(&k[k.len() - 3..], [Newline, Dedent, Eof]);
    let k = kinds("def f():\n    return 1");
    assert_eq!(&k[k.len() - 3..], [Newline, Dedent, Eof]);
}

#[test]
fn blank_and_comment_only_lines_do_not_affect_indentation() {
    let src = "def f():\n    a = 1\n\n# column-0 comment\n        # deep comment\n  \n    b = 2\n";
    assert_eq!(
        kinds(src),
        [
            Keyword(Kw::Def),
            Ident,
            LParen,
            RParen,
            Colon,
            Newline,
            Indent,
            Ident,
            Assign,
            Int,
            Newline,
            Ident,
            Assign,
            Int,
            Newline,
            Dedent,
            Eof,
        ]
    );
}

#[test]
fn dedent_after_blank_line_is_at_next_token() {
    let src = "if a:\n    b\n\n\nc\n";
    let l = lexed(src);
    let dedent = l.tokens.iter().find(|t| t.kind == Dedent).unwrap();
    assert_eq!(dedent.span, Span::point(src.find('c').unwrap() as u32));
}

#[test]
fn tab_indentation_is_fine_on_its_own() {
    assert_eq!(
        kinds("if a:\n\tb\n\tif c:\n\t\td\ne\n"),
        [
            Keyword(Kw::If),
            Ident,
            Colon,
            Newline,
            Indent,
            Ident,
            Newline,
            Keyword(Kw::If),
            Ident,
            Colon,
            Newline,
            Indent,
            Ident,
            Newline,
            Dedent,
            Dedent,
            Ident,
            Newline,
            Eof,
        ]
    );
}

#[test]
fn separate_blocks_may_use_different_whitespace() {
    kinds("if a:\n    b\nif c:\n\td\n");
}

#[test]
fn indented_first_line_emits_indent() {
    // Rejecting this is the parser's job ("unexpected indent").
    assert_eq!(kinds("  x\n"), [Indent, Ident, Newline, Dedent, Eof]);
}

#[test]
fn inconsistent_dedent_is_an_error() {
    let src = "if a:\n    b\n  c\n";
    let e = assert_err(src, LexErrorKind::InconsistentDedent, "  ");
    assert_eq!(e.diagnostic.primary.span, sp(12, 14));
}

#[test]
fn tabs_mixed_with_spaces_in_one_line_is_an_error() {
    assert_err("if a:\n \tb\n", LexErrorKind::MixedIndentation, " \t");
    assert_err("if a:\n\t    b\n", LexErrorKind::MixedIndentation, "\t    ");
}

#[test]
fn tabs_vs_spaces_across_lines_is_an_error() {
    let e = assert_err(
        "if a:\n    b\n    if c:\n\t\td\n",
        LexErrorKind::MixedIndentation,
        "\t\t",
    );
    assert!(e
        .diagnostic
        .message
        .contains("uses tabs, but the enclosing block uses spaces"));
    assert_err(
        "if a:\n\tb\n        c\n",
        LexErrorKind::MixedIndentation,
        "        ",
    );
}

#[test]
fn mixed_whitespace_in_blank_or_comment_lines_is_ignored() {
    kinds("if a:\n    b\n \t \n \t# c\n    c\n");
}

// ---- implicit line joining ------------------------------------------------

#[test]
fn no_newline_or_indent_inside_parens() {
    let src = "x = f(1,\n      2)\ny = 3\n";
    assert_eq!(
        kinds(src),
        [
            Ident, Assign, Ident, LParen, Int, Comma, Int, RParen, Newline, Ident, Assign, Int,
            Newline, Eof,
        ]
    );
}

#[test]
fn no_newline_or_dedent_inside_brackets() {
    let src = "def f():\n    m = [[1, 2],\n# comment\n\n[3, 4]]\n    return m\n";
    assert_eq!(
        kinds(src),
        [
            Keyword(Kw::Def),
            Ident,
            LParen,
            RParen,
            Colon,
            Newline,
            Indent,
            Ident,
            Assign,
            LBracket,
            LBracket,
            Int,
            Comma,
            Int,
            RBracket,
            Comma,
            LBracket,
            Int,
            Comma,
            Int,
            RBracket,
            RBracket,
            Newline,
            Keyword(Kw::Return),
            Ident,
            Newline,
            Dedent,
            Eof,
        ]
    );
}

#[test]
fn mixed_whitespace_inside_brackets_is_not_indentation() {
    kinds("x = (1 +\n \t 2)\n");
}

#[test]
fn brackets_must_match() {
    assert_err("x = 1)\n", LexErrorKind::UnmatchedBracket, ")");
    let e = assert_err("x = (1]\n", LexErrorKind::MismatchedBracket, "]");
    assert_eq!(e.diagnostic.secondary[0].span, sp(4, 5));
    let e = assert_err("x = f([1,\n  2)\n", LexErrorKind::MismatchedBracket, ")");
    assert_eq!(e.diagnostic.secondary[0].span, sp(6, 7));
    assert_err("x = f(1,\n  2\n", LexErrorKind::UnclosedBracket, "(");
    assert_err("m = [[1]\n", LexErrorKind::UnclosedBracket, "[");
}

// ---- operators ------------------------------------------------------------

#[test]
fn every_operator_ascii_forms() {
    assert_eq!(
        texts("( ) [ ] , : = . -> + - * / ^ @ == != < <= > >=\n"),
        [
            (LParen, "("),
            (RParen, ")"),
            (LBracket, "["),
            (RBracket, "]"),
            (Comma, ","),
            (Colon, ":"),
            (Assign, "="),
            (Dot, "."),
            (Arrow, "->"),
            (Plus, "+"),
            (Minus, "-"),
            (Star, "*"),
            (Slash, "/"),
            (Caret, "^"),
            (At, "@"),
            (Eq, "=="),
            (Ne, "!="),
            (Lt, "<"),
            (Le, "<="),
            (Gt, ">"),
            (Ge, ">="),
            (Newline, "\n"),
            (Eof, ""),
        ]
    );
}

#[test]
fn every_operator_glyph_forms() {
    assert_eq!(
        texts("⁻¹ ∂ ∇ ≤ ≥ ≠\n"),
        [
            (Inverse, "⁻¹"),
            (Partial, "∂"),
            (Nabla, "∇"),
            (Le, "≤"),
            (Ge, "≥"),
            (Ne, "≠"),
            (Newline, "\n"),
            (Eof, ""),
        ]
    );
}

#[test]
fn operators_without_spaces_use_longest_match() {
    assert_eq!(line("a==b\n"), [Ident, Eq, Ident]);
    assert_eq!(line("a=-b\n"), [Ident, Assign, Minus, Ident]);
    assert_eq!(line("a<=b>=c\n"), [Ident, Le, Ident, Ge, Ident]);
    assert_eq!(
        line("f(x)->y\n"),
        [Ident, LParen, Ident, RParen, Arrow, Ident]
    );
    assert_eq!(line("a+-b\n"), [Ident, Plus, Minus, Ident]);
    assert_eq!(line("a+ -b\n"), [Ident, Plus, Minus, Ident]);
    assert_eq!(line("a- >b\n"), [Ident, Minus, Gt, Ident]);
    assert_eq!(line("a**b\n"), [Ident, Star, Star, Ident]);
    assert_eq!(line("A⁻¹⁻¹\n"), [Ident, Inverse, Inverse]);
}

#[test]
fn glyph_comparisons_map_to_ascii_kinds() {
    assert_eq!(line("a ≤ b"), line("a <= b"));
    assert_eq!(line("a ≥ b"), line("a >= b"));
    assert_eq!(line("a ≠ b"), line("a != b"));
}

/// Sky v0.1 has no uncertainty: `+-` is `a + (-b)` as in Python, and the
/// glyph `±` is rejected with help rather than lexed as an operator.
#[test]
fn plus_minus_is_plus_then_minus_and_glyph_is_rejected() {
    assert_eq!(
        texts("a+-b\n")[..4],
        [(Ident, "a"), (Plus, "+"), (Minus, "-"), (Ident, "b")]
    );
    assert_eq!(line("a+-b\n"), line("a + -b\n"));
    assert_eq!(
        line("5.31 +- 0.02 kg\n"),
        [Float, Plus, Minus, Float, Ident]
    );
    assert_eq!(line("a+--b\n"), [Ident, Plus, Minus, Minus, Ident]);
    assert_eq!(
        line("f(x)+->y\n"),
        [Ident, LParen, Ident, RParen, Plus, Arrow, Ident]
    );

    let e = assert_err("m = 5.31 ± 0.02 kg\n", LexErrorKind::UnexpectedChar, "±");
    assert_eq!(e.diagnostic.message, "unexpected character '±' (U+00B1)");
    assert_eq!(
        e.diagnostic.help,
        ["uncertainty (±) is not supported in v0.1"]
    );
    assert_err("a±b\n", LexErrorKind::UnexpectedChar, "±");
}

#[test]
fn spec_examples() {
    assert_eq!(line("d = 3\n"), [Ident, Assign, Int]);
    assert_eq!(texts("10 km\n")[..2], [(Int, "10"), (Ident, "km")]);
    assert_eq!(
        texts("x = A⁻¹ @ b\n")[..6],
        [
            (Ident, "x"),
            (Assign, "="),
            (Ident, "A"),
            (Inverse, "⁻¹"),
            (At, "@"),
            (Ident, "b")
        ]
    );
    assert_eq!(
        line("x = inv(A) @ b\n"),
        [Ident, Assign, Ident, LParen, Ident, RParen, At, Ident]
    );
    assert_eq!(
        line("df = ∂f/∂x\n"),
        [Ident, Assign, Partial, Ident, Slash, Partial, Ident]
    );
    assert_eq!(line("df = d(f)/d(x)\n")[2], Ident);
    assert_eq!(line("g = ∇φ\n"), [Ident, Assign, Nabla, Ident]);
    assert_eq!(line("g = grad(φ)\n")[2], Ident);
    assert_eq!(
        line("x = 5 meters + 20 kelvin\n"),
        [Ident, Assign, Int, Ident, Plus, Int, Ident]
    );
    assert_eq!(
        line("C = Matrix[3,4] @ Matrix[4,8]\n"),
        [
            Ident, Assign, Ident, LBracket, Int, Comma, Int, RBracket, At, Ident, LBracket, Int,
            Comma, Int, RBracket,
        ]
    );
    assert_eq!(line("E = ħ * ω\n"), [Ident, Assign, Ident, Star, Ident]);
}

#[test]
fn lone_superscript_minus_and_bang_are_errors() {
    let e = assert_err("A⁻ @ b\n", LexErrorKind::UnexpectedChar, "⁻");
    assert!(e.diagnostic.help[0].contains("⁻¹"));
    assert_err("a ! b\n", LexErrorKind::UnexpectedChar, "!");
    assert_err("x = A¹\n", LexErrorKind::UnexpectedChar, "¹");
}

#[test]
fn unicode_lookalike_operators_get_help() {
    let e = assert_err("a − b\n", LexErrorKind::UnexpectedChar, "\u{2212}");
    assert!(e.diagnostic.help[0].contains('-'));
    assert_err("a × b\n", LexErrorKind::UnexpectedChar, "×");
    assert_err("a·b\n", LexErrorKind::UnexpectedChar, "·");
    assert_err("x = 'a'\n", LexErrorKind::UnexpectedChar, "'");
    assert_err("x = {}\n", LexErrorKind::UnexpectedChar, "{");
    let e = assert_err("x =\u{00A0}1\n", LexErrorKind::UnexpectedChar, "\u{00A0}");
    // Non-printing characters are described by code point only.
    assert_eq!(e.diagnostic.message, "unexpected character U+00A0");
    let e = assert_err("x = \u{1B}\n", LexErrorKind::UnexpectedChar, "\u{1B}");
    assert_eq!(e.diagnostic.message, "unexpected character U+001B");
}

// ---- keywords and identifiers ---------------------------------------------

#[test]
fn keywords() {
    let src = "def return if else for in import equation solve field\n";
    let expected: Vec<_> = [
        Kw::Def,
        Kw::Return,
        Kw::If,
        Kw::Else,
        Kw::For,
        Kw::In,
        Kw::Import,
        Kw::Equation,
        Kw::Solve,
        Kw::Field,
    ]
    .into_iter()
    .map(Keyword)
    .collect();
    assert_eq!(line(src), expected);
    for kw in &expected {
        let Keyword(kw) = kw else { unreachable!() };
        assert_eq!(Kw::from_ident(kw.as_str()), Some(*kw));
    }
}

#[test]
fn stdlib_names_and_keyword_prefixes_are_identifiers() {
    assert_eq!(
        line("d inv grad define ifs input _field Def IF\n"),
        [Ident; 9]
    );
}

#[test]
fn unicode_identifiers() {
    let src = "ψ ħ λ_max Δt μm é φ2 _x 変数\n";
    assert_eq!(line(src), [Ident; 9]);
    assert!(lexed(src).warnings.is_empty());
    // Subscript digits are not XID_Continue.
    assert_err("ψ₀ = 1\n", LexErrorKind::UnexpectedChar, "₀");
}

#[test]
fn invisible_characters_break_identifiers() {
    assert_err(
        "ab\u{200D}c = 1\n",
        LexErrorKind::UnexpectedChar,
        "\u{200D}",
    );
    assert_err(
        "ab\u{200C}c = 1\n",
        LexErrorKind::UnexpectedChar,
        "\u{200C}",
    );
    assert_err("a\u{200B}b = 1\n", LexErrorKind::UnexpectedChar, "\u{200B}");
    assert_err("a\u{00AD}b = 1\n", LexErrorKind::UnexpectedChar, "\u{00AD}");
}

#[test]
fn hangul_fillers_cannot_start_an_identifier() {
    // XID_Start, but they render as nothing: `\u{3164} = 1` would bind a
    // variable whose name is invisible.
    for c in ['\u{115F}', '\u{1160}', '\u{3164}', '\u{FFA0}'] {
        let s = c.to_string();
        let e = assert_err(&format!("{c} = 1\n"), LexErrorKind::UnexpectedChar, &s);
        assert_eq!(
            e.diagnostic.help,
            ["invisible characters are not allowed in source code"]
        );
        assert_err(&format!("x = {c}y\n"), LexErrorKind::UnexpectedChar, &s);
        assert_err(&format!("a{c}b = 1\n"), LexErrorKind::UnexpectedChar, &s);
    }
}

// ---- numbers --------------------------------------------------------------

#[test]
fn numbers() {
    assert_eq!(
        texts("0 42 1_000 3.14 1e-10 1E+5 2e3 6.022_140_76e23 0.5\n")
            .into_iter()
            .map(|(k, t)| (k, t.to_string()))
            .take(9)
            .collect::<Vec<_>>(),
        [
            (Int, "0"),
            (Int, "42"),
            (Int, "1_000"),
            (Float, "3.14"),
            (Float, "1e-10"),
            (Float, "1E+5"),
            (Float, "2e3"),
            (Float, "6.022_140_76e23"),
            (Float, "0.5"),
        ]
        .map(|(k, t)| (k, t.to_string()))
    );
}

#[test]
fn dot_after_int_without_digits_is_a_dot() {
    assert_eq!(line("1.\n"), [Int, Dot]);
    assert_eq!(line("x.y\n"), [Ident, Dot, Ident]);
    assert_eq!(line(".5\n"), [Dot, Int]);
    assert_eq!(line("v.x.y\n"), [Ident, Dot, Ident, Dot, Ident]);
}

#[test]
fn invalid_numbers() {
    assert_err("1__0\n", LexErrorKind::InvalidNumber, "_");
    assert_err("1_\n", LexErrorKind::InvalidNumber, "_");
    assert_err("1_000_\n", LexErrorKind::InvalidNumber, "_");
    assert_err("1.5_\n", LexErrorKind::InvalidNumber, "_");
    assert_err("1e5_\n", LexErrorKind::InvalidNumber, "_");
    // `_5` after a dot is an identifier, not a fraction.
    assert_eq!(line("1._5\n"), [Int, Dot, Ident]);
}

#[test]
fn number_glued_to_letters_is_an_error() {
    let e = assert_err("x = 10km\n", LexErrorKind::InvalidNumber, "10km");
    assert!(e.diagnostic.help[0].contains("10 km"));
    assert_err("x = 1e\n", LexErrorKind::InvalidNumber, "1e");
    assert_err("x = 2eV\n", LexErrorKind::InvalidNumber, "2eV");
    assert_err("x = 0x1F\n", LexErrorKind::InvalidNumber, "0x1F");
    assert_err("x = 1.5e3m\n", LexErrorKind::InvalidNumber, "1.5e3m");
    assert_eq!(line("x = 2 eV\n"), [Ident, Assign, Int, Ident]);
}

// ---- strings --------------------------------------------------------------

#[test]
fn strings() {
    let src = r#"s = "hi # not a comment" + "a\"b\\c\n\t\r\0" + "ψ ∂""#;
    let t = texts(src);
    assert_eq!(t[2], (Str, "\"hi # not a comment\""));
    assert_eq!(t[4], (Str, r#""a\"b\\c\n\t\r\0""#));
    assert_eq!(t[6], (Str, "\"ψ ∂\""));
    assert_eq!(unescape_str(t[4].1), "a\"b\\c\n\t\r\0");
    assert_eq!(unescape_str("\"\""), "");
}

#[test]
fn unterminated_strings() {
    assert_err(
        "s = \"abc\nx = 1\n",
        LexErrorKind::UnterminatedString,
        "\"abc",
    );
    assert_err("s = \"abc", LexErrorKind::UnterminatedString, "\"abc");
    assert_err(
        "s = \"abc\\\"\n",
        LexErrorKind::UnterminatedString,
        "\"abc\\\"",
    );
    assert_err(
        "s = \"abc\\\n\"\n",
        LexErrorKind::UnterminatedString,
        "\"abc\\",
    );
}

#[test]
fn invalid_escape() {
    let e = assert_err(r#"s = "a\qb""#, LexErrorKind::InvalidEscape, r"\q");
    assert_eq!(e.diagnostic.message, "unknown escape `\\q`");
    assert_err("s = \"a\\u{41}\"", LexErrorKind::InvalidEscape, r"\u");
}

// ---- security -------------------------------------------------------------

#[test]
fn bidi_controls_are_rejected_everywhere() {
    for (src, at) in [
        ("x = 1 \u{202E} # comment\n", 6),
        ("x = 1 # comment \u{2066}\n", 16),
        ("s = \"\u{202B}\"\n", 5),
        ("x\u{200F} = 1\n", 1),
        ("x = 1\n\u{061C}", 6),
    ] {
        let e = lex_err(src);
        assert_eq!(e.kind, LexErrorKind::BidiControl, "{src:?}");
        let span = e.diagnostic.primary.span;
        assert_eq!(span.start, at, "{src:?}");
        let ch = src[at as usize..].chars().next().unwrap();
        assert_eq!(span.len() as usize, ch.len_utf8());
        assert!(e
            .diagnostic
            .message
            .contains(&format!("U+{:04X}", ch as u32)));
    }
}

#[test]
fn size_limit_is_checked_before_bidi_scan() {
    // `lex` uses `MAX_SOURCE_BYTES` (16 MiB); a tiny limit exercises the same
    // order without a 16 MiB input. The bidi control lies past the limit, so
    // the file must be rejected for its size before the bidi scan runs.
    let src = "x = 1 # \u{202E}\n";
    let e = check_whole_file(src, 8).unwrap_err();
    assert_eq!(e.kind, LexErrorKind::SourceTooLarge);
    assert_eq!(e.diagnostic.primary.span, Span::point(0));
    assert_eq!(e.diagnostic.message, "source file is larger than 8 bytes");
    // At or under the limit, the bidi check still runs.
    let e = check_whole_file(src, src.len()).unwrap_err();
    assert_eq!(e.kind, LexErrorKind::BidiControl);
    assert!(check_whole_file("x = 1\n", 6).is_ok());
}

#[test]
fn size_limit_is_human_readable() {
    assert_eq!(human_size(MAX_SOURCE_BYTES), "16 MiB");
    assert_eq!(human_size(4 << 30), "4 GiB");
    assert_eq!(human_size(3 << 10), "3 KiB");
    assert_eq!(human_size((1 << 20) + 1), "1048577 bytes");
    assert_eq!(human_size(0), "0 bytes");
}

/// `x = 1`, then a comment that pads the file to exactly `len` bytes.
fn source_of_len(len: usize) -> String {
    let mut s = String::with_capacity(len);
    s.push_str("x = 1\n#");
    s.extend(std::iter::repeat_n('a', len - s.len() - 1));
    s.push('\n');
    assert_eq!(s.len(), len);
    s
}

#[test]
fn source_of_exactly_max_source_bytes_lexes() {
    assert_eq!(MAX_SOURCE_BYTES, 16 * 1024 * 1024);
    let src = source_of_len(MAX_SOURCE_BYTES);
    assert_eq!(kinds(&src), [Ident, Assign, Int, Newline, Eof]);
}

#[test]
fn source_one_byte_over_max_source_bytes_is_rejected() {
    // The bidi control would be an error too; the size check must win.
    let mut src = source_of_len(MAX_SOURCE_BYTES + 1);
    src.replace_range(8..11, "\u{202E}");
    assert_eq!(src.len(), MAX_SOURCE_BYTES + 1);
    let e = lex(&src).unwrap_err();
    assert_eq!(e.kind, LexErrorKind::SourceTooLarge);
    assert_eq!(e.diagnostic.primary.span, Span::point(0));
    assert_eq!(e.diagnostic.message, "source file is larger than 16 MiB");
    assert_eq!(
        e.diagnostic.notes,
        ["the file is 16777217 bytes; the limit is 16777216 bytes"]
    );
}

/// One-way drift check between the lexer and the diagnostic renderer: every
/// character the lexer rejects because it is invisible or a bidi control
/// must be escaped by the renderer, or a diagnostic about it would print the
/// very character it complains about. The renderer may escape more (tab and
/// C0 controls, which are legal in source), so the reverse is not checked.
#[test]
fn lexer_rejected_invisible_and_bidi_chars_are_display_unsafe() {
    let mut rejected = 0;
    let mut not_escaped = Vec::new();
    for c in (0..=0x10FFFFu32).filter_map(char::from_u32) {
        if is_invisible(c) || crate::source::is_bidi_control(c) {
            rejected += 1;
            if !sky_diag::is_display_unsafe(c) {
                not_escaped.push(format!("U+{:04X}", c as u32));
            }
        }
    }
    assert!(
        not_escaped.is_empty(),
        "rejected by the lexer but not escaped by sky_diag::is_display_unsafe: {not_escaped:?}"
    );
    // 38 invisible characters plus 12 bidi controls. Update this when either
    // set changes; it keeps the loop above from silently checking nothing.
    assert_eq!(rejected, 50);
}

/// The predicates the drift test uses are the ones `lex` enforces.
#[test]
fn drift_test_predicates_match_lexer_behaviour() {
    for c in (0..=0x10FFFFu32).filter_map(char::from_u32) {
        if crate::source::is_bidi_control(c) {
            // Rejected anywhere, even in a comment.
            let e = lex(&format!("x = 1 # {c}\n")).unwrap_err();
            assert_eq!(e.kind, LexErrorKind::BidiControl, "U+{:04X}", c as u32);
        } else if is_invisible(c) && c != '\u{FEFF}' {
            // Rejected inside an identifier and as a token of its own.
            // (U+FEFF is skipped only as a leading BOM; covered elsewhere.)
            for src in [format!("a{c}b = 1\n"), format!("x = {c}1\n")] {
                let e = lex(&src).unwrap_err();
                assert_eq!(e.kind, LexErrorKind::UnexpectedChar, "U+{:04X}", c as u32);
            }
        }
    }
}

#[test]
fn confusable_identifier_warns_once_per_name() {
    // Cyrillic 'а' (U+0430) in a Latin word.
    let src = "pаth = 1\ny = pаth\n";
    let l = lexed(src);
    assert_eq!(l.warnings.len(), 1);
    let w = &l.warnings[0];
    assert_eq!(w.severity, sky_diag::Severity::Warning);
    assert_eq!(w.primary.span, sp(0, 5));
    assert_eq!(
        w.message,
        "identifier `pаth` contains Cyrillic 'а' (U+0430), which looks like Latin 'a'"
    );
    // It is still an ordinary identifier.
    assert_eq!(l.tokens[0].kind, Ident);
}

#[test]
fn confusable_whole_script_and_mixed_script_warnings() {
    let l = lexed("х = 1\nxЖ = 2\nµm = 3\n");
    let msgs: Vec<_> = l.warnings.iter().map(|w| w.message.as_str()).collect();
    assert_eq!(
        msgs,
        [
            "identifier `х` contains Cyrillic 'х' (U+0445), which looks like Latin 'x'",
            "identifier `xЖ` mixes Latin and Cyrillic scripts",
            "identifier `µm` contains MICRO SIGN U+00B5, which looks like 'μ' (U+03BC); use 'μ' instead",
        ]
    );
}

#[test]
fn legitimate_greek_identifiers_do_not_warn() {
    let l = lexed("ψ = 1\nω = 2\nħ = 3\nλ = 4\nα = 5\nν = 6\nΔt = 7\nμm = 8\nσ_x = 9\n");
    assert!(l.warnings.is_empty(), "{:?}", l.warnings);
}

#[test]
fn keywords_are_not_checked_for_confusables() {
    assert!(lexed("def f():\n    return 1\n").warnings.is_empty());
}

#[test]
fn warnings_before_an_error_are_kept() {
    let e = lex_err("pаth = 1\ny = \"unterminated\n");
    assert_eq!(e.kind, LexErrorKind::UnterminatedString);
    assert_eq!(e.warnings.len(), 1);
    let order: Vec<_> = e.diagnostics().map(|d| d.severity).collect();
    assert_eq!(
        order,
        [sky_diag::Severity::Warning, sky_diag::Severity::Error]
    );
}

// ---- invariants over a mixed program ---------------------------------------

#[test]
fn invariants_hold_on_a_realistic_program() {
    let src = "\
# Kinematics
import units

def position(x0, v, t) -> Length:
    # comment in body
    return x0 + v * t   # trailing

def energy(m, ψ):
    E = ħ * ω + 0.1 eV
    if E ≥ 0 J:
        M = [[1.0, 2.0],
             [3.0, 4.0]]
        return M⁻¹ @ ψ
    else:
        return ∇ψ

v = 10 km / 30 min
s = \"done\"";
    let l = lexed(src);
    assert!(l.warnings.is_empty());
    let k: Vec<_> = l.tokens.iter().map(|t| t.kind).collect();
    let count = |kind| k.iter().filter(|&&x| x == kind).count();
    assert_eq!(count(Indent), 4);
    assert_eq!(count(Dedent), 4);
}
