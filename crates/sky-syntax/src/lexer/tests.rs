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
    // Inside a string too, not "unterminated string".
    assert_err("s = \"a\rb\"\n", LexErrorKind::BareCarriageReturn, "\r");
    assert_err("s = \"a\r", LexErrorKind::BareCarriageReturn, "\r");
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
fn compound_assignment_operators() {
    assert_eq!(
        texts("+= -= *= /= ^= @=\n"),
        [
            (PlusAssign, "+="),
            (MinusAssign, "-="),
            (StarAssign, "*="),
            (SlashAssign, "/="),
            (CaretAssign, "^="),
            (AtAssign, "@="),
            (Newline, "\n"),
            (Eof, ""),
        ]
    );
    // Greedy: the two characters must touch.
    assert_eq!(line("x+=1\n"), [Ident, PlusAssign, Int]);
    assert_eq!(line("x^=2\n"), [Ident, CaretAssign, Int]);
    assert_eq!(line("A@=B\n"), [Ident, AtAssign, Ident]);
    assert_eq!(line("a+ =b\n"), [Ident, Plus, Assign, Ident]);
    assert_eq!(line("a + = b\n"), [Ident, Plus, Assign, Ident]);
    // `=` binds to the operator before it, never to a following `=`.
    assert_eq!(line("a+==b\n"), [Ident, PlusAssign, Assign, Ident]);
    assert_eq!(line("a=+b\n"), [Ident, Assign, Plus, Ident]);
    assert_eq!(line("a==-b\n"), [Ident, Eq, Minus, Ident]);
    // `->` still wins over `-=` where both could start.
    assert_eq!(line("f(x)->y\n")[4], Arrow);
    assert_eq!(line("a-=-b\n"), [Ident, MinusAssign, Minus, Ident]);
    // `**=` is `*` then `*=`; the parser reports it like `**`.
    assert_eq!(line("x**=2\n"), [Ident, Star, StarAssign, Int]);
}

#[test]
fn glyph_comparisons_map_to_ascii_kinds() {
    assert_eq!(line("a ≤ b"), line("a <= b"));
    assert_eq!(line("a ≥ b"), line("a >= b"));
    assert_eq!(line("a ≠ b"), line("a != b"));
}

/// `+-` is `a + (-b)` as in Python, and not an uncertainty.
#[test]
fn plus_dash_is_plus_then_minus() {
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
}

/// `±` and `+/-` are the same token (spec "Uncertainty").
#[test]
fn plus_minus_glyph_and_ascii_spelling() {
    assert_eq!(
        texts("m = 5.31 ± 0.02 kg\n")[..6],
        [
            (Ident, "m"),
            (Assign, "="),
            (Float, "5.31"),
            (PlusMinus, "±"),
            (Float, "0.02"),
            (Ident, "kg"),
        ]
    );
    assert_eq!(line("5.31 +/- 0.02 kg\n"), line("5.31 ± 0.02 kg\n"));
    assert_eq!(texts("a+/-b\n")[1], (PlusMinus, "+/-"));
    assert_eq!(line("a±b\n"), [Ident, PlusMinus, Ident]);
    // The ASCII spelling is greedy, but only for exactly `+/-`.
    assert_eq!(line("a +/-= b\n"), [Ident, PlusMinus, Assign, Ident]);
    assert_eq!(line("a += b\n"), [Ident, PlusAssign, Ident]);
    assert_eq!(line("a + / - b\n"), [Ident, Plus, Slash, Minus, Ident]);
    // `±` is a single token of two bytes.
    assert_eq!(lexed("±\n").tokens[0].span.range(), 0..2);
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
fn transpose_glyph_is_a_postfix_operator_not_an_identifier_letter() {
    // U+1D40 is a letter (XID_Continue), so it must be cut out of
    // identifiers or `Aᵀ` would be one (non-NFKC, hence rejected) name.
    assert_eq!(line("Aᵀ\n"), [Ident, Transpose]);
    assert_eq!(line("Aᵀᵀ\n"), [Ident, Transpose, Transpose]);
    assert_eq!(line("(A @ B)ᵀ\n").last(), Some(&Transpose));
    assert_eq!(line("A⁻¹ᵀ\n"), [Ident, Inverse, Transpose]);
    assert_eq!(line("xᵀy\n"), [Ident, Transpose, Ident]);
    assert_eq!(texts("ᵀ\n")[0], (Transpose, "ᵀ"));
    // Other modifier letters stay identifier characters, so `Aᴬ` is still
    // rejected as non-NFKC rather than read as an operator.
    assert_err("Aᴬ\n", LexErrorKind::NonNfkcIdentifier, "Aᴬ");
}

#[test]
fn transpose_glyph_is_not_nfkc_stable_like_the_inverse_glyph() {
    // The glyph is a compatibility character: NFKC maps it to a plain `T`,
    // so a tool that normalizes source would silently turn `Aᵀ` into the
    // name `AT`. The lexer never normalizes, and, as with `⁻¹` (NFKC `-1`),
    // the ASCII spelling `transpose(A)` is immune.
    assert_eq!(crate::nfkc::nfkc("ᵀ"), "T");
    assert!(!crate::nfkc::is_nfkc("ᵀ"));
    assert!(!is_ident_start('ᵀ') && !is_ident_continue('ᵀ'));
}

#[test]
fn look_alike_transpose_marks_suggest_the_glyph() {
    for src in ["A⊤\n", "A⊺\n"] {
        let e = assert_err(src, LexErrorKind::UnexpectedChar, &src[1..src.len() - 1]);
        assert!(e.diagnostic.help[0].contains("transpose"), "{src}");
    }
}

#[test]
fn lone_superscript_minus_and_bang_are_errors() {
    let e = assert_err("A⁻ @ b\n", LexErrorKind::UnexpectedChar, "⁻");
    assert!(e.diagnostic.help[0].contains("⁻¹"));
    assert_err("a ! b\n", LexErrorKind::UnexpectedChar, "!");
    assert_err("x = A¹\n", LexErrorKind::UnexpectedChar, "¹");
}

#[test]
fn c_style_logical_operators_suggest_keywords() {
    for (src, at, word) in [
        ("x = !a\n", "!", "`not`"),
        ("x = a && b\n", "&", "`and`"),
        ("x = a || b\n", "|", "`or`"),
        ("x = a & b\n", "&", "`and`"),
        ("x = a | b\n", "|", "`or`"),
    ] {
        let e = assert_err(src, LexErrorKind::UnexpectedChar, at);
        assert!(
            e.diagnostic.help[0].contains(word),
            "{src:?}: {:?}",
            e.diagnostic.help
        );
    }
    // `!=` is still "not equal".
    assert_eq!(line("x = a != b\n"), [Ident, Assign, Ident, Ne, Ident]);
}

#[test]
fn bool_literals_are_lowercase_keywords() {
    assert_eq!(
        line("x = true or false\n"),
        [
            Ident,
            Assign,
            Keyword(Kw::True),
            Keyword(Kw::Or),
            Keyword(Kw::False),
        ]
    );
    // Python's spellings are ordinary identifiers, as are longer words.
    assert_eq!(
        line("x = True + False + truth + falsey\n"),
        [Ident, Assign, Ident, Plus, Ident, Plus, Ident, Plus, Ident]
    );
}

#[test]
fn logical_keywords_are_whole_words_only() {
    assert_eq!(
        line("x = a and not b or c\n"),
        [
            Ident,
            Assign,
            Ident,
            Keyword(Kw::And),
            Keyword(Kw::Not),
            Ident,
            Keyword(Kw::Or),
            Ident,
        ]
    );
    // Longer identifiers that start with a keyword stay identifiers.
    assert_eq!(
        line("android = order + nothing + and_x\n"),
        [Ident, Assign, Ident, Plus, Ident, Plus, Ident]
    );
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
    let src = "def return if elif else while for in as import pass break continue equation solve field and or not true false\n";
    let expected: Vec<_> = [
        Kw::Def,
        Kw::Return,
        Kw::If,
        Kw::Elif,
        Kw::Else,
        Kw::While,
        Kw::For,
        Kw::In,
        Kw::As,
        Kw::Import,
        Kw::Pass,
        Kw::Break,
        Kw::Continue,
        Kw::Equation,
        Kw::Solve,
        Kw::Field,
        Kw::And,
        Kw::Or,
        Kw::Not,
        Kw::True,
        Kw::False,
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
    // Computed from the tables, so it needs no hand-updating, but the loop
    // above cannot silently check nothing. The three sets are disjoint
    // (`invisible_tables_are_well_formed`, and bidi controls are neither
    // XID nor in `INVISIBLE_EXTRA`).
    let expected = table_len(DEFAULT_IGNORABLE_IDENT)
        + table_len(INVISIBLE_EXTRA)
        + crate::source::BIDI_CONTROLS.len();
    assert!(expected > 0);
    assert_eq!(rejected, expected);
}

fn table_len(ranges: &[(u32, u32)]) -> usize {
    ranges.iter().map(|&(lo, hi)| (hi - lo + 1) as usize).sum()
}

#[test]
fn invisible_tables_are_well_formed() {
    assert!(!DEFAULT_IGNORABLE_IDENT.is_empty());
    for table in [DEFAULT_IGNORABLE_IDENT, INVISIBLE_EXTRA] {
        for &(lo, hi) in table {
            assert!(lo <= hi && hi <= 0x10FFFF, "{lo:X}..{hi:X}");
        }
        for w in table.windows(2) {
            assert!(w[0].1 + 1 < w[1].0, "unsorted or adjacent: {w:X?}");
        }
    }
    // The extras are exactly the default-ignorables that are not XID, so
    // none may overlap the generated table.
    for c in (0..=0x10FFFFu32).filter_map(char::from_u32) {
        assert!(
            !(in_ranges(DEFAULT_IGNORABLE_IDENT, c) && in_ranges(INVISIBLE_EXTRA, c)),
            "U+{:04X} is in both tables",
            c as u32
        );
        if in_ranges(INVISIBLE_EXTRA, c) {
            assert!(
                !unicode_ident::is_xid_start(c) && !unicode_ident::is_xid_continue(c),
                "U+{:04X} is XID: it belongs in the generated table",
                c as u32
            );
        }
    }
    // ZWNJ, ZWJ, and the Hangul fillers come from the generated table (they
    // are Default_Ignorable_Code_Point and XID in Unicode 18.0.0), not from
    // an explicit list.
    for c in [
        '\u{200C}', '\u{200D}', '\u{115F}', '\u{1160}', '\u{3164}', '\u{FFA0}',
    ] {
        assert!(in_ranges(DEFAULT_IGNORABLE_IDENT, c), "U+{:04X}", c as u32);
    }
}

/// Everything the old hand-typed `is_invisible` list (#11 through #16)
/// rejected is still invisible, so no character is newly accepted.
#[test]
fn previously_invisible_characters_stay_invisible() {
    let old = [
        0x00AD..=0x00AD,
        0x034F..=0x034F,
        0x115F..=0x1160,
        0x17B4..=0x17B5,
        0x180B..=0x180F,
        0x200B..=0x200D,
        0x2060..=0x2064,
        0x3164..=0x3164,
        0xFE00..=0xFE0F,
        0xFEFF..=0xFEFF,
        0xFFA0..=0xFFA0,
    ];
    for cp in old.into_iter().flatten() {
        assert!(is_invisible(char::from_u32(cp).unwrap()), "U+{cp:04X}");
    }
}

/// Default_Ignorable_Code_Point ranges from the verbatim UCD records the
/// generator also wrote out, parsed here independently of the generated
/// intersection.
fn default_ignorable_from_ucd() -> Vec<(u32, u32)> {
    const SOURCE: &str = include_str!("../../tests/data/ucd-default-ignorable.txt");
    let (a, b, c) = sky_diag::DISPLAY_UNSAFE_UNICODE_VERSION;
    assert!(
        SOURCE
            .lines()
            .next()
            .unwrap()
            .contains(&format!("Unicode {a}.{b}.{c} UCD")),
        "sky-syntax and sky-diag tables must come from the same Unicode version"
    );
    let out: Vec<(u32, u32)> = SOURCE
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let (cps, prop) = l.split('#').next().unwrap().split_once(';').unwrap();
            assert_eq!(prop.trim(), "Default_Ignorable_Code_Point");
            let cps = cps.trim();
            let (lo, hi) = cps.split_once("..").unwrap_or((cps, cps));
            (
                u32::from_str_radix(lo, 16).unwrap(),
                u32::from_str_radix(hi, 16).unwrap(),
            )
        })
        .collect();
    assert!(!out.is_empty());
    out
}

/// Every default-ignorable code point that `unicode-ident` counts as
/// XID_Start or XID_Continue is rejected by `lex` at the start, middle, and
/// end of an identifier (at the start of a line and after `=`), with the
/// error pointing at exactly that character. Every other default-ignorable
/// code point is rejected there too (as a bidi control or an unexpected
/// character).
#[test]
fn default_ignorable_identifier_characters_are_rejected_everywhere() {
    let di = default_ignorable_from_ucd();
    let mut xid_count = 0;
    for c in (0..=0x10FFFFu32).filter_map(char::from_u32) {
        if !di.iter().any(|&(lo, hi)| lo <= c as u32 && c as u32 <= hi) {
            continue;
        }
        let xid = unicode_ident::is_xid_start(c) || unicode_ident::is_xid_continue(c);
        xid_count += usize::from(xid);
        let cases = [
            (format!("{c}ab = 1\n"), 0),
            (format!("a{c}b = 1\n"), 1),
            (format!("ab{c} = 1\n"), 2),
            (format!("x = {c}ab\n"), 4),
            (format!("x = a{c}b\n"), 5),
            (format!("x = ab{c}\n"), 6),
        ];
        for (src, at) in cases {
            if c == '\u{FEFF}' && at == 0 {
                continue; // a leading BOM is skipped by design
            }
            let e = lex(&src).expect_err(&format!("U+{:04X} in {src:?}", c as u32));
            if xid {
                assert_eq!(e.kind, LexErrorKind::UnexpectedChar, "U+{:04X}", c as u32);
                let span = e.diagnostic.primary.span;
                assert_eq!(span.start as usize, at, "U+{:04X} in {src:?}", c as u32);
                assert_eq!(span.len() as usize, c.len_utf8());
                assert_eq!(
                    e.diagnostic.help,
                    ["invisible characters are not allowed in source code"]
                );
            }
        }
    }
    // Cross-check: `unicode-ident`'s XID data agrees with the generated
    // table on which default-ignorables are identifier characters.
    assert!(xid_count > 0);
    assert_eq!(xid_count, table_len(DEFAULT_IGNORABLE_IDENT));
}

#[test]
fn supplementary_variation_selector_does_not_make_a_second_variable() {
    // `ab` and `a\u{E0100}b` render the same; the second must not lex as a
    // distinct identifier.
    let e = assert_err(
        "ab = 1\na\u{E0100}b = 2\n",
        LexErrorKind::UnexpectedChar,
        "\u{E0100}",
    );
    assert_eq!(e.diagnostic.message, "unexpected character U+E0100");
    assert_eq!(
        e.diagnostic.help,
        ["invisible characters are not allowed in source code"]
    );
    assert_err(
        "ab\u{E01EF} = 1\n",
        LexErrorKind::UnexpectedChar,
        "\u{E01EF}",
    );
    assert_err("a\u{FE0F}b = 1\n", LexErrorKind::UnexpectedChar, "\u{FE0F}");
}

#[test]
fn zero_width_joiner_and_non_joiner_are_rejected_in_identifiers() {
    for c in ['\u{200C}', '\u{200D}'] {
        let s = c.to_string();
        let e = assert_err(&format!("ab{c}c = 1\n"), LexErrorKind::UnexpectedChar, &s);
        assert_eq!(
            e.diagnostic.message,
            format!("unexpected character U+{:04X}", c as u32)
        );
        assert_err(&format!("ab{c} = 1\n"), LexErrorKind::UnexpectedChar, &s);
        assert_err(&format!("{c}ab = 1\n"), LexErrorKind::UnexpectedChar, &s);
    }
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
    // (MICRO SIGN `µm` used to warn here; it is now a non-NFKC error, see
    // `micro_kelvin_and_planck_signs_name_the_intended_letter`.)
    let l = lexed("х = 1\nxЖ = 2\n");
    let msgs: Vec<_> = l.warnings.iter().map(|w| w.message.as_str()).collect();
    assert_eq!(
        msgs,
        [
            "identifier `х` contains Cyrillic 'х' (U+0445), which looks like Latin 'x'",
            "identifier `xЖ` mixes Latin and Cyrillic scripts",
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

// ---- NFKC identifiers -----------------------------------------------------

/// Asserts `src` fails with a non-NFKC error on exactly `ident`, and returns
/// the error.
fn assert_non_nfkc(src: &str, ident: &str) -> LexError {
    let e = assert_err(src, LexErrorKind::NonNfkcIdentifier, ident);
    assert_eq!(
        e.diagnostic.message,
        "identifier is not in Unicode NFKC form"
    );
    assert_eq!(
        e.diagnostic.primary.message.as_deref(),
        Some("not in NFKC form")
    );
    assert_eq!(
        e.diagnostic.notes,
        [format!(
            "`{}` is not in NFKC (Unicode Normalization Form KC), and Sky never normalizes identifiers",
            escape_non_ascii(ident)
        )]
    );
    e
}

#[test]
fn decomposed_identifier_is_rejected_not_normalized() {
    // `é` precomposed is fine; `e` + COMBINING ACUTE ACCENT looks the same
    // but would be a different name, so it is an error.
    let src = "\u{00E9} = 1\ne\u{0301} = 2\n";
    let e = assert_non_nfkc(src, "e\u{0301}");
    assert_eq!(e.diagnostic.primary.span, Span::new(7, 10));
    assert_eq!(e.diagnostic.help, ["write it as `\u{00E9}` (`\\u{00E9}`)"]);
    assert_eq!(lexed("\u{00E9} = 1\n").tokens[0].kind, Ident);
}

#[test]
fn compatibility_variants_are_rejected() {
    for (src, ident, nf) in [
        ("\u{FB01}le = 1\n", "\u{FB01}le", "file"), // LATIN SMALL LIGATURE FI
        ("\u{FF58} = 1\n", "\u{FF58}", "x"),        // FULLWIDTH x
        ("\u{1D431} = 1\n", "\u{1D431}", "x"),      // MATHEMATICAL BOLD SMALL X
        ("\u{1D465} = 1\n", "\u{1D465}", "x"),      // MATHEMATICAL ITALIC SMALL X
        ("y = a\u{1D465}\n", "a\u{1D465}", "ax"),
    ] {
        let e = assert_non_nfkc(src, ident);
        assert_eq!(
            e.diagnostic.help.last().unwrap(),
            &format!("write it as `{nf}`")
        );
    }
}

#[test]
fn keywords_match_raw_text_only() {
    // Fullwidth `ｉｆ` is not the keyword `if`: it is an error whose help
    // names the keyword.
    let e = assert_non_nfkc("\u{FF49}\u{FF46} x:\n", "\u{FF49}\u{FF46}");
    assert_eq!(e.diagnostic.help, ["did you mean `if`"]);
    let e = assert_non_nfkc("y = \u{FF44}ef\n", "\u{FF44}ef");
    assert_eq!(e.diagnostic.help, ["did you mean `def`"]);
    let e = assert_non_nfkc(
        "x = \u{FF54}\u{FF52}\u{FF55}\u{FF45}\n",
        "\u{FF54}\u{FF52}\u{FF55}\u{FF45}",
    );
    assert_eq!(e.diagnostic.help, ["did you mean `true`"]);
    let e = assert_non_nfkc("x = \u{FF46}alse\n", "\u{FF46}alse");
    assert_eq!(e.diagnostic.help, ["did you mean `false`"]);
    // The raw keyword is still a keyword.
    assert_eq!(line("x = true\n")[2], Keyword(Kw::True));
}

#[test]
fn micro_kelvin_and_planck_signs_name_the_intended_letter() {
    let e = assert_non_nfkc("size = 5 \u{00B5}m\n", "\u{00B5}m");
    assert_eq!(
        e.diagnostic.help,
        [
            "MICRO SIGN U+00B5 is not a letter for identifiers; use GREEK SMALL LETTER MU U+03BC (`\u{03BC}`)",
            "write it as `\u{03BC}m` (`\\u{03BC}m`)",
        ]
    );
    let e = assert_non_nfkc("T = 300 \u{212A}\n", "\u{212A}");
    assert_eq!(
        e.diagnostic.help,
        [
            "KELVIN SIGN U+212A is not a letter for identifiers; use LATIN CAPITAL LETTER K U+004B (`K`)",
            "write it as `K`",
        ]
    );
    let e = assert_non_nfkc("E = \u{210F} * w\n", "\u{210F}");
    assert_eq!(
        e.diagnostic.help,
        [
            "PLANCK CONSTANT OVER TWO PI U+210F is not a letter for identifiers; use LATIN SMALL LETTER H WITH STROKE U+0127 (`\u{0127}`)",
            "write it as `\u{0127}` (`\\u{0127}`)",
        ]
    );
    // The intended letters themselves are fine.
    assert!(lexed("size = 5 \u{03BC}m\nT = 300 K\nE = \u{0127} * w\n")
        .warnings
        .is_empty());
}

#[test]
fn nfkc_maybe_characters_are_compared_in_full() {
    // COMBINING GRAVE ACCENT is NFKC_QC=Maybe: after `a` it composes to `à`,
    // after `x` (no precomposed form) it is fine.
    let e = assert_non_nfkc("a\u{0300} = 1\n", "a\u{0300}");
    assert_eq!(e.diagnostic.help, ["write it as `\u{00E0}` (`\\u{00E0}`)"]);
    assert_eq!(lexed("x\u{0300} = 1\n").tokens[0].kind, Ident);
    // Hangul jamo compose into a syllable.
    let e = assert_non_nfkc("\u{1100}\u{1161} = 1\n", "\u{1100}\u{1161}");
    assert_eq!(e.diagnostic.help, ["write it as `\u{AC00}` (`\\u{AC00}`)"]);
    assert_eq!(lexed("\u{AC00} = 1\n").tokens[0].kind, Ident);
}

#[test]
fn non_xid_compatibility_characters_are_invalid_characters_not_nfkc_errors() {
    // `½` is NFKC_QC=No but not XID_Continue, so it ends the identifier `x`
    // and is an unexpected character.
    let e = assert_err("y = x\u{00BD}\n", LexErrorKind::UnexpectedChar, "\u{00BD}");
    assert_eq!(
        e.diagnostic.message,
        "unexpected character '\u{00BD}' (U+00BD)"
    );
    // `ŀ`'s NFKC form contains the MIDDLE DOT, so it is not an identifier
    // character at all.
    assert_err("\u{0140} = 1\n", LexErrorKind::UnexpectedChar, "\u{0140}");
    assert_err("a\u{013F}b = 1\n", LexErrorKind::UnexpectedChar, "\u{013F}");
}

#[test]
fn escape_non_ascii_matches_the_renderer_escape() {
    // Same `\u{XXXX}` spelling as sky_diag's escaping of unsafe characters.
    for c in ['\u{200D}', '\u{E0100}', '\u{00AD}'] {
        assert_eq!(
            escape_non_ascii(&c.to_string()),
            sky_diag::display_safe(&c.to_string())
        );
    }
    assert_eq!(escape_non_ascii("e\u{0301}x"), "e\\u{0301}x");
}

/// Every NFKC_QC=No code point is rejected inside an identifier: as a
/// non-NFKC identifier when the lexer accepts it as an identifier
/// character, and otherwise as an unexpected character.
#[test]
fn every_nfkc_qc_no_code_point_is_rejected_in_identifiers() {
    let mut as_ident = 0;
    let mut as_unexpected = 0;
    for &(lo, hi) in crate::nfkc_table::NFKC_QC_NO {
        for c in (lo..=hi).filter_map(char::from_u32) {
            let src = format!("a{c}b = 1\n");
            if c == crate::lexer::TRANSPOSE_GLYPH {
                // The one exception: it is the transpose operator, so it
                // ends the identifier instead of joining it.
                let kinds: Vec<_> = lex(&src).unwrap().tokens.iter().map(|t| t.kind).collect();
                assert_eq!(kinds[..3], [Ident, Transpose, Ident]);
                // Counted with the characters that never join an identifier.
                as_unexpected += 1;
                continue;
            }
            let e = lex(&src).expect_err(&format!("U+{:04X}", c as u32));
            if is_ident_continue(c) {
                assert_eq!(
                    e.kind,
                    LexErrorKind::NonNfkcIdentifier,
                    "U+{:04X}",
                    c as u32
                );
                assert_eq!(e.diagnostic.primary.span.range(), 0..2 + c.len_utf8());
                as_ident += 1;
            } else {
                assert_eq!(e.kind, LexErrorKind::UnexpectedChar, "U+{:04X}", c as u32);
                assert_eq!(e.diagnostic.primary.span.range(), 1..1 + c.len_utf8());
                as_unexpected += 1;
            }
            // At the start of an identifier too.
            let e = lex(&format!("{c}b = 1\n")).expect_err(&format!("U+{:04X}", c as u32));
            if is_ident_start(c) {
                assert_eq!(
                    e.kind,
                    LexErrorKind::NonNfkcIdentifier,
                    "U+{:04X}",
                    c as u32
                );
            } else {
                assert_eq!(e.kind, LexErrorKind::UnexpectedChar, "U+{:04X}", c as u32);
            }
        }
    }
    assert!(as_ident > 0 && as_unexpected > 0);
    let total: u32 = crate::nfkc_table::NFKC_QC_NO
        .iter()
        .map(|&(lo, hi)| hi - lo + 1)
        .sum();
    assert_eq!((as_ident + as_unexpected) as u32, total);
}

mod nfkc_properties {
    use super::*;
    use proptest::prelude::*;
    use std::sync::OnceLock;

    /// All identifier-start and identifier-continue characters, plus a few
    /// that exercise composition (combining marks, Hangul jamo) and
    /// compatibility variants.
    fn pools() -> &'static (Vec<char>, Vec<char>) {
        static POOLS: OnceLock<(Vec<char>, Vec<char>)> = OnceLock::new();
        POOLS.get_or_init(|| {
            let all: Vec<char> = (0..=0x10FFFFu32).filter_map(char::from_u32).collect();
            let starts = all.iter().copied().filter(|&c| is_ident_start(c)).collect();
            let conts = all
                .iter()
                .copied()
                .filter(|&c| is_ident_continue(c))
                .collect();
            (starts, conts)
        })
    }

    const INTERESTING: &[char] = &[
        'a',
        'e',
        'x',
        'K',
        '\u{0300}',
        '\u{0301}',
        '\u{0323}',
        '\u{0340}',
        '\u{00E9}',
        '\u{1100}',
        '\u{1161}',
        '\u{11A8}',
        '\u{AC00}',
        '\u{FB01}',
        '\u{FF58}',
        '\u{1D431}',
        '\u{00B5}',
        '\u{212A}',
        '\u{210F}',
        '\u{2126}',
        '\u{3099}',
        '\u{304B}',
        '\u{0F73}',
        '\u{1E0A}',
        '\u{0CCA}',
        '_',
        '1',
    ];

    fn start_char() -> impl Strategy<Value = char> {
        prop_oneof![
            (0..pools().0.len()).prop_map(|i| pools().0[i]),
            prop::sample::select(INTERESTING).prop_filter("start", |&c| is_ident_start(c)),
        ]
    }

    fn continue_char() -> impl Strategy<Value = char> {
        prop_oneof![
            (0..pools().1.len()).prop_map(|i| pools().1[i]),
            prop::sample::select(INTERESTING),
        ]
    }

    fn identifier() -> impl Strategy<Value = String> {
        (start_char(), prop::collection::vec(continue_char(), 0..8)).prop_map(|(s, rest)| {
            let mut out = String::from(s);
            out.extend(rest);
            out
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        /// The NFKC form of any identifier lexes as exactly one identifier
        /// (or keyword) token, and the identifier itself is accepted exactly
        /// when it is already in NFKC.
        #[test]
        fn nfkc_form_of_an_identifier_is_one_identifier(id in identifier()) {
            prop_assume!(is_identifier(&id));
            let nf = nfkc::nfkc(&id);
            prop_assert_eq!(nfkc::nfkc(&nf), nf.clone(), "NFKC is idempotent");
            let l = lex(&nf).map_err(|e| TestCaseError::fail(format!("{nf:?}: {:?}", e.diagnostic)))?;
            let kinds: Vec<_> = l.tokens.iter().map(|t| t.kind).collect();
            prop_assert!(
                matches!(kinds[0], Ident | Keyword(_)) && kinds[1..] == [Newline, Eof],
                "{:?} lexes as {:?}", nf, kinds
            );
            prop_assert_eq!(l.tokens[0].span.range(), 0..nf.len());
            match lex(&id) {
                Ok(_) => prop_assert!(nfkc::is_nfkc(&id)),
                Err(e) => {
                    prop_assert_eq!(e.kind, LexErrorKind::NonNfkcIdentifier);
                    prop_assert!(!nfkc::is_nfkc(&id));
                    prop_assert_eq!(e.diagnostic.primary.span.range(), 0..id.len());
                }
            }
        }
    }
}

/// The lexer's `NFKC_NON_IDENTIFIER` exceptions are computed here from the
/// Unicode data and the lexer's own banned-character rules: every
/// identifier character (before the exceptions) whose NFKC form contains a
/// banned character or anything else that is not an identifier character.
/// Exactly those are rejected, as unexpected characters.
#[test]
fn nfkc_non_identifier_list_is_exactly_the_computed_set() {
    let mut computed = Vec::new();
    for c in (0..=0x10FFFFu32).filter_map(char::from_u32) {
        if !is_ident_continue_base(c) && !is_ident_start_base(c) {
            continue;
        }
        let n = crate::nfkc::nfkc(&c.to_string());
        let mut chars = n.chars();
        let ok = if is_ident_start_base(c) {
            chars.next().is_some_and(is_ident_start_base) && chars.all(is_ident_continue_base)
        } else {
            n.chars().all(is_ident_continue_base)
        };
        if !ok {
            computed.push(c);
        }
    }
    assert_eq!(
        computed, NFKC_NON_IDENTIFIER,
        "the identifier characters whose NFKC form is not an identifier changed \
         (Unicode data or the banned set changed): review NFKC_NON_IDENTIFIER in \
         crates/sky-syntax/src/lexer.rs and update it to the computed list"
    );
    // The exceptions are not NFKC themselves, so they cannot appear in the
    // NFKC form of another character (no fixpoint needed).
    for &c in NFKC_NON_IDENTIFIER {
        assert!(!crate::nfkc::is_nfkc(&c.to_string()), "U+{:04X}", c as u32);
        for src in [
            format!("{c}x = 1\n"),
            format!("a{c}b = 1\n"),
            format!("y = x{c}\n"),
        ] {
            let s = c.to_string();
            let e = assert_err(&src, LexErrorKind::UnexpectedChar, &s);
            assert_eq!(
                e.diagnostic.help,
                ["its NFKC form is not a valid identifier (`ŀ` is `l·`, and `·` reads as multiplication)"]
            );
        }
    }
}

/// The span and the carets are the raw source bytes of the identifier, not
/// offsets or widths of its (longer or shorter) NFKC form.
#[test]
fn non_nfkc_carets_point_at_the_raw_source() {
    // `ﬁ` is 3 bytes and 1 column; its NFKC `fi` is 2 bytes and 2 columns.
    // U+FDF8 is 3 bytes and 1 column; its NFKC form is 4 Arabic letters.
    for (src, start, ident, carets) in [
        ("x = \u{FB01}le + 1\n", 4, "\u{FB01}le", "    ^^^"),
        ("y = 2\nz = a\u{FDF8}b\n", 10, "a\u{FDF8}b", "    ^^^"),
        (
            "q = e\u{0301}\u{0323}x\n",
            4,
            "e\u{0301}\u{0323}x",
            "    ^^^^",
        ),
    ] {
        let e = assert_non_nfkc(src, ident);
        let span = e.diagnostic.primary.span;
        assert_eq!(span.range(), start..start + ident.len(), "{src:?}");
        let file = sky_diag::SourceFile::new("t.sky", src);
        let out = e.diagnostic.render_to_string(&file);
        let caret_line = out
            .lines()
            .find(|l| l.contains('^'))
            .unwrap()
            .split_once("| ")
            .unwrap()
            .1;
        assert_eq!(caret_line, format!("{carets} not in NFKC form"), "{out}");
    }
}

/// An identifier that is both non-NFKC and next to an invisible character
/// produces exactly one error: whichever comes first in the source, and no
/// warnings.
#[test]
fn non_nfkc_and_invisible_character_give_one_error() {
    for (src, kind, at) in [
        (
            "e\u{0301}\u{200D}x = 1\n",
            LexErrorKind::NonNfkcIdentifier,
            "e\u{0301}",
        ),
        (
            "\u{FB01}\u{FE0F} = 1\n",
            LexErrorKind::NonNfkcIdentifier,
            "\u{FB01}",
        ),
        (
            "a\u{200D}\u{FB01} = 1\n",
            LexErrorKind::UnexpectedChar,
            "\u{200D}",
        ),
        (
            "x = \u{E0100}\u{00B5}\n",
            LexErrorKind::UnexpectedChar,
            "\u{E0100}",
        ),
    ] {
        let e = assert_err(src, kind, at);
        assert!(e.warnings.is_empty(), "{src:?}: {:?}", e.warnings);
        assert_eq!(e.diagnostics().count(), 1, "{src:?}");
    }
}

#[test]
fn micro_sign_in_a_unit_gets_the_micro_sign_help() {
    let e = assert_non_nfkc("d = 10 \u{00B5}m\n", "\u{00B5}m");
    assert_eq!(
        e.diagnostic.help,
        [
            "MICRO SIGN U+00B5 is not a letter for identifiers; use GREEK SMALL LETTER MU U+03BC (`\u{03BC}`)",
            "write it as `\u{03BC}m` (`\\u{03BC}m`)",
        ]
    );
    // Compound units too: the unit name is still an identifier token.
    assert_non_nfkc("v = 3 \u{00B5}m/s\n", "\u{00B5}m");
}

/// A non-NFKC identifier as long as the source limit allows gets a small
/// diagnostic: an escaped prefix and the byte length, no NFKC form (so no
/// normalization of 16 MiB) and no keyword check.
#[test]
fn huge_non_nfkc_identifier_has_a_bounded_diagnostic() {
    let mut src = String::from("a");
    while src.len() + 4 < crate::MAX_SOURCE_BYTES {
        src.push_str("\u{0301}\u{0316}");
    }
    src.push('\n');
    assert!(src.len() <= crate::MAX_SOURCE_BYTES);
    assert!(src.len() > crate::MAX_SOURCE_BYTES - 5);
    let e = lex(&src).unwrap_err();
    assert_eq!(e.kind, LexErrorKind::NonNfkcIdentifier);
    assert_eq!(e.diagnostic.primary.span.range(), 0..src.len() - 1);
    let text: usize = e
        .diagnostic
        .notes
        .iter()
        .chain(&e.diagnostic.help)
        .map(String::len)
        .sum();
    assert!(text < 1024, "{text} bytes of note and help");
    assert!(e.diagnostic.help.is_empty(), "{:?}", e.diagnostic.help);
    assert!(e.diagnostic.notes[0].starts_with("`a\\u{0301}\\u{0316}"));
    assert!(e.diagnostic.notes[0].contains(&format!("…` ({} bytes) is not in NFKC", src.len() - 1)));
}

/// 64 characters still get the full help; 65 get the short form.
#[test]
fn full_nfkc_help_stops_at_64_characters() {
    let at_limit = format!("\u{FB01}{}", "a".repeat(FULL_NFKC_HELP_MAX_CHARS - 1));
    let e = lex(&at_limit).unwrap_err();
    assert_eq!(
        e.diagnostic.help,
        [format!(
            "write it as `fi{}`",
            "a".repeat(FULL_NFKC_HELP_MAX_CHARS - 1)
        )]
    );
    let over = format!("{at_limit}a");
    let e = lex(&over).unwrap_err();
    assert_eq!(e.kind, LexErrorKind::NonNfkcIdentifier);
    assert!(e.diagnostic.help.is_empty());
    assert_eq!(
        e.diagnostic.notes[0],
        format!(
            "`\\u{{FB01}}{}…` ({} bytes) is not in NFKC (Unicode Normalization Form KC), and Sky never normalizes identifiers",
            "a".repeat(FULL_NFKC_HELP_MAX_CHARS - 1),
            over.len()
        )
    );
    // The compatibility-letter help is short and still given.
    let micro = format!("\u{00B5}{}", "m".repeat(FULL_NFKC_HELP_MAX_CHARS));
    let e = lex(&micro).unwrap_err();
    assert_eq!(e.diagnostic.help.len(), 1);
    assert!(e.diagnostic.help[0].starts_with("MICRO SIGN U+00B5"));
}

#[test]
fn write_as_help_shows_characters_and_escapes() {
    assert_eq!(
        write_as_help("\u{03BC}m"),
        "write it as `\u{03BC}m` (`\\u{03BC}m`)"
    );
    assert_eq!(write_as_help("fi"), "write it as `fi`");
    // Anything the renderer would escape is shown only as escapes (not
    // reachable from `lex` with the pinned data: the NFKC form of an
    // identifier has no invisible characters).
    assert_eq!(
        write_as_help("a\u{200D}b"),
        "write it in NFKC form: `a\\u{200D}b`"
    );
}

/// More than 30 combining marks in a row (after decomposition) is an error
/// at the 31st, with a constant-size message, found before any
/// normalization, even in a 16 MiB identifier.
#[test]
fn too_many_combining_marks() {
    let ok = format!("x{} = 1\n", "\u{0300}".repeat(30));
    assert!(lex(&ok).is_ok());
    let bad = format!("y = x{} + 1\n", "\u{0300}".repeat(31));
    let e = lex(&bad).unwrap_err();
    assert_eq!(e.kind, LexErrorKind::TooManyCombiningMarks);
    let at = "y = x".len() + 30 * 2;
    assert_eq!(e.diagnostic.primary.span.range(), at..at + 2);
    assert_eq!(
        e.diagnostic.message,
        "too many combining marks in a row (at most 30)"
    );
    // U+0344 is two marks after decomposition.
    let e = lex(&format!("x{}\u{0344}", "\u{0316}".repeat(29))).unwrap_err();
    assert_eq!(e.kind, LexErrorKind::TooManyCombiningMarks);
    assert_eq!(
        e.diagnostic.primary.span.range(),
        1 + 29 * 2..1 + 29 * 2 + 2
    );
    // A 16 MiB run: same message, same size, no normalization.
    let mut huge = String::from("x");
    while huge.len() + 2 < crate::MAX_SOURCE_BYTES {
        huge.push('\u{0300}');
    }
    huge.push('\n');
    let big = lex(&huge).unwrap_err();
    let small = lex(&bad).unwrap_err();
    assert_eq!(big.kind, LexErrorKind::TooManyCombiningMarks);
    assert_eq!(big.diagnostic.primary.span.range(), 1 + 60..1 + 62);
    assert_eq!(big.diagnostic.message, small.diagnostic.message);
    assert_eq!(big.diagnostic.notes, small.diagnostic.notes);
    assert_eq!(big.diagnostic.help, small.diagnostic.help);
}

/// The escaped prefix of a long non-NFKC identifier does not end between a
/// letter and its combining marks: it stops before the letter.
#[test]
fn long_identifier_prefix_keeps_marks_with_their_letter() {
    // Characters 0..63: U+FB01 and 62 `a`s, then `e`; the 65th character
    // (the cut) is a U+0301 on that `e`.
    let id = format!("\u{FB01}{}e\u{0301}\u{0323}bbb", "a".repeat(62));
    let e = lex(&id).unwrap_err();
    assert_eq!(e.kind, LexErrorKind::NonNfkcIdentifier);
    assert_eq!(
        e.diagnostic.notes[0],
        format!(
            "`\\u{{FB01}}{}…` ({} bytes) is not in NFKC (Unicode Normalization Form KC), and Sky never normalizes identifiers",
            "a".repeat(62),
            id.len()
        )
    );
    // A starter at the cut is not moved.
    let id = format!("\u{FB01}{}b", "a".repeat(64));
    let e = lex(&id).unwrap_err();
    assert!(e.diagnostic.notes[0].starts_with(&format!("`\\u{{FB01}}{}…`", "a".repeat(63))));
}
