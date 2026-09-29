//! The indentation-aware lexer.
//!
//! [`lex`] turns a whole source file into a token vector. It never produces
//! tokens for whitespace or comments: the lossless CST rebuilds those from the
//! gaps between token spans.
//!
//! # Span rules
//!
//! - Spans are half-open byte ranges, in increasing order, and never overlap:
//!   `tokens[i].span.end <= tokens[i + 1].span.start`.
//! - `Newline` covers the line terminator (`\n` or `\r\n`). Only the end of a
//!   logical line produces one: blank lines, comment-only lines, and line
//!   breaks inside `(` `)` / `[` `]` produce no token.
//! - `Indent` and `Dedent` are zero-width at the start of the first token of
//!   the line that changes the indentation.
//! - At end of input the lexer emits a `Newline` if the last logical line was
//!   not terminated (zero-width at `src.len()`), then one `Dedent` per open
//!   indentation level, then `Eof`, all zero-width at `src.len()`.
//!
//! # Errors and warnings
//!
//! The first error stops lexing; warnings (confusable identifiers) found
//! before it are kept in [`LexError::warnings`]. A leading UTF-8 BOM is
//! skipped. Rejected outright:
//!
//! - bidirectional control characters anywhere, including comments and
//!   strings (Trojan Source, spec v0.1 non-negotiable 4);
//! - a lone `\r` not followed by `\n`;
//! - invisible characters (every `Default_Ignorable_Code_Point` that is
//!   `XID_Start` or `XID_Continue`, e.g. ZERO WIDTH JOINER, HANGUL FILLER,
//!   VARIATION SELECTOR-17 U+E0100, plus a few non-identifier ones such as
//!   ZERO WIDTH SPACE) anywhere outside strings and comments, so at any
//!   position in an identifier, and `·` (U+00B7) inside identifiers;
//! - tabs and spaces mixed in one line's indentation, or an indented line
//!   whose indentation uses tabs where the enclosing block uses spaces (or
//!   the reverse);
//! - a dedent to a column that matches no enclosing block;
//! - unterminated strings, unknown escapes, unmatched or mismatched brackets;
//! - an identifier that is not in Unicode NFKC form (`e` + U+0301 instead
//!   of `é`, `ﬁ`, fullwidth `ｘ`, mathematical `𝑥`, MICRO SIGN `µ`). The
//!   source is never normalized, and keywords match the raw text only;
//!   the NFKC form appears only in the error's help;
//! - the glyph `±`: uncertainty is not part of Sky v0.1. The ASCII sequence
//!   `+-` is simply `Plus` then `Minus`, so `a+-b` is `a + (-b)` as in Python.

use std::collections::HashSet;

use sky_diag::Diagnostic;

use crate::confusable::check_identifier;
use crate::invisible_table::DEFAULT_IGNORABLE_IDENT;
use crate::nfkc;
use crate::source::find_bidi_control;
use crate::token::{Kw, Token, TokenKind};
use crate::{Span, MAX_SOURCE_BYTES};

/// A successfully lexed file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lexed {
    /// Always ends with [`TokenKind::Eof`].
    pub tokens: Vec<Token>,
    /// Warnings (confusable identifiers), in source order.
    pub warnings: Vec<Diagnostic>,
}

/// The error that stopped lexing.
///
/// The diagnostic is boxed so that `Result<_, LexError>` stays small on the
/// happy path; field access (`e.diagnostic.message`) works through `Deref`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexError {
    pub kind: LexErrorKind,
    pub diagnostic: Box<Diagnostic>,
    /// Warnings found before the error, in source order.
    pub warnings: Vec<Diagnostic>,
}

impl LexError {
    /// Warnings, then the error: the order `sky check` prints them in.
    pub fn diagnostics(&self) -> impl Iterator<Item = &Diagnostic> {
        self.warnings
            .iter()
            .chain(std::iter::once(&*self.diagnostic))
    }
}

/// What went wrong, for callers and tests that match on errors; the
/// human-readable details are in [`LexError::diagnostic`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LexErrorKind {
    BidiControl,
    SourceTooLarge,
    BareCarriageReturn,
    UnexpectedChar,
    /// An identifier that is not in Unicode NFKC form.
    NonNfkcIdentifier,
    /// An identifier with more than 30 combining marks in a row (the
    /// Unicode stream-safe limit).
    TooManyCombiningMarks,
    MixedIndentation,
    InconsistentDedent,
    InvalidNumber,
    UnterminatedString,
    InvalidEscape,
    UnmatchedBracket,
    MismatchedBracket,
    UnclosedBracket,
}

/// Lexes a whole source file. See the [module docs](self) for the rules.
pub fn lex(src: &str) -> Result<Lexed, LexError> {
    check_whole_file(src, MAX_SOURCE_BYTES)?;
    let mut lexer = Lexer::new(src);
    match lexer.run() {
        Ok(()) => Ok(Lexed {
            tokens: lexer.tokens,
            warnings: lexer.warnings,
        }),
        Err((kind, diagnostic)) => Err(LexError {
            kind,
            diagnostic,
            warnings: lexer.warnings,
        }),
    }
}

/// Checks that apply to the whole file before tokenizing: first that it is
/// at most `max_len` bytes ([`MAX_SOURCE_BYTES`] in [`lex`]), then that it
/// contains no bidi control. The size check must come first: spans are
/// `u32`, so the bidi scan (and every later span) relies on the size check
/// having kept offsets in range, and an oversized file should be refused
/// without scanning it. `max_len` is a parameter only so tests can check
/// this order without a 16 MiB input.
fn check_whole_file(src: &str, max_len: usize) -> Result<(), LexError> {
    if src.len() > max_len {
        let diag = Diagnostic::error(
            format!("source file is larger than {}", human_size(max_len)),
            Span::point(0),
        )
        .with_note(format!(
            "the file is {} bytes; the limit is {} bytes",
            src.len(),
            max_len
        ));
        return Err(LexError {
            kind: LexErrorKind::SourceTooLarge,
            diagnostic: Box::new(diag),
            warnings: Vec::new(),
        });
    }
    if let Some(b) = find_bidi_control(src) {
        let start = b.byte_offset as u32;
        let span = Span::new(start, start + b.ch.len_utf8() as u32);
        let diag = Diagnostic::error(
            format!(
                "bidirectional control character U+{:04X} is not allowed",
                b.ch as u32
            ),
            span,
        )
        .with_label("invisible bidi control here")
        .with_note("bidi controls can make source display differently from how it is parsed");
        return Err(LexError {
            kind: LexErrorKind::BidiControl,
            diagnostic: Box::new(diag),
            warnings: Vec::new(),
        });
    }
    Ok(())
}

/// `n` bytes in human form: `16 MiB`, `4 KiB`, or `8 bytes` (the largest unit
/// that divides `n` exactly, so the limit is never rounded).
fn human_size(n: usize) -> String {
    const UNITS: [(usize, &str); 3] = [(1 << 30, "GiB"), (1 << 20, "MiB"), (1 << 10, "KiB")];
    for (size, unit) in UNITS {
        if n >= size && n % size == 0 {
            return format!("{} {unit}", n / size);
        }
    }
    format!("{n} bytes")
}

/// Decodes the escapes in a [`TokenKind::Str`] token's text (quotes
/// included). Assumes the text came from the lexer, which has validated it.
pub fn unescape_str(text: &str) -> String {
    let inner = text
        .strip_prefix('"')
        .and_then(|t| t.strip_suffix('"'))
        .unwrap_or(text);
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('0') => out.push('\0'),
            Some(other) => out.push(other),
            None => {}
        }
    }
    out
}

/// Escapes the lexer accepts after `\` in a string.
const ESCAPES: &[char] = &['\\', '"', 'n', 't', 'r', '0'];

/// Default-ignorable characters the lexer rejects that are *not* in the
/// generated [`DEFAULT_IGNORABLE_IDENT`] table because they are neither
/// `XID_Start` nor `XID_Continue`. They could never be part of an identifier
/// anyway; listing them keeps them described as invisible (`U+XXXX`, with
/// the "invisible characters" help) wherever they appear outside strings and
/// comments. All are `Default_Ignorable_Code_Point` in Unicode 18.0.0.
///
/// Sorted, non-overlapping, and disjoint from the generated table (checked
/// by a test).
pub(crate) static INVISIBLE_EXTRA: &[(u32, u32)] = &[
    (0x00AD, 0x00AD), // SOFT HYPHEN (Cf)
    (0x180E, 0x180E), // MONGOLIAN VOWEL SEPARATOR (Cf)
    (0x200B, 0x200B), // ZERO WIDTH SPACE (Cf)
    (0x2060, 0x2064), // WORD JOINER, FUNCTION APPLICATION .. INVISIBLE PLUS (Cf)
    (0xFEFF, 0xFEFF), // ZERO WIDTH NO-BREAK SPACE / BOM (Cf); skipped only as a leading BOM
];

fn in_ranges(ranges: &[(u32, u32)], c: char) -> bool {
    let c = c as u32;
    ranges
        .binary_search_by(|&(lo, hi)| {
            if hi < c {
                std::cmp::Ordering::Less
            } else if lo > c {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// Characters that render as nothing and must never appear in source
/// outside strings and comments, and never in an identifier at any position.
///
/// This is the generated [`DEFAULT_IGNORABLE_IDENT`] table
/// (`Default_Ignorable_Code_Point` ∩ (`XID_Start` ∪ `XID_Continue`), Unicode
/// 18.0.0, from `crates/sky-diag/tools/gen_display_unsafe.py`) plus
/// [`INVISIBLE_EXTRA`]. The generated table includes ZWNJ/ZWJ (U+200C,
/// U+200D), the Hangul fillers (U+115F, U+1160, U+3164, U+FFA0), and all
/// variation selectors (U+180B..U+180D, U+180F, U+FE00..U+FE0F,
/// U+E0100..U+E01EF).
///
/// Every one of these, and every bidi control, must also be escaped by the
/// diagnostic renderer (`sky_diag::is_display_unsafe`); the test
/// `lexer_rejected_invisible_and_bidi_chars_are_display_unsafe` checks that
/// for every code point. The lexer must not use `is_display_unsafe` itself:
/// that set includes tab and the other C0 controls, which are legal in
/// source (tab as whitespace, the rest inside strings and comments).
fn is_invisible(c: char) -> bool {
    in_ranges(DEFAULT_IGNORABLE_IDENT, c) || in_ranges(INVISIBLE_EXTRA, c)
}

/// Characters the lexer bans from identifiers even though they are
/// `XID_Continue` (see [`is_ident_start_base`]): invisible characters, and
/// MIDDLE DOT / GREEK ANO TELEIA, which read as multiplication.
pub(crate) fn is_banned_in_identifiers(c: char) -> bool {
    is_invisible(c) || c == '\u{00B7}' || c == '\u{0387}'
}

/// Identifier start before [`NFKC_NON_IDENTIFIER`] is removed. Some
/// invisible characters are XID_Start (the Hangul fillers U+115F, U+1160,
/// U+3164, U+FFA0), so an identifier could otherwise start with, or consist
/// entirely of, an invisible character.
pub(crate) fn is_ident_start_base(c: char) -> bool {
    c == '_' || (unicode_ident::is_xid_start(c) && !is_banned_in_identifiers(c))
}

/// Identifier continue before [`NFKC_NON_IDENTIFIER`] is removed.
pub(crate) fn is_ident_continue_base(c: char) -> bool {
    unicode_ident::is_xid_continue(c) && !is_banned_in_identifiers(c)
}

/// Identifier characters (by [`is_ident_start_base`] /
/// [`is_ident_continue_base`]) whose NFKC form is not an identifier by those
/// same rules, because it contains a banned character: `Ŀ` (U+013F) and `ŀ`
/// (U+0140) are `L·` and `l·`. They are not identifier characters, so the
/// NFKC form of every identifier is an identifier.
///
/// This list is the single source for the lexer and for the test
/// `lexer::tests::nfkc_non_identifier_list_is_exactly_the_computed_set`,
/// which recomputes the set from the Unicode data and fails if it changes.
/// Sorted.
pub(crate) const NFKC_NON_IDENTIFIER: &[char] = &['\u{013F}', '\u{0140}'];

pub(crate) fn is_ident_start(c: char) -> bool {
    is_ident_start_base(c) && NFKC_NON_IDENTIFIER.binary_search(&c).is_err()
}

pub(crate) fn is_ident_continue(c: char) -> bool {
    is_ident_continue_base(c) && NFKC_NON_IDENTIFIER.binary_search(&c).is_err()
}

/// `'c' (U+XXXX)`, or just `U+XXXX` for characters that are unsafe or
/// meaningless to print (controls, whitespace, invisible characters).
fn describe(c: char) -> String {
    if c.is_control() || c.is_whitespace() || is_invisible(c) {
        format!("U+{:04X}", c as u32)
    } else {
        format!("'{c}' (U+{:04X})", c as u32)
    }
}

fn unexpected_char_help(c: char) -> Option<&'static str> {
    Some(match c {
        '\u{2212}' => "for subtraction, use ASCII `-`",
        '×' | '\u{22C5}' | '·' => "for multiplication, use `*` (or `@` for matrix product)",
        _ if NFKC_NON_IDENTIFIER.contains(&c) => {
            "its NFKC form is not a valid identifier (`ŀ` is `l·`, and `·` reads as multiplication)"
        }
        '÷' => "for division, use `/`",
        '±' => "uncertainty (±) is not supported in v0.1",
        '⁻' => "the inverse operator is the glyph `⁻¹` (or the function `inv`)",
        '!' => "for logical negation, use `not`; for \"not equal\", use `!=` or `≠`",
        '&' => "for logical and, use `and`",
        '|' => "for logical or, use `or`",
        '\'' | '‘' | '’' | '“' | '”' => "strings use ASCII double quotes: \"...\"",
        '\\' => "backslash line continuation is not supported; wrap the expression in ( )",
        '₀'..='₉' | '⁰' | '¹' | '²' | '³' | '⁴'..='⁹' => {
            "superscript and subscript digits are not part of identifiers or numbers"
        }
        _ if is_invisible(c) => "invisible characters are not allowed in source code",
        _ if c.is_whitespace() => "only ASCII spaces and tabs are whitespace in Sky",
        _ => return None,
    })
}

/// The diagnostic is boxed for the same reason as in [`LexError`]: every
/// helper returns this, and an unboxed `Diagnostic` would make each `Ok`
/// carry the size of the error.
type LexResult<T> = Result<T, (LexErrorKind, Box<Diagnostic>)>;

fn err<T>(kind: LexErrorKind, diagnostic: Diagnostic) -> LexResult<T> {
    Err((kind, Box::new(diagnostic)))
}

struct Lexer<'a> {
    src: &'a str,
    pos: usize,
    tokens: Vec<Token>,
    warnings: Vec<Diagnostic>,
    /// Indentation strings of the open blocks; `indents[0]` is `""`.
    indents: Vec<&'a str>,
    /// Open `(` / `[` with their spans.
    brackets: Vec<(char, Span)>,
    /// True when the next token starts a new logical line.
    at_line_start: bool,
    /// Identifiers already checked for confusables (warn once per name).
    checked_idents: HashSet<&'a str>,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self {
        Lexer {
            src,
            pos: 0,
            tokens: Vec::new(),
            warnings: Vec::new(),
            indents: vec![""],
            brackets: Vec::new(),
            at_line_start: true,
            checked_idents: HashSet::new(),
        }
    }

    fn peek(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn peek_nth(&self, n: usize) -> Option<char> {
        self.src[self.pos..].chars().nth(n)
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.pos += c.len_utf8();
            true
        } else {
            false
        }
    }

    fn offset(&self) -> u32 {
        self.pos as u32
    }

    fn span_from(&self, start: usize) -> Span {
        Span::new(start as u32, self.pos as u32)
    }

    fn push(&mut self, kind: TokenKind, span: Span) {
        self.tokens.push(Token { kind, span });
    }

    fn run(&mut self) -> LexResult<()> {
        self.eat('\u{FEFF}');
        loop {
            if self.at_line_start && self.brackets.is_empty() {
                let ws_start = self.pos;
                while matches!(self.peek(), Some(' ' | '\t')) {
                    self.pos += 1;
                }
                match self.peek() {
                    None => break,
                    // Blank or comment-only line: no tokens, indentation ignored.
                    Some('\n' | '\r') => {
                        self.newline()?;
                        continue;
                    }
                    Some('#') => {
                        self.skip_comment();
                        continue;
                    }
                    Some(_) => {
                        self.indentation(ws_start)?;
                        self.at_line_start = false;
                    }
                }
            }

            let start = self.pos;
            let Some(c) = self.bump() else { break };
            let kind = match c {
                ' ' | '\t' => continue,
                '#' => {
                    self.pos = start;
                    self.skip_comment();
                    continue;
                }
                '\n' | '\r' => {
                    self.pos = start;
                    let span = self.newline()?;
                    if self.brackets.is_empty() {
                        self.push(TokenKind::Newline, span);
                        self.at_line_start = true;
                    }
                    continue;
                }
                '0'..='9' => self.number(start)?,
                '"' => self.string(start)?,
                c if is_ident_start(c) => self.ident(start)?,
                '(' | '[' => {
                    self.brackets.push((c, self.span_from(start)));
                    if c == '(' {
                        TokenKind::LParen
                    } else {
                        TokenKind::LBracket
                    }
                }
                ')' | ']' => self.close_bracket(c, start)?,
                ',' => TokenKind::Comma,
                ':' => TokenKind::Colon,
                '.' => TokenKind::Dot,
                '*' => TokenKind::Star,
                '/' => TokenKind::Slash,
                '^' => TokenKind::Caret,
                '@' => TokenKind::At,
                '=' if self.eat('=') => TokenKind::Eq,
                '=' => TokenKind::Assign,
                '!' if self.eat('=') => TokenKind::Ne,
                '<' if self.eat('=') => TokenKind::Le,
                '<' => TokenKind::Lt,
                '>' if self.eat('=') => TokenKind::Ge,
                '>' => TokenKind::Gt,
                '+' => TokenKind::Plus,
                '-' if self.eat('>') => TokenKind::Arrow,
                '-' => TokenKind::Minus,
                '≤' => TokenKind::Le,
                '≥' => TokenKind::Ge,
                '≠' => TokenKind::Ne,
                '∂' => TokenKind::Partial,
                '∇' => TokenKind::Nabla,
                '⁻' if self.eat('¹') => TokenKind::Inverse,
                other => {
                    let mut diag = Diagnostic::error(
                        format!("unexpected character {}", describe(other)),
                        self.span_from(start),
                    );
                    if let Some(help) = unexpected_char_help(other) {
                        diag = diag.with_help(help);
                    }
                    return err(LexErrorKind::UnexpectedChar, diag);
                }
            };
            let span = self.span_from(start);
            self.push(kind, span);
        }

        if let Some(&(open, span)) = self.brackets.last() {
            let close = if open == '(' { ')' } else { ']' };
            return err(
                LexErrorKind::UnclosedBracket,
                Diagnostic::error(format!("unclosed `{open}`"), span)
                    .with_label(format!("this `{open}` is never closed"))
                    .with_help(format!("add a matching `{close}`")),
            );
        }
        let end = Span::point(self.src.len() as u32);
        if !self.at_line_start {
            self.push(TokenKind::Newline, end);
        }
        while self.indents.len() > 1 {
            self.indents.pop();
            self.push(TokenKind::Dedent, end);
        }
        self.push(TokenKind::Eof, end);
        Ok(())
    }

    /// Consumes `\n` or `\r\n` at `self.pos` and returns its span.
    fn newline(&mut self) -> LexResult<Span> {
        let start = self.pos;
        if self.eat('\r') && !self.eat('\n') {
            return err(
                LexErrorKind::BareCarriageReturn,
                Diagnostic::error(
                    "carriage return (U+000D) not followed by a line feed",
                    self.span_from(start),
                )
                .with_help("use `\\n` or `\\r\\n` line endings"),
            );
        }
        self.eat('\n');
        Ok(self.span_from(start))
    }

    /// Skips `#` up to (not including) the line terminator.
    fn skip_comment(&mut self) {
        let rest = &self.src[self.pos..];
        self.pos += rest.find(['\n', '\r']).unwrap_or(rest.len());
    }

    /// Handles the indentation `src[ws_start..pos]` of a logical line whose
    /// first token starts at `pos`.
    fn indentation(&mut self, ws_start: usize) -> LexResult<()> {
        let ws = &self.src[ws_start..self.pos];
        let ws_span = self.span_from(ws_start);
        if ws.contains(' ') && ws.contains('\t') {
            return err(
                LexErrorKind::MixedIndentation,
                Diagnostic::error("indentation mixes tabs and spaces", ws_span)
                    .with_label("tabs and spaces here")
                    .with_help("indent with spaces only, or with tabs only"),
            );
        }
        let top = *self.indents.last().expect("indent stack is never empty");
        if ws == top {
            return Ok(());
        }
        if !ws.is_empty() && !top.is_empty() && ws.as_bytes()[0] != top.as_bytes()[0] {
            let (this, block) = if ws.starts_with('\t') {
                ("tabs", "spaces")
            } else {
                ("spaces", "tabs")
            };
            return err(
                LexErrorKind::MixedIndentation,
                Diagnostic::error(
                    format!("indentation uses {this}, but the enclosing block uses {block}"),
                    ws_span,
                )
                .with_label(format!("indented with {this}"))
                .with_help("indent with spaces only, or with tabs only"),
            );
        }
        let here = Span::point(self.offset());
        if ws.len() > top.len() {
            // Same kind of whitespace and longer, so `top` is a prefix of `ws`.
            self.indents.push(ws);
            self.push(TokenKind::Indent, here);
            return Ok(());
        }
        while self.indents.last().is_some_and(|t| t.len() > ws.len()) {
            self.indents.pop();
            self.push(TokenKind::Dedent, here);
        }
        if *self.indents.last().expect("indent stack is never empty") != ws {
            return err(
                LexErrorKind::InconsistentDedent,
                Diagnostic::error(
                    "unindent does not match any outer indentation level",
                    ws_span,
                )
                .with_label(format!("dedented to column {}", ws.len() + 1))
                .with_help("dedent to the indentation of an enclosing block"),
            );
        }
        Ok(())
    }

    /// ASCII digits with single `_` separators between them.
    fn digits(&mut self) -> LexResult<()> {
        loop {
            match self.peek() {
                Some('0'..='9') => self.pos += 1,
                Some('_') if matches!(self.peek_nth(1), Some('0'..='9')) => self.pos += 1,
                Some('_') => {
                    let start = self.pos;
                    self.pos += 1;
                    return err(
                        LexErrorKind::InvalidNumber,
                        Diagnostic::error(
                            "`_` in a number must be between two digits",
                            self.span_from(start),
                        ),
                    );
                }
                _ => return Ok(()),
            }
        }
    }

    fn number(&mut self, start: usize) -> LexResult<TokenKind> {
        let mut kind = TokenKind::Int;
        self.digits()?;
        if self.peek() == Some('.') && matches!(self.peek_nth(1), Some('0'..='9')) {
            self.pos += 1;
            self.digits()?;
            kind = TokenKind::Float;
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            let sign = matches!(self.peek_nth(1), Some('+' | '-'));
            let digit_at = if sign { 2 } else { 1 };
            if matches!(self.peek_nth(digit_at), Some('0'..='9')) {
                self.pos += digit_at;
                self.digits()?;
                kind = TokenKind::Float;
            }
        }
        if self.peek().is_some_and(is_ident_continue) {
            while self.peek().is_some_and(is_ident_continue) {
                self.bump();
            }
            let span = self.span_from(start);
            return err(
                LexErrorKind::InvalidNumber,
                Diagnostic::error(
                    format!("invalid number literal `{}`", &self.src[span.range()]),
                    span,
                )
                .with_help("to give a number a unit, put a space before the unit: `10 km`"),
            );
        }
        Ok(kind)
    }

    fn string(&mut self, start: usize) -> LexResult<TokenKind> {
        loop {
            let at = self.pos;
            match self.peek() {
                None | Some('\n' | '\r') => {
                    return err(
                        LexErrorKind::UnterminatedString,
                        Diagnostic::error("unterminated string", self.span_from(start))
                            .with_label("string starts here and is not closed on this line")
                            .with_help("add a closing `\"`"),
                    );
                }
                Some('"') => {
                    self.pos += 1;
                    return Ok(TokenKind::Str);
                }
                Some('\\') => {
                    self.pos += 1;
                    match self.peek() {
                        // Let the unterminated-string branch report it.
                        None | Some('\n' | '\r') => {}
                        Some(e) if ESCAPES.contains(&e) => self.pos += 1,
                        Some(e) => {
                            self.bump();
                            return err(
                                LexErrorKind::InvalidEscape,
                                Diagnostic::error(
                                    format!("unknown escape `\\{}`", describe_escape(e)),
                                    self.span_from(at),
                                )
                                .with_help("valid escapes are \\\\ \\\" \\n \\t \\r \\0"),
                            );
                        }
                    }
                }
                Some(c) => self.pos += c.len_utf8(),
            }
        }
    }

    fn ident(&mut self, start: usize) -> LexResult<TokenKind> {
        while self.peek().is_some_and(is_ident_continue) {
            self.bump();
        }
        let text = &self.src[start..self.pos];
        // Keywords match the raw text only; the NFKC form never decides the
        // token kind (`ｉｆ` is an error, not `if`).
        if let Some(kw) = Kw::from_ident(text) {
            return Ok(TokenKind::Keyword(kw));
        }
        if !text.is_ascii() {
            match nfkc::check(text) {
                nfkc::Check::Nfkc => {}
                nfkc::Check::NotNfkc => {
                    return err(
                        LexErrorKind::NonNfkcIdentifier,
                        non_nfkc_diagnostic(text, self.span_from(start)),
                    );
                }
                nfkc::Check::TooManyNonStarters { at } => {
                    let mark = text[at..].chars().next().expect("offset of a character");
                    let lo = start + at;
                    return err(
                        LexErrorKind::TooManyCombiningMarks,
                        Diagnostic::error(
                            format!(
                                "too many combining marks in a row (at most {})",
                                nfkc::MAX_NON_STARTERS
                            ),
                            Span::new(lo as u32, (lo + mark.len_utf8()) as u32),
                        )
                        .with_label(format!(
                            "combining mark {} in a row",
                            nfkc::MAX_NON_STARTERS + 1
                        ))
                        .with_note(format!(
                            "an identifier may have at most {} combining marks (characters with a nonzero canonical combining class, counted after decomposition) in a row, the Unicode stream-safe limit (UAX #15)",
                            nfkc::MAX_NON_STARTERS
                        )),
                    );
                }
            }
        }
        if self.checked_idents.insert(text) {
            if let Some(finding) = check_identifier(text) {
                self.warnings.push(
                    Diagnostic::warning(
                        format!("identifier `{text}` {finding}"),
                        self.span_from(start),
                    )
                    .with_label("possibly confusable identifier"),
                );
            }
        }
        Ok(TokenKind::Ident)
    }

    fn close_bracket(&mut self, close: char, start: usize) -> LexResult<TokenKind> {
        let span = self.span_from(start);
        let expected_open = if close == ')' { '(' } else { '[' };
        match self.brackets.pop() {
            Some((open, _)) if open == expected_open => {}
            Some((open, open_span)) => {
                let expected = if open == '(' { ')' } else { ']' };
                return err(
                    LexErrorKind::MismatchedBracket,
                    Diagnostic::error(format!("mismatched `{close}`: expected `{expected}`"), span)
                        .with_label(format!("expected `{expected}`"))
                        .with_secondary(open_span, format!("`{open}` opened here")),
                );
            }
            None => {
                return err(
                    LexErrorKind::UnmatchedBracket,
                    Diagnostic::error(format!("unmatched `{close}`"), span)
                        .with_label("no matching opening bracket"),
                );
            }
        }
        Ok(if close == ')' {
            TokenKind::RParen
        } else {
            TokenKind::RBracket
        })
    }
}

/// `s` with every non-ASCII character written as the renderer writes an
/// escaped character (`\u{00E9}`), so differences that do not show when the
/// text is printed (a combining mark, a compatibility variant) are visible.
fn escape_non_ascii(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            out.push_str(&format!("\\u{{{:04X}}}", c as u32));
        }
    }
    out
}

/// How many characters of a string the non-NFKC error spells out.
const SHOWN_CHARS: usize = 64;
/// Longest identifier (in characters) that gets the full non-NFKC error:
/// the whole identifier quoted, its NFKC form computed and shown, and the
/// keyword check. Longer identifiers get an escaped prefix and their byte
/// length only, so the message for a 16 MiB identifier costs no
/// normalization and stays small (the check that found the error,
/// `nfkc::check`, normalizes only short segments). No longer identifier can
/// normalize to a keyword: keywords are at most 8 ASCII characters, every
/// character's NFKD is non-empty, and composition only produces non-ASCII
/// characters, so an ASCII NFKC form has at least as many characters as
/// the identifier.
const FULL_NFKC_HELP_MAX_CHARS: usize = 64;

/// [`escape_non_ascii`] of at most [`SHOWN_CHARS`] characters of `s` in
/// backticks, then how many were left out.
fn quote_escaped(s: &str) -> String {
    match s.char_indices().nth(SHOWN_CHARS) {
        None => format!("`{}`", escape_non_ascii(s)),
        Some((cut, _)) => format!(
            "`{}…` ({} more characters)",
            escape_non_ascii(&s[..cut]),
            s[cut..].chars().count()
        ),
    }
}

/// The help that spells out the NFKC form `n`, typeable: the characters
/// themselves, then their escapes (`write it as `μm` (`\u{03BC}m`)`). ASCII
/// needs no escapes. If `n` had a character the renderer escapes (none do
/// with the pinned data, since the lexer bans invisible characters), only
/// the escapes are shown. Not truncated: `n` is the NFKC form of at most
/// [`FULL_NFKC_HELP_MAX_CHARS`] characters, and one character's NFKC form
/// has at most 18 characters.
fn write_as_help(n: &str) -> String {
    if n.chars().any(sky_diag::is_display_unsafe) {
        format!("write it in NFKC form: {}", quote_escaped(n))
    } else if n.is_ascii() {
        format!("write it as `{n}`")
    } else {
        format!("write it as `{n}` (`{}`)", escape_non_ascii(n))
    }
}

/// Compatibility characters whose NFKC form is a different, specific letter
/// that scientific code means: `(char, its name, intended char, its name)`.
const INTENDED_LETTERS: &[(char, &str, char, &str)] = &[
    (
        '\u{00B5}',
        "MICRO SIGN",
        '\u{03BC}',
        "GREEK SMALL LETTER MU",
    ),
    ('\u{212A}', "KELVIN SIGN", 'K', "LATIN CAPITAL LETTER K"),
    (
        '\u{210F}',
        "PLANCK CONSTANT OVER TWO PI",
        '\u{0127}',
        "LATIN SMALL LETTER H WITH STROKE",
    ),
    (
        '\u{2126}',
        "OHM SIGN",
        '\u{03A9}',
        "GREEK CAPITAL LETTER OMEGA",
    ),
    (
        '\u{212B}',
        "ANGSTROM SIGN",
        '\u{00C5}',
        "LATIN CAPITAL LETTER A WITH RING ABOVE",
    ),
    ('\u{210E}', "PLANCK CONSTANT", 'h', "LATIN SMALL LETTER H"),
];

/// True if `s` would lex as one identifier (or keyword) token under the
/// lexer's own character rules.
fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    chars.next().is_some_and(is_ident_start) && chars.all(is_ident_continue)
}

/// The error for an identifier that is not in NFKC. The NFKC form is only
/// used to write the message; the source is never normalized.
fn non_nfkc_diagnostic(text: &str, span: Span) -> Diagnostic {
    let diag = Diagnostic::error("identifier is not in Unicode NFKC form", span)
        .with_label("not in NFKC form");
    let long_prefix = text.char_indices().nth(FULL_NFKC_HELP_MAX_CHARS);
    let Some((cut, at_cut)) = long_prefix else {
        return short_non_nfkc_diagnostic(diag, text);
    };
    // Do not cut between a letter and its combining marks: end the prefix
    // before the letter instead (at most 64 characters back).
    let cut = text[..cut + at_cut.len_utf8()]
        .char_indices()
        .rev()
        .find(|&(_, c)| !nfkc::is_non_starter(c))
        .map_or(cut, |(i, _)| if i == 0 { cut } else { i });
    let diag = diag
        .with_note(format!(
            "`{}…` ({} bytes) is not in NFKC (Unicode Normalization Form KC), and Sky never normalizes identifiers",
            escape_non_ascii(&text[..cut]),
            text.len()
        ))
        .with_note(format!(
            "the NFKC form is not shown for identifiers longer than {FULL_NFKC_HELP_MAX_CHARS} characters"
        ));
    intended_letter_help(diag, text)
}

/// The full non-NFKC error for an identifier of at most
/// [`FULL_NFKC_HELP_MAX_CHARS`] characters.
fn short_non_nfkc_diagnostic(diag: Diagnostic, text: &str) -> Diagnostic {
    let diag = diag.with_note(format!(
        "{} is not in NFKC (Unicode Normalization Form KC), and Sky never normalizes identifiers",
        quote_escaped(text)
    ));
    let normalized = nfkc::nfkc(text);
    // XID_Start/XID_Continue are closed under NFKC and the lexer bans the
    // exceptions (`NFKC_NON_IDENTIFIER`), so the NFKC form of an identifier
    // is always an identifier; tests check this for every identifier
    // character. No user-facing message for the impossible case.
    debug_assert!(
        is_identifier(&normalized),
        "NFKC form {normalized:?} of identifier {text:?} is not an identifier"
    );
    if let Some(kw) = Kw::from_ident(&normalized) {
        return diag.with_help(format!("did you mean `{}`", kw.as_str()));
    }
    intended_letter_help(diag, text).with_help(write_as_help(&normalized))
}

/// Help for each compatibility letter in `text` that has an intended letter
/// ([`INTENDED_LETTERS`]); at most one help per entry.
fn intended_letter_help(mut diag: Diagnostic, text: &str) -> Diagnostic {
    for &(c, name, intended, intended_name) in INTENDED_LETTERS {
        if text.contains(c) {
            diag = diag.with_help(format!(
                "{name} U+{:04X} is not a letter for identifiers; use {intended_name} U+{:04X} (`{intended}`)",
                c as u32, intended as u32
            ));
        }
    }
    diag
}

fn describe_escape(c: char) -> String {
    if c.is_control() || c.is_whitespace() || is_invisible(c) {
        format!("U+{:04X}", c as u32)
    } else {
        c.to_string()
    }
}

#[cfg(test)]
mod tests;
