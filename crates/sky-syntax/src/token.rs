//! The token interface shared by the lexer and the parser.

use crate::Span;

/// One token. Its text, where relevant (identifiers, literals), is
/// `&src[token.span.range()]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenKind {
    // Literals and names. Text via `src[span]`.
    /// `42`, `1_000`. Underscores only between digits.
    Int,
    /// `3.14`, `1e-10`, `6.022_140_76e23`.
    Float,
    /// XID identifier, including Greek letters (`ψ`, `ħ`), unit names
    /// (`km`) and the stdlib functions `d`, `inv`, `grad`.
    Ident,
    /// Double-quoted string. The span includes both quotes; escapes are
    /// left as written (see [`crate::lexer::unescape_str`]).
    Str,

    // Layout.
    /// End of a logical line; the span covers `\n` or `\r\n`. At end of
    /// input without a final newline it is zero-width at `src.len()`.
    Newline,
    /// Zero-width, at the start of the first token of the indented line.
    Indent,
    /// Zero-width, at the start of the first token of the dedented line
    /// (or at `src.len()` for the dedents emitted at end of input).
    Dedent,
    /// Zero-width at `src.len()`. Always the last token.
    Eof,

    // Punctuation.
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Colon,
    /// `=`
    Assign,
    Dot,
    /// `->`
    Arrow,

    // Operators.
    Plus,
    Minus,
    Star,
    Slash,
    /// `^`
    Caret,
    /// `@` (matrix product)
    At,
    /// The glyph `⁻¹` (ASCII spelling is the stdlib function `inv`).
    Inverse,
    /// The glyph `∂` (ASCII spelling is the stdlib function `d`).
    Partial,
    /// The glyph `∇` (ASCII spelling is the stdlib function `grad`).
    Nabla,
    /// `==`
    Eq,
    /// `!=` or `≠`
    Ne,
    Lt,
    /// `<=` or `≤`
    Le,
    Gt,
    /// `>=` or `≥`
    Ge,

    Keyword(Kw),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kw {
    Def,
    Return,
    If,
    Else,
    For,
    In,
    Import,
    Equation,
    Solve,
    Field,
    And,
    Or,
    Not,
}

impl Kw {
    pub fn from_ident(s: &str) -> Option<Kw> {
        Some(match s {
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
            "and" => Kw::And,
            "or" => Kw::Or,
            "not" => Kw::Not,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Kw::Def => "def",
            Kw::Return => "return",
            Kw::If => "if",
            Kw::Else => "else",
            Kw::For => "for",
            Kw::In => "in",
            Kw::Import => "import",
            Kw::Equation => "equation",
            Kw::Solve => "solve",
            Kw::Field => "field",
            Kw::And => "and",
            Kw::Or => "or",
            Kw::Not => "not",
        }
    }
}
