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
    /// `'` directly after a name (`r'`, `r''`): the derivative mark of an
    /// `equation`. Anywhere else `'` is still an unexpected character (Sky's
    /// strings use `\"`).
    Prime,

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
    /// The glyph `ᵀ` (U+1D40 MODIFIER LETTER CAPITAL T; ASCII spelling is the
    /// stdlib function `transpose`).
    Transpose,
    /// The glyph `∂` (ASCII spelling is the stdlib function `d`).
    Partial,
    /// The glyph `∇` (ASCII spelling is the stdlib function `grad`).
    Nabla,
    /// `±` or `+/-`: `value ± uncertainty` (spec "Uncertainty"). The ASCII
    /// `+-` is not this token: it is `Plus` then `Minus`.
    PlusMinus,
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

    // Compound assignment. Lexed greedily: `x+=1` is `x`, `+=`, `1`, while
    // `a+ =b` (with a space) is `+` then `=`.
    /// `+=`
    PlusAssign,
    /// `-=`
    MinusAssign,
    /// `*=`
    StarAssign,
    /// `/=`
    SlashAssign,
    /// `^=`
    CaretAssign,
    /// `@=`
    AtAssign,

    Keyword(Kw),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kw {
    Def,
    Return,
    If,
    Elif,
    Else,
    While,
    For,
    In,
    /// `x as km/h`: show a value in a chosen unit (spec "Showing values in a unit").
    As,
    Import,
    Pass,
    Break,
    Continue,
    Equation,
    Solve,
    Field,
    And,
    Or,
    Not,
    True,
    False,
}

impl Kw {
    pub fn from_ident(s: &str) -> Option<Kw> {
        Some(match s {
            "def" => Kw::Def,
            "return" => Kw::Return,
            "if" => Kw::If,
            "elif" => Kw::Elif,
            "else" => Kw::Else,
            "while" => Kw::While,
            "for" => Kw::For,
            "in" => Kw::In,
            "as" => Kw::As,
            "import" => Kw::Import,
            "pass" => Kw::Pass,
            "break" => Kw::Break,
            "continue" => Kw::Continue,
            "equation" => Kw::Equation,
            "solve" => Kw::Solve,
            "field" => Kw::Field,
            "and" => Kw::And,
            "or" => Kw::Or,
            "not" => Kw::Not,
            "true" => Kw::True,
            "false" => Kw::False,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Kw::Def => "def",
            Kw::Return => "return",
            Kw::If => "if",
            Kw::Elif => "elif",
            Kw::Else => "else",
            Kw::While => "while",
            Kw::For => "for",
            Kw::In => "in",
            Kw::As => "as",
            Kw::Import => "import",
            Kw::Pass => "pass",
            Kw::Break => "break",
            Kw::Continue => "continue",
            Kw::Equation => "equation",
            Kw::Solve => "solve",
            Kw::Field => "field",
            Kw::And => "and",
            Kw::Or => "or",
            Kw::Not => "not",
            Kw::True => "true",
            Kw::False => "false",
        }
    }
}
